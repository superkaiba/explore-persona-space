"""Smoke tests for ``scripts/workflow_lint.py``.

Asserts that the committed ``.claude/workflow.yaml`` lints cleanly so
the /issue HARD GATE (Phase A.0 of the restoration plan, see
``.claude/plans/restore-issue-skill-richness.md``) doesn't silently
regress. The lint covers schema validation, cross-reference
resolution, and AUTO-GENERATED fence-block alignment with SKILL.md
and markers.md.

Also covers the ``--check-asks`` mode: every ``AskUserQuestion``
mention in .claude/agents/**.md and .claude/skills/**/SKILL.md must
be anchored to a documented gate (task #372).

Also covers the ``--check-script-refs`` mode: every
``scripts/<name>.py`` reference in .claude/agents/**.md and
.claude/skills/**/SKILL.md must resolve to a real file under
``scripts/`` (dead-tool / invented-tool failure class).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    _MARKER_RECIPE_PINS,
    _ROOT_GUARD_HOOK,
    _ROOT_GUARD_SELFTEST_BLOCKED,
    BATCH_JUDGE_LEGACY_ALLOWLIST,
    HUB_VERIFY_LEGACY_ALLOWLIST,
    SKILL_REF_ALLOWLIST,
    SKILL_REF_FS_ROOTS,
    UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST,
    UPLOAD_PREFIX_CLOBBER_ALLOWLIST,
    _head_is_main,
    _iter_ask_target_files,
    _live_skill_names,
    _other_worktree_prefix,
    _run_root_guard,
    _tracked_on_main_ref,
    _values_equal,
    check_agent_model_pins,
    check_agents_note_argv_verdict,
    check_asks,
    check_autonomous_asks,
    check_awk_elision_parity,
    check_bare_commit_pathspec,
    check_batch_judge_client,
    check_compute_shape_review_lens,
    check_crash_fix_relaunch_contract,
    check_dispatcher_cvd_pin,
    check_gate_ids_unique,
    check_git_recipes_root_guard,
    check_grep_qv,
    check_heredoc_dotenv,
    check_hollow_verification_gate_review_lens,
    check_hub_dir_filecount_guard,
    check_hub_verify_retry,
    check_lessons_index,
    check_long_loop_restartability_review_lens,
    check_marker_recipe_snippets,
    check_marker_registry,
    check_marker_scalar_integrity,
    check_no_literal_round_marker_versions,
    check_no_workflow_improver_spawn,
    check_pipe_python,
    check_piped_git_push,
    check_poller_marker_consumers,
    check_push_failure_swallow,
    check_rule_frontmatter_parses,
    check_script_references,
    check_section_reference_pointer_coverage,
    check_sh_function_rc_capture,
    check_skill_bang_backtick,
    check_skill_references,
    check_smoke_architecture_review_lens,
    check_smoke_output_hygiene,
    check_stale_label_disposition_clause,
    check_upload_as_file,
    check_upload_file_in_loop,
    check_upload_prefix_clobber,
    check_vm_thread_cap_guidance,
    check_wandb_required,
)

from explore_persona_space.workflow import (  # noqa: E402
    MarkerEntry,
    WorkflowYaml,
    load_workflow_yaml,
)


def _run(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_LINT), *flags],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_workflow_lint_default_exits_zero():
    """No-args invocation must succeed (schema check + bundled
    script-reference check on the committed tree)."""
    result = _run()
    assert result.returncode == 0, (
        f"workflow_lint default failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_references_exits_zero():
    """The HARD GATE: every ``(see workflow.yaml § X)`` reference in
    CLAUDE.md / SKILL.md / markers.md must resolve to a real key. This
    is the gate that Phase A's restored SKILL.md depends on; if it
    regresses, the restored cross-refs are dangling."""
    result = _run("--check-references")
    assert result.returncode == 0, (
        f"workflow_lint --check-references failed (HARD GATE regressed):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_tables_exits_zero():
    """The AUTO-GENERATED fence blocks in SKILL.md and markers.md must
    match the renderer's output (no hand-edits inside the fences)."""
    result = _run("--check-tables")
    assert result.returncode == 0, (
        f"workflow_lint --check-tables failed (AUTO-GENERATED tables drifted):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_asks_repo_passes():
    """Repo-level check: the committed agent + skill specs must already
    satisfy the auto-continuation contract. If this fails, the audit
    cleanup from task #372 has regressed (someone added a bare
    AskUserQuestion mention outside any gate)."""
    result = _run("--check-asks")
    assert result.returncode == 0, (
        f"workflow_lint --check-asks failed at repo scope:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_autonomous_asks_repo_passes():
    """Repo-level check: every committed AskUserQuestion mention in
    .claude/skills/issue/SKILL.md + .claude/agents/*.md must document
    its autonomous-mode behavior. If this fails, the #503/#504/#505
    closure has regressed (someone added an AskUserQuestion without an
    autonomous-mode auto-resolve / skip / block-and-fail annotation)."""
    result = _run("--check-autonomous-asks")
    assert result.returncode == 0, (
        f"workflow_lint --check-autonomous-asks failed at repo scope:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# Unit tests for the ``check_asks`` function (task #372).
# Each case writes a tiny markdown file under ``tmp_path``, calls
# ``check_asks(workflow, roots=[tmp_path])``, and inspects the error
# list. PASS = empty list; FAIL = at least one error string.
# ─────────────────────────────────────────────────────────────────────


def _workflow():
    return load_workflow_yaml(_REPO_ROOT / ".claude" / "workflow.yaml")


def test_check_asks_pass_inline_gate_annotation(tmp_path):
    """PASS — line carries an inline ``<!-- gate: gates.plan_approval -->``
    annotation that resolves to a real workflow.yaml gate."""
    (tmp_path / "SKILL.md").write_text(
        "Use `AskUserQuestion` for plan approval. <!-- gate: gates.plan_approval -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_pass_gate_annotation_line_above(tmp_path):
    """PASS — annotation on the line immediately above the mention."""
    (tmp_path / "SKILL.md").write_text(
        "<!-- gate: gates.experiment_goal -->\n"
        "Ask via `AskUserQuestion`: what is the one-sentence Goal?\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_fail_unannotated(tmp_path):
    """FAIL — bare ``AskUserQuestion`` mention with no annotation, no
    anti-pattern marker, and no gate citation in the paragraph."""
    (tmp_path / "SKILL.md").write_text(
        "Whenever you feel like it, just use `AskUserQuestion` and the user will reply.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "bare 'AskUserQuestion'" in errors[0]


def test_check_asks_fail_nonexistent_gate_key(tmp_path):
    """FAIL — ``<!-- gate: ... -->`` annotation references a key that
    does NOT resolve in workflow.yaml § gates."""
    (tmp_path / "SKILL.md").write_text(
        "Use `AskUserQuestion`. <!-- gate: gates.NONEXISTENT_GATE -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "does not" in errors[0] and "resolve" in errors[0]


def test_check_asks_pass_anti_pattern_marker(tmp_path):
    """PASS — paragraph carries the ``<!-- example: anti-pattern -->``
    marker, signalling this is documentation of misuse, not a live call
    site."""
    (tmp_path / "SKILL.md").write_text(
        "<!-- example: anti-pattern -->\n"
        "Do NOT use `AskUserQuestion` outside the documented gates.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_pass_existing_workflow_yaml_citation(tmp_path):
    """PASS — paragraph already cites a gate via the existing
    ``(see workflow.yaml § gates.X)`` convention; no need to also stamp
    a redundant ``<!-- gate: ... -->`` annotation."""
    (tmp_path / "SKILL.md").write_text(
        "The clarifier gate (see workflow.yaml § gates.clarifier_blocking)\n"
        "is implemented by asking the user via `AskUserQuestion`.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_mixed_file_passes_and_fails(tmp_path):
    """Multi-mention file: properly annotated mentions PASS, bare
    mentions FAIL with line-specific errors."""
    (tmp_path / "SKILL.md").write_text(
        # line 1: PASS via gate annotation
        "Use `AskUserQuestion` here. <!-- gate: gates.plan_approval -->\n"
        # line 2: PASS via anti-pattern marker on line above
        "<!-- example: anti-pattern -->\n"
        "Do NOT call `AskUserQuestion` outside gates.\n"
        # line 4: blank
        "\n"
        # line 5: FAIL — bare, no annotation, no citation
        "Stray `AskUserQuestion` mention without anchor.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly 1 error, got: {errors}"
    assert ":5:" in errors[0]


def test_check_asks_pass_anti_pattern_marker_after_mention(tmp_path):
    """The anti-pattern marker MUST be at or above the mention — markers
    that appear AFTER the mention do not anchor it. This test guards
    against a regression where the lookback window is accidentally
    flipped to a look-ahead."""
    (tmp_path / "SKILL.md").write_text(
        "Stray `AskUserQuestion` mention with marker below.\n<!-- example: anti-pattern -->\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert ":1:" in errors[0]


def test_check_asks_pass_citation_below_mention_same_paragraph(tmp_path):
    """Rule 3 scans forward within the same paragraph too: a
    ``workflow.yaml § gates.X`` citation BELOW the mention (but still in
    the same prose paragraph, bounded by blank lines) anchors it. This
    is the case for prose like ``ask the user via X (see workflow.yaml §
    gates.Y for the gate)`` where the parenthetical lands on the next
    wrapped line."""
    (tmp_path / "SKILL.md").write_text(
        "Ask the user via `AskUserQuestion` for plan approval\n"
        "(see workflow.yaml § gates.plan_approval).\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_asks_fail_citation_in_next_paragraph(tmp_path):
    """Rule 3's forward scan STOPS at paragraph boundaries: a citation
    that appears after a blank line does NOT anchor the mention. Without
    this guard, a single citation could anchor every AskUserQuestion in
    the rest of the document."""
    (tmp_path / "SKILL.md").write_text(
        "Stray `AskUserQuestion` mention.\n"
        "\n"
        "Unrelated next paragraph (see workflow.yaml § gates.plan_approval).\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert ":1:" in errors[0]


def test_check_asks_pass_bare_citation_without_parens(tmp_path):
    """Rule 3's permissive regex also accepts the bare-prose form
    ``workflow.yaml § gates.X`` (no opening paren), used by existing
    documentation like Step 0c's "gate #6 — see workflow.yaml §
    gates.inline" preamble. Without this, prose that already references
    a gate would need a redundant ``<!-- gate: -->`` stamp."""
    (tmp_path / "SKILL.md").write_text(
        "This is a legitimate `AskUserQuestion` use because the gate IS a\n"
        "gate (see workflow.yaml § gates.experiment_goal). It does not\n"
        "violate the auto-continuation policy.\n"
    )
    errors = check_asks(_workflow(), roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


# ---------------------------------------------------------------------------
# Unit tests for the ``check_autonomous_asks`` function (proposal #4,
# 2026-06-06). Each case writes a tiny markdown file under ``tmp_path``,
# calls ``check_autonomous_asks(roots=[tmp_path])``, and inspects the
# error list.
# ---------------------------------------------------------------------------


def test_check_autonomous_asks_pass_interactive_mode_keyword(tmp_path):
    """The literal phrase 'Interactive mode' anywhere in the section block
    satisfies the rule."""
    (tmp_path / "SKILL.md").write_text(
        "**Interactive mode** (user is in chat): raise `AskUserQuestion`\nand wait for reply.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_env_keyword(tmp_path):
    """The literal 'EPM_AUTONOMOUS_SESSION' anywhere in the section block
    satisfies the rule."""
    (tmp_path / "SKILL.md").write_text(
        "With `EPM_AUTONOMOUS_SESSION=1`, auto-resolve; else raise\n"
        "`AskUserQuestion` for the user.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_annotation_auto_resolve(tmp_path):
    """The `<!-- autonomous-mode: auto-resolve -->` annotation in the
    same section block satisfies the rule."""
    (tmp_path / "SKILL.md").write_text(
        "Raise `AskUserQuestion` <!-- autonomous-mode: auto-resolve -->\nto pick the option.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_annotation_skip(tmp_path):
    """The `<!-- autonomous-mode: skip -->` annotation also satisfies."""
    (tmp_path / "SKILL.md").write_text("Raise `AskUserQuestion` <!-- autonomous-mode: skip -->\n")
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_annotation_block_and_fail(tmp_path):
    """The `<!-- autonomous-mode: block-and-fail -->` annotation also satisfies."""
    (tmp_path / "SKILL.md").write_text(
        "Raise `AskUserQuestion` <!-- autonomous-mode: block-and-fail -->\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_annotation_gate_allowed(tmp_path):
    """The `<!-- autonomous-mode: gate-allowed -->` annotation also satisfies."""
    (tmp_path / "SKILL.md").write_text(
        "Raise `AskUserQuestion` <!-- autonomous-mode: gate-allowed -->\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_fail_unannotated(tmp_path):
    """A bare `AskUserQuestion` mention with no autonomous-mode keyword
    or annotation in the section FAILs the check."""
    (tmp_path / "SKILL.md").write_text("Raise `AskUserQuestion` to pick the option.\n")
    errors = check_autonomous_asks(roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "missing autonomous-mode documentation" in errors[0]


def test_check_autonomous_asks_fail_invalid_annotation_value(tmp_path):
    """The annotation must be one of the four valid values; a typo'd
    action (e.g. `auto-pick` instead of `auto-resolve`) FAILs."""
    (tmp_path / "SKILL.md").write_text(
        "Raise `AskUserQuestion` <!-- autonomous-mode: auto-pick -->\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"


def test_check_autonomous_asks_pass_anti_pattern_exempt(tmp_path):
    """`<!-- example: anti-pattern -->` paragraphs are documentation,
    not real call sites — same exemption as ``check_asks``."""
    (tmp_path / "SKILL.md").write_text(
        "Do not raise `AskUserQuestion` <!-- example: anti-pattern -->\nfor design forks.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_pass_keyword_above_via_wider_bounds(tmp_path):
    """The wider section bounds walk back to the nearest blank line
    above (uncapped), so a parent paragraph saying 'Interactive mode'
    satisfies a sub-bullet's `AskUserQuestion` mention."""
    (tmp_path / "SKILL.md").write_text(
        "**Interactive mode** (user is in chat). The orchestrator\n"
        "branches on session mode.\n"
        "- Sub-bullet 1: do thing A.\n"
        "- Sub-bullet 2: raise `AskUserQuestion` to confirm.\n"
        "- Sub-bullet 3: post the marker.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_autonomous_asks_stops_at_header_boundary(tmp_path):
    """The forward walk stops at the next markdown header so we don't
    leak into the next section's content."""
    (tmp_path / "SKILL.md").write_text(
        "Raise `AskUserQuestion` to confirm.\n"
        "### Next section heading\n"
        "Interactive mode handling here doesn't help the section above.\n"
    )
    errors = check_autonomous_asks(roots=[tmp_path])
    assert len(errors) == 1, f"expected 1 error, got: {errors}"


# ---------------------------------------------------------------------------
# Unit tests for ``check_script_references`` (dead-tool / invented-tool
# failure class). Each case writes a tiny markdown file under ``tmp_path``
# referencing ``scripts/<name>.py`` and a fixture ``scripts/`` dir, then
# calls ``check_script_references(roots=[tmp_path], scripts_dir=...)``.
# ---------------------------------------------------------------------------


def test_check_script_refs_pass_existing_script(tmp_path):
    """A reference to a script that exists under scripts/ PASSes."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "task.py").write_text("# real helper\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("Run `uv run python scripts/task.py find <N>`.\n")
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_script_refs_fail_dangling_script(tmp_path):
    """A reference to a script that does NOT exist under scripts/ FAILs."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text(
        "Before provisioning, run `scripts/hf_gate_accept.py --from-plan P`.\n"
    )
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "scripts/hf_gate_accept.py" in errors[0]
    assert "does not exist" in errors[0]
    assert "SKILL.md:1" in errors[0]


def test_check_script_refs_mixed_good_and_dangling(tmp_path):
    """A file with one good and one dangling reference reports only the
    dangling one."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "pod.py").write_text("# real helper\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "agent.md").write_text(
        "Good: `scripts/pod.py provision`.\nBad: `scripts/sample_outputs.py --n 3`.\n"
    )
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "scripts/sample_outputs.py" in errors[0]
    assert "agent.md:2" in errors[0]


def test_check_script_refs_does_not_match_other_prefixes(tmp_path):
    """A path like `my_scripts/foo.py` is NOT a `scripts/foo.py` reference
    and must not be flagged."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("See `external/my_scripts/foo.py` for details.\n")
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert errors == [], f"expected PASS (non-scripts/ prefix), got: {errors}"


def test_check_script_refs_historical_opt_out_passes(tmp_path):
    """A dead reference on a line carrying the `<!-- lint: historical-ref -->`
    opt-out comment is a narrative incident citation and must NOT be
    flagged (task #545: second hit of the incident-citation class)."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "agent.md").write_text(
        "(Incident #528: the branch-only `scripts/run_experiment_528.py` "
        "dispatcher silently skipped phase 2.) <!-- lint: historical-ref -->\n"
    )
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert errors == [], f"expected PASS (opted-out historical ref), got: {errors}"


def test_check_script_refs_opt_out_is_per_line(tmp_path):
    """The opt-out covers ONLY its own line: a dead reference on another
    line of the same file still FAILs, and the error names the opt-out."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "agent.md").write_text(
        "(Incident: `scripts/dead_dispatcher.py` ate a phase.) "
        "<!-- lint: historical-ref -->\n"
        "Then run `scripts/dead_dispatcher.py --resume`.\n"
    )
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "scripts/dead_dispatcher.py" in errors[0]
    assert "agent.md:2" in errors[0]
    assert "<!-- lint: historical-ref -->" in errors[0]


def test_check_script_refs_repo_tree_is_clean():
    """The committed .claude/ tree must carry no dangling script
    references — this is the regression guard the durable fix installs."""
    errors = check_script_references()
    assert errors == [], (
        "committed .claude/ agents/skills reference scripts that do not "
        "exist under scripts/:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Staleness-aware degrade for check_script_references / check_skill_references
# (#1622/#1672): inside a stale non-``main`` worktree, a reference whose
# target is missing locally but tracked at main/origin/main WARNs instead of
# FAILing (the Step 5a sync deliberately never syncs scripts/ helpers).
# Test notes: the two production-mode tests
# ``test_check_script_refs_repo_tree_is_clean`` (above) and
# ``test_check_skill_refs_repo_tree_is_clean`` (below) call the checks with
# NO fixture args, so post-#1672 they become degrade-ACTIVE when Step 9c runs
# them inside a worktree (a target missing locally but present on main WARNs
# instead of FAILing) and stay fully strict on the main checkout — intended
# behavior (#1672 plan §7 risk 1): merge-time strictness lives in the
# Step 10d landing-tree gate (non-git tree ⇒ strict) plus the main-checkout
# run of these same tests.
# ---------------------------------------------------------------------------


def test_check_script_refs_stale_tree_present_on_main_warns(tmp_path):
    """A dangling ref whose target IS on main degrades to one WARN, no error
    (#1622/#1672 stale-worktree tolerance); the WARN names the script, the
    main ref, and the incident."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("Run `uv run python scripts/plan_patch.py <draft>`.\n")
    sink: list[str] = []
    errors = check_script_references(
        roots=[docs], scripts_dir=scripts_dir, main_probe=lambda n: True, warn_sink=sink
    )
    assert errors == [], f"expected WARN degrade, got errors: {errors}"
    assert len(sink) == 1, f"expected exactly one WARN, got: {sink}"
    assert "scripts/plan_patch.py" in sink[0]
    assert "main" in sink[0]
    assert "#1622" in sink[0]


def test_check_script_refs_stale_tree_absent_on_main_fails(tmp_path):
    """A dangling ref whose target is ALSO absent on main still FAILs with
    the unchanged strict message — the degrade is not a bypass."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("Run `scripts/zz_invented_helper.py --go`.\n")
    sink: list[str] = []
    errors = check_script_references(
        roots=[docs], scripts_dir=scripts_dir, main_probe=lambda n: False, warn_sink=sink
    )
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "scripts/zz_invented_helper.py" in errors[0]
    assert "does not exist" in errors[0]
    assert sink == [], f"expected no WARN, got: {sink}"


def test_check_script_refs_fixture_mode_never_probes_ambiently(tmp_path):
    """Fixture-scoped calls (scripts_dir set, no probe injected) stay strict
    even when the referenced script exists on main and pytest runs inside a
    non-main worktree: the ambient degrade is gated on production scope
    (``scripts_dir is None``), keeping existing unit tests hermetic."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    # scripts/task.py exists on main — if the ambient probe fired in fixture
    # mode this would WARN instead of erroring.
    (docs / "SKILL.md").write_text("Run `uv run python scripts/task.py view 1`.\n")
    errors = check_script_references(roots=[docs], scripts_dir=scripts_dir)
    assert len(errors) == 1, f"expected one strict error (fixture mode), got: {errors}"
    assert "scripts/task.py" in errors[0]


def test_head_is_main_fail_safe_nongit(tmp_path):
    """A non-git tree (the Step 10d /tmp landing-tree gate copy) reads as
    'main' — i.e. the STRICT regime — by fail-safe."""
    assert _head_is_main(tmp_path) is True


def test_head_is_main_git_fixture(tmp_path):
    """On a real git fixture: True on branch main, False on a feature branch."""

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.email=t@e.st", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-b", "main")
    git("commit", "--allow-empty", "-m", "root")
    assert _head_is_main(tmp_path) is True
    git("checkout", "-b", "feature")
    assert _head_is_main(tmp_path) is False


def test_tracked_on_main_ref_git_fixture(tmp_path):
    """A committed path on a fixture repo's main resolves True; a missing
    path resolves False. Fresh tmp repo per test so the lru_cache key
    (relpath, repo_root) never collides across tests."""
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), "-c", "user.email=t@e.st", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-b", "main")
    (root / "scripts").mkdir()
    (root / "scripts" / "x.py").write_text("# helper\n")
    git("add", "scripts/x.py")
    git("commit", "-m", "add x")
    assert _tracked_on_main_ref("scripts/x.py", str(root)) is True
    assert _tracked_on_main_ref("scripts/y.py", str(root)) is False


def test_check_skill_refs_stale_tree_present_on_main_warns(tmp_path):
    """A /skill token unresolved locally but present on main degrades to one
    WARN, no error — the skill-refs sibling of the script-refs degrade."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/ghost-skill` here.\n")
    sink: list[str] = []
    errors = check_skill_references(
        roots=[docs],
        skills_dir=skills,
        allowlist=frozenset(),
        main_probe=lambda r: True,
        warn_sink=sink,
    )
    assert errors == [], f"expected WARN degrade, got errors: {errors}"
    assert len(sink) == 1, f"expected exactly one WARN, got: {sink}"
    assert "/ghost-skill" in sink[0]
    assert "main" in sink[0]
    assert "#1622" in sink[0]


def test_check_skill_refs_stale_tree_absent_on_main_fails(tmp_path):
    """A /skill token absent locally AND on main still FAILs with the
    unchanged strict message."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/ghost-skill` here.\n")
    sink: list[str] = []
    errors = check_skill_references(
        roots=[docs],
        skills_dir=skills,
        allowlist=frozenset(),
        main_probe=lambda r: False,
        warn_sink=sink,
    )
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/ghost-skill" in errors[0]
    assert "SKILL_REF_ALLOWLIST" in errors[0]
    assert sink == [], f"expected no WARN, got: {sink}"


def test_check_script_refs_ambient_wiring_degrade_active(tmp_path, monkeypatch, capsys):
    """Pins the AMBIENT production wiring line (``main_probe is None and
    scripts_dir is None and not _head_is_main(...)``): with a non-main HEAD
    and a main-tracked target (both monkeypatched), a dangling ref in an
    ambient call (scripts_dir=None) WARNs to stderr and returns no error —
    so a later refactor cannot silently retire the degrade while every
    injected-probe test stays green."""
    import workflow_lint as wl

    monkeypatch.setattr(wl, "_head_is_main", lambda _root: False)
    monkeypatch.setattr(wl, "_tracked_on_main_ref", lambda _rel, _root: True)
    docs = tmp_path / "docs"
    docs.mkdir()
    # The fixture name must not exist under the real scripts/ dir.
    (docs / "SKILL.md").write_text("Run `scripts/zz_nonexistent_1672.py` now.\n")
    errors = check_script_references(roots=[docs])
    assert errors == [], f"expected ambient WARN degrade, got errors: {errors}"
    err = capsys.readouterr().err
    assert "WARN:" in err
    assert "scripts/zz_nonexistent_1672.py" in err


# ---------------------------------------------------------------------------
# Unit tests for ``check_skill_references`` (skill-rename / skill-retirement
# rot class, #713/#714). Each case writes a tiny markdown file under
# ``tmp_path`` carrying a backtick-delimited ``/<skill-name>`` slash-command
# token plus a fixture ``skills/`` dir, then calls
# ``check_skill_references(roots=[...], skills_dir=..., allowlist=...)``.
# Mirrors the ``check_script_references`` block above, swapping the
# scripts_dir hook for skills_dir + an explicit allowlist override.
# ---------------------------------------------------------------------------


def test_check_skill_refs_pass_live_skill(tmp_path):
    """A `/skill` token resolving to a live skill dir under skills/ PASSes."""
    skills = tmp_path / "skills"
    (skills / "weekly").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("Run `/weekly` on Wednesdays.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (live skill dir), got: {errors}"


def test_check_skill_refs_pass_live_skill_with_args(tmp_path):
    """A `/skill <args>` invocation matches only the skill name (the trailing
    lookahead closes on whitespace) and resolves to the live dir."""
    skills = tmp_path / "skills"
    (skills / "issue").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Boot with `/issue <N>` then `/issue 137 --auto`.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (args-form, live skill), got: {errors}"


def test_check_skill_refs_pass_dir_without_skill_md(tmp_path):
    """Resolution is by skill DIRECTORY existence, not `*/SKILL.md`:
    clean-results is a live skill dir with no SKILL.md (SPEC.md/exemplars/),
    so `/clean-results` must PASS."""
    skills = tmp_path / "skills"
    (skills / "clean-results").mkdir(parents=True)
    (skills / "clean-results" / "SPEC.md").write_text("spec\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("See `/clean-results`.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (dir without SKILL.md), got: {errors}"


def test_check_skill_refs_fail_unresolved(tmp_path):
    """A `/skill` token resolving neither to a live dir nor the allowlist FAILs
    with a file:line-anchored error naming the token."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SKILL.md").write_text("Run `/ghost-skill` here.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/ghost-skill" in errors[0]
    assert "SKILL.md:1" in errors[0]


def test_check_skill_refs_pass_allowlist_exact(tmp_path):
    """An allowlisted exact token (user-global / builtin command) PASSes even
    with an empty skills dir."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Self-pass `/humanize quick` before returning.\n")
    errors = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset({"humanize"})
    )
    assert errors == [], f"expected PASS (allowlisted exact token), got: {errors}"


def test_check_skill_refs_pass_namespace_prefix(tmp_path):
    """An allowlisted `<plugin>:` namespace prefix resolves every
    `/<plugin>:<member>` without per-member enumeration."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/codex:rescue` then `/codex:setup`.\n")
    errors = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset({"codex:"})
    )
    assert errors == [], f"expected PASS (namespace prefix), got: {errors}"


def test_check_skill_refs_fail_unallowlisted_namespace(tmp_path):
    """A plugin-namespaced token whose prefix is NOT allowlisted FAILs."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/other:thing`.\n")
    errors = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset({"codex:"})
    )
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/other:thing" in errors[0]


def test_check_skill_refs_pass_namespace_on_disk(tmp_path):
    """Forward-compat: a colon-named skill DIRECTORY resolves a namespaced
    token even with no allowlist prefix."""
    skills = tmp_path / "skills"
    (skills / "code-review:code-review").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/code-review:code-review`.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (namespace dir on disk), got: {errors}"


def test_check_skill_refs_does_not_match_path_token(tmp_path):
    """The central FP control: a backticked PATH (`/workspace/logs/x`, trailing
    `/`) and bare prose paths (`/tmp/foo`, `/mnt/eps`, no backtick) must NOT
    match — zero false positives."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Sentinel at `/workspace/logs/x` and bare /tmp/foo and /mnt/eps.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (path tokens not matched), got: {errors}"


def test_check_skill_refs_ignores_fenced_code(tmp_path):
    """Lines inside fenced code blocks (``` / ~~~) are skipped: an HTML close
    tag / sed / regex fragment carrying `/ghost` must NOT FAIL."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("```\n`/ghost` inside a fence\nsed -n '/^---$/p'\n```\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (fenced code skipped), got: {errors}"


def test_check_skill_refs_historical_opt_out(tmp_path):
    """A dead `/skill` ref on a line carrying the shared
    `<!-- lint: historical-ref -->` opt-out is a one-off narrative citation
    and must NOT FAIL."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Retired `/oldskill`. <!-- lint: historical-ref -->\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (opted-out historical ref), got: {errors}"


def test_check_skill_refs_opt_out_is_per_line(tmp_path):
    """The opt-out covers ONLY its own line: a dead ref on another line of the
    same file still FAILs and the error names the second line."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "Retired `/oldskill`. <!-- lint: historical-ref -->\nThen run `/oldskill` again.\n"
    )
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "a.md:2" in errors[0]


def test_check_skill_refs_mixed_good_and_dangling(tmp_path):
    """A file with one resolving and one dangling ref reports only the
    dangling one, anchored to its line."""
    skills = tmp_path / "skills"
    (skills / "issue").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Good `/issue`.\nBad `/ghost`.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/ghost" in errors[0]
    assert "a.md:2" in errors[0]


def test_check_skill_refs_no_fp_on_codespan_closing_backtick_slash(tmp_path):
    """Regression guard (#814): the closing-backtick-mistaken-for-opening FP.

    Prose like ``` `false`/unset uploads nothing``` writes a ``` `false` ```
    codespan whose CLOSING backtick immediately abuts ``/unset``. Before the
    `(?<!\\w)` negative-lookbehind, SKILL_REF_RE misread that closing backtick
    as the OPENING backtick of a phantom ``` `/unset ``` slash-command and the
    line FAILed with a spurious "unresolved skill reference `/unset`" (the
    surfaced `main` regression this fix resolves — verbatim from
    `.claude/rules/upload-policy.md:171`). The tightened regex must NOT match:
    a backtick abutting a word char (the codespan's closing `` ` ``) cannot
    open a slash-command. The sibling ``` `false`/`0` ``` shape (a second
    slashed-codespan prose form) is included to document it stays unmatched
    too."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "`false`/unset uploads nothing.\nWhen `false`/`0` it uploads nothing.\n"
    )
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (codespan closing backtick not an opener), got: {errors}"


def test_check_skill_refs_matches_legit_ref_after_boundary(tmp_path):
    """The negative-lookbehind must NOT drop a legitimate `/name` ref reached
    across a non-word-char boundary — start-of-line, whitespace, `(`, and the
    plugin-namespace form all still resolve — AND a genuinely dead `/ghost`
    ref after such a boundary still surfaces as the one unresolved error, so
    detection strength is preserved. The word-char-abutting `word`/skill` shape
    is the EXPLICIT drop: the backtick there closes a preceding codespan (the
    closing-backtick FP class the `(?<!\\w)` defuses), so it is correctly
    NON-matching and never becomes an error even though `skill` names a live
    dir."""
    skills = tmp_path / "skills"
    (skills / "weekly").mkdir(parents=True)
    (skills / "skill").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    # Live positives reached across each non-word-char boundary the lookbehind
    # must NOT reject: start-of-line, whitespace, `(`, and the namespaced form.
    (docs / "a.md").write_text(
        "`/weekly` at column zero.\n"
        "Run `/weekly` after a space.\n"
        "Wrapped (`/weekly`) in parens.\n"
        "Namespaced `/codex:rescue` still resolves.\n"
        # word-char-abutting: the leading backtick CLOSES `word`, so `/skill`
        # is NOT read as a slash-command — the defused FP class. Even though a
        # `skill` dir exists, this must never produce a match/error.
        "Prose word`/skill` is a closing-codespan backtick, not an opener.\n"
        # A genuinely dangling ref after a legitimate boundary still FAILs.
        "Dead `/ghost` after a space.\n"
    )
    errors = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset({"codex:"})
    )
    assert len(errors) == 1, f"expected exactly one error (the dangling /ghost), got: {errors}"
    assert "/ghost" in errors[0], f"expected the /ghost ref flagged, got: {errors}"
    assert "a.md:6" in errors[0], f"expected the error anchored to the /ghost line, got: {errors}"


@pytest.mark.parametrize("fs_root", sorted(SKILL_REF_FS_ROOTS))
def test_check_skill_refs_pass_bare_fs_root_backticked(fs_root, tmp_path):
    """Regression (#1445): a bare single-segment backticked filesystem root
    (`/tmp`, `/workspace`, `/mnt`, ...) closes on a backtick, so SKILL_REF_RE's
    trailing lookahead cannot reject it; the SKILL_REF_FS_ROOTS carve-out must
    resolve EVERY member — zero FPs with an empty skills dir + empty allowlist
    (acceptance criterion 1, pinned per member). The fs_roots-off arm proves
    the CARVE-OUT (not a regex non-match) is what resolves each member."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(f"Write it under `/{fs_root}` first.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert errors == [], f"expected PASS (bare fs root `/{fs_root}` carved out), got: {errors}"
    errors_off = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset(), fs_roots=frozenset()
    )
    assert len(errors_off) == 1, (
        f"expected exactly one error with fs_roots emptied (proves SKILL_REF_RE "
        f"extracts `/{fs_root}` and only the carve-out resolves it), got: {errors_off}"
    )
    assert f"/{fs_root}" in errors_off[0]


def test_check_skill_refs_fs_root_carveout_is_load_bearing(tmp_path):
    """The same bare `/tmp` token FAILs when fs_roots is emptied via the
    override hook — proving the carve-out (not the regex/allowlist) resolves it."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Write it under `/tmp` first.\n")
    errors = check_skill_references(
        roots=[docs], skills_dir=skills, allowlist=frozenset(), fs_roots=frozenset()
    )
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/tmp" in errors[0]


def test_check_skill_refs_fail_dead_ref_with_production_fs_roots(tmp_path):
    """True-positive power preserved (#1445): a genuinely dead `/ghost-skill`
    still FAILs — as the ONE error — with the production fs-roots default
    active and a carved-out `/tmp` on the same line."""
    skills = tmp_path / "skills"
    skills.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Run `/ghost-skill` here, then clean `/tmp` up.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "/ghost-skill" in errors[0]


def test_skill_ref_fs_roots_disjoint_from_live_skills_and_allowlist():
    """Collision guard (#1445): an fs-root entry naming a live
    .claude/skills/ dir would silently disable rot detection for that skill;
    double-membership with SKILL_REF_ALLOWLIST would blur the two sets'
    contracts. Both intersections must stay empty. REMEDY on failure: drop
    the colliding member(s) from SKILL_REF_FS_ROOTS (scripts/workflow_lint.py)
    — do NOT delete this test (plan #1445 §6)."""
    live = _live_skill_names(_REPO_ROOT / ".claude" / "skills")
    assert SKILL_REF_FS_ROOTS & live == set(), (
        "fs-root carve-out member(s) name a live skill dir — drop them from "
        f"SKILL_REF_FS_ROOTS in scripts/workflow_lint.py: {sorted(SKILL_REF_FS_ROOTS & live)}"
    )
    assert set() == SKILL_REF_FS_ROOTS & SKILL_REF_ALLOWLIST, (
        "fs-root carve-out member(s) double-listed in SKILL_REF_ALLOWLIST — keep "
        "each token in exactly one set (scripts/workflow_lint.py): "
        f"{sorted(SKILL_REF_FS_ROOTS & SKILL_REF_ALLOWLIST)}"
    )


def test_check_skill_refs_flags_fs_root_live_skill_collision(tmp_path):
    """The in-function collision guard (#1445): a live skill dir named like an
    fs-root member is reported by check_skill_references itself, so the
    no-flags lint run catches the collision without pytest. REMEDY: drop the
    colliding member from SKILL_REF_FS_ROOTS (scripts/workflow_lint.py)."""
    skills = tmp_path / "skills"
    (skills / "tmp").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("No slash-command tokens here.\n")
    errors = check_skill_references(roots=[docs], skills_dir=skills, allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one collision error, got: {errors}"
    assert "SKILL_REF_FS_ROOTS" in errors[0]
    assert "tmp" in errors[0]


def test_check_skill_refs_repo_tree_is_clean():
    """Green-on-main guard: the committed workflow surface (agents + skills +
    rules + CLAUDE.md + workflow.yaml) carries no unresolved skill refs under
    the production SKILL_REF_ALLOWLIST. This is the regression backstop the
    durable fix installs — it FAILs naming any new dangling token."""
    errors = check_skill_references()
    assert errors == [], (
        "committed workflow surface carries unresolved skill references "
        "(not a live skill dir and not in SKILL_REF_ALLOWLIST):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_skill_refs_cli_exits_zero():
    """The dedicated --check-skill-refs flag must exist and pass on the
    committed tree (mirrors test_workflow_lint_check_heredoc_dotenv_cli...)."""
    result = _run("--check-skill-refs")
    assert result.returncode == 0, (
        f"workflow_lint --check-skill-refs failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_wandb_required`` (task #496 post-mortem). Each
# case writes a tiny .py file under ``tmp_path`` that mimics a
# trainer-config call site and calls
# ``check_wandb_required(experiments_dir=tmp_path)``.
# ---------------------------------------------------------------------------


_TRAINER_HEADER = "from explore_persona_space.train.sft import TrainLoraConfig, train_lora\n"


def test_check_wandb_required_fail_bare_report_to_none(tmp_path):
    """FAIL — `report_to="none"` inside a TrainLoraConfig call site with
    no waiver comment. This is the exact #496 anti-pattern."""
    pkg = tmp_path / "warmth_sycophancy_496"
    pkg.mkdir()
    (pkg / "train_one_cell.py").write_text(
        _TRAINER_HEADER + 'cfg = TrainLoraConfig(\n    run_name="x",\n    report_to="none",\n)\n'
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert len(errors) == 1, f"expected 1 error, got: {errors}"
    assert "report_to" in errors[0]
    assert "WANDB_INTENTIONALLY_DISABLED" in errors[0]
    assert "train_one_cell.py:4" in errors[0]


def test_check_wandb_required_fail_report_to_none_literal(tmp_path):
    """FAIL — `report_to=None` (Python None, not the string) also
    disables WandB and must carry a waiver."""
    pkg = tmp_path / "exp_a"
    pkg.mkdir()
    (pkg / "train.py").write_text(
        _TRAINER_HEADER + "cfg = TrainLoraConfig(\n    report_to=None,\n)\n"
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert len(errors) == 1, f"expected 1 error, got: {errors}"


def test_check_wandb_required_fail_report_to_empty_list(tmp_path):
    """FAIL — `report_to=[]` is the HuggingFace-canonical "send nowhere"
    value and must carry a waiver too."""
    pkg = tmp_path / "exp_b"
    pkg.mkdir()
    (pkg / "train.py").write_text(
        _TRAINER_HEADER + "cfg = TrainLoraConfig(\n    report_to=[],\n)\n"
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert len(errors) == 1, f"expected 1 error, got: {errors}"


def test_check_wandb_required_pass_waiver_same_line(tmp_path):
    """PASS — waiver comment on the same line as the kwarg."""
    pkg = tmp_path / "exp_c"
    pkg.mkdir()
    (pkg / "train.py").write_text(
        _TRAINER_HEADER
        + "cfg = TrainLoraConfig(\n"
        + '    report_to="none",  # WANDB_INTENTIONALLY_DISABLED: smoke-only run\n'
        + ")\n"
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_wandb_required_pass_waiver_line_above(tmp_path):
    """PASS — waiver comment on the immediately preceding non-blank line."""
    pkg = tmp_path / "exp_d"
    pkg.mkdir()
    (pkg / "train.py").write_text(
        _TRAINER_HEADER
        + "cfg = TrainLoraConfig(\n"
        + "    # WANDB_INTENTIONALLY_DISABLED: deterministic replay rig\n"
        + '    report_to="none",\n'
        + ")\n"
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_wandb_required_fail_waiver_reason_too_short(tmp_path):
    """FAIL — waiver present but reason after the colon is below the
    ≥10-char minimum (token-shaped bypass)."""
    pkg = tmp_path / "exp_e"
    pkg.mkdir()
    (pkg / "train.py").write_text(
        _TRAINER_HEADER
        + 'cfg = TrainLoraConfig(\n    report_to="none",  # WANDB_INTENTIONALLY_DISABLED: x\n)\n'
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert len(errors) == 1, f"expected 1 error, got: {errors}"


def test_check_wandb_required_skips_file_without_trainer_config(tmp_path):
    """PASS — a file that does not mention any trainer-config builder
    (e.g. an eval-only or analyzer module) is skipped even if it carries
    a bare `report_to="none"` literal in a docstring or comment example."""
    pkg = tmp_path / "exp_f"
    pkg.mkdir()
    (pkg / "analyze.py").write_text(
        '"""Pure analyzer module."""\n'
        '# Example trainer config: cfg = SomeConfig(report_to="none")\n'
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert errors == [], f"expected PASS (no trainer-config builder), got: {errors}"


def test_check_wandb_required_passthrough_default_does_not_match(tmp_path):
    """PASS — `report_to: str = "wandb"` (the POSITIVE default in a
    passthrough kwarg signature, e.g. contrastive_neg_geometry_472's
    `train_cell.py:355`) must NOT trigger the lint. The regex is
    pinned to disabling literals only."""
    pkg = tmp_path / "exp_g"
    pkg.mkdir()
    (pkg / "train_cell.py").write_text(
        _TRAINER_HEADER
        + 'def build(\n    report_to: str = "wandb",\n) -> TrainLoraConfig:\n'
        + "    return TrainLoraConfig(report_to=report_to)\n"
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert errors == [], f"expected PASS (positive default), got: {errors}"


def test_check_wandb_required_ternary_with_wandb_branch_does_not_match(tmp_path):
    """PASS — `report_to="wandb" if wandb_project else "none"` (the
    factor_screen_365 conditional shape) puts the disabling literal on
    the FALSE branch, not directly after `report_to=`. The regex is
    anchored to `report_to=` immediately followed by the disabling
    value, so this should not match."""
    pkg = tmp_path / "exp_h"
    pkg.mkdir()
    (pkg / "training.py").write_text(
        _TRAINER_HEADER
        + 'cfg = TrainLoraConfig(\n    report_to="wandb" if wandb_project else "none",\n)\n'
    )
    errors = check_wandb_required(experiments_dir=tmp_path)
    assert errors == [], f"expected PASS (ternary with wandb branch), got: {errors}"


def test_check_wandb_required_repo_tree_is_clean():
    """The committed src/explore_persona_space/experiments/ tree must
    carry no un-waived WandB-disabled training-config builders. This is
    the regression guard the durable fix installs."""
    errors = check_wandb_required()
    assert errors == [], (
        "src/explore_persona_space/experiments/ has un-waived WandB-disabled "
        "trainer-config builders (CLAUDE.md 'Upload Policy' violation, "
        "#496 class):\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Unit tests for the worktree-aware scan-root logic (``_other_worktree_prefix``
# / ``_is_other_worktree_path`` / ``_iter_ask_target_files`` walking).
#
# Bug fixed: when ``scripts/workflow_lint.py`` was invoked from inside a
# worktree at ``<repo>/.claude/worktrees/<X>/``, the previous exclusion
# rule (``".claude/worktrees/" not in str(p)``) silently dropped ALL
# files under the current worktree's ``.claude/skills/``, so a
# workflow-improver running inside a worktree got a FALSE PASS from
# ``--check-asks`` because its edited SKILL.md was never scanned. The
# fix scans the CURRENT worktree's files while still excluding sibling
# worktrees, and preserves the "all worktrees excluded" behaviour when
# the lint runs from the main checkout.
# ---------------------------------------------------------------------------


def test_other_worktree_prefix_returns_none_for_main_checkout():
    """Running from a plain main checkout (no ``.claude/worktrees/<X>``
    segment in the parent chain) → None, meaning "no current worktree
    to exempt, exclude every ``.claude/worktrees/`` path"."""
    from pathlib import Path as _P

    from workflow_lint import _other_worktree_prefix as _otp

    assert _otp(_P("/home/user/explore-persona-space")) is None
    assert _otp(_P("/tmp/some/random/dir")) is None


def test_other_worktree_prefix_extracts_worktree_name():
    """Running from inside a worktree → returns the
    ``.claude/worktrees/<X>/`` substring so callers can use it to
    distinguish "our worktree" from sibling worktrees."""
    from pathlib import Path as _P

    from workflow_lint import _other_worktree_prefix as _otp

    assert (
        _otp(_P("/home/user/explore-persona-space/.claude/worktrees/agent-abc"))
        == ".claude/worktrees/agent-abc/"
    )
    assert (
        _otp(_P("/home/user/explore-persona-space/.claude/worktrees/fix-bug-42"))
        == ".claude/worktrees/fix-bug-42/"
    )


def test_other_worktree_prefix_ignores_unrelated_worktrees_segment():
    """A path with ``worktrees`` that is NOT inside ``.claude/`` (e.g.
    a directory literally named ``worktrees`` somewhere else) does NOT
    activate the worktree-aware mode."""
    from pathlib import Path as _P

    from workflow_lint import _other_worktree_prefix as _otp

    # `worktrees` is not preceded by `.claude/`
    assert _otp(_P("/some/repo/git/worktrees/foo")) is None
    # `.claude/worktrees` with no name segment after
    assert _otp(_P("/home/user/repo/.claude/worktrees")) is None


def test_other_worktree_prefix_skips_unrelated_worktrees_dir_higher_up():
    """A path with an unrelated ``worktrees`` segment HIGHER up the
    chain (not preceded by ``.claude``) must NOT short-circuit the
    search — if a real ``.claude/worktrees/<name>`` segment appears
    further down, the function must find IT, not the unrelated higher
    segment."""
    from pathlib import Path as _P

    from workflow_lint import _other_worktree_prefix as _otp

    # First `worktrees` is bare (preceded by `foo`); second is preceded
    # by `.claude` — function must skip the first and match the second.
    assert (
        _otp(_P("/home/foo/worktrees/baz/.claude/worktrees/wt-real"))
        == ".claude/worktrees/wt-real/"
    )


def test_is_other_worktree_path_main_excludes_all_worktrees():
    """From a main checkout (``current_worktree_prefix is None``) every
    ``.claude/worktrees/`` path is "other"."""
    from pathlib import Path as _P

    from workflow_lint import _is_other_worktree_path as _iow

    assert _iow(_P("/repo/.claude/worktrees/wt-a/.claude/skills/foo/SKILL.md"), None) is True
    assert _iow(_P("/repo/.claude/worktrees/wt-b/.claude/agents/x.md"), None) is True
    # Non-worktree paths are NOT "other".
    assert _iow(_P("/repo/.claude/skills/foo/SKILL.md"), None) is False


def test_is_other_worktree_path_worktree_includes_self_excludes_siblings():
    """From inside ``<repo>/.claude/worktrees/wt-a``: paths under
    ``wt-a`` are NOT other; paths under ``wt-b`` ARE other; the
    workflow-improver-running-in-its-own-worktree path PASSes through."""
    from pathlib import Path as _P

    from workflow_lint import _is_other_worktree_path as _iow

    prefix = ".claude/worktrees/wt-a/"
    # Same worktree → not other (this is the fix).
    assert _iow(_P("/repo/.claude/worktrees/wt-a/.claude/skills/foo/SKILL.md"), prefix) is False
    assert _iow(_P("/repo/.claude/worktrees/wt-a/.claude/agents/x.md"), prefix) is False
    # Sibling worktree → other.
    assert _iow(_P("/repo/.claude/worktrees/wt-b/.claude/skills/foo/SKILL.md"), prefix) is True
    # Path without `.claude/worktrees/` at all (e.g. a main-checkout fixture
    # path accidentally passed in) → not other.
    assert _iow(_P("/repo/.claude/skills/foo/SKILL.md"), prefix) is False


def test_is_other_worktree_path_prefix_with_trailing_slash_disambiguates_siblings():
    """The trailing slash in ``current_worktree_prefix`` is load-bearing:
    a sibling worktree named ``wt-a-other`` MUST be detected as "other"
    even though its name STARTS WITH our worktree's name ``wt-a``."""
    from pathlib import Path as _P

    from workflow_lint import _is_other_worktree_path as _iow

    prefix = ".claude/worktrees/wt-a/"
    assert (
        _iow(
            _P("/repo/.claude/worktrees/wt-a-other/.claude/skills/foo/SKILL.md"),
            prefix,
        )
        is True
    )


def test_iter_ask_target_files_scans_current_worktree_self(tmp_path):
    """End-to-end on a synthetic tree: when ``repo_root`` looks like
    ``<base>/.claude/worktrees/<wt-a>``, the file iterator returns the
    worktree's OWN ``.claude/agents`` + ``.claude/skills/**/SKILL.md``
    files (regression guard for the silent-drop bug)."""
    # Build a synthetic worktree: .../base/.claude/worktrees/wt-a/.claude/{agents,skills}/...
    worktree = tmp_path / "base" / ".claude" / "worktrees" / "wt-a"
    agents_dir = worktree / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "alpha.md").write_text("# alpha\n")
    skills_subdir = worktree / ".claude" / "skills" / "demo"
    skills_subdir.mkdir(parents=True)
    (skills_subdir / "SKILL.md").write_text("# demo skill\n")

    files = _iter_ask_target_files(worktree)
    rels = sorted(str(p.relative_to(worktree)) for p in files)
    assert rels == [
        ".claude/agents/alpha.md",
        ".claude/skills/demo/SKILL.md",
    ], rels


def test_iter_ask_target_files_excludes_sibling_worktrees(tmp_path):
    """When ``repo_root`` is main-checkout-shaped (no
    ``.claude/worktrees/<X>`` segment in its parent chain), files under
    nested ``.claude/worktrees/*`` directories are EXCLUDED. Preserves
    the original behaviour for the main-checkout invocation."""
    # Main checkout under tmp_path/main: a regular SKILL.md plus a
    # nested worktree containing a "stale" SKILL.md that must NOT be
    # picked up.
    main_root = tmp_path / "main"
    main_skills = main_root / ".claude" / "skills" / "real"
    main_skills.mkdir(parents=True)
    (main_skills / "SKILL.md").write_text("# real skill on main\n")
    main_agents = main_root / ".claude" / "agents"
    main_agents.mkdir(parents=True)
    (main_agents / "real_agent.md").write_text("# real agent on main\n")
    # Stale worktree copy nested inside the main tree.
    stale_skill = main_root / ".claude" / "worktrees" / "wt-x" / ".claude" / "skills" / "stale"
    stale_skill.mkdir(parents=True)
    (stale_skill / "SKILL.md").write_text("# stale duplicate inside worktree\n")
    stale_agent = main_root / ".claude" / "worktrees" / "wt-x" / ".claude" / "agents"
    stale_agent.mkdir(parents=True)
    (stale_agent / "stale_agent.md").write_text("# stale duplicate agent\n")

    files = _iter_ask_target_files(main_root)
    rels = sorted(str(p.relative_to(main_root)) for p in files)
    # Only the main-checkout files; both worktree copies excluded.
    assert rels == [
        ".claude/agents/real_agent.md",
        ".claude/skills/real/SKILL.md",
    ], rels


def test_iter_ask_target_files_excludes_only_siblings_from_worktree(tmp_path):
    """From inside worktree ``wt-a``, files under ``wt-a/.claude/skills``
    ARE included, but a sibling worktree ``wt-b`` is excluded (catches
    the case where multiple worktrees coexist under the same
    ``.claude/worktrees/`` parent and the lint must not pick up siblings)."""
    base = tmp_path / "base"
    # Our worktree (wt-a).
    wt_a = base / ".claude" / "worktrees" / "wt-a"
    wt_a_skills = wt_a / ".claude" / "skills" / "mine"
    wt_a_skills.mkdir(parents=True)
    (wt_a_skills / "SKILL.md").write_text("# my skill\n")
    wt_a_agents = wt_a / ".claude" / "agents"
    wt_a_agents.mkdir(parents=True)
    (wt_a_agents / "my_agent.md").write_text("# my agent\n")
    # Sibling worktree (wt-b) under the SAME `.claude/worktrees/` parent.
    wt_b_skills = base / ".claude" / "worktrees" / "wt-b" / ".claude" / "skills" / "theirs"
    wt_b_skills.mkdir(parents=True)
    (wt_b_skills / "SKILL.md").write_text("# their skill\n")
    wt_b_agents = base / ".claude" / "worktrees" / "wt-b" / ".claude" / "agents"
    wt_b_agents.mkdir(parents=True)
    (wt_b_agents / "their_agent.md").write_text("# their agent\n")
    # workflow_lint is invoked from inside wt-a → wt-a's files only.
    # But _iter_ask_target_files only walks repo_root/.claude/{agents,skills}
    # (NOT base/.claude/worktrees/wt-b/...), so for this configuration we
    # need to also confirm: walking from wt-a returns ONLY wt-a's files
    # (because wt-b is outside repo_root entirely from wt-a's perspective).
    files = _iter_ask_target_files(wt_a)
    rels = sorted(str(p) for p in files)
    assert any("wt-a/.claude/skills/mine/SKILL.md" in r for r in rels), rels
    assert any("wt-a/.claude/agents/my_agent.md" in r for r in rels), rels
    assert not any("wt-b" in r for r in rels), rels


def test_iter_ask_target_files_from_worktree_excludes_nested_other_worktrees(tmp_path):
    """From inside worktree ``wt-a``, if (pathologically) ``wt-a`` itself
    contains a nested ``.claude/worktrees/wt-c`` subdirectory, that
    nested directory's files are EXCLUDED. Guards the case where a
    worktree's own working tree contains a stale snapshot of another
    worktree."""
    base = tmp_path / "base"
    wt_a = base / ".claude" / "worktrees" / "wt-a"
    wt_a_skills = wt_a / ".claude" / "skills" / "mine"
    wt_a_skills.mkdir(parents=True)
    (wt_a_skills / "SKILL.md").write_text("# my skill\n")
    # Nested worktree inside wt-a's own .claude/worktrees/ — must be excluded.
    nested = wt_a / ".claude" / "worktrees" / "wt-c" / ".claude" / "skills" / "stale"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("# stale nested\n")

    files = _iter_ask_target_files(wt_a)
    rels = sorted(str(p) for p in files)
    assert any("wt-a/.claude/skills/mine/SKILL.md" in r for r in rels), rels
    assert not any("wt-c" in r for r in rels), rels


def test_workflow_lint_check_asks_scans_skill_files_from_worktree():
    """End-to-end: the production ``check_asks(workflow)`` call from
    within this worktree MUST actually scan ``.claude/skills/**/SKILL.md``
    files (regression guard: before the fix, 0 SKILL.md files were
    scanned and ``--check-asks`` gave a false PASS for any SKILL.md edit
    a workflow-improver made inside a worktree)."""
    from workflow_lint import _REPO_ROOT  # the worktree we are running from

    files = _iter_ask_target_files(_REPO_ROOT)
    skill_files = [f for f in files if "SKILL.md" in str(f)]
    assert len(skill_files) > 0, (
        "expected ≥1 SKILL.md file in --check-asks scope from the current "
        "tree, got 0 — the worktree-aware exclusion has regressed and "
        "workflow-improver edits to SKILL.md will silently false-PASS"
    )
    # Smoke: every SKILL.md path must belong to THIS worktree (or to the
    # main checkout if this test runs from main). No sibling worktree paths.
    prefix = _other_worktree_prefix(_REPO_ROOT)
    if prefix is not None:
        for sf in skill_files:
            # Either it's not under .claude/worktrees/ at all (impossible
            # when prefix is set), or it must contain our prefix.
            assert prefix in str(sf), (
                f"SKILL.md {sf} is not under our worktree prefix {prefix}; "
                f"sibling-worktree exclusion regressed"
            )


# ---------------------------------------------------------------------------
# Unit tests for ``check_marker_registry`` (task #555 drift class). Each
# fixture case writes a tiny SKILL.md under ``tmp_path`` and calls
# ``check_marker_registry(workflow, skill_md=<fixture>)`` against the REAL
# committed workflow.yaml registry (so "registered" means actually
# registered, and the sentinel kind below stays unregistered by design).
# ---------------------------------------------------------------------------

# Deliberately absurd kind that must never be registered; used to assert
# the FAIL paths without depending on registry contents.
_UNREGISTERED_KIND = "epm:zz-test-sentinel-unregistered"


def test_workflow_lint_check_marker_registry_repo_passes():
    """Repo-level check: every marker kind any committed skill's SKILL.md
    under .claude/skills/**/ AND every committed agent spec under
    .claude/agents/*.md instructs posting must be declared in
    workflow.yaml § markers. If this fails, a skill or agent edit added a
    posting site for an unregistered kind (the task #555 drift class)."""
    errors = check_marker_registry(_workflow())
    assert errors == [], (
        "committed SKILL.md / agent specs post marker kinds missing from "
        "workflow.yaml § markers:\n" + "\n".join(errors)
    )


def test_check_marker_registry_pass_registered_cli_post(tmp_path):
    """A `task.py post-marker` invocation with a registered kind PASSes."""
    skill = tmp_path / "SKILL.md"
    skill.write_text("Run `uv run python scripts/task.py post-marker <N> epm:plan --note '...'`.\n")
    errors = check_marker_registry(_workflow(), skill_md=skill)
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_marker_registry_fail_unregistered_cli_post(tmp_path):
    """A `task.py post-marker` invocation with an unregistered kind FAILs."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(f"Run `task.py post-marker <N> {_UNREGISTERED_KIND} --note 'x'`.\n")
    errors = check_marker_registry(_workflow(), skill_md=skill)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert _UNREGISTERED_KIND in errors[0]
    assert "SKILL.md:1" in errors[0]
    assert "not declared in workflow.yaml" in errors[0]


def test_check_marker_registry_fail_unregistered_prose_post(tmp_path):
    """Posting prose ('post `epm:<kind> v1`') with an unregistered kind
    FAILs — the prose form is how most SKILL.md steps instruct posts."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(f"On classifier error, post `{_UNREGISTERED_KIND} v1` with the stderr.\n")
    errors = check_marker_registry(_workflow(), skill_md=skill)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert _UNREGISTERED_KIND in errors[0]


def test_check_marker_registry_comment_form_post_matches(tmp_path):
    """The `<!-- epm:<kind> v1 -->` comment form after a post-verb also
    counts as a posting site."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(f"Post a `<!-- {_UNREGISTERED_KIND} v1 -->` event on the task.\n")
    errors = check_marker_registry(_workflow(), skill_md=skill)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert _UNREGISTERED_KIND in errors[0]


def test_check_marker_registry_read_mention_does_not_match(tmp_path):
    """Read-side mentions ('the latest `epm:<kind>` marker') are NOT
    posting sites and never FAIL, even for unregistered kinds."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        f"Read the latest `{_UNREGISTERED_KIND} v<n>` marker on the source task.\n"
        f"If an `{_UNREGISTERED_KIND}` event exists, resume from it.\n"
    )
    errors = check_marker_registry(_workflow(), skill_md=skill)
    assert errors == [], f"read-side mention tripped the posting check: {errors}"


def test_check_marker_registry_missing_skill_md_returns_empty(tmp_path):
    """A nonexistent SKILL.md path returns no errors (mirrors the other
    checks' missing-file behavior)."""
    errors = check_marker_registry(_workflow(), skill_md=tmp_path / "nope" / "SKILL.md")
    assert errors == [], f"expected empty on missing file, got: {errors}"


def test_check_marker_registry_agents_dir_fail_unregistered_post(tmp_path):
    """Agent specs are posting surface too (task #555 follow-up): a
    `task.py post-marker` invocation with an unregistered kind inside a
    fixture agents dir FAILs, naming the agent file."""
    agents = tmp_path / "agents"
    agents.mkdir()
    agent = agents / "some-agent.md"
    agent.write_text(f"Run `task.py post-marker <N> {_UNREGISTERED_KIND} --note 'x'`.\n")
    errors = check_marker_registry(_workflow(), agents_dir=agents)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert _UNREGISTERED_KIND in errors[0]
    assert "some-agent.md:1" in errors[0]


def test_check_marker_registry_agents_dir_pass_registered_post(tmp_path):
    """Posting prose in an agent spec with a registered kind PASSes."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "analyzer-like.md").write_text(
        "When done, post `epm:analysis v1` with the fact sheet.\n"
    )
    errors = check_marker_registry(_workflow(), agents_dir=agents)
    assert errors == [], f"expected PASS for a registered kind, got: {errors}"


def test_check_marker_registry_combined_overrides_scan_both(tmp_path):
    """Passing skill_md AND agents_dir scans both overridden surfaces
    (and only them): one unregistered posting site in each yields two
    errors, one per file."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(f"Post a `{_UNREGISTERED_KIND} v1` event on the task.\n")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "agent.md").write_text(
        f"Run `task.py post-marker <N> {_UNREGISTERED_KIND} --note 'x'`.\n"
    )
    errors = check_marker_registry(_workflow(), skill_md=skill, agents_dir=agents)
    assert len(errors) == 2, f"expected one error per fixture file, got: {errors}"
    assert any("SKILL.md:1" in e for e in errors)
    assert any("agent.md:1" in e for e in errors)


def test_check_marker_registry_skills_dir_fail_unregistered_post(tmp_path):
    """NON-issue skills are posting surface too (task #555 chain, final
    fix): a `task.py post-marker` invocation with an unregistered kind in
    a nested `<skill>/SKILL.md` under a fixture skills dir FAILs — the
    recursive walk the production scan uses for `.claude/skills/**/
    SKILL.md` must reach it. (The real instance was promote-clean-result's
    `epm:consolidated-into` site, unlinted until the walk was widened.)"""
    skills = tmp_path / "skills"
    nested = skills / "promote-foo"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        f"Run `uv run python scripts/task.py post-marker <M> {_UNREGISTERED_KIND} "
        f"--by promote-foo`.\n"
    )
    errors = check_marker_registry(_workflow(), skills_dir=skills)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert _UNREGISTERED_KIND in errors[0]
    assert "SKILL.md:1" in errors[0]


def test_check_marker_registry_skills_dir_pass_registered_post(tmp_path):
    """The promote-clean-result posting shape PASSes now that
    `epm:consolidated-into` is registered in workflow.yaml § markers —
    pins both the skills_dir walk and the registration itself."""
    skills = tmp_path / "skills"
    nested = skills / "promote-clean-result"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "Run `uv run python scripts/task.py post-marker <M> epm:consolidated-into "
        "--by promote-clean-result`.\n"
    )
    errors = check_marker_registry(_workflow(), skills_dir=skills)
    assert errors == [], f"expected PASS for a registered kind, got: {errors}"


def test_check_marker_registry_pins_failure_lesson_field_contract(tmp_path):
    """#712 §4f: ``check_marker_registry`` ALSO pins the ``epm:failure-lesson``
    field contract — its registry ``fields:`` string MUST contain the literal
    tokens ``root_cause_confirmed`` AND ``supersedes``, and its ``when:`` string
    MUST contain ``root_cause_confirmed=yes`` — so a future edit that drops or
    renames a field FAILs the lint.

    FAIL leg — a synthetic workflow whose ``epm:failure-lesson`` marker is
    MISSING the tokens produces a field-contract error (empty override dirs
    isolate the new field-contract assertion from the posting-site scan).

    PASS leg — the REAL committed ``workflow.yaml`` carries the tokens, so the
    repo-level check produces no ``epm:failure-lesson`` field-contract error.
    (In TDD pass 1 BOTH legs fail: the field-contract assertion does not exist
    yet AND the real workflow.yaml has not yet gained the tokens — the
    implementation pass adds both, after which this test pins the contract.)
    """
    empty_skills = tmp_path / "skills"
    empty_agents = tmp_path / "agents"
    empty_skills.mkdir()
    empty_agents.mkdir()

    # FAIL: a failure-lesson marker stripped of the required field tokens.
    stripped = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(
                kind="epm:failure-lesson",
                posted_by="orchestrator",
                when="fires when a crash-fix round resolves the failure",
                fields="failure_class, phase, lesson, generalizes, owning_agent, gotcha_candidate",
            )
        ],
    )
    fail_errors = check_marker_registry(stripped, skills_dir=empty_skills, agents_dir=empty_agents)
    assert any(
        "epm:failure-lesson" in e
        and ("root_cause_confirmed" in e or "supersedes" in e or "field" in e.lower())
        for e in fail_errors
    ), (
        "expected a field-contract FAIL naming epm:failure-lesson + the missing "
        f"token(s); got: {fail_errors}"
    )

    # PASS: the real workflow.yaml satisfies the field contract (after the
    # implementation pass fills the fields:/when: tokens).
    repo_errors = check_marker_registry(_workflow())
    assert not any(
        "epm:failure-lesson" in e
        and ("root_cause_confirmed" in e or "supersedes" in e or "field" in e.lower())
        for e in repo_errors
    ), (
        "the committed workflow.yaml epm:failure-lesson marker is missing a "
        f"required field token: {repo_errors}"
    )


def test_no_flags_default_run_pins_failure_lesson_field_contract(tmp_path):
    """#712 §4f: the BARE ``workflow_lint.py`` (no check/emit flags) MUST run the
    ``epm:failure-lesson`` field-contract assertion, so a future edit dropping
    ``root_cause_confirmed`` / ``supersedes`` TRIPS the default pre-commit lint.

    The round-1 review found the assertion was opt-in — gated on
    ``--check-marker-registry`` alone, NOT bundled into the no-flags default —
    so the mechanical pin that justifies ``architectural: false`` was not
    load-bearing on the default path. This is the END-TO-END CLI regression
    (the sibling ``test_check_marker_registry_pins_...`` calls the helper
    directly and so cannot catch the bundling gap).

    Writes a copy of the REAL committed ``workflow.yaml`` with the
    ``root_cause_confirmed`` / ``supersedes`` field tokens renamed away (which
    also breaks the ``root_cause_confirmed=yes`` ``when:`` token), then invokes
    the no-flags CLI via ``--file`` and asserts it exits non-zero with the
    ``#712 §4f`` field-contract diagnostic.
    """
    real_yaml = (_REPO_ROOT / ".claude" / "workflow.yaml").read_text()
    # Rename the two field tokens away. They occur ONLY on the
    # epm:failure-lesson marker's `fields:`/`when:` lines, so this strip is
    # surgically scoped and leaves the rest of the schema loadable.
    stripped = real_yaml.replace("root_cause_confirmed", "rc_confirmed_renamed").replace(
        "supersedes", "replaces_renamed"
    )
    assert stripped != real_yaml, "the field tokens were not present to strip — fixture stale"
    stripped_yaml = tmp_path / "workflow.yaml"
    stripped_yaml.write_text(stripped)

    result = _run("--file", str(stripped_yaml))
    assert result.returncode != 0, (
        "the no-flags default run did NOT fail on a field-token-stripped "
        f"workflow.yaml — the field-contract check is not bundled into no_flags:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "#712 §4f" in result.stderr, (
        "expected the #712 §4f field-contract diagnostic in stderr; got:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_marker_scalar_integrity`` +
# ``check_poller_marker_consumers`` (task #1191; incident #873: an unquoted
# workflow.yaml plain scalar containing ' #' silently truncated at the
# comment marker and --check-references passed on the truncated parse; the
# same task's poller runtime-tripwire claim shipped with no poll_pipeline
# code until a critic caught it).
# ---------------------------------------------------------------------------

# The verbatim #873 offender value (unquoted → YAML truncates at ' #').
_I873_POSTED_BY = "skill (via experiment-implementer); poll_pipeline (runtime tripwire, #873)"


def test_check_marker_scalar_integrity_fail_truncated_comment_scalar(tmp_path):
    """END-TO-END #873 repro: the UNQUOTED offender value written to a
    minimal fixture workflow.yaml parses to
    'skill (via experiment-implementer); poll_pipeline (runtime tripwire,'
    (trailing comma AND 2-vs-1 parens — both heuristic legs), so the check
    FAILs with exactly one error naming the kind + the ``posted_by`` field."""
    fixture = tmp_path / "workflow.yaml"
    fixture.write_text(
        "version: 1\n"
        "markers:\n"
        "  - kind: epm:compute-deviation\n"
        f"    posted_by: {_I873_POSTED_BY}\n"
        "    when: fires on a >2x wall-time deviation\n"
        "    fields: projected_hours\n"
    )
    wf = load_workflow_yaml(fixture)
    # Precondition of the repro: YAML really did truncate at the comment.
    assert wf.markers[0].posted_by.endswith("(runtime tripwire,"), wf.markers[0].posted_by
    errors = check_marker_scalar_integrity(wf)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "epm:compute-deviation" in errors[0]
    assert "'posted_by'" in errors[0]


def test_check_marker_scalar_integrity_pass_quoted_scalar(tmp_path):
    """The same value DOUBLE-QUOTED (as live workflow.yaml carries it today)
    parses in full — balanced parens, no trailing ','/'(' — and PASSes."""
    fixture = tmp_path / "workflow.yaml"
    fixture.write_text(
        "version: 1\n"
        "markers:\n"
        "  - kind: epm:compute-deviation\n"
        f'    posted_by: "{_I873_POSTED_BY}"\n'
        "    when: fires on a >2x wall-time deviation\n"
        "    fields: projected_hours\n"
    )
    wf = load_workflow_yaml(fixture)
    assert wf.markers[0].posted_by.endswith("#873)"), wf.markers[0].posted_by
    errors = check_marker_scalar_integrity(wf)
    assert errors == [], f"expected PASS for the quoted scalar, got: {errors}"


def test_check_marker_scalar_integrity_fail_unbalanced_only():
    """A value with unbalanced parens but NO trailing ','/'(' still fires
    (the second heuristic leg alone)."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="skill (runtime tripwire", when="w", fields="f")
        ],
    )
    errors = check_marker_scalar_integrity(wf)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "unbalanced parens" in errors[0]


def test_check_marker_scalar_integrity_fail_non_posted_by_field():
    """Round-1 Statistics Must-Fix 2: the truncation signature in a
    NON-``posted_by`` field (``when``) fires with an error naming that
    field — discriminates a broken implementation that scans only
    ``posted_by`` (§11.2's all-four-fields decision)."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(
                kind="epm:x",
                posted_by="skill",
                when="fires after phase 2 (see runbook,",
                fields="f",
            )
        ],
    )
    errors = check_marker_scalar_integrity(wf)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "'when'" in errors[0], f"error must name the 'when' field: {errors[0]}"


def test_check_marker_scalar_integrity_fail_trailing_open_paren():
    """A value ending in '(' fires (the second ``endswith`` branch)."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="posted by the watcher (", when="w", fields="f")
        ],
    )
    errors = check_marker_scalar_integrity(wf)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "trailing ,/(" in errors[0]


def test_check_marker_scalar_integrity_pass_balanced_prose():
    """Ordinary prose with balanced parens ending in '.' PASSes — the check
    keys on the truncation AFTERMATH, not on comment-ish content."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(
                kind="epm:x",
                posted_by="skill (via experiment-implementer); poll_pipeline (tripwire, #873).",
                when="fires when the projection exceeds 2x (see #873).",
                fields="projected_hours",
            )
        ],
    )
    errors = check_marker_scalar_integrity(wf)
    assert errors == [], f"expected PASS for balanced prose, got: {errors}"


def test_check_marker_scalar_integrity_allowlist_waives():
    """A (kind, field) allowlist entry waives an otherwise-failing value."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="skill (runtime tripwire", when="w", fields="f")
        ],
    )
    errors = check_marker_scalar_integrity(
        wf, allowlist={("epm:x", "posted_by"): "deliberate enumeration prose"}
    )
    assert errors == [], f"expected the allowlist to waive, got: {errors}"


def test_workflow_lint_check_marker_scalar_integrity_repo_passes():
    """Live-tree invariant: the committed workflow.yaml has 0 truncation
    signatures across all markers x 4 string fields (probe 2026-07-09)."""
    errors = check_marker_scalar_integrity(_workflow())
    assert errors == [], (
        "committed workflow.yaml § markers carries a truncated-comment "
        "signature:\n" + "\n".join(errors)
    )


def test_workflow_lint_check_marker_scalar_integrity_flag_exits_zero():
    """CLI flag path exits 0 on the live tree."""
    result = _run("--check-marker-scalar-integrity")
    assert result.returncode == 0, (
        f"workflow_lint --check-marker-scalar-integrity failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_poller_marker_consumers_fail_no_reference(tmp_path):
    """A poller-posted kind referenced by NO consumer surface AND absent
    from its declared poster file fails BOTH legs (2 errors)."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="poll_pipeline.py (tripwire)", when="w", fields="f")
        ],
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("No markers mentioned here.\n")
    poller = tmp_path / "poll_pipeline.py"
    poller.write_text("# no kinds posted here\n")
    errors = check_poller_marker_consumers(
        wf, consumer_paths=[skill], poller_file_map={"poll_pipeline": poller}
    )
    assert len(errors) == 2, f"expected Leg A + Leg B errors, got: {errors}"
    assert any("NO consumer surface" in e for e in errors), errors
    assert any("declared poster" in e for e in errors), errors


def test_check_poller_marker_consumers_pass_referenced(tmp_path):
    """The same kind mentioned in a consumer surface AND in the poster
    file PASSes both legs."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="poll_pipeline.py (tripwire)", when="w", fields="f")
        ],
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("The poller posts `epm:x v1` on a tripwire hit.\n")
    poller = tmp_path / "poll_pipeline.py"
    poller.write_text('KIND = "epm:x"\n')
    errors = check_poller_marker_consumers(
        wf, consumer_paths=[skill], poller_file_map={"poll_pipeline": poller}
    )
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_poller_marker_consumers_skips_non_poller(tmp_path):
    """A non-poller ``posted_by`` (e.g. 'skill') is never checked, even
    with zero references anywhere."""
    wf = WorkflowYaml(
        version=1,
        markers=[MarkerEntry(kind="epm:x", posted_by="skill", when="w", fields="f")],
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("Nothing here.\n")
    errors = check_poller_marker_consumers(wf, consumer_paths=[skill], poller_file_map={})
    assert errors == [], f"expected non-poller posted_by to be skipped, got: {errors}"


def test_check_poller_marker_consumers_leg_b_only(tmp_path):
    """Kind present in a consumer surface but ABSENT from the mapped
    poster file yields exactly one Leg-B error (the #873 pre-fix state:
    the claim is documented but the posting code does not exist)."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="poll_pipeline.py (tripwire)", when="w", fields="f")
        ],
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("The poller posts `epm:x v1` on a tripwire hit.\n")
    poller = tmp_path / "poll_pipeline.py"
    poller.write_text("# posting code never written\n")
    errors = check_poller_marker_consumers(
        wf, consumer_paths=[skill], poller_file_map={"poll_pipeline": poller}
    )
    assert len(errors) == 1, f"expected exactly one Leg-B error, got: {errors}"
    assert "declared poster" in errors[0]
    assert "poll_pipeline" in errors[0]


def test_check_poller_marker_consumers_allowlist_waives(tmp_path):
    """An allowlisted kind is waived from both legs."""
    wf = WorkflowYaml(
        version=1,
        markers=[
            MarkerEntry(kind="epm:x", posted_by="poll_pipeline.py (tripwire)", when="w", fields="f")
        ],
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("Nothing here.\n")
    poller = tmp_path / "poll_pipeline.py"
    poller.write_text("# nothing here\n")
    errors = check_poller_marker_consumers(
        wf,
        consumer_paths=[skill],
        poller_file_map={"poll_pipeline": poller},
        allowlist={"epm:x": "deliberate out-of-band consumer (dashboard)"},
    )
    assert errors == [], f"expected the allowlist to waive, got: {errors}"


def test_workflow_lint_check_poller_marker_consumers_repo_passes():
    """Live-tree invariant: every committed poller-posted marker kind has
    >=1 consumer reference and its declared poster mentions it (probe
    2026-07-09: 5/5)."""
    errors = check_poller_marker_consumers(_workflow())
    assert errors == [], (
        "committed workflow.yaml § markers carries a poller-posted kind "
        "with no consumer / poster reference:\n" + "\n".join(errors)
    )


def test_workflow_lint_check_poller_marker_consumers_flag_exits_zero():
    """CLI flag path exits 0 on the live tree."""
    result = _run("--check-poller-marker-consumers")
    assert result.returncode == 0, (
        f"workflow_lint --check-poller-marker-consumers failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_new_checks_bundled(monkeypatch, capsys):
    """Round-1 Statistics Must-Fix 1: BUNDLING is DISCRIMINATED, not
    assumed. The live tree passes both new checks, so an exit-0 no-flags
    run cannot distinguish 'bundled and passing' from 'never registered'
    (the #873 gap re-created at main()). With each new check
    monkeypatch-sentineled, an in-process ``main()`` with no flags AND one
    with ``--check-references`` must BOTH surface the sentinel and return
    nonzero — a forgotten dispatch line / --check-references bundle fails
    this test.

    Every OTHER check function is patched to a no-op so the in-process
    runs stay fast (the real no-flags run takes minutes) — this does not
    weaken the discrimination: with all other checks silenced, an exit-0 /
    sentinel-free run can only mean the new check is not registered on
    that path.
    """
    import workflow_lint as wl

    for name in dir(wl):
        if name.startswith(("check_", "_check_")) and callable(getattr(wl, name)):
            monkeypatch.setattr(wl, name, lambda *a, **k: [])
    monkeypatch.setattr(wl, "emit_tables", lambda *a, **k: [])
    monkeypatch.setattr(
        wl, "check_marker_scalar_integrity", lambda *a, **k: ["SENTINEL-scalar-integrity-bundling"]
    )
    monkeypatch.setattr(
        wl, "check_poller_marker_consumers", lambda *a, **k: ["SENTINEL-poller-consumers-bundling"]
    )

    # Path 1: the no-flags default run.
    rc_default = wl.main([])
    err_default = capsys.readouterr().err
    assert rc_default != 0, "no-flags main() exited 0 with sentinel-failing new checks"
    assert "SENTINEL-scalar-integrity-bundling" in err_default, (
        f"check_marker_scalar_integrity not bundled into the no-flags run:\n{err_default}"
    )
    assert "SENTINEL-poller-consumers-bundling" in err_default, (
        f"check_poller_marker_consumers not bundled into the no-flags run:\n{err_default}"
    )

    # Path 2: the --check-references (pre-commit) run.
    rc_refs = wl.main(["--check-references"])
    err_refs = capsys.readouterr().err
    assert rc_refs != 0, "--check-references main() exited 0 with sentinel-failing new checks"
    assert "SENTINEL-scalar-integrity-bundling" in err_refs, (
        f"check_marker_scalar_integrity not bundled into --check-references:\n{err_refs}"
    )
    assert "SENTINEL-poller-consumers-bundling" in err_refs, (
        f"check_poller_marker_consumers not bundled into --check-references:\n{err_refs}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_heredoc_dotenv`` (incident class #552/#612: a
# no-arg python-dotenv ``load_dotenv()`` inside a heredoc feeding a python
# interpreter's stdin crashes at runtime via find_dotenv()'s frame-walk
# ``assert frame.f_back is not None``). Each fixture case writes a tiny
# ``*.sh`` under ``tmp_path`` and calls
# ``check_heredoc_dotenv(scripts_dir=tmp_path)``.
# ---------------------------------------------------------------------------


def test_check_heredoc_dotenv_fail_issue612_driver_shape(tmp_path):
    """FAIL — the exact pre-fix #612 production-driver shape: opener line
    backslash-continued into an `|| fail` line, body imports + calls the
    no-arg python-dotenv ``load_dotenv()``. This is the live incident the
    check exists to catch (4 reviewers + smoke runs missed it)."""
    (tmp_path / "driver.sh").write_text(
        "#!/usr/bin/env bash\n"
        'uv run python - "$PANEL_POLL_TIMEOUT_S" "$PANEL_POLL_INTERVAL_S" <<\'PY\' \\\n'
        '  || fail "panel_set.json did not appear on HF within the timeout" 3\n'
        "import sys, time\n"
        "from dotenv import load_dotenv\n"
        "load_dotenv()\n"
        "from huggingface_hub import hf_hub_download\n"
        "PY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "driver.sh:6" in errors[0]
    assert "load_dotenv()" in errors[0]
    assert "stdin" in errors[0]


def test_check_heredoc_dotenv_fail_simple_python_stdin(tmp_path):
    """FAIL — plain `uv run python - <<'PY'` (no continuation) with the
    dangerous import + call."""
    (tmp_path / "x.sh").write_text(
        "uv run python - <<'PY'\nfrom dotenv import load_dotenv\nload_dotenv()\nPY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


def test_check_heredoc_dotenv_fail_python3_bare_no_dash(tmp_path):
    """FAIL — `python3 <<EOF` (no `-` arg) also executes the heredoc from
    stdin; the bare-interpreter-as-last-token form must match too."""
    (tmp_path / "x.sh").write_text(
        "python3 <<EOF\nfrom dotenv import load_dotenv\nload_dotenv()\nEOF\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_heredoc_dotenv_fail_qualified_call(tmp_path):
    """FAIL — `import dotenv` + qualified no-arg `dotenv.load_dotenv()`."""
    (tmp_path / "x.sh").write_text(
        "uv run python - <<'PY'\nimport dotenv\ndotenv.load_dotenv()\nPY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


def test_check_heredoc_dotenv_pass_explicit_path_arg(tmp_path):
    """PASS — `load_dotenv(dotenv_path=...)` skips the frame-walking
    find_dotenv() entirely; only the NO-ARG call is the crash."""
    (tmp_path / "x.sh").write_text(
        "uv run python - <<'PY'\n"
        "from dotenv import load_dotenv\n"
        'load_dotenv(dotenv_path="/workspace/explore-persona-space/.env")\n'
        "PY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (explicit path), got: {errors}"


def test_check_heredoc_dotenv_pass_project_wrapper(tmp_path):
    """PASS — the stdin-safe project wrapper (resolves .env via
    resolve_dotenv_path() cwd-walking, no frame inspection). This is the
    canonical in-heredoc shape (#585 round-2 review fix; live exemplar
    scripts/i556_run_all_1gpu.sh) and must NOT be flagged."""
    (tmp_path / "x.sh").write_text(
        "uv run python - <<'PYEOF'\n"
        "from explore_persona_space.orchestrate.env import load_dotenv\n"
        "load_dotenv()\n"
        "PYEOF\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (stdin-safe project wrapper), got: {errors}"


def test_check_heredoc_dotenv_pass_non_python_heredoc(tmp_path):
    """PASS — a heredoc that does NOT feed a python interpreter's stdin
    (here: generating a .py file via `cat`) is data, not stdin-executed
    code; the generated file runs with a real __file__ later."""
    (tmp_path / "x.sh").write_text(
        "cat > /tmp/gen.py <<'EOF'\nfrom dotenv import load_dotenv\nload_dotenv()\nEOF\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (non-python heredoc), got: {errors}"


def test_check_heredoc_dotenv_pass_heredoc_is_data_for_python_script(tmp_path):
    """PASS — `python scripts/foo.py <<EOF` feeds the heredoc to the
    SCRIPT as stdin data; the body is not executed as python source, so
    a load_dotenv-shaped line in it is not a call site."""
    (tmp_path / "x.sh").write_text(
        "uv run python scripts/foo.py <<'EOF'\nfrom dotenv import load_dotenv\nload_dotenv()\nEOF\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (heredoc is script data), got: {errors}"


def test_check_heredoc_dotenv_pass_commented_call(tmp_path):
    """PASS — a commented-out `# load_dotenv()` line (the post-fix #612
    driver carries exactly this as an explanatory comment) is not a call."""
    (tmp_path / "x.sh").write_text(
        "uv run python - <<'PY'\n"
        "from dotenv import load_dotenv\n"
        "# NO bare load_dotenv() here: it crashes from stdin\n"
        "PY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (commented call only), got: {errors}"


def test_check_heredoc_dotenv_second_heredoc_after_safe_one_still_scanned(tmp_path):
    """A dangerous python-fed heredoc AFTER an earlier safe heredoc in the
    same file is still caught (the body-skipping parser must resume opener
    detection after each terminator, not swallow the rest of the file)."""
    (tmp_path / "x.sh").write_text(
        "cat <<'EOF'\nplain text body\nEOF\n"
        "uv run python - <<'PY'\nfrom dotenv import load_dotenv\nload_dotenv()\nPY\n"
    )
    errors = check_heredoc_dotenv(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:6" in errors[0]


def test_check_heredoc_dotenv_repo_tree_is_clean():
    """The committed scripts/*.sh tree must carry no no-arg python-dotenv
    load_dotenv() calls inside python-stdin heredocs — this is the
    regression guard the durable fix installs (the #612 hot-fix removed
    the live one; i556's project-wrapper shape is stdin-safe by design)."""
    errors = check_heredoc_dotenv()
    assert errors == [], (
        "scripts/*.sh has no-arg python-dotenv load_dotenv() calls inside "
        "python-stdin heredocs (#552/#612 crash class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_heredoc_dotenv_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-heredoc-dotenv")
    assert result.returncode == 0, (
        f"workflow_lint --check-heredoc-dotenv failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_pipe_python`` (incident class #753: a bare
# ``... | python -c/-m`` pipe consumer dies on this VM with
# ``python: command not found`` (exit 127) — no ``python`` on PATH, only
# ``python3`` and ``uv run python``; ~41 violations across 4+ sessions on
# 2026-06-29). Each fixture case writes a tiny ``*.sh`` under ``tmp_path``
# and calls ``check_pipe_python(scripts_dir=tmp_path)``. The dual-engine
# test additionally asserts the Python ``re`` lint and the POSIX
# ``grep -qE`` hook (SOURCED from ``.claude/settings.json``, not a
# hard-coded copy — F2) AGREE on the §4 example set, including the F3
# attached-arg (``python -c'code'``) shapes; the F1 fix flags
# ``echo ... | python -c`` producer pipes (no longer skipped).
# ---------------------------------------------------------------------------

# The §4 example sets, shared by the function tests and the dual-engine test.
# MATCHES = the real failures the check must catch; NOMATCH = the false
# positives the anchor avoids.
_PIPE_PYTHON_MATCHES = [
    'cat x.json | python3 -c "import sys,json; ..."',  # offender #1 shape
    'task.py view 1 --json | python3 -c "import sys,json"',  # offender #2 shape
    "echo '{}' | python -c \"print(1)\"",  # echo producer pipe — F1: a REAL pipe
    'foo | python3.11 -c "x"',
    "foo | python -m json.tool",
    'foo |python -c "x"',  # no space after pipe
    'cat x | python -u -c "x"',  # intervening single-dash flag
    "cat x | python -c'print(1)'",  # F3: attached arg (quote glued to -c)
    "foo | python3 -c'x'",  # F3: attached arg on python3
    "foo | python -m'json.tool'",  # F3: attached arg on -m
]
_PIPE_PYTHON_NOMATCH = [
    'echo "use uv run python instead"',  # literal docs string
    "curl https://pypi.org/python-3.12/",  # URL containing "python"
    "cat setup.py | grep foo",  # consumer is grep, not python
    "which python",  # informational, no pipe consumer
    "apt-get install python3",  # informational, no pipe consumer
    'cat x | uv run python -c "x"',  # CORRECT usage — token after | is `uv`
    "python scripts/foo.py < input",  # start-of-command, no pipe consumer
    "ls python_helpers.py | wc -l",  # filename containing "python"
    "foo | python -compose x",  # `-co` is a different flag prefix, not `-c`
]


def test_check_pipe_python_fail_simple_pipe(tmp_path):
    """FAIL — a plain `cat x | python3 -c "..."` consumer-side pipe (the
    exact offender #1 shape) must be flagged."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        'cat x.json | python3 -c "import sys,json;print(json.load(sys.stdin))"\n'
    )
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]
    assert "uv run python" in errors[0]


def test_check_pipe_python_fail_backslash_continued(tmp_path):
    """FAIL — the backslash-continued shape both real #753 offenders use:
    a `cat ... \\` newline `| python3 -c` logical line. The error must
    point at the FIRST physical line of the logical command."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        "cat $REPO/eval.json 2>/dev/null \\\n"
        '    | python3 -c "import sys,json;print(json.load(sys.stdin))" || true\n'
    )
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    # The logical line starts at physical line 2 (the `cat ... \` line).
    assert "x.sh:2" in errors[0]


def test_check_pipe_python_fail_dash_m_and_python311(tmp_path):
    """FAIL — `| python -m json.tool` and `| python3.11 -c` are both
    bare-interpreter consumer pipes."""
    (tmp_path / "x.sh").write_text('foo | python -m json.tool\nbar | python3.11 -c "x"\n')
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert len(errors) == 2, f"expected exactly two errors, got: {errors}"


def test_check_pipe_python_pass_uv_run_python(tmp_path):
    """PASS — the CORRECT `| uv run python -c` shape: the token right
    after the pipe is `uv`, not the bare interpreter."""
    (tmp_path / "x.sh").write_text(
        'cat x.json | uv run python -c "import sys,json;print(json.load(sys.stdin))"\n'
    )
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (uv run python), got: {errors}"


def test_check_pipe_python_pass_comment_line_skipped(tmp_path):
    """PASS — only `#`-comment lines are skipped: a comment that carries the
    bad `| python -c` substring is documentation, not a live pipe."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        '# bad shape to avoid: cat x | python -c "..."\n'
        'echo "all good"\n'  # plain echo, no `| python -c` — not flagged
    )
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (comment line skipped), got: {errors}"


def test_check_pipe_python_fail_echo_producer_pipe(tmp_path):
    """FAIL (F1, the round-1 merge-blocker) — `echo '{}' | python -c "..."`
    is a REAL producer pipe: echo's stdout is consumed by bare `python`,
    which crashes exit 127 on this VM. The earlier blanket `echo `-skip
    silently missed exactly this must-catch shape. `echo `-prefixed lines
    are NOT skipped; the check must return exactly one error for this
    line."""
    (tmp_path / "x.sh").write_text("#!/usr/bin/env bash\necho '{}' | python -c \"print(1)\"\n")
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert len(errors) == 1, (
        f"an `echo ... | python -c` producer pipe must be flagged (F1): {errors}"
    )
    assert "x.sh:2" in errors[0]
    assert "uv run python" in errors[0]


def test_check_pipe_python_fail_substring_in_nonskipped_quoted_string(tmp_path):
    """FAIL (the honest 'known limitation') — a NON-comment line whose
    quoted string merely CONTAINS `| python -c` WOULD match the line-local
    regex. This documents that the lint is not quote-aware and the ONLY
    skipped class is `#`-comment lines (post-F1, `echo` lines are no longer
    special) — keeping the 'known limitation' prose honest rather than
    silently broader than stated. To document the bad pattern, use a
    `#`-comment, not a quoted/echo string."""
    (tmp_path / "x.sh").write_text('MSG="bad shape: cat x | python -c foo"\n')
    errors = check_pipe_python(scripts_dir=tmp_path)
    assert len(errors) == 1, (
        "a non-comment line with `| python -c` inside a quoted "
        f"string is expected to match (known recall-vs-precision edge): {errors}"
    )


def test_check_pipe_python_pass_no_files(tmp_path):
    """PASS — an empty scripts dir (no `*.sh`) yields no errors."""
    assert check_pipe_python(scripts_dir=tmp_path) == []


def test_check_pipe_python_repo_tree_is_clean():
    """The committed scripts/*.sh tree must carry no bare `| python -c/-m`
    consumer pipes — this is the regression guard the durable fix installs
    (the #753 change rewired the 2 existing offenders,
    run_issue452_deconfound.sh and run_program_orchestrator.sh, to
    `| uv run python -c`)."""
    errors = check_pipe_python()
    assert errors == [], (
        "scripts/*.sh has bare `| python -c/-m` consumer pipes "
        "(#753 exit-127 crash class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_pipe_python_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-pipe-python")
    assert result.returncode == 0, (
        f"workflow_lint --check-pipe-python failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_pipe_python_bundled_in_no_flags_source_pin():
    """NON-VACUOUS no-flags bundling pin (#1233; the #712 §4f
    opt-in-not-bundled shipping class): `check_pipe_python` must be
    dispatched by the BARE ``workflow_lint.py`` run. Source-inspection
    assert on the dispatch branch + the no_flags detection-tuple
    membership — the prior exit-0-on-a-clean-tree assert was vacuous (a
    clean tree exits 0 whether or not the check is dispatched) and
    burned a redundant full no-flags subprocess run
    (``test_workflow_lint_default_exits_zero`` keeps the behavioral
    clean-tree cover; the planted-offender function tests above cover
    detection)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_pipe_python or no_flags:\s*\n"
        r"\s*errors\.extend\(check_pipe_python\(\)\)",
        src,
    ), "check_pipe_python is not dispatched on the no-flags branch"
    assert "or args.check_pipe_python" in src, (
        "--check-pipe-python is missing from the no_flags detection tuple"
    )


def _load_shipped_pipe_python_hook_ere() -> str:
    """Extract the pipe-into-python PreToolUse hook's POSIX ERE from the
    SHIPPED `.claude/settings.json` (F2: source the regex from the live
    config, never a hard-coded second copy — a hard-coded copy stays green
    while the production hook drifts, defeating acceptance-criterion-4).

    The pipe-python hook is identified by its unique BLOCKED message marker
    (`BLOCKED: bare \\`| python -c/-m\\` pipe`), which disambiguates it from
    the 4 OTHER PreToolUse Bash hooks. Returns the ERE between the hook's
    `grep -qE '...'` single quotes (un-escaping the JSON `\\\\` → `\\`)."""
    import json
    import re

    settings = json.loads((_REPO_ROOT / ".claude" / "settings.json").read_text())
    for hook_block in settings.get("hooks", {}).get("PreToolUse", []):
        if hook_block.get("matcher") != "Bash":
            continue
        for hook in hook_block.get("hooks", []):
            cmd = hook.get("command", "")
            if "BLOCKED: bare `| python -c/-m` pipe" not in cmd:
                continue
            m = re.search(r"grep -qE '([^']+)'", cmd)
            assert m, f"pipe-python hook found but no `grep -qE '...'` pattern in: {cmd!r}"
            # The command is a JSON string; `\\` in the file is a single
            # backslash in the parsed value — json.loads already un-escaped
            # it, so `m.group(1)` is the literal ERE the shell `grep` sees.
            return m.group(1)
    raise AssertionError("pipe-python PreToolUse Bash hook not found in .claude/settings.json")


def test_check_pipe_python_dual_engine_agreement_on_example_set():
    """Implementer-note dual-engine test (acceptance criterion 4): the
    Python `re` lint regex (`PIPE_PYTHON_RE`) and the POSIX-ERE hook regex
    run through a real `grep -qE` subprocess must AGREE on every §4
    example — match the MATCHES set (incl. the F3 attached-arg shapes),
    reject the NOMATCH set. Post-F3 the two boundaries (`-[cm]\\b` /
    `-[cm]([^A-Za-z0-9_]|$)`) are semantically identical, so there is no
    longer a divergence edge — the engines agree everywhere.

    The hook regex is SOURCED FROM `.claude/settings.json` (F2): the test
    parses the shipped `PreToolUse` Bash hook and extracts its
    `grep -qE '...'` pattern, so the production hook cannot drift without
    breaking this test (a hard-coded copy would stay green on drift)."""
    from workflow_lint import PIPE_PYTHON_RE  # the Python `re` lint regex

    # F2: the POSIX-ERE hook regex, extracted from the SHIPPED settings.json.
    hook_ere = _load_shipped_pipe_python_hook_ere()

    def grep_matches(s: str) -> bool:
        """Run the hook's POSIX engine exactly as the hook does:
        `echo "$cmd" | grep -qE '<hook_ere>'`."""
        proc = subprocess.run(
            ["grep", "-qE", hook_ere],
            input=s + "\n",
            text=True,
            check=False,
        )
        return proc.returncode == 0

    # Both engines MATCH every real failure (incl. the F3 attached-arg shapes).
    for s in _PIPE_PYTHON_MATCHES:
        lint = bool(PIPE_PYTHON_RE.search(s))
        hook = grep_matches(s)
        assert lint, f"lint regex must match must-catch case: {s!r}"
        assert hook, f"hook regex must match must-catch case: {s!r}"
        assert lint == hook, f"engines diverge on a MATCH case (should not): {s!r}"

    # Both engines REJECT every clean (non-offender) shape.
    for s in _PIPE_PYTHON_NOMATCH:
        lint = bool(PIPE_PYTHON_RE.search(s))
        hook = grep_matches(s)
        assert not lint, f"lint regex must NOT match clean case: {s!r}"
        assert not hook, f"hook regex must NOT match clean case: {s!r}"
        assert lint == hook, f"engines diverge on a NOMATCH case (should not): {s!r}"

    # F3 regression: the in-string substring + attached-arg edges now AGREE
    # across engines (the old divergence is gone with the aligned boundary).
    for s in ('MSG="bad: foo | python -c"', "cat x | python -c'print(1)'"):
        assert bool(PIPE_PYTHON_RE.search(s)) == grep_matches(s), (
            f"engines must agree on the F3 edge case: {s!r}"
        )


def test_pipe_python_hook_subprocess_blocks_attached_arg():
    """F3 hook-subprocess test — the SHIPPED PreToolUse hook command, fed
    the harness JSON-stdin shape, must `exit 2` on the attached-argument
    form `cat x | python -c'print(1)'` (valid shell that crashes exit 127
    on this VM) and exit 0 on the correct `| uv run python -c` form. Runs
    the real hook command string from settings.json end-to-end (not just
    the extracted regex), so an escaping break in the JSON `command` is
    caught too."""
    import json

    settings = json.loads((_REPO_ROOT / ".claude" / "settings.json").read_text())
    hook_cmd = None
    for hook_block in settings.get("hooks", {}).get("PreToolUse", []):
        if hook_block.get("matcher") != "Bash":
            continue
        for hook in hook_block.get("hooks", []):
            cmd = hook.get("command", "")
            if "BLOCKED: bare `| python -c/-m` pipe" in cmd:
                hook_cmd = cmd
    assert hook_cmd, "pipe-python PreToolUse Bash hook not found"

    def run_hook(command: str) -> int:
        stdin = json.dumps({"tool_input": {"command": command}})
        proc = subprocess.run(["bash", "-c", hook_cmd], input=stdin, text=True, check=False)
        return proc.returncode

    assert run_hook("cat x | python -c'print(1)'") == 2, "attached-arg form must be blocked (F3)"
    assert run_hook('cat x.json | python3 -c "import sys"') == 2, "plain pipe must be blocked"
    assert run_hook('cat x | uv run python -c "x"') == 0, "uv run python must pass"


# ---------------------------------------------------------------------------
# Unit tests for ``check_piped_git_push`` (incident class #957 / #1048: a
# `git push` / `git merge` / `git commit` / `gh pr merge|create` piped into
# a filter masks
# the producer's non-zero exit code — bash makes the pipeline's status the
# LAST stage's — so a rejected push reads as success; 4 sessions hit the
# class on 2026-07-02 and #957's Step 10d push was masked 2026-07-04; the
# commit verb was added by #1591 after #1584's piped commit was
# SIGPIPE-killed mid-pre-commit-hook). Each
# fixture case writes a tiny ``*.sh`` under ``tmp_path`` and calls
# ``check_piped_git_push(scripts_dir=tmp_path)``. The hook/lint agreement
# test drives the SHARED semantic subset through BOTH the
# ``PIPED_GIT_PUSH_RE`` lint predicate and the
# ``.claude/hooks/guard_piped_git_push.sh`` subprocess.
# ---------------------------------------------------------------------------


def test_check_piped_git_push_fail_simple_pipe(tmp_path):
    """FAIL — the flagship incident shape `git push origin main 2>&1 |
    tail -20` (the pipe masks a rejected push; #957)."""
    (tmp_path / "x.sh").write_text("#!/usr/bin/env bash\ngit push origin main 2>&1 | tail -20\n")
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]
    assert "#957" in errors[0]
    assert "pipefail" in errors[0]


def test_check_piped_git_push_fail_gh_pr_merge(tmp_path):
    """FAIL — `gh pr merge ... | head` masks a failed merge the same way
    (the prose rule's 'merge/PR/`git commit` command' clause)."""
    (tmp_path / "x.sh").write_text("gh pr merge 123 --squash | head\n")
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:1" in errors[0]


def test_check_piped_git_push_fail_backslash_continued(tmp_path):
    """FAIL — the backslash-continued shape (`git push ... \\` newline
    `| tail`), merged into one logical line; the error points at the FIRST
    physical line (the #753 offender-shape analog)."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\ngit push origin main 2>&1 \\\n    | tail -20\n"
    )
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]


def test_check_piped_git_push_fail_pipe_amp_shorthand(tmp_path):
    """FAIL — `|&` (bash's `2>&1 |` shorthand) is normalized to `|` on the
    logical line before matching."""
    (tmp_path / "x.sh").write_text("git push |& tail -5\n")
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_piped_git_push_pass_or_chain(tmp_path):
    """PASS — `git push ... || echo failed` is a disjunction, not a pipe
    (the sole real tree shape, issue931_dispatch.sh:253)."""
    (tmp_path / "x.sh").write_text(
        'git push origin "issue-931" || echo "[i931] WARNING: push failed"\n'
    )
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_comment_line_skipped(tmp_path):
    """PASS — a `#`-comment carrying the bad pattern is documentation."""
    (tmp_path / "x.sh").write_text("#!/usr/bin/env bash\n# never do: git push | tail -5\necho ok\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_pipefail_header_file(tmp_path):
    """PASS — a `set -euo pipefail` header makes every later pipe propagate
    the producer's failure (the rule's own sanctioned escape)."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ngit push 2>&1 | tee push.log\n"
    )
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_fail_offense_before_later_pipefail(tmp_path):
    """FAIL (plan #1048 MF3, fires-direction) — an offense BEFORE a LATER
    `set -o pipefail` line yields EXACTLY ONE error: the pipefail tracking
    skips only the REST of the file after the first pipefail line, never a
    whole-file pre-scan (which would false-allow the earlier offense)."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        "git push origin main | tail -5\n"
        "echo mid\n"
        "set -o pipefail\n"
        "git push 2>&1 | tee log\n"
    )
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]


def test_check_piped_git_push_fail_raw_newline_logical_lines(tmp_path):
    """FAIL exactly once (the hook B10/A16 mirror) — physical lines are
    independent logical lines: a cross-line `git status | grep x` +
    `git push origin main | tail -5` file flags only the piped-push line."""
    (tmp_path / "x.sh").write_text("git status | grep x\ngit push origin main | tail -5\n")
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]


def test_check_piped_git_push_pass_merge_base(tmp_path):
    """PASS — `git merge-base ... | head -1` (a canonical
    .claude/rules/diff-size-budget.md probe): the verb must be followed by
    whitespace-or-pipe, so `merge-base` never matches."""
    (tmp_path / "x.sh").write_text("git merge-base --all main HEAD | head -1\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_producer_as_consumer(tmp_path):
    """PASS — `echo foo | git push`: the producer is the FINAL stage, whose
    exit code IS the pipeline's — nothing is masked."""
    (tmp_path / "x.sh").write_text("echo foo | git push\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_dry_run(tmp_path):
    """PASS — a `--dry-run` push may pipe: it lands nothing, so masking its
    exit code cannot cause the proceeded-on-a-rejected-push incident."""
    (tmp_path / "x.sh").write_text("git push --dry-run 2>&1 | head -5\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_no_files(tmp_path):
    """PASS — an empty scripts dir (no `*.sh`) yields no errors."""
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_fail_piped_commit(tmp_path):
    """FAIL — a piped `git commit` (#1591): the masked-exit harm plus the
    #1584 SIGPIPE-mid-pre-commit-hook kill."""
    (tmp_path / "x.sh").write_text('#!/usr/bin/env bash\ngit commit -m "wip" 2>&1 | head -20\n')
    errors = check_piped_git_push(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert "commit" in errors[0]


def test_check_piped_git_push_pass_commit_dry_run(tmp_path):
    """PASS — the verb-independent `--dry-run` span skip covers the commit
    form (a dry-run commit lands nothing and runs no pre-commit hook)."""
    (tmp_path / "x.sh").write_text("#!/usr/bin/env bash\ngit commit --dry-run 2>&1 | head -5\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_pass_commit_as_consumer(tmp_path):
    """PASS — a message piped INTO commit (`cat msg | git commit -F -`) is
    producer-as-consumer/final stage: nothing is masked."""
    (tmp_path / "x.sh").write_text("#!/usr/bin/env bash\ncat msg.txt | git commit -F -\n")
    assert check_piped_git_push(scripts_dir=tmp_path) == []


def test_check_piped_git_push_repo_tree_is_clean():
    """The committed scripts/*.sh tree must carry no piped push/merge/
    commit-class commands — the regression lock (the plan #1048 §2 item-8
    scan found the tree clean: the sole `git push`+`|` hit,
    issue931_dispatch.sh:253, is an `||` disjunction; the #1591 widened-verb
    re-scan measured 0 commit-class hits too)."""
    errors = check_piped_git_push()
    assert errors == [], (
        "scripts/*.sh has piped git push/merge-class commands "
        "(#957 masked-exit-code class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_piped_git_push_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-piped-git-push")
    assert result.returncode == 0, (
        f"workflow_lint --check-piped-git-push failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_piped_git_push_bundled_in_no_flags_source_pin():
    """NON-VACUOUS no-flags bundling pin (#1293, the #1233 exemplar-2
    shape; the #712 §4f opt-in-not-bundled shipping class):
    `check_piped_git_push` must be dispatched by the BARE
    ``workflow_lint.py`` run. Source-inspection assert on the dispatch
    branch + the no_flags detection-tuple membership — the prior
    exit-0-on-a-clean-tree assert was vacuous (a clean tree exits 0
    whether or not the check is dispatched) and burned a redundant full
    no-flags subprocess run (``test_workflow_lint_default_exits_zero``
    keeps the behavioral clean-tree cover; the ``check_piped_git_push``
    function tests above and the ``_PIPED_PUSH_SHARED`` hook/lint
    agreement suite below cover detection)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_piped_git_push or no_flags:\s*\n"
        r"\s*errors\.extend\(check_piped_git_push\(\)\)",
        src,
    ), "check_piped_git_push is not dispatched on the no-flags branch"
    assert "or args.check_piped_git_push" in src, (
        "--check-piped-git-push is missing from the no_flags detection tuple"
    )


# The SHARED semantic subset for the hook/lint agreement test: shapes whose
# verdict both engines must agree on. Deliberate DIVERGENCES excluded here
# (named per plan §4.6): the HOOK alone carries the heredoc blanket-allow
# and the EPM_ALLOW_PIPED_PUSH inline/env escape hatch (runtime-only
# affordances); the LINT alone carries file-level pipefail tracking, which
# the single-command hook expresses as a whole-command `pipefail` substring
# check — so pipefail/heredoc/escape-hatch shapes are NOT in the subset.
_PIPED_PUSH_SHARED = [
    # (command, must_flag)
    ("git push | tail -5", True),  # B1 plain pipe
    ("git push origin main 2>&1 | grep -v x", True),  # B2 redirection crossing
    ("gh pr merge 123 --squash | head", True),  # B3 gh producer
    ("git merge issue-x 2>&1 | tail -5", True),  # B7 git merge
    ("git push |& tail -5", True),  # B9 |& shorthand
    ('git commit -m "wip" 2>&1 | head -20', True),  # B12 piped commit (#1584/#1591)
    ('git commit -m "wip" |& head -3', True),  # B17 |& shorthand, commit form
    ('git push origin main || echo "push failed"', False),  # A7 || chain
    ("git merge-base --all main HEAD | head -1", False),  # A9 merge-base
    ("echo foo | git push", False),  # A14 producer as consumer
    ("git status | grep x && git push", False),  # A5 different segment
    ("git push --dry-run 2>&1 | head -5", False),  # A8 dry-run carve-out
    ("cat msg.txt | git commit -F -", False),  # A21 commit as consumer/final stage
    ("git commit --dry-run 2>&1 | head -5", False),  # A20 dry-run carve-out, commit form
    ("git commit-tree HEAD^{tree} -m x | head -1", False),  # A22 verb-prefix word
]


def test_piped_git_push_hook_lint_agreement_on_shared_cases():
    """Hook/lint dual-engine agreement on the SHARED semantic subset: the
    lint predicate (`PIPED_GIT_PUSH_RE` + the `|&` normalization + the
    `--dry-run` span skip, exactly as `check_piped_git_push` applies them)
    and the shipped hook script driven as a subprocess must agree on every
    shared case — plain-pipe blocks, `||` allows, merge-base allows,
    producer-as-consumer allows, cross-segment allows, dry-run allows.

    FULL equivalence is deliberately NOT asserted: the hook alone carries
    the heredoc blanket-allow + the EPM_ALLOW_PIPED_PUSH escape hatch
    (runtime affordances a committed script must not rely on), and the lint
    alone carries file-level pipefail tracking (the hook sees ONE command
    and uses a whole-command `pipefail` substring check instead). Those
    divergent shapes are excluded from the subset above.
    """
    import json as _json
    import os as _os

    from workflow_lint import PIPED_GIT_PUSH_RE

    hook = _REPO_ROOT / ".claude" / "hooks" / "guard_piped_git_push.sh"
    assert hook.exists(), hook
    env = {k: v for k, v in _os.environ.items() if k != "EPM_ALLOW_PIPED_PUSH"}

    def lint_flags(cmd: str) -> bool:
        m = PIPED_GIT_PUSH_RE.search(cmd.replace("|&", "|"))
        return bool(m) and "--dry-run" not in m.group(0)

    def hook_flags(cmd: str) -> bool:
        proc = subprocess.run(
            [str(hook)],
            input=_json.dumps({"tool_input": {"command": cmd}}),
            text=True,
            capture_output=True,
            env=env,
        )
        assert proc.returncode in (0, 2), (proc.returncode, proc.stderr)
        return proc.returncode == 2

    for cmd, must_flag in _PIPED_PUSH_SHARED:
        lint = lint_flags(cmd)
        hook_v = hook_flags(cmd)
        assert lint == must_flag, f"lint verdict wrong for {cmd!r}: {lint} != {must_flag}"
        assert hook_v == must_flag, f"hook verdict wrong for {cmd!r}: {hook_v} != {must_flag}"
        assert lint == hook_v, f"engines diverge on shared case {cmd!r}"


# ---------------------------------------------------------------------------
# Unit tests for ``check_push_failure_swallow`` (incident class #825
# r6/r7/r8, task #1205: a workload's `git push ... || echo WARNING`
# swallowed a deterministic auth failure, the step declared success, and
# the self-DELETEing GCE instance held the only copy of 73 committed eval
# JSONs). Each fixture writes a tiny ``*.sh`` under ``tmp_path`` and calls
# ``check_push_failure_swallow(scripts_dir=tmp_path)``. The workload-side
# `||` sibling of ``check_piped_git_push`` — NO pipefail escape here
# (pipefail never applies to `||` disjunctions).
# ---------------------------------------------------------------------------


def test_check_push_failure_swallow_fail_echo(tmp_path):
    """FAIL — the flagship incident shape `git push origin x || echo warn`
    (#825 r8: issue825_sampled_sep_dispatch.sh:502)."""
    (tmp_path / "x.sh").write_text(
        '#!/usr/bin/env bash\ngit push origin "issue-9" || echo "WARNING: push failed" >&2\n'
    )
    errors = check_push_failure_swallow(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]
    assert "#825" in errors[0]
    assert "PUSH_SWALLOW_EXEMPT" in errors[0]


def test_check_push_failure_swallow_fail_true_colon_printf(tmp_path):
    """FAIL — the `|| true`, `|| :`, and `|| printf` swallow variants each
    flag (one error per line), including the flag-tolerant `git -C` form."""
    (tmp_path / "x.sh").write_text(
        "git push || true\n"
        'git -C "$ROOT" push origin main || :\n'
        "git push origin main || printf 'warn\\n'\n"
    )
    errors = check_push_failure_swallow(scripts_dir=tmp_path)
    assert len(errors) == 3, f"expected three errors, got: {errors}"


def test_check_push_failure_swallow_fail_backslash_continued(tmp_path):
    """FAIL — `git push origin x \\` newline `  || echo warn` is ONE
    logical line (backslash continuations merged before matching); the
    error points at the FIRST physical line."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\ngit push origin main \\\n  || echo 'push failed'\n"
    )
    errors = check_push_failure_swallow(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:2" in errors[0]


def test_check_push_failure_swallow_fail_despite_pipefail(tmp_path):
    """FAIL — unlike the piped-push sibling, a `set -euo pipefail` header
    is NO escape: pipefail never applies to `||` disjunctions."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ngit push origin main || echo warn\n"
    )
    errors = check_push_failure_swallow(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_push_failure_swallow_pass_safe_shapes(tmp_path):
    """PASS — the three verified safe shapes on the live tree: an
    if-condition (auto_push_main.sh:23 — the rc is CONSUMED), a bare push
    (cron_export_literature.sh:41 — set -e propagates), and the
    `|| { retry; } || true` group (the rendered #1205 GCE leg's own retry:
    the re-count after it is the verification)."""
    (tmp_path / "x.sh").write_text(
        "if git push origin main; then\n"
        "  echo pushed\n"
        "fi\n"
        "git push origin main\n"
        'git push origin "HEAD:main" || { sleep 20; git push origin "HEAD:main"; } || true\n'
    )
    assert check_push_failure_swallow(scripts_dir=tmp_path) == []


def test_check_push_failure_swallow_pass_comment_line(tmp_path):
    """PASS — a `#`-comment carrying the bad pattern is documentation."""
    (tmp_path / "x.sh").write_text("# never do: git push || echo warn\necho ok\n")
    assert check_push_failure_swallow(scripts_dir=tmp_path) == []


def test_check_push_failure_swallow_pass_waiver(tmp_path):
    """PASS — a reason-bearing `# PUSH_SWALLOW_EXEMPT:` waiver on the same
    line (and the preceding-line placement for continued commands)."""
    (tmp_path / "x.sh").write_text(
        "git push origin main || echo warn  "
        "# PUSH_SWALLOW_EXEMPT: mirror push, verified by the next step\n"
        "# PUSH_SWALLOW_EXEMPT: preceding-line waiver for the continued form\n"
        "git push origin main \\\n  || echo warn\n"
    )
    assert check_push_failure_swallow(scripts_dir=tmp_path) == []


def test_check_push_failure_swallow_pass_frozen_allowlist(tmp_path):
    """PASS — a file whose repo-root-relative path is in the FROZEN
    PUSH_SWALLOW_LEGACY_ALLOWLIST (the on-main issue931 offender + the
    pre-seeded issue-825 sep-dispatch siblings) is skipped wholesale; a
    same-shape NEW script is still flagged."""
    (tmp_path / "issue931_dispatch.sh").write_text(
        'git push origin "issue-931" || echo "[i931] WARNING: git push failed" >&2\n'
    )
    (tmp_path / "issue825_sampled_sep_dispatch.sh").write_text(
        'git push origin "issue-825" || echo "[i825-ss] WARNING: git push failed" >&2\n'
    )
    (tmp_path / "new_dispatch.sh").write_text("git push origin main || echo warn\n")
    errors = check_push_failure_swallow(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected only the NEW script flagged, got: {errors}"
    assert "new_dispatch.sh" in errors[0]


def test_check_push_failure_swallow_pass_no_files(tmp_path):
    """PASS — an empty scripts dir (no `*.sh`) yields no errors."""
    assert check_push_failure_swallow(scripts_dir=tmp_path) == []


def test_check_push_failure_swallow_repo_tree_is_clean():
    """The committed scripts/*.sh tree must carry no push-failure swallows
    outside the frozen allowlist — the regression lock (the #1205 scan of
    main + every issue-* branch found exactly the four allowlisted
    offenders)."""
    errors = check_push_failure_swallow()
    assert errors == [], (
        "scripts/*.sh has git-push failure swallows (#825 r6-r8 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_push_failure_swallow_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-push-failure-swallow")
    assert result.returncode == 0, (
        f"workflow_lint --check-push-failure-swallow failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_push_failure_swallow_bundled_in_no_flags_source_pin():
    """`check_push_failure_swallow` is wired into the no-flags default run
    — pinned STRUCTURALLY (the source carries the `or no_flags` dispatch
    and the no_flags-tuple membership) so the bundling cannot silently
    drop; the sibling clean-tree CLI test plus the shared no-flags run
    cover the behavioral side."""
    src = (_REPO_ROOT / "scripts" / "workflow_lint.py").read_text()
    assert "args.check_push_failure_swallow or no_flags" in src
    assert "or args.check_push_failure_swallow" in src


# ---------------------------------------------------------------------------
# Unit tests for ``check_sh_function_rc_capture`` (incident class #1426,
# task #1516: `run_seed "$s" || rc=$?` on a same-file function under
# `set -euo pipefail` — bash disables errexit inside the function BODY in
# an `||` context, so a Gate-1 terminal failure + a manifest SystemExit
# collapsed to rc=0 and the `[phase=done]` success sentinel fired). Each
# fixture writes a tiny ``*.sh`` under ``tmp_path`` and calls
# ``check_sh_function_rc_capture(scripts_dir=tmp_path)``. Single
# external-command captures (the entire live `|| rc=$?` population) never
# match — the invocation regex requires a collected same-file function
# name at command position.
# ---------------------------------------------------------------------------


def test_check_sh_function_rc_capture_fail_rc_capture(tmp_path):
    """FAIL — the #1426 reconstruction: `run_seed "$s" || rc=$?` on a
    same-file function under `set -euo pipefail` collapses mid-function
    failures to the last command's rc."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "run_seed() {\n"
        "  echo running\n"
        "}\n"
        'run_seed "$s" || rc=$?\n'
    )
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:6" in errors[0]
    assert "run_seed" in errors[0]
    assert "#1426" in errors[0]
    assert "RC_CAPTURE_EXEMPT" in errors[0]


def test_check_sh_function_rc_capture_fail_or_true(tmp_path):
    """FAIL — the `|| true` swallow variant on a same-file function under
    `set -e` is the same body-errexit-suppression footgun."""
    (tmp_path / "x.sh").write_text("set -e\nmyfn() { echo hi; }\nmyfn || true\n")
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]
    assert "myfn" in errors[0]


def test_check_sh_function_rc_capture_fail_shebang_e(tmp_path):
    """FAIL — no `set` line at all; the `#!/bin/bash -e` shebang sets the
    initial errexit state ON."""
    (tmp_path / "x.sh").write_text("#!/bin/bash -e\nmyfn() { echo hi; }\nmyfn || true\n")
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


def test_check_sh_function_rc_capture_fail_backslash_continued(tmp_path):
    """FAIL — `myfn "$a" \\` newline `  || rc=$?` is ONE logical line
    (backslash continuations merged); the error points at the FIRST
    physical line."""
    (tmp_path / "x.sh").write_text('set -e\nmyfn() { echo hi; }\nmyfn "$a" \\\n  || rc=$?\n')
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


def test_check_sh_function_rc_capture_fail_mid_chain(tmp_path):
    """FAIL — `cd "$d" && myfn || rc=$?`: the function sits at command
    position after `&&`, and the `||` context still disables body errexit
    (equal-precedence left-associative chaining)."""
    (tmp_path / "x.sh").write_text('set -e\nmyfn() { echo hi; }\ncd "$d" && myfn || rc=$?\n')
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


def test_check_sh_function_rc_capture_pass_external_command(tmp_path):
    """PASS — the entire live `|| rc=$?` population: single EXTERNAL
    commands captured in a file that also defines a function. A child
    process keeps its own set -e, so these are safe by construction."""
    (tmp_path / "x.sh").write_text(
        "set -e\n"
        "helper() { echo hi; }\n"
        "uv run python scripts/x.py || rc=$?\n"
        "git push origin main || rc=$?\n"
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_no_set_e(tmp_path):
    """PASS — identical to the #1426 reconstruction minus the
    `set -euo pipefail` line: without errexit the function body never had
    implicit protection, so the capture is the author's explicit error
    handling, not the footgun."""
    (tmp_path / "x.sh").write_text(
        '#!/usr/bin/env bash\nrun_seed() {\n  echo running\n}\nrun_seed "$s" || rc=$?\n'
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_set_plus_e_region_then_reflag(tmp_path):
    """Line-order state tracking: a hit inside a `set +e` region does not
    flag; the SAME shape after `set -e` re-enables does — exactly one
    error, at the second site."""
    (tmp_path / "x.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "myfn() { echo hi; }\n"
        "set +e\n"
        "myfn || rc=$?\n"
        "set -e\n"
        "myfn || rc=$?\n"
    )
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:7" in errors[0]


def test_check_sh_function_rc_capture_pass_bare_invocation(tmp_path):
    """PASS — a bare invocation and a `;`-separated `rc=$?` (no `||`)
    leave errexit live inside the body; neither is the footgun."""
    (tmp_path / "x.sh").write_text('set -e\nmyfn() { echo hi; }\nmyfn "$s"\nmyfn; rc=$?\n')
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_def_line_one_liner(tmp_path):
    """PASS — the issue1345 shape: a one-liner DEFINITION whose body
    legitimately ends `|| true` is not an invocation."""
    (tmp_path / "x.sh").write_text(
        'set -e\nno_r4() { [[ -f "$F" ]] && echo "--x" || true; }\necho done\n'
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_quoted_string(tmp_path):
    """PASS — the bootstrap_pod shape: `ssh_cmd 'cd /x && do_thing ||
    true'` where ssh_cmd IS a same-file function; the `|| true` is
    remote-side TEXT inside the quoted argument (quote masking kills the
    guaranteed false positive)."""
    (tmp_path / "x.sh").write_text(
        "set -e\nssh_cmd() { ssh pod \"$1\"; }\nssh_cmd 'cd /x && do_thing || true'\n"
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_semicolon_segment(tmp_path):
    """PASS — `myfn; cleanup_external || true`: the suppressor sits in a
    LATER `;`-segment and belongs to that segment's own (external)
    command, not to the function invocation."""
    (tmp_path / "x.sh").write_text("set -e\nmyfn() { echo hi; }\nmyfn; cleanup_external || true\n")
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_waiver(tmp_path):
    """PASS — a reason-bearing `# RC_CAPTURE_EXEMPT:` waiver on the same
    line (and the preceding-line placement for continued commands); a
    short reason (<10 chars) does NOT waive."""
    (tmp_path / "ok.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "myfn() { echo hi; }\n"
        "myfn || true  # RC_CAPTURE_EXEMPT: body handles its own errors\n"
        "# RC_CAPTURE_EXEMPT: preceding-line waiver for the continued form\n"
        "myfn \\\n"
        "  || rc=$?\n"
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []
    (tmp_path / "short.sh").write_text(
        "set -e\nmyfn() { echo hi; }\nmyfn || true  # RC_CAPTURE_EXEMPT: x\n"
    )
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected only the short-reason file flagged, got: {errors}"
    assert "short.sh" in errors[0]


def test_check_sh_function_rc_capture_pass_comment_and_heredoc(tmp_path):
    """PASS — a commented-out invocation is documentation, and a heredoc
    BODY carrying `myfn || true` (remote-command text) is consumed and
    never scanned as shell."""
    (tmp_path / "x.sh").write_text(
        "set -e\nmyfn() { echo hi; }\n# myfn || true\ncat <<'EOF'\nmyfn || true\nEOF\necho done\n"
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_comment_e_token_on_set_line(tmp_path):
    """PASS — a `set` line whose only `-e`-bearing token lives in a trailing
    COMMENT (`set -uo pipefail  # NOT set -e -- ...`) must not flip errexit
    ON: the transition scanner comment-strips first (live shapes:
    i632_dispatch_with_log_capture.sh:12, issue683_dispatch.sh:30)."""
    (tmp_path / "x.sh").write_text(
        "#!/bin/bash\n"
        "set -uo pipefail  # NOT set -e -- workers own their rc handling\n"
        "myfn() { echo hi; }\n"
        'myfn "$a" || rc=$?\n'
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_no_files(tmp_path):
    """PASS — an empty scripts dir (no `*.sh`) yields no errors."""
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_repo_tree_is_clean():
    """The committed scripts/**/*.sh tree must carry no same-file
    function `|| rc=$?`/`|| true` captures under set -e — the 0-findings
    state the no-flags bundling was preconditioned on (#1516 plan A3:
    prototype run over all 149 rglob files -> 0 findings)."""
    errors = check_sh_function_rc_capture()
    assert errors == [], (
        "scripts/**/*.sh has same-file function rc-captures (#1426 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_sh_function_rc_capture_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-sh-function-rc-capture")
    assert result.returncode == 0, (
        f"workflow_lint --check-sh-function-rc-capture failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_sh_function_rc_capture_bundled_in_no_flags_source_pin():
    """`check_sh_function_rc_capture` is wired into the no-flags default
    run — pinned STRUCTURALLY (the source carries the `or no_flags`
    dispatch, the no_flags-tuple membership, AND the callee itself: a
    copy-paste dispatch of the wrong check under the correct conditional
    would otherwise pass the sibling pins green)."""
    src = (_REPO_ROOT / "scripts" / "workflow_lint.py").read_text()
    assert "args.check_sh_function_rc_capture or no_flags" in src
    assert "or args.check_sh_function_rc_capture" in src
    assert "errors.extend(check_sh_function_rc_capture())" in src


def test_check_sh_function_rc_capture_fail_spelling_variants(tmp_path):
    """FAIL — capture-SPELLING variants: `|| status=$?` (any simple var
    name, not the literal `rc`) and `|| :` (the colon builtin, an exact
    `|| true` synonym) each flag — exactly two errors."""
    (tmp_path / "x.sh").write_text("set -e\nmyfn() { echo hi; }\nmyfn || status=$?\nmyfn || :\n")
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 2, f"expected exactly two errors, got: {errors}"


def test_check_sh_function_rc_capture_pass_brace_group_handler(tmp_path):
    """PASS — brace-group handlers (`|| { echo FATAL; exit 1; }`,
    `|| { rc=$?; }`) are documented out-of-scope residuals: pins that the
    suppress regex does NOT over-reach into brace groups (they are a
    pervasive live fail-loud idiom; flagging them would break the
    0-findings bundling precondition)."""
    (tmp_path / "x.sh").write_text(
        "set -e\nmyfn() { echo hi; }\nmyfn || { echo FATAL; exit 1; }\nmyfn || { rc=$?; }\n"
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_pass_trailing_comment(tmp_path):
    """PASS — suppressor text ONLY in a trailing comment (`# TODO: add
    || rc=$? capture`) never flags: pins the quote-aware
    `_strip_sh_trailing_comment` path."""
    (tmp_path / "x.sh").write_text(
        'set -e\nmyfn() { echo hi; }\nmyfn "$s"  # TODO: add || rc=$? capture\n'
    )
    assert check_sh_function_rc_capture(scripts_dir=tmp_path) == []


def test_check_sh_function_rc_capture_fail_set_o_errexit(tmp_path):
    """FAIL then unflag — the long-form `set -o errexit` / `set +o
    errexit` arm (0 live uses; pinned so the completeness arm cannot ship
    untested): exactly one error, at the first site."""
    (tmp_path / "x.sh").write_text(
        "myfn() { echo hi; }\nset -o errexit\nmyfn || true\nset +o errexit\nmyfn || true\n"
    )
    errors = check_sh_function_rc_capture(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:3" in errors[0]


# ---------------------------------------------------------------------------
# Unit tests for ``check_grep_qv`` (incident class #928 -> #1125: ugrep
# 7.5.0's quiet+invert exit status diverges from GNU — rc=1 even when
# non-matching lines are selected — so an rc-consumed q+v grep trigger in an
# executable workflow snippet silently fails OPEN under a PATH-shadowed
# grep; the Step 10d pre-push lint gate classified a 12-file code-bearing
# payload as skip-artifact-only). Each fixture writes a tiny ``*.md`` (with
# a fenced code block, the SKILL.md scan shape) or ``*.sh`` under
# ``tmp_path`` and calls ``check_grep_qv(roots=[tmp_path])``.
# ---------------------------------------------------------------------------


def test_check_grep_qv_flags_combined_token(tmp_path):
    """FAIL — the live #928 trigger shape verbatim: a fenced,
    backslash-continued elif consuming the combined-token quiet+invert
    exit status; the error points at the FIRST physical line."""
    (tmp_path / "SKILL.md").write_text(
        "Prose above.\n"
        "```bash\n"
        "elif grep -qvE '^(tasks/|figures/)' \\\n"
        "    /tmp/issue-1-own-diff.txt; then\n"
        "  echo armed\n"
        "```\n"
    )
    errors = check_grep_qv(roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "SKILL.md:3" in errors[0]
    assert "#928" in errors[0]
    assert "#1125" in errors[0]


def test_check_grep_qv_flags_separated_tokens(tmp_path):
    """FAIL — separated tokens (`-q ... -vE`) in a `.sh` logical line
    combine across the option run exactly as the fused token does."""
    (tmp_path / "x.sh").write_text("if grep -q -vE '^tasks/' files.txt; then\n  echo y\nfi\n")
    errors = check_grep_qv(roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.sh:1" in errors[0]


def test_check_grep_qv_flags_vq_token_order(tmp_path):
    """FAIL — the reversed combined token (`-vq`) is the same rc-consumed
    quiet+invert combination (flag-set membership, not token spelling)."""
    (tmp_path / "x.sh").write_text("grep -vq '^tasks/' files.txt\n")
    errors = check_grep_qv(roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_grep_qv_flags_long_form(tmp_path):
    """FAIL — the long forms (`--quiet --invert-match`, and the `--silent`
    alias) are the same combination spelled out."""
    (tmp_path / "x.sh").write_text(
        "grep --quiet --invert-match '^tasks/' files.txt\n"
        "grep --silent --invert-match '^tasks/' files.txt\n"
    )
    errors = check_grep_qv(roots=[tmp_path])
    assert len(errors) == 2, f"expected exactly two errors, got: {errors}"


def test_check_grep_qv_flags_path_pinned_ugrep(tmp_path):
    """FAIL — a path-pinned `ugrep` is broken BY CONSTRUCTION (its
    quiet+invert rc diverges wherever the binary lives), so the
    path-pin exemption is grep-only."""
    (tmp_path / "x.sh").write_text("/usr/bin/ugrep -qv '^a' f.txt\n")
    errors = check_grep_qv(roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "`ugrep`" in errors[0]


def test_check_grep_qv_allows_path_pinned(tmp_path):
    """PASS — `/usr/bin/grep -qvE` is the sanctioned GNU pin (the #928
    incident's own verified workaround)."""
    (tmp_path / "x.sh").write_text("/usr/bin/grep -qvE '^tasks/' files.txt\n")
    assert check_grep_qv(roots=[tmp_path]) == []


def test_check_grep_qv_allows_single_flag_and_git_grep(tmp_path):
    """PASS — plain `-q` without `-v` (match-found rc agrees across
    implementations: the gate's `grep -qxE` verdict consumers), plain
    `-vE` without `-q` (the output-test rewrite itself), and `git grep`
    (git's own engine, not PATH-shadowable)."""
    (tmp_path / "x.sh").write_text(
        "grep -qxE 'pass|skip-artifact-only' verdict.txt\n"
        "if [ -n \"$(grep -vE '^tasks/' files.txt)\" ]; then echo armed; fi\n"
        "git grep -qv something -- scripts/\n"
    )
    assert check_grep_qv(roots=[tmp_path]) == []


def test_check_grep_qv_pass_pipeline_split(tmp_path):
    """PASS — `-v` and `-q` on DIFFERENT pipeline commands never combine:
    each command word's contiguous option run is evaluated independently."""
    (tmp_path / "x.sh").write_text("grep -v x f | grep -q y f2\n")
    assert check_grep_qv(roots=[tmp_path]) == []


def test_check_grep_qv_skips_prose_and_comments(tmp_path):
    """PASS — the pattern in `.md` prose OUTSIDE a fence and on a
    `#`-comment line INSIDE a fence is documentation, not an executable
    snippet."""
    (tmp_path / "SKILL.md").write_text(
        "Never write grep -qvE in a trigger (prose mention).\n"
        "```bash\n"
        "# banned shape: grep -qvE '^tasks/' file\n"
        "echo ok\n"
        "```\n"
    )
    (tmp_path / "x.sh").write_text("# doc: grep -qvE '^tasks/' file\necho ok\n")
    assert check_grep_qv(roots=[tmp_path]) == []


def test_check_grep_qv_live_tree_passes():
    """The committed workflow surface must carry no unpinned q+v grep
    trigger — the regression lock for the #1125 two-site fix (the
    `test_live_trees_pass` pattern from the judge-pin check). No
    grandfather allowlist exists by design: the post-fix tree is clean."""
    errors = check_grep_qv()
    assert errors == [], (
        "workflow surface has unpinned quiet+invert grep triggers "
        "(#928 ugrep fail-open class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_grep_qv_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-grep-qv")
    assert result.returncode == 0, (
        f"workflow_lint --check-grep-qv failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_dispatcher_cvd_pin`` (incident class #523 Phase B,
# recurred #541/#543/#557; recipe fix #578: the in-process CVD clobber is
# defeated by import-time cuInit, so backgrounded parallel per-cell python
# launches passing --gpu-id/+gpu_id= MUST also pin CUDA_VISIBLE_DEVICES= in
# the launcher env on the same command). Each fixture case writes a tiny
# ``*.sh`` under ``tmp_path`` and calls
# ``check_dispatcher_cvd_pin(scripts_dir=tmp_path)``.
# ---------------------------------------------------------------------------


def test_check_dispatcher_cvd_pin_fail_backgrounded_wave_shape(tmp_path):
    """FAIL — the pre-waiver i460/#523 wave shape: backslash-continued
    backgrounded launch with --gpu-id and no CUDA_VISIBLE_DEVICES=."""
    (tmp_path / "dispatch.sh").write_text(
        "#!/usr/bin/env bash\n"
        'for cond in "${CONDS[@]}"; do\n'
        "    uv run python scripts/foo_train.py \\\n"
        '        --conds "$cond" --gpu-id "$cvd" \\\n'
        '        > "$log" 2>&1 &\n'
        "done\n"
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "dispatch.sh:3" in errors[0]
    assert "CUDA_VISIBLE_DEVICES" in errors[0]
    assert "CVD_PIN_EXEMPT" in errors[0]


def test_check_dispatcher_cvd_pin_fail_nohup_hydra_gpu_id(tmp_path):
    """FAIL — single-line nohup launch with the Hydra ``+gpu_id=`` form
    and no env pin."""
    (tmp_path / "x.sh").write_text(
        'nohup uv run python scripts/train.py +gpu_id=${gpu} > "$log" 2>&1 &\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_dispatcher_cvd_pin_pass_cvd_prefixed(tmp_path):
    """PASS — the compliant #578 reference shape (i474): env CVD pin AND
    matching --gpu-id on the same backgrounded command."""
    (tmp_path / "x.sh").write_text(
        'CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/foo_train.py \\\n'
        '    --conds "$cond" --gpu-id "$cvd" \\\n'
        '    > "$log" 2>&1 &\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (env CVD pinned), got: {errors}"


def test_check_dispatcher_cvd_pin_pass_sequential_launch(tmp_path):
    """PASS — a sequential (non-backgrounded) launch cannot co-locate
    siblings; --gpu-id without env CVD is not the parallel bug class."""
    (tmp_path / "x.sh").write_text(
        'uv run python scripts/foo_train.py --gpu-id 0 \\\n    > "$log" 2>&1\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (sequential), got: {errors}"


def test_check_dispatcher_cvd_pin_pass_and_and_chain(tmp_path):
    """PASS — a trailing ``&&`` is a command chain, not a background
    token; must not parse as backgrounded."""
    (tmp_path / "x.sh").write_text(
        'uv run python scripts/foo_train.py --gpu-id 0 &&\n    echo "done"\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (&& chain), got: {errors}"


def test_check_dispatcher_cvd_pin_pass_waiver_previous_line(tmp_path):
    """PASS — a ``# CVD_PIN_EXEMPT: <reason>`` waiver on the immediately
    preceding non-blank line (the only valid placement for a
    backslash-continued launch) is honored."""
    (tmp_path / "x.sh").write_text(
        "# CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim\n"
        "uv run python scripts/foo_train.py \\\n"
        '    --gpu-id "$cvd" > "$log" 2>&1 &\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (waived), got: {errors}"


def test_check_dispatcher_cvd_pin_pass_waiver_same_line(tmp_path):
    """PASS — a same-line trailing waiver on a single-line launch."""
    (tmp_path / "x.sh").write_text(
        "uv run python scripts/foo.py --gpu-id 0 &  "
        "# CVD_PIN_EXEMPT: single process on a 1-GPU pod, no sibling\n"
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (same-line waiver), got: {errors}"


def test_check_dispatcher_cvd_pin_fail_waiver_reason_too_short(tmp_path):
    """FAIL — a waiver with a reason shorter than the minimum is a
    token-shaped bypass, not a justification (same convention as
    WANDB_INTENTIONALLY_DISABLED)."""
    (tmp_path / "x.sh").write_text(
        '# CVD_PIN_EXEMPT: x\nuv run python scripts/foo_train.py --gpu-id "$cvd" > "$log" 2>&1 &\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_dispatcher_cvd_pin_pass_commented_and_echo_lines(tmp_path):
    """PASS — commented-out launches and echo dry-run previews are not
    launch sites."""
    (tmp_path / "x.sh").write_text(
        '# uv run python scripts/foo.py --gpu-id 0 > "$log" 2>&1 &\n'
        'echo "would run: uv run python scripts/foo.py --gpu-id 0" &\n'
    )
    errors = check_dispatcher_cvd_pin(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (comment/echo), got: {errors}"


def test_check_dispatcher_cvd_pin_repo_tree_is_clean():
    """The committed scripts/*.sh tree must carry no unwaived backgrounded
    --gpu-id/+gpu_id= python launches without an env CVD pin. Pre-#578
    completed-task dispatchers carry explicit CVD_PIN_EXEMPT waivers."""
    errors = check_dispatcher_cvd_pin()
    assert errors == [], (
        "scripts/*.sh has backgrounded --gpu-id/+gpu_id= python launches "
        "without a CUDA_VISIBLE_DEVICES= pin (#523/#541/#543/#557 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_dispatcher_cvd_pin_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-dispatcher-cvd-pin")
    assert result.returncode == 0, (
        f"workflow_lint --check-dispatcher-cvd-pin failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# Unit tests for the ``check_agent_model_pins`` function
# (d07424178 / task #545 incident class, 2026-06-09 → 2026-06-12).
# Each case writes a tiny .md file under ``tmp_path`` with a YAML
# frontmatter ``model: "..."`` pin and calls
# ``check_agent_model_pins(roots=[tmp_path])``.
# ─────────────────────────────────────────────────────────────────────


def _write_agent(path, model_pin):
    """Write a minimal agent .md file with a YAML frontmatter model pin."""
    path.write_text(
        f"---\nname: test-agent\nmodel: {model_pin!r}\n---\n\nAgent body.\n",
        encoding="utf-8",
    )


def test_check_agent_model_pins_pass_opus_with_1m_suffix(tmp_path):
    """PASS — the current canonical pin: opus-4-7 with the [1m] routing
    suffix (a 1M-context-supporting base)."""
    _write_agent(tmp_path / "analyzer.md", "claude-opus-4-7[1m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_agent_model_pins_pass_fable_without_suffix(tmp_path):
    """PASS — fable-5 has 1M native context and no [1m] suffix. Naming the
    base alone is fine; this case is the correct rewrite of the d07424178
    pin if Thomas decides to move to fable-5."""
    _write_agent(tmp_path / "analyzer.md", "claude-fable-5")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_agent_model_pins_pass_sonnet_46(tmp_path):
    """PASS — sonnet-4-6 also has 1M native context, no suffix."""
    _write_agent(tmp_path / "x.md", "claude-sonnet-4-6")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_agent_model_pins_pass_haiku_no_suffix(tmp_path):
    """PASS — haiku-4-5 is a 200K-context tier, no [1m] suffix."""
    _write_agent(tmp_path / "x.md", "claude-haiku-4-5")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_agent_model_pins_fail_fable_with_1m_suffix(tmp_path):
    """FAIL — the d07424178 / task #545 regression test: fable-5 is a
    real base but does NOT expose a [1m] routing variant. Pinning the
    suffixed id killed every subagent fleet-wide for ~72h."""
    _write_agent(tmp_path / "analyzer.md", "claude-fable-5[1m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "claude-fable-5[1m]" in errors[0]
    assert "does not expose a '[1m]'" in errors[0]
    assert "d07424178" in errors[0] or "#545" in errors[0]


def test_check_agent_model_pins_fail_sonnet45_with_1m_suffix(tmp_path):
    """FAIL — sonnet-4-5 is a 200K-context tier with no [1m] variant."""
    _write_agent(tmp_path / "x.md", "claude-sonnet-4-5[1m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 1, f"expected one error, got: {errors}"
    assert "claude-sonnet-4-5[1m]" in errors[0]
    assert "does not expose a '[1m]'" in errors[0]


def test_check_agent_model_pins_fail_unknown_base(tmp_path):
    """FAIL — a base id that is not in the allowlist (typo or
    aspirational id; the harness rejects it at spawn)."""
    _write_agent(tmp_path / "x.md", "claude-galaxy-9")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 1, f"expected one error, got: {errors}"
    assert "claude-galaxy-9" in errors[0]
    assert "not in the allowlist" in errors[0]


def test_check_agent_model_pins_fail_unknown_suffix_treated_as_unknown_base(tmp_path):
    """FAIL — a non-[1m] suffix like '[2m]' is glued to the base by
    :func:`_split_agent_model_pin` (intentional: only the literal '[1m]'
    is a recognized routing suffix). The result is reported as an
    unknown base, which is the correct outcome — the harness would
    reject it too."""
    _write_agent(tmp_path / "x.md", "claude-opus-4-7[2m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 1, f"expected one error, got: {errors}"
    assert "claude-opus-4-7[2m]" in errors[0]
    assert "not in the allowlist" in errors[0]


def test_check_agent_model_pins_pass_missing_frontmatter(tmp_path):
    """PASS — an agent file with no ``model:`` line inherits the parent
    model (CLAUDE.md 'Prompt-cache key discipline' explicitly allows it);
    no runtime contract to validate."""
    (tmp_path / "x.md").write_text("---\nname: x\n---\n\nBody.\n", encoding="utf-8")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert errors == [], f"expected PASS (no pin), got: {errors}"


def test_check_agent_model_pins_d07424178_regression_full_fleet(tmp_path):
    """FAIL on the EXACT shape of the d07424178 commit: bulk-rename of
    all agents to ``claude-fable-5[1m]``. The check must report one
    error per file (so the lint output names every offending pin, not
    just the first)."""
    for name in ("analyzer.md", "code-reviewer.md", "planner.md", "experimenter.md"):
        _write_agent(tmp_path / name, "claude-fable-5[1m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 4, f"expected one error per file, got: {len(errors)}: {errors}"
    # Every error should be the suffix-on-non-1m-base shape (not the
    # unknown-base shape — fable-5 itself IS in the allowlist).
    for e in errors:
        assert "does not expose a '[1m]'" in e
        assert "claude-fable-5" in e


def test_check_agent_model_pins_mixed_pass_and_fail(tmp_path):
    """A directory with a mix of valid and invalid pins reports only
    the invalid ones, with file:line precision."""
    _write_agent(tmp_path / "ok1.md", "claude-opus-4-7[1m]")
    _write_agent(tmp_path / "ok2.md", "claude-fable-5")
    _write_agent(tmp_path / "bad.md", "claude-fable-5[1m]")
    errors = check_agent_model_pins(roots=[tmp_path])
    assert len(errors) == 1, f"expected one error (bad.md only), got: {errors}"
    assert "bad.md:" in errors[0]


def test_check_agent_model_pins_repo_tree_is_clean():
    """The committed .claude/agents tree must already pass — the regression
    guard. If this fails, someone re-introduced the d07424178 / task #545
    pin shape and every subagent will die at spawn."""
    errors = check_agent_model_pins()
    assert errors == [], (
        "committed .claude/agents/*.md has invalid model pins "
        "(d07424178 / task #545 incident class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_agent_model_pins_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-agent-model-pins")
    assert result.returncode == 0, (
        f"workflow_lint --check-agent-model-pins failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# Unit tests for the ``check_upload_as_file`` function
# (#595 → #640 → #612 recurrence class): hub._upload raises ValueError
# unconditionally on a single-file path without upload_as_file=True, so a
# per-file upload loop crashes on the FIRST file after the expensive
# phases. Each case writes a tiny .py under ``tmp_path`` and calls
# ``check_upload_as_file(scripts_dir=tmp_path)``.
# ─────────────────────────────────────────────────────────────────────


def test_check_upload_as_file_fail_named_arg_no_kwarg(tmp_path):
    """FAIL — the #612 offender shape: a file-named variable (``summary_path``)
    passed to _upload with the upload_as_file kwarg entirely absent."""
    (tmp_path / "driver.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def phase_c(summary_path):\n"
        '    hub._upload(summary_path, repo_id="r", repo_type="dataset", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "driver.py:4" in errors[0]
    assert "upload_as_file=True" in errors[0]
    assert "summary_path" in errors[0]


def test_check_upload_as_file_fail_string_literal_no_kwarg(tmp_path):
    """FAIL — a decidable single-file string literal (.json) without the kwarg."""
    (tmp_path / "x.py").write_text(
        '_upload("out/summary.json", repo_id="r", repo_type="dataset", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "single-file path literal" in errors[0]


def test_check_upload_as_file_fail_pathdiv_literal_no_kwarg(tmp_path):
    """FAIL — the ``out_dir / "shift.pt"`` path-division shape (decidable file)."""
    (tmp_path / "x.py").write_text(
        'def f(out_dir):\n    _upload(out_dir / "shift.pt", repo_id="r", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_as_file_fail_decidable_file_explicit_false(tmp_path):
    """FAIL — a decidable file literal with an explicit upload_as_file=False
    is the #595 silent-no-op shape; an explicit False does NOT excuse a
    literal file (unlike a heuristic name signal)."""
    (tmp_path / "x.py").write_text('_upload("out/x.json", repo_id="r", upload_as_file=False)\n')
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_as_file_pass_named_arg_with_kwarg_true(tmp_path):
    """PASS — the correct single-file shape: upload_as_file=True."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def f(summary_path):\n"
        '    hub._upload(summary_path, repo_id="r", path_in_repo="p", upload_as_file=True)\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (upload_as_file=True), got: {errors}"


def test_check_upload_as_file_pass_folder_variable(tmp_path):
    """PASS — a generic folder variable (no file-suffix name, no literal)
    correctly relies on the upload_folder default; not flagged."""
    (tmp_path / "x.py").write_text(
        "def f(local_dir, staging):\n"
        '    _upload(local_dir, repo_id="r", repo_type="dataset", path_in_repo="p")\n'
        '    _upload(staging, repo_id="r", repo_type="dataset", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (folder vars), got: {errors}"


def test_check_upload_as_file_pass_named_arg_explicit_false(tmp_path):
    """PASS — a HEURISTIC name signal (``results_path``) with an EXPLICIT
    upload_as_file=False is the author's deliberate folder declaration and
    is deferred to (a name suffix must not override an explicit choice — a
    ``*_path`` variable can legitimately hold a directory)."""
    (tmp_path / "x.py").write_text(
        'def f(results_path):\n    _upload(results_path, repo_id="r", upload_as_file=False)\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (explicit False on name signal), got: {errors}"


def test_check_upload_as_file_pass_waiver_previous_line(tmp_path):
    """PASS — a ``# UPLOAD_AS_FILE_EXEMPT: <reason>`` waiver on the
    immediately preceding non-blank line is honored (a file-named var that
    is really a directory)."""
    (tmp_path / "x.py").write_text(
        "def f(results_path):\n"
        "    # UPLOAD_AS_FILE_EXEMPT: results_path is actually a directory here\n"
        '    _upload(results_path, repo_id="r", repo_type="dataset", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (waived), got: {errors}"


def test_check_upload_as_file_fail_waiver_reason_too_short(tmp_path):
    """FAIL — a waiver with a reason shorter than the minimum is a
    token-shaped bypass, not a justification."""
    (tmp_path / "x.py").write_text(
        "def f(results_path):\n"
        "    # UPLOAD_AS_FILE_EXEMPT: x\n"
        '    _upload(results_path, repo_id="r", repo_type="dataset", path_in_repo="p")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_as_file_fail_glob_loop_two_statement(tmp_path):
    """FAIL — the EXACT #640/#595 production offender shape: a two-statement
    ``files = sorted(dir.glob("*.json"))`` then ``for f in files: _upload(f, ...)``
    with ``path_in_repo=f"...{f.name}"``. The current name/literal heuristics
    miss it (``f`` has no file-suffix, no literal); the glob-loop + path_in_repo
    signals catch it."""
    (tmp_path / "carrier.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def upload_raw_completions(raw_dir):\n"
        '    files = sorted(raw_dir.glob("*.json"))\n'
        "    for f in files:\n"
        "        hub._upload(\n"
        "            f,\n"
        '            repo_id="r",\n'
        '            repo_type="dataset",\n'
        '            path_in_repo=f"issue640/raw_completions/{f.name}",\n'
        "        )\n"
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "upload_as_file=True" in errors[0]


def test_check_upload_as_file_fail_inline_glob_loop(tmp_path):
    """FAIL — ``for p in dir.glob("*.json"): _upload(p, ...)`` (inline glob
    loop, bare loop var, no path_in_repo .name signal)."""
    (tmp_path / "x.py").write_text(
        "def f(d):\n"
        '    for p in d.glob("*.json"):\n'
        '        _upload(p, repo_id="r", repo_type="dataset", path_in_repo="x")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "per-file glob/iterdir loop variable ('p')" in errors[0]


def test_check_upload_as_file_fail_iterdir_loop(tmp_path):
    """FAIL — ``for path in dir.iterdir(): _upload(path, ...)`` (flat per-file
    sweep; the canonical iterdir use)."""
    (tmp_path / "x.py").write_text(
        "def f(d):\n"
        "    for path in d.iterdir():\n"
        '        _upload(path, repo_id="r", repo_type="dataset", path_in_repo="x")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "per-file glob/iterdir loop variable ('path')" in errors[0]


def test_check_upload_as_file_pass_dir_shaped_glob_loop(tmp_path):
    """PASS — ``for d in dir.glob("*/"): _upload(d, ...)`` iterates
    DIRECTORIES (trailing-slash pattern), so the glob-loop single-file signal
    must NOT fire — it correctly relies on the upload_folder default. The
    candidate's ambiguous-``glob("*/")`` defer-to-folder case."""
    (tmp_path / "x.py").write_text(
        "def f(root):\n"
        '    for d in root.glob("*/"):\n'
        '        _upload(d, repo_id="r", repo_type="dataset", path_in_repo="x")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (dir-shaped glob loop), got: {errors}"


def test_check_upload_as_file_pass_extensionless_glob_loop(tmp_path):
    """PASS — ``for x in dir.glob("*"): _upload(x, ...)`` has NO file-extension
    token in the pattern, so the file-vs-dir intent is undecidable. The
    glob-loop signal defers (conservative — never manufacture a false positive
    on a possible directory sweep; the candidate's "no extension token →
    defer" rule). The riskiest per-file cases are caught by the
    path_in_repo=f'...{x.name}' signal instead."""
    (tmp_path / "x.py").write_text(
        "def f(d):\n"
        '    for x in d.glob("*"):\n'
        '        _upload(x, repo_id="r", repo_type="dataset", path_in_repo="x")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (extensionless glob loop), got: {errors}"


def test_check_upload_as_file_fail_path_in_repo_name_kwarg(tmp_path):
    """FAIL — the ``path_in_repo=f"...{item.name}"`` idiom alone (a non-glob
    loop over a bare ``items`` iterable): taking ``.name`` on a per-item path
    you upload individually is a single-file signal independent of the loop
    iterator."""
    (tmp_path / "x.py").write_text(
        "def f(items):\n"
        "    for item in items:\n"
        '        _upload(item, repo_id="r", path_in_repo=f"x/{item.name}")\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "item.name" in errors[0]


def test_check_upload_as_file_pass_glob_loop_with_kwarg_true(tmp_path):
    """PASS — the CORRECT fixed shape: the #640 glob loop now passing
    upload_as_file=True (this is what the production carriers look like
    post-fix; the lint must not re-flag them)."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def upload_raw(raw_dir):\n"
        '    for f in sorted(raw_dir.glob("*.json")):\n'
        "        hub._upload(\n"
        "            f,\n"
        '            repo_id="r",\n'
        '            path_in_repo=f"x/{f.name}",\n'
        "            upload_as_file=True,\n"
        "        )\n"
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (glob loop with kwarg True), got: {errors}"


def test_check_upload_as_file_pass_glob_loop_explicit_false_waived(tmp_path):
    """PASS — a glob-loop signal with an EXPLICIT upload_as_file=False is the
    author's deliberate folder declaration and is deferred to (the glob-loop /
    path_in_repo signals are heuristic name-context signals, same deferral
    policy as the file-named-arg signal: they fire only when the kwarg is
    entirely absent)."""
    (tmp_path / "x.py").write_text(
        "def f(d):\n"
        '    for sub in d.glob("*.json"):\n'
        '        _upload(sub, repo_id="r", path_in_repo="x", upload_as_file=False)\n'
    )
    errors = check_upload_as_file(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (explicit False on glob-loop signal), got: {errors}"


def test_check_upload_as_file_repo_tree_is_clean():
    """The committed scripts/**/*.py tree must carry no unwaived single-file
    _upload calls missing upload_as_file=True (#595/#640/#612 class). This
    is the systemic regression guard the candidate asked for."""
    errors = check_upload_as_file()
    assert errors == [], (
        "scripts/**/*.py has _upload(...) single-file calls missing "
        "upload_as_file=True (#595/#640/#612 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_upload_as_file_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-upload-as-file")
    assert result.returncode == 0, (
        f"workflow_lint --check-upload-as-file failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# Unit tests for ``check_hub_dir_filecount_guard`` (#1190; incident #658:
# the Hub rejects >10k files per repo directory at COMMIT time with a
# NON-retriable BadRequestError AFTER all bytes are staged). Direct
# ``upload_folder(`` call sites in scripts/ must reference the hub.py
# runtime guard ``assert_hub_dir_filecounts``, carry a
# ``# HUB_DIR_FILECOUNT_EXEMPT: <reason>`` waiver, or be grandfathered.
# Each case writes a tiny .py under ``tmp_path`` and calls
# ``check_hub_dir_filecount_guard(scripts_dir=tmp_path)``.
# ─────────────────────────────────────────────────────────────────────


def test_check_hub_dir_filecount_fail_attribute_call(tmp_path):
    """(a) FAIL — the #658 incident shape: a direct ``api.upload_folder(``
    call in a module with no guard reference."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        "    api = HfApi()\n"
        '    api.upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "x.py:5" in errors[0]
    assert "assert_hub_dir_filecounts" in errors[0]
    assert "HUB_DIR_FILECOUNT_EXEMPT" in errors[0]
    assert "gotchas.md" in errors[0]
    assert "transient-retry wrapper" in errors[0]


def test_check_hub_dir_filecount_pass_module_references_guard(tmp_path):
    """(b) PASS — the module references the guard helper (the one-line fix)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n"
        "from explore_persona_space.orchestrate.hub import assert_hub_dir_filecounts\n\n"
        "def push(d):\n"
        '    assert_hub_dir_filecounts(d, "p")\n'
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (module references the guard), got: {errors}"


def test_check_hub_dir_filecount_pass_attribute_guard_reference(tmp_path):
    """(b') PASS — a ``hub.assert_hub_dir_filecounts(...)`` attribute
    reference counts as a guard reference too."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n"
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        '    hub.assert_hub_dir_filecounts(d, "p")\n'
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (hub.assert_... reference), got: {errors}"


def test_check_hub_dir_filecount_pass_waiver(tmp_path):
    """(c) PASS — a waiver with a real reason on the preceding line."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        "    # HUB_DIR_FILECOUNT_EXEMPT: tiny fixed 3-file tree, cap unreachable\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (waived), got: {errors}"


def test_check_hub_dir_filecount_fail_waiver_reason_too_short(tmp_path):
    """(d) FAIL — a waiver whose reason is under the 10-char floor is not a
    waiver (the reason is a justification, not a token bypass)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        "    # HUB_DIR_FILECOUNT_EXEMPT: ok\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error (short reason), got: {errors}"


def test_check_hub_dir_filecount_pass_injectable_allowlist(tmp_path):
    """(e) PASS — the injectable ``legacy_allowlist=`` grandfathers a file by
    its walk-root-parent-relative posix path (the production path shape is
    ``scripts/<name>.py``)."""
    (tmp_path / "legacy.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    rel = f"{tmp_path.name}/legacy.py"
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path, legacy_allowlist=frozenset({rel}))
    assert errors == [], f"expected PASS (allowlisted {rel}), got: {errors}"
    # Sanity: without the allowlist the same file IS flagged (non-vacuous).
    errors_unlisted = check_hub_dir_filecount_guard(
        scripts_dir=tmp_path, legacy_allowlist=frozenset()
    )
    assert len(errors_unlisted) == 1, errors_unlisted


def test_check_hub_dir_filecount_fail_bare_name_hf_import(tmp_path):
    """(f) FAIL — the ast.Name arm: a ``from huggingface_hub import
    upload_folder`` caller (the issue667_save_maps.py / issue825 shape)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_folder\n\n"
        "def push(d):\n"
        '    upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error (bare-name arm), got: {errors}"


def test_check_hub_dir_filecount_pass_local_def_carveout(tmp_path):
    """(g) NOT flagged — a module defining its own ``def upload_folder``
    calls the LOCAL wrapper, not the huggingface_hub function (the
    scripts/issue623_upload.py shape; the carve-out, not the allowlist, is
    its pass condition)."""
    (tmp_path / "x.py").write_text(
        "def upload_folder(folder_path, repo_id, path_in_repo):\n"
        "    return None\n\n"
        "def push(d):\n"
        '    upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (local def upload_folder carve-out), got: {errors}"


def test_check_hub_dir_filecount_pass_exact_name_only(tmp_path):
    """PASS — exact-name match only: differently-named wrappers
    (``upload_folder_verified`` / ``_upload_folder_filtered``) do NOT match."""
    (tmp_path / "x.py").write_text(
        "from issue1073_common import upload_folder_verified\n"
        "from explore_persona_space.orchestrate.hub import _upload_folder_filtered\n\n"
        "def push(d, api):\n"
        '    upload_folder_verified(api, folder_path=str(d), repo_id="r")\n'
        '    api.upload_folder_scoped_verify(folder_path=str(d), repo_id="r")\n'
        '    _upload_folder_filtered(d, "r", "dataset", "p", ["*.json"], [])\n'
    )
    errors = check_hub_dir_filecount_guard(scripts_dir=tmp_path)
    assert errors == [], f"expected PASS (no exact-name upload_folder call), got: {errors}"


def test_check_hub_dir_filecount_bundled_in_no_flags():
    """(h) NON-VACUOUS no-flags bundling pin: the check must be dispatched by
    the BARE ``workflow_lint.py`` run. Source-inspection assert on the
    dispatch branch + the no_flags tuple membership (exit-0-on-a-clean-tree
    is vacuous — it passes whether or not the check is dispatched)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_hub_dir_filecount or no_flags:\s*\n"
        r"\s*errors\.extend\(check_hub_dir_filecount_guard\(\)\)",
        src,
    ), "check_hub_dir_filecount_guard is not dispatched on the no-flags branch"
    assert "or args.check_hub_dir_filecount" in src, (
        "--check-hub-dir-filecount is missing from the no_flags detection tuple"
    )


def test_check_hub_dir_filecount_live_tree_passes():
    """The committed scripts/**/*.py tree must pass — pins the grandfather
    allowlist's completeness so the no-flags default run (pre-commit /
    Step 9c) cannot break on a stale allowlist. A NEW direct upload_folder
    caller must call assert_hub_dir_filecounts or carry a
    HUB_DIR_FILECOUNT_EXEMPT waiver — never extend the allowlist."""
    errors = check_hub_dir_filecount_guard()
    assert errors == [], (
        "scripts/**/*.py has direct upload_folder(...) call sites missing the "
        "hub dir-filecount guard (#658/#1190 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_hub_dir_filecount_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-hub-dir-filecount")
    assert result.returncode == 0, (
        f"workflow_lint --check-hub-dir-filecount failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────
# --check-upload-prefix-clobber (task #1452 / incident #1005): hardcoded
# issue-prefix HF upload DESTINATIONS — Rule A (cross-issue dest token)
# and Rule B (own-issue token via a fallback channel: `x or CONST`,
# argparse default=, wrapper-param default). Fixture matrix per plan
# §6.2. Each tmp case writes tiny .py files under ``tmp_path`` and calls
# ``check_upload_prefix_clobber(scripts_dir=tmp_path)``.
# ─────────────────────────────────────────────────────────────────────

_UPC_WRAPPER_COMMON = (
    'FIT_RESULTS_PREFIX = "issue928_cot/fit_results"\n'
    'PREFIX = "issue928_foo"\n\n'
    "def upload_folder_scoped_verify(folder, path_in_repo, expected_names):\n"
    "    from huggingface_hub import HfApi\n\n"
    "    HfApi().upload_folder(\n"
    '        folder_path=str(folder), repo_id="r", repo_type="dataset",\n'
    "        path_in_repo=path_in_repo,\n"
    "    )\n"
)


def test_check_upload_prefix_clobber_fail_cross_issue_literal(tmp_path):
    """FAIL (Rule A) — §6.2 case 1: an issue1005_ script uploading into a
    literal issue928_ destination (the copied-uploader parent-clobber class)."""
    (tmp_path / "issue1005_x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", '
        'path_in_repo="issue928_foo/bar")\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "cross-issue" in errors[0]
    assert "issue928_" in errors[0]


def test_check_upload_prefix_clobber_fail_cross_issue_import_wrapper(tmp_path):
    """FAIL (Rule A) — §6.2 case 2: a foreign constant imported from an
    issue928_common module, fed through the parent's own upload wrapper
    (exercises pass-1 wrapper inference + import-map tokens)."""
    (tmp_path / "issue928_common.py").write_text(_UPC_WRAPPER_COMMON)
    (tmp_path / "issue1005_x.py").write_text(
        "from issue928_common import upload_folder_scoped_verify, PREFIX\n\n"
        "def push(d):\n"
        '    upload_folder_scoped_verify(d, PREFIX, ["a.json"])\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "issue1005_x.py" in errors[0]
    assert "cross-issue" in errors[0]


def test_check_upload_prefix_clobber_fail_or_fallback_same_issue(tmp_path):
    """FAIL (Rule B) — §6.2 case 3: the verbatim #1005 incident shape,
    ``args.p or FIT_RESULTS_PREFIX`` at the wrapper's destination slot."""
    (tmp_path / "issue928_common.py").write_text(_UPC_WRAPPER_COMMON)
    (tmp_path / "issue928_x.py").write_text(
        "from issue928_common import upload_folder_scoped_verify, FIT_RESULTS_PREFIX\n\n"
        "def main(args, d):\n"
        '    upload_folder_scoped_verify(d, args.p or FIT_RESULTS_PREFIX, ["a.json"])\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "issue928_x.py" in errors[0]
    assert "or-fallback" in errors[0]


def test_check_upload_prefix_clobber_fail_argparse_default_transitive(tmp_path):
    """FAIL (Rule B) — §6.2 case 4: the #928 post-incident remediated shape —
    an argparse ``default=CONST`` where CONST is a TRANSITIVE f-string const
    (also covers §4.3.2 transitive const resolution)."""
    (tmp_path / "issue928_y.py").write_text(
        "import argparse\n\n"
        'BASE = "issue928_foo"\n'
        'CONST = f"{BASE}/analysis_tensors"\n\n'
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        '    ap.add_argument("--tensors-upload-prefix", default=CONST)\n'
        "    args = ap.parse_args()\n"
        "    from huggingface_hub import HfApi\n\n"
        "    HfApi().upload_folder(\n"
        '        folder_path="d", repo_id="r", path_in_repo=args.tensors_upload_prefix\n'
        "    )\n"
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "argparse-default" in errors[0]


def test_check_upload_prefix_clobber_fail_waiver_reason_too_short(tmp_path):
    """FAIL — §6.2 case 5: a waiver whose reason is under 10 chars does NOT
    suppress the finding."""
    (tmp_path / "issue1005_x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        "    # UPLOAD_PREFIX_EXEMPT: short\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", '
        'path_in_repo="issue928_foo/bar")\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error (short waiver), got: {errors}"


def test_check_upload_prefix_clobber_fail_without_allowlist(tmp_path):
    """§6.2 case 6 / acceptance criterion 5(a): the incident-shape fixture
    FAILs with an empty allowlist and PASSes once its relpath is
    allowlisted (Rule B only — the allowlist override hook)."""
    (tmp_path / "issue928_common.py").write_text(_UPC_WRAPPER_COMMON)
    (tmp_path / "issue928_x.py").write_text(
        "from issue928_common import upload_folder_scoped_verify, FIT_RESULTS_PREFIX\n\n"
        "def main(args, d):\n"
        '    upload_folder_scoped_verify(d, args.p or FIT_RESULTS_PREFIX, ["a.json"])\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert errors, "expected the #928 incident shape to FAIL with an empty allowlist"
    rel = f"{tmp_path.name}/issue928_x.py"
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset({rel}))
    assert errors == [], f"expected PASS with the file allowlisted, got: {errors}"


def test_check_upload_prefix_clobber_pass_read_side_foreign_prefix(tmp_path):
    """PASS — §6.2 case 7: cross-issue READS (list_repo_tree /
    hf_hub_download) are the sanctioned reuse default and never flag."""
    (tmp_path / "issue1073_x.py").write_text(
        "from huggingface_hub import HfApi, hf_hub_download\n\n"
        "def fetch():\n"
        "    rows = HfApi().list_repo_tree(\n"
        '        "r", path_in_repo="issue779_monitoring/r_b", repo_type="dataset"\n'
        "    )\n"
        '    hf_hub_download("r", "issue779_monitoring/r_b/x.pt", repo_type="dataset")\n'
        "    return rows\n"
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert errors == [], f"expected PASS (read-side foreign prefix), got: {errors}"


def test_check_upload_prefix_clobber_pass_direct_own_prefix(tmp_path):
    """PASS — §6.2 case 8: a DIRECT own-issue prefix hardcode is the
    sanctioned Upload Policy norm (keeps ~100 issue scripts green)."""
    (tmp_path / "issue1333_x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d, cell):\n"
        "    HfApi().upload_folder(\n"
        '        folder_path=str(d), repo_id="r", path_in_repo=f"issue1333_store/{cell}"\n'
        "    )\n"
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert errors == [], f"expected PASS (direct own-prefix), got: {errors}"


def test_check_upload_prefix_clobber_pass_valid_waiver_preceding_line(tmp_path):
    """PASS — §6.2 case 9: a valid waiver (reason ≥ 10 chars) on the
    immediately preceding non-blank line suppresses the finding."""
    (tmp_path / "issue1005_x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        "    # UPLOAD_PREFIX_EXEMPT: deliberate mirror into the parent bucket\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", '
        'path_in_repo="issue928_foo/bar")\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert errors == [], f"expected PASS (valid waiver), got: {errors}"


def test_check_upload_prefix_clobber_pass_non_issue_named_script(tmp_path):
    """PASS — §6.2 case 10: a non-issue-named script is skipped entirely
    (disclosed scope: the incident class is reused issue scripts)."""
    (tmp_path / "build_foo.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", '
        'path_in_repo="issue928_foo/bar")\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert errors == [], f"expected PASS (non-issue-named script skipped), got: {errors}"


def test_check_upload_prefix_clobber_fail_function_local_assign(tmp_path):
    """FAIL (Rule B) — §6.2 case 14: the issue1092_figures shape — a
    function-LOCAL ``path_in_repo = f"{args.hf_prefix}/…"`` assignment
    feeding the upload, with a token-carrying argparse default."""
    (tmp_path / "issue1092_x.py").write_text(
        "import argparse\n\n"
        'HF_PREFIX_DEFAULT = "issue1092_realistic"\n\n'
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        '    ap.add_argument("--hf-prefix", default=HF_PREFIX_DEFAULT)\n'
        "    args = ap.parse_args()\n"
        "    from huggingface_hub import HfApi\n\n"
        '    path_in_repo = f"{args.hf_prefix}/p7/analysis_tensors/nulls"\n'
        "    HfApi().upload_folder(\n"
        '        folder_path="d", repo_id="r", path_in_repo=path_in_repo\n'
        "    )\n"
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "argparse-default" in errors[0]


def test_check_upload_prefix_clobber_fail_wrapper_param_default(tmp_path):
    """FAIL (Rule B) — recommended fixture: a wrapper whose dest param
    carries a signature default resolving to an issue token (the pass-1
    wrapper-fallback finding at the def line)."""
    (tmp_path / "issue928_w.py").write_text(
        'PREFIX = "issue928_foo"\n\n'
        "def upload_verified(folder, path_in_repo=PREFIX):\n"
        "    from huggingface_hub import HfApi\n\n"
        "    HfApi().upload_folder(\n"
        '        folder_path=str(folder), repo_id="r", path_in_repo=path_in_repo\n'
        "    )\n"
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "wrapper-param-default" in errors[0]


def test_check_upload_prefix_clobber_fail_raw_completions_positional_dest(tmp_path):
    """FAIL (Rule A) — recommended fixture: the
    ``upload_raw_completions_to_data_repo`` positional-0 dest slot
    (``experiment_name`` carries the issue prefix)."""
    (tmp_path / "issue1005_r.py").write_text(
        "from explore_persona_space.orchestrate.hub import "
        "upload_raw_completions_to_data_repo\n\n"
        "def push(rows):\n"
        '    upload_raw_completions_to_data_repo("issue928_foo", rows)\n'
    )
    errors = check_upload_prefix_clobber(scripts_dir=tmp_path, legacy_allowlist=frozenset())
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "cross-issue" in errors[0]


def test_check_upload_prefix_clobber_bundled_in_no_flags():
    """§6.2 case 12 — NON-VACUOUS no-flags bundling pin: source-inspection
    assert on the dispatch branch + the no_flags tuple membership
    (exit-0-on-a-clean-tree is vacuous — it passes whether or not the
    check is dispatched). House pattern:
    test_check_hub_dir_filecount_bundled_in_no_flags."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_upload_prefix_clobber or no_flags:\s*\n"
        r"\s*errors\.extend\(check_upload_prefix_clobber\(\)\)",
        src,
    ), "check_upload_prefix_clobber is not dispatched on the no-flags branch"
    assert "or args.check_upload_prefix_clobber" in src, (
        "--check-upload-prefix-clobber is missing from the no_flags detection tuple"
    )


def test_check_upload_prefix_clobber_live_trees_pass():
    """§6.2 case 11 — the committed scripts/**/*.py tree must pass with the
    committed allowlist (locks the grandfather set to today's tree, the
    JUDGE_PIN pattern). A NEW fallback destination must use default=None +
    a fail-loud raise or an UPLOAD_PREFIX_EXEMPT waiver — never extend the
    allowlist."""
    errors = check_upload_prefix_clobber()
    assert errors == [], (
        "scripts/**/*.py has hardcoded issue-prefix upload destinations "
        "(#1005 parent-clobber class):\n" + "\n".join(errors)
    )


def test_check_upload_prefix_clobber_allowlist_load_bearing():
    """§6.2 case 13 / acceptance criterion 5(b) — anti-vacuity pin: with the
    allowlist EMPTIED on the REAL tree, (i) every committed allowlist
    relpath appears among the findings (each entry load-bearing — catches
    silent under-trigger AND stale entries in one assert) and (ii) the
    #1005 incident file is flagged."""
    errors = check_upload_prefix_clobber(legacy_allowlist=frozenset())
    flagged_files = {e.split(": ", 1)[0].rsplit(":", 1)[0] for e in errors}
    assert UPLOAD_PREFIX_CLOBBER_ALLOWLIST, "committed allowlist unexpectedly empty"
    for rel in sorted(UPLOAD_PREFIX_CLOBBER_ALLOWLIST):
        assert str(_REPO_ROOT / rel) in flagged_files, (
            f"allowlist entry {rel} is NOT load-bearing (no finding with an "
            f"empty allowlist) — stale entry or silent under-trigger.\n"
            f"flagged: {sorted(flagged_files)}"
        )
    incident = str(_REPO_ROOT / "scripts/issue928_fit_decomposition.py")
    assert incident in flagged_files, (
        "the #1005 incident file (issue928_fit_decomposition.py) is not flagged "
        "with an empty allowlist — the check no longer covers the incident shape"
    )


def test_workflow_lint_check_upload_prefix_clobber_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree
    (acceptance criterion 2)."""
    result = _run("--check-upload-prefix-clobber")
    assert result.returncode == 0, (
        f"workflow_lint --check-upload-prefix-clobber failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ── --check-upload-file-in-loop (task #1544; incidents #664 / #658 r4 / #1481) ─
# Each test writes a tiny fixture into ``tmp_path`` and calls
# ``check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})``
# unless it is exercising the committed allowlist / real tree.


def test_check_upload_file_in_loop_fail_for_loop(tmp_path):
    """§4.7 item 1 — FAIL: the #664/#1481 shape, an attribute-form
    ``api.upload_file(...)`` inside a ``for`` loop."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert ":6:" in errors[0], errors[0]
    assert "inside a loop" in errors[0]
    assert "upload_folder" in errors[0]


def test_check_upload_file_in_loop_fail_bare_name_import(tmp_path):
    """§4.7 item 2 — FAIL: a bare-name ``upload_file`` caller
    (``from huggingface_hub import upload_file``) inside a loop."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_file_in_loop_fail_while_loop(tmp_path):
    """§4.7 item 3 — FAIL: an ``upload_file`` call inside a ``while`` loop."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def drain(queue):\n"
        "    while queue:\n"
        "        f = queue.pop()\n"
        '        upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_file_in_loop_fail_comprehension(tmp_path):
    """§4.7 item 4 — FAIL: an ``upload_file`` call inside a list
    comprehension (a loop in expression form)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "def push(files):\n"
        '    return [api.upload_file(path_or_fileobj=str(f), repo_id="r") for f in files]\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_file_in_loop_fail_nested_in_loop_if(tmp_path):
    """§4.7 item 5 — FAIL: nesting under a non-loop statement (``if``)
    inside a loop still flags (the run_experiment_444 shape)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "def push(files):\n"
        "    for f in files:\n"
        "        if f.exists():\n"
        '            api.upload_file(path_or_fileobj=str(f), repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"


def test_check_upload_file_in_loop_pass_bulk_upload_folder(tmp_path):
    """§4.7 item 6 — PASS: the recommended fix shape, one loop-free
    ``upload_folder`` bulk call."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "def push(d):\n"
        '    HfApi().upload_folder(folder_path=str(d), repo_id="r", path_in_repo="p")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (bulk upload_folder), got: {errors}"


def test_check_upload_file_in_loop_pass_single_call_no_loop(tmp_path):
    """§4.7 item 7 — PASS: loop-free ``upload_file`` calls (one
    module-level + one in-function) are legitimate single-file uploads."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        'upload_file(path_or_fileobj="a.json", path_in_repo="a.json", repo_id="r")\n\n'
        "def push_one(p):\n"
        '    upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (no loop), got: {errors}"


def test_check_upload_file_in_loop_pass_def_inside_loop(tmp_path):
    """§4.7 item 8 — PASS: a function *defined* in a loop resets the loop
    context (its body executes elsewhere)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "def build(xs):\n"
        "    cbs = []\n"
        "    for x in xs:\n"
        "        def cb(p=x):\n"
        '            api.upload_file(path_or_fileobj=str(p), repo_id="r")\n'
        "        cbs.append(cb)\n"
        "    return cbs\n"
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (boundary reset), got: {errors}"


def test_check_upload_file_in_loop_pass_helper_called_from_loop(tmp_path):
    """§4.7 item 9 — PASS: a module-level helper *called* from a loop is a
    deliberate lexical false negative (no dataflow in v1)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n\n"
        "api = HfApi()\n"
        "def _push(f):\n"
        '    api.upload_file(path_or_fileobj=str(f), repo_id="r")\n\n'
        "def main(files):\n"
        "    for f in files:\n"
        "        _push(f)\n"
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (lexical-only), got: {errors}"


def test_check_upload_file_in_loop_pass_waiver_same_line(tmp_path):
    """§4.7 item 10a — PASS: a same-line ``# UPLOAD_LOOP_EXEMPT`` waiver
    with a reason ≥ 10 chars silences the finding."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r")'
        "  # UPLOAD_LOOP_EXEMPT: bounded 2-file retry wrapper\n"
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (same-line waiver), got: {errors}"


def test_check_upload_file_in_loop_pass_waiver_previous_line(tmp_path):
    """§4.7 item 10b — PASS: the waiver on the immediately preceding
    non-blank line also silences."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        "        # UPLOAD_LOOP_EXEMPT: bounded 2-file retry wrapper\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (previous-line waiver), got: {errors}"


def test_check_upload_file_in_loop_fail_waiver_reason_too_short(tmp_path):
    """§4.7 item 11 — FAIL: a waiver with a < 10-char reason does NOT
    silence (the reason is a justification, not a token bypass)."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        "        # UPLOAD_LOOP_EXEMPT: ok\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error (short reason), got: {errors}"


def test_check_upload_file_in_loop_pass_injectable_allowlist(tmp_path):
    """§4.7 item 12 — PASS: the injectable ``legacy_allowlist=`` dict
    grandfathers a file by its walk-root-parent-relative posix path up to
    the allowed site COUNT (the production path shape is
    ``scripts/<name>.py``)."""
    (tmp_path / "legacy.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r")\n'
    )
    rel = f"{tmp_path.name}/legacy.py"
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={rel: 1})
    assert errors == [], f"expected PASS (allowlisted {rel}: 1), got: {errors}"
    # Sanity: without the allowlist the same file IS flagged (non-vacuous).
    errors_unlisted = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors_unlisted) == 1, errors_unlisted


def test_check_upload_file_in_loop_fail_allowlist_count_exceeded(tmp_path):
    """§4.7 item 12 companion — FAIL: a SECOND offending call in a file
    grandfathered at count 1 reports (the count-gate pin: a new offense in
    a grandfathered file surfaces instead of hiding behind the entry)."""
    (tmp_path / "legacy.py").write_text(
        "from huggingface_hub import upload_file\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r")\n\n'
        "def push_more(files):\n"
        "    for f in files:\n"
        '        upload_file(path_or_fileobj=str(f), repo_id="r2")\n'
    )
    rel = f"{tmp_path.name}/legacy.py"
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={rel: 1})
    assert len(errors) == 3, f"expected count-exceeded note + both findings, got: {errors}"
    assert "exceed" in errors[0] and "grandfathered" in errors[0], errors[0]
    assert "inside a loop" in errors[1] and "inside a loop" in errors[2]


def test_check_upload_file_in_loop_live_tree_passes():
    """§4.7 item 13 — the committed scripts/**/*.py tree must pass with the
    committed allowlist (locks the grandfather set to today's tree). A NEW
    in-loop per-file upload must batch into one upload_folder commit or
    carry an UPLOAD_LOOP_EXEMPT waiver — never extend the allowlist."""
    errors = check_upload_file_in_loop()
    assert errors == [], (
        "scripts/**/*.py has ungrandfathered per-file upload calls inside loops "
        "(#664/#1481 storm class):\n" + "\n".join(errors)
    )


def test_check_upload_file_in_loop_allowlist_load_bearing():
    """§4.7 item 14 — anti-vacuity pin with EXACT per-file counts: with the
    allowlist EMPTIED on the REAL tree, the findings grouped per file must
    match the committed dict EXACTLY (same file set, same per-file counts) —
    catches silent under-trigger, stale entries, AND a new offense netting
    into a grandfathered file, in one assert."""
    errors = check_upload_file_in_loop(legacy_allowlist={})
    per_file: dict[str, int] = {}
    for e in errors:
        path = e.split(": ", 1)[0].rsplit(":", 1)[0]
        per_file[path] = per_file.get(path, 0) + 1
    assert UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST, "committed allowlist unexpectedly empty"
    expected = {str(_REPO_ROOT / rel): n for rel, n in UPLOAD_FILE_IN_LOOP_LEGACY_ALLOWLIST.items()}
    assert per_file == expected, (
        "empty-allowlist findings do not match the committed allowlist dict "
        "(stale entry, silent under-trigger, or a new offense in a grandfathered "
        f"file).\nexpected: {expected}\ngot: {per_file}"
    )


def test_check_upload_file_in_loop_bundled_in_no_flags():
    """§4.7 item 15 — NON-VACUOUS no-flags bundling pin: source-inspection
    assert on the dispatch branch + the no_flags tuple membership
    (exit-0-on-a-clean-tree is vacuous — it passes whether or not the
    check is dispatched). House pattern:
    test_check_upload_prefix_clobber_bundled_in_no_flags."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_upload_file_in_loop or no_flags:\s*\n"
        r"\s*errors\.extend\(check_upload_file_in_loop\(\)\)",
        src,
    ), "check_upload_file_in_loop is not dispatched on the no-flags branch"
    assert "or args.check_upload_file_in_loop" in src, (
        "--check-upload-file-in-loop is missing from the no_flags detection tuple"
    )


def test_workflow_lint_check_upload_file_in_loop_cli_exits_zero():
    """§4.7 item 16 — the dedicated flag must exist and pass on the
    committed tree (acceptance criterion 7)."""
    result = _run("--check-upload-file-in-loop")
    assert result.returncode == 0, (
        f"workflow_lint --check-upload-file-in-loop failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_upload_file_in_loop_fail_hub_upload_as_file_true(tmp_path):
    """§4.7 item 17 — FAIL (shape B): the literal #664 form,
    ``hub._upload(f, ..., upload_as_file=True)`` inside a loop —
    check_upload_as_file deliberately DEFERS explicit-kwarg calls, so this
    check is the only static gate on the shape."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def push(files):\n"
        "    for f in files:\n"
        '        hub._upload(f, repo_id="r", upload_as_file=True)\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "_upload(..., upload_as_file=True)" in errors[0], errors[0]


def test_check_upload_file_in_loop_pass_hub_upload_single_file(tmp_path):
    """§4.7 item 18 — PASS: one loop-free ``hub._upload(p,
    upload_as_file=True)`` (the legitimate #595-guard single-file caller)."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def push_one(p):\n"
        '    hub._upload(p, repo_id="r", upload_as_file=True)\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (no loop), got: {errors}"


def test_check_upload_file_in_loop_pass_hub_upload_no_kwarg_in_loop(tmp_path):
    """§4.7 item 19 — PASS: an in-loop ``_upload`` WITHOUT
    ``upload_as_file=True`` is not this check's shape (a per-dir folder
    loop is legitimate; the #595 runtime ValueError guard and
    check_upload_as_file own the single-file forms)."""
    (tmp_path / "x.py").write_text(
        "from explore_persona_space.orchestrate import hub\n\n"
        "def push(dirs):\n"
        "    for d in dirs:\n"
        '        hub._upload(d, repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (no upload_as_file=True kwarg), got: {errors}"


def test_check_upload_file_in_loop_pass_other_upload_names_in_loop(tmp_path):
    """§4.7 item 20 — PASS: exact-name negative pin — ``upload_files`` /
    ``api.upload_folder`` inside loops do NOT match."""
    (tmp_path / "x.py").write_text(
        "from huggingface_hub import HfApi\n"
        "from mylib import upload_files\n\n"
        "api = HfApi()\n"
        "def push(files, dirs):\n"
        "    for f in files:\n"
        "        upload_files(f)\n"
        "    for d in dirs:\n"
        '        api.upload_folder(folder_path=str(d), repo_id="r")\n'
    )
    errors = check_upload_file_in_loop(scripts_dir=tmp_path, legacy_allowlist={})
    assert errors == [], f"expected PASS (exact-name only), got: {errors}"


# ── --check-batch-judge-client (task #658/#663 post-mortem) ───────────────────
# Each test writes a fixture into ``tmp_path`` and calls
# ``check_batch_judge_client(scripts_dir=tmp_path, src_dir=<empty>)``. The
# `src_dir` is pointed at an empty dir so only the scripts fixture is scanned
# unless a test needs both.


def _empty(tmp_path):
    d = tmp_path / "_empty_src"
    d.mkdir()
    return d


def test_check_batch_judge_client_fail_inline_create_call(tmp_path):
    """The #658 offender shape: an inline messages.batches.create + a
    deadline-less while-True poller in a non-sanctioned script."""
    (tmp_path / "issue658_judge_e0_batch.py").write_text(
        "import anthropic, time\n"
        "client = anthropic.Anthropic()\n"
        "def go(requests):\n"
        "    batch = client.messages.batches.create(requests=requests)\n"
        "    while True:\n"
        "        b = client.messages.batches.retrieve(batch.id)\n"
        "        if b.processing_status == 'ended':\n"
        "            break\n"
        "        time.sleep(30)\n"
    )
    errors = check_batch_judge_client(scripts_dir=tmp_path, src_dir=_empty(tmp_path))
    assert len(errors) == 1, errors
    assert "messages.batches.create" in errors[0]


def test_check_batch_judge_client_fail_to_thread_reference_form(tmp_path):
    """The bare-reference form passed to asyncio.to_thread (the shape
    judge_dispatch itself uses) is also flagged outside the sanctioned set."""
    (tmp_path / "issue999_thread.py").write_text(
        "import asyncio, anthropic\n"
        "client = anthropic.Anthropic()\n"
        "async def go(requests):\n"
        "    return await asyncio.to_thread("
        "client.messages.batches.create, requests=requests)\n"
    )
    errors = check_batch_judge_client(scripts_dir=tmp_path, src_dir=_empty(tmp_path))
    assert len(errors) == 1, errors


def test_check_batch_judge_client_pass_openai_batches_create(tmp_path):
    """OpenAI's client.batches.create (no `messages` segment) is a DIFFERENT
    API and must NOT be flagged."""
    (tmp_path / "openai_gen.py").write_text(
        "client = object()\n"
        "def go():\n"
        "    return client.batches.create(input_file_id='x', endpoint='/v1/chat')\n"
    )
    errors = check_batch_judge_client(scripts_dir=tmp_path, src_dir=_empty(tmp_path))
    assert errors == [], errors


def test_check_batch_judge_client_pass_sanctioned_client_file(tmp_path):
    """A file at the sanctioned-client path suffix is exempt even with an
    inline messages.batches.create."""
    src = tmp_path / "src" / "explore_persona_space" / "eval"
    src.mkdir(parents=True)
    (src / "batch_judge.py").write_text(
        "client = object()\n"
        "def go(chunk):\n"
        "    return client.messages.batches.create(requests=chunk)\n"
    )
    errors = check_batch_judge_client(scripts_dir=_empty(tmp_path), src_dir=tmp_path / "src")
    assert errors == [], errors


def test_check_batch_judge_client_pass_waiver_previous_line(tmp_path):
    """A '# BATCH_JUDGE_CLIENT_EXEMPT: <reason>' waiver on the previous
    non-blank line suppresses the flag."""
    (tmp_path / "datagen.py").write_text(
        "client = object()\n"
        "def go(r):\n"
        "    # BATCH_JUDGE_CLIENT_EXEMPT: legitimate non-judge data-gen batch\n"
        "    return client.messages.batches.create(requests=r)\n"
    )
    errors = check_batch_judge_client(scripts_dir=tmp_path, src_dir=_empty(tmp_path))
    assert errors == [], errors


def test_check_batch_judge_client_fail_waiver_reason_too_short(tmp_path):
    """A waiver with a < 10-char reason does not suppress the flag."""
    (tmp_path / "datagen.py").write_text(
        "client = object()\n"
        "def go(r):\n"
        "    # BATCH_JUDGE_CLIENT_EXEMPT: short\n"
        "    return client.messages.batches.create(requests=r)\n"
    )
    errors = check_batch_judge_client(scripts_dir=tmp_path, src_dir=_empty(tmp_path))
    assert len(errors) == 1, errors


def test_check_batch_judge_client_repo_tree_is_clean():
    """The committed scripts/**/*.py + src/explore_persona_space/**/*.py tree
    must carry no unwaived inline messages.batches.create outside the
    sanctioned batch clients (#658/#663 class). Legacy data-gen offenders are
    grandfathered in BATCH_JUDGE_LEGACY_ALLOWLIST; a NEW offender FAILs."""
    errors = check_batch_judge_client()
    assert errors == [], (
        "scripts/ or src/explore_persona_space/ has an inline "
        "messages.batches.create outside the sanctioned batch clients "
        "(#658/#663 class):\n" + "\n".join(errors)
    )


def test_check_batch_judge_client_legacy_allowlist_entry_is_exempt(tmp_path):
    """A file at a legacy-allowlist path is exempt even with an inline
    messages.batches.create — locks the grandfathering behavior. The surviving
    data-gen / analysis siblings (e.g. analyze_axis_tails.py) demonstrate that
    the allowlist is file-granular, not call-granular: membership exempts the
    whole file's inline batch-create calls. The pre-#663 i528 judge was migrated
    onto the sanctioned eval.batch_judge helper (#668) and DROPPED from the
    allowlist, so it must no longer be a member (the lint now governs it)."""
    sd = tmp_path / "scripts"
    sd.mkdir()
    # i528_phase4_judge.py was migrated off the inline batch-create path (#668),
    # so it is NO LONGER grandfathered — the lint governs the file directly now.
    assert "scripts/i528_phase4_judge.py" not in BATCH_JUDGE_LEGACY_ALLOWLIST
    # analyze_axis_tails.py is the surviving non-data-gen sibling demonstrating
    # the per-path exemption mechanism is still exercised.
    assert "scripts/analyze_axis_tails.py" in BATCH_JUDGE_LEGACY_ALLOWLIST
    # And a NON-allowlisted offender at the same dir IS flagged (the exemption
    # is per-path, not blanket-scripts/).
    (sd / "issue777_new_judge.py").write_text(
        "client = object()\ndef go(r):\n    return client.messages.batches.create(requests=r)\n"
    )
    errors = check_batch_judge_client(scripts_dir=sd, src_dir=_empty(tmp_path))
    assert len(errors) == 1, errors


def test_workflow_lint_check_batch_judge_client_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-batch-judge-client")
    assert result.returncode == 0, (
        f"workflow_lint --check-batch-judge-client failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ── --check-hub-verify-retry (task #920/#997/#1202) ──────────────────────────
# Each test writes a fixture into ``tmp_path`` and calls
# ``check_hub_verify_retry(scripts_dir=tmp_path)`` (scripts/-only scan; no
# src_dir arg — #997 owns the library path).


def test_check_hub_verify_retry_fail_bare_attr_call(tmp_path):
    """The #920 offender shape: a bare api.list_repo_files( verify leg in a
    non-grandfathered script. The error must route authors at the retried
    hub helpers (verify_repo_paths_uploaded et al.)."""
    (tmp_path / "issue9999_verify.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def verify(repo):\n"
        "    return api.list_repo_files(repo, repo_type='dataset')\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert "issue9999_verify.py" in errors[0]
    assert ".list_repo_files(" in errors[0]
    assert "verify_repo_paths_uploaded" in errors[0]
    assert "list_hf_files_under_path" in errors[0]
    assert "list_repo_files_complete" in errors[0]
    assert "HUB_VERIFY_RETRY_EXEMPT" in errors[0]
    assert "#920" in errors[0]


def test_check_hub_verify_retry_fail_bare_file_exists(tmp_path):
    """A bare api.file_exists( single-path probe is the same un-retried
    class and is flagged."""
    (tmp_path / "issue9999_probe.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def probe(repo, path):\n"
        "    return api.file_exists(repo, path, repo_type='dataset')\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ".file_exists(" in errors[0]


def test_check_hub_verify_retry_fail_imported_name_form(tmp_path):
    """The ``from huggingface_hub import <target>`` bare-Name form is
    flagged — both the plain import and the aliased ``as lrt`` form (the
    asname-aware bound-name map; an alias cannot evade the Name leg)."""
    (tmp_path / "issue9999_name_form.py").write_text(
        "from huggingface_hub import list_repo_files\n"
        "from huggingface_hub import list_repo_tree as lrt\n"
        "def go(repo):\n"
        "    files = list_repo_files(repo)\n"
        "    tree = lrt(repo, path_in_repo='p')\n"
        "    return files, tree\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 2, errors
    assert any("list_repo_files(" in e for e in errors), errors
    # The aliased hit reports the CANONICAL symbol, not the alias.
    assert any("list_repo_tree(" in e for e in errors), errors


def test_check_hub_verify_retry_fail_bare_list_repo_tree(tmp_path):
    """list_repo_tree( is the SAME un-retried pagination class (gotchas.md
    names it as the recommended large-repo listing) and is flagged."""
    (tmp_path / "issue9999_tree.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def scan(repo):\n"
        "    return list(api.list_repo_tree(repo, path_in_repo='x', recursive=True))\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ".list_repo_tree(" in errors[0]


def test_check_hub_verify_retry_pass_local_name_without_hf_import(tmp_path):
    """A script-local ``def file_exists`` helper (no huggingface_hub import
    of the symbol) is NOT flagged — the Name leg is gated on the import."""
    (tmp_path / "local_helper.py").write_text(
        "import os\n"
        "def file_exists(p):\n"
        "    return os.path.exists(p)\n"
        "def go(p):\n"
        "    return file_exists(p)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_pass_hub_helper_usage(tmp_path):
    """Compliant usage of the retried orchestrate.hub helpers is
    structurally invisible to the detector."""
    (tmp_path / "compliant.py").write_text(
        "from explore_persona_space.orchestrate.hub import (\n"
        "    list_hf_files_under_path,\n"
        "    verify_repo_paths_uploaded,\n"
        ")\n"
        "def verify(api, repo, paths):\n"
        "    verify_repo_paths_uploaded(api, repo, paths, repo_type='dataset')\n"
        "    return list_hf_files_under_path(api, repo, 'prefix/')\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_pass_comment_and_docstring_mentions(tmp_path):
    """Prose mentions in comments/docstrings can never match — AST has no
    comment nodes and a string mention is an ast.Constant."""
    (tmp_path / "prose_only.py").write_text(
        '"""Mentions list_repo_files( and .file_exists( in prose only."""\n'
        "# a list_repo_tree( mention in a comment\n"
        "X = 'list_repo_files(...) as a string literal'\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_pass_waiver_previous_line(tmp_path):
    """A '# HUB_VERIFY_RETRY_EXEMPT: <reason>' waiver on the previous
    non-blank line suppresses the flag."""
    (tmp_path / "waived.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def go(repo):\n"
        "    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient by the caller\n"
        "    return api.list_repo_files(repo)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_pass_waiver_same_line(tmp_path):
    """The waiver also binds on the call's OWN line (the same-line branch
    of the waiver helper; the previous-line branch is covered above)."""
    (tmp_path / "waived_inline.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def go(repo):\n"
        "    return api.list_repo_files(repo)  "
        "# HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient here\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_fail_waiver_reason_too_short(tmp_path):
    """A waiver with a < 10-char reason does not suppress the flag."""
    (tmp_path / "waived_short.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def go(repo):\n"
        "    # HUB_VERIFY_RETRY_EXEMPT: short\n"
        "    return api.list_repo_files(repo)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_check_hub_verify_retry_pass_store_ctx_assignment_target(tmp_path):
    """A Store-ctx assignment TARGET (the monkeypatch patch shape,
    ``HfApi.list_repo_tree = _fake``) can never be a call and is exempt
    without a waiver (#1482/#1561)."""
    (tmp_path / "issue9999_selftest.py").write_text(
        "import huggingface_hub\n"
        "def _fake(*a, **k):\n"
        "    raise RuntimeError('stub')\n"
        "def patch():\n"
        "    huggingface_hub.HfApi.list_repo_tree = _fake\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_pass_del_ctx_attribute(tmp_path):
    """A Del-ctx attribute (``del api.file_exists`` — the monkeypatch
    cleanup shape) is a deletion target, not a call; exempt without a
    waiver."""
    (tmp_path / "issue9999_unpatch.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def unpatch():\n"
        "    del api.file_exists\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_fail_load_ctx_monkeypatch_save(tmp_path):
    """The Load-ctx monkeypatch SAVE (``orig = HfApi.list_repo_tree``) is
    STILL flagged — the saved alias is call-equivalent (alias-later-called
    evasion) — while the Store-ctx restore line yields no second error.
    The error message names the monkeypatch-save waiver requirement
    (#1482/#1561)."""
    (tmp_path / "issue9999_save.py").write_text(
        "import huggingface_hub\n"
        "orig = huggingface_hub.HfApi.list_repo_tree\n"
        "huggingface_hub.HfApi.list_repo_tree = orig\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ":2:" in errors[0]
    assert ".list_repo_tree(" in errors[0]
    assert "monkeypatch" in errors[0].lower()


def test_check_hub_verify_retry_pass_monkeypatch_block_waiver_on_save_only(tmp_path):
    """The full #1482 monkeypatch block (save / patch / restore, the
    issue779:618-635 shape) needs exactly ONE waiver — on the Load-ctx
    save line; the Store-ctx patch/restore lines are exempt by ctx
    (pre-fix this block required 4 waivers)."""
    (tmp_path / "issue9999_block.py").write_text(
        "import huggingface_hub\n"
        "def _fake(*a, **k):\n"
        "    raise RuntimeError('stub')\n"
        "def selftest():\n"
        "    # HUB_VERIFY_RETRY_EXEMPT: attribute reference only (monkeypatch save), no call\n"
        "    orig_lrt = huggingface_hub.HfApi.list_repo_tree\n"
        "    huggingface_hub.HfApi.list_repo_tree = _fake\n"
        "    try:\n"
        "        pass\n"
        "    finally:\n"
        "        huggingface_hub.HfApi.list_repo_tree = orig_lrt\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_fail_bare_reference_argument_position(tmp_path):
    """A bare Load-ctx reference passed as a wrapper ARGUMENT
    (``retry_transient(api.list_repo_files)``) stays flagged — guards
    against a future over-broad 'only flag ast.Call' change (the exact
    wrong fix the ctx gate must not become)."""
    (tmp_path / "issue9999_wrapper_arg.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def retry_transient(fn):\n"
        "    return fn\n"
        "wrapped = retry_transient(api.list_repo_files)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ".list_repo_files(" in errors[0]


def test_check_hub_verify_retry_name_leg_store_ctx(tmp_path):
    """Name-leg ctx gate: a pure rebind of the imported name
    (``list_repo_files = None``) is no longer flagged (a binding is not a
    call), while a wrap-rebind (``list_repo_tree = my_retry(list_repo_tree)``)
    still yields exactly one hit via its RHS Load usage."""
    (tmp_path / "rebind_only.py").write_text(
        "from huggingface_hub import list_repo_files\nlist_repo_files = None\n"
    )
    (tmp_path / "wrap_rebind.py").write_text(
        "from huggingface_hub import list_repo_tree\n"
        "def my_retry(fn):\n"
        "    return fn\n"
        "list_repo_tree = my_retry(list_repo_tree)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert "wrap_rebind.py" in errors[0]


def test_check_hub_verify_retry_pass_augassign_store_target(tmp_path):
    """An AugAssign target (``api.list_repo_tree += x``) reads ctx=Store —
    the one Store shape with a runtime getattr; getattr of a method never
    calls it, so the ctx gate exempts it (optional pin from the #1561 plan
    review round)."""
    (tmp_path / "issue9999_augassign.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def bump(x):\n"
        "    api.list_repo_tree += x\n"
    )
    errors = check_hub_verify_retry(scripts_dir=tmp_path)
    assert errors == [], errors


def test_check_hub_verify_retry_allowlist_is_file_granular(tmp_path):
    """The grandfather allowlist exempts by exact rel path (whole file), and
    a NON-allowlisted offender at the same dir IS flagged — the exemption is
    per-path, not blanket-scripts/. verify_uploads.py is the known
    workflow-helper member (migration onto the hub helpers is a named
    follow-up)."""
    assert "scripts/verify_uploads.py" in HUB_VERIFY_LEGACY_ALLOWLIST
    sd = tmp_path / "scripts"
    sd.mkdir()
    (sd / "issue9999_new_verify.py").write_text(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "def go(repo):\n"
        "    return api.list_repo_files(repo)\n"
    )
    errors = check_hub_verify_retry(scripts_dir=sd)
    assert len(errors) == 1, errors


def test_check_hub_verify_retry_repo_tree_is_clean():
    """The committed scripts/**/*.py tree must carry no unwaived bare Hub
    verify call outside HUB_VERIFY_LEGACY_ALLOWLIST — locks the allowlist to
    the land-time tree so the no-flags default run cannot break, and makes
    every NEW bare caller a reviewed diff (#920/#997/#1202)."""
    errors = check_hub_verify_retry()
    assert errors == [], (
        "scripts/**/*.py has a bare list_repo_files( / list_repo_tree( / "
        ".file_exists( Hub call outside the grandfathered set (#920 class):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_hub_verify_retry_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-hub-verify-retry")
    assert result.returncode == 0, (
        f"workflow_lint --check-hub-verify-retry failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_hub_verify_retry_bundled_in_no_flags():
    """NON-VACUOUS no-flags bundling pin: the check must be dispatched by
    the BARE ``workflow_lint.py`` run. Source-inspection assert on the
    dispatch branch + the no_flags tuple membership (exit-0-on-a-clean-tree
    is vacuous — it passes whether or not the check is dispatched)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_hub_verify_retry or no_flags:\s*\n"
        r"\s*errors\.extend\(check_hub_verify_retry\(\)\)",
        src,
    ), "check_hub_verify_retry is not dispatched on the no-flags branch"
    assert "or args.check_hub_verify_retry" in src, (
        "--check-hub-verify-retry is missing from the no_flags detection tuple"
    )


# ─── --check-no-workflow-improver-spawn (#678 S2) ──────────────────────────


def test_no_stale_workflow_improver_spawn():
    """No live Agent(subagent_type="workflow-improver", ...) spawn survives
    anywhere in the committed workflow surface (#678). The frozen agent file is
    excluded; a live spawn instruction anywhere else is a regression."""
    errors = check_no_workflow_improver_spawn()
    assert errors == [], (
        'stale Agent(subagent_type="workflow-improver", ...) spawn found in the '
        "workflow surface (retired by #678):\n" + "\n".join(errors)
    )


def test_check_no_workflow_improver_spawn_flags_a_stray_spawn(tmp_path):
    """A live Agent(subagent_type="workflow-improver", ...) spawn in any
    in-scope file (here a rule .md) IS flagged — the guard actually trips."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "stray.md").write_text(
        "When a bug is hit, the orchestrator runs\n"
        'Agent(subagent_type="workflow-improver", run_in_background=true)\n'
        "to apply the fix.\n"
    )
    errors = check_no_workflow_improver_spawn(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "stale Agent" in errors[0]


def test_check_no_workflow_improver_spawn_hermetic_under_cache_nested_root(tmp_path):
    """A tmp repo rooted inside a directory literally named .claude/cache/ is
    still scanned — the cache/agent-memory exclusion matches relative to the
    repo root, not the absolute path (#1174: a repo-nested TMPDIR wholesale-
    excluded the whole tmp tree). Files genuinely under the tmp repo's OWN
    .claude/cache/ and .claude/agent-memory/ stay excluded."""
    outer = tmp_path / ".claude" / "cache" / "tmprepo"
    rules = outer / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "stray.md").write_text(
        'Agent(subagent_type="workflow-improver", run_in_background=true)\n'
    )
    cache = outer / ".claude" / "cache"
    cache.mkdir(parents=True)
    (cache / "planted.md").write_text(
        'Agent(subagent_type="workflow-improver", run_in_background=true)\n'
    )
    agent_mem = outer / ".claude" / "agent-memory"
    agent_mem.mkdir(parents=True)
    (agent_mem / "planted_mem.md").write_text(
        'Agent(subagent_type="workflow-improver", run_in_background=true)\n'
    )
    errors = check_no_workflow_improver_spawn(repo_root=outer)
    assert len(errors) == 1, errors
    assert "stale Agent" in errors[0]
    assert "stray.md" in errors[0]


def test_check_no_workflow_improver_spawn_excludes_frozen_agent_file(tmp_path):
    """The frozen .claude/agents/workflow-improver.md is excluded even if its
    (deprecated, historical) body still shows a spawn shape."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "workflow-improver.md").write_text(
        "---\nname: workflow-improver\n---\n"
        '> DEPRECATED. (historical) Agent(subagent_type="workflow-improver", ...)\n'
    )
    errors = check_no_workflow_improver_spawn(repo_root=tmp_path)
    assert errors == [], errors


def test_workflow_lint_check_no_workflow_improver_spawn_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-no-workflow-improver-spawn")
    assert result.returncode == 0, (
        f"workflow_lint --check-no-workflow-improver-spawn failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ─── --check-no-literal-round-marker-versions (#917) ────────────────────────


def test_no_literal_round_marker_versions_live_tree():
    """No checked-in workflow prose instructs a literal v1 for a
    round-versioned marker kind (#917 — the #825/#389 collision class).
    The E1-E7 sweep of #917 established zero hits at introduction."""
    errors = check_no_literal_round_marker_versions()
    assert errors == [], (
        "literal round-marker version instruction found in the workflow "
        "surface (rephrase to `v<n>` / max+1, #917):\n" + "\n".join(errors)
    )


def test_check_no_literal_round_marker_versions_flags_literal_v1(tmp_path):
    """A literal `epm:results v1` posting instruction in an in-scope file
    (here a rule .md) IS flagged — the guard actually trips."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "stray.md").write_text("On completion, post `epm:results v1` on the task.\n")
    errors = check_no_literal_round_marker_versions(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "max+1" in errors[0]
    assert "stray.md:1" in errors[0]


def test_check_no_literal_round_marker_versions_wrapped_pair_trips(tmp_path):
    """A line-wrapped kind/version pair still trips — the scan is whole-file,
    so the `\\s+` between kind and `v1` may span a newline."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "a.md").write_text("then post `epm:proposed-tests\n  v1` and EXIT.\n")
    errors = check_no_literal_round_marker_versions(repo_root=tmp_path)
    assert len(errors) == 1, errors


def test_check_no_literal_round_marker_versions_no_false_positives(tmp_path):
    """`v<n>`, `v12`, and a genuinely-once kind (`epm:failure v1`) never
    match — the pattern is restricted to the 3 round-versioned kinds and
    `v1` is word-bounded."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "ok.md").write_text(
        "post `epm:results v<n>` (max+1 per the rule); the legitimate\n"
        "`epm:experiment-implementation v12` round-12 example; and\n"
        "`epm:failure v1` is a genuinely-once marker, out of scope.\n"
    )
    errors = check_no_literal_round_marker_versions(repo_root=tmp_path)
    assert errors == [], errors


def test_check_no_literal_round_marker_versions_excluded_paths_pass(tmp_path):
    """A hit under an EXCLUDED path (.claude/plans/, .claude/agent-memory/)
    passes — pins the exclusion set (archives may legitimately quote the
    incident text; a mis-enumerated exclusion would false-FAIL the default
    lint bundle on archive text)."""
    plans = tmp_path / ".claude" / "plans"
    plans.mkdir(parents=True)
    (plans / "issue-825.md").write_text("the brief said: post `epm:results v1` (historical)\n")
    mem = tmp_path / ".claude" / "agent-memory" / "implementer"
    mem.mkdir(parents=True)
    (mem / "note.md").write_text("brief instructed `epm:experiment-implementation v1`\n")
    errors = check_no_literal_round_marker_versions(repo_root=tmp_path)
    assert errors == [], errors


def test_workflow_lint_check_no_literal_round_marker_versions_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-no-literal-round-marker-versions")
    assert result.returncode == 0, (
        f"workflow_lint --check-no-literal-round-marker-versions failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_workflow_lint_check_gate_ids_unique_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree
    (all current gate ids are unique)."""
    result = _run("--check-gate-ids-unique")
    assert result.returncode == 0, (
        f"workflow_lint --check-gate-ids-unique failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_gate_ids_unique_repo_passes():
    """Repo-level: the committed gate ids are all unique."""
    errors = check_gate_ids_unique(_workflow())
    assert errors == [], f"expected PASS, got: {errors}"


def test_check_gate_ids_unique_flags_a_duplicate():
    """A collision across two gate sub-lists is flagged, naming BOTH gate
    names and the duplicated id."""
    wf = _workflow().model_copy(deep=True)
    # Force a collision: give a park_and_wait gate the same id as an inline
    # gate, on the deep copy so the cached real object is untouched.
    dup_id = wf.gates.inline[0].id
    wf.gates.park_and_wait[0].id = dup_id
    errors = check_gate_ids_unique(wf)
    assert len(errors) >= 1, errors
    # Both offending gate names appear in the error.
    assert wf.gates.inline[0].name in errors[0]
    assert wf.gates.park_and_wait[0].name in errors[0]
    assert f"duplicate gate id {dup_id}" in errors[0]


# ---------------------------------------------------------------------------
# Regression guard for analyzer.md Step 3.5 inherited-figure cross-check (#729).
# Step 3.5 must require, on a same-issue follow-up re-fold, cross-checking an
# INHERITED figure's .meta.json `points` against the NEW round's result JSON
# (regenerate on mismatch). The rule is load-bearing prose that is easy to lose
# silently in a future analyzer.md edit, so pin the stable concept tokens.
# Scoped to the Step 3.5 slice (its `### Step 3.5` H3 up to the next H3/H2) to
# avoid false matches elsewhere in the file. #667 a36 round 2 caught a stale
# inherited figure post-hoc by diligence, not a guardrail.
# ---------------------------------------------------------------------------


def test_analyzer_step35_cross_checks_inherited_figures():
    """analyzer.md Step 3.5 must require cross-checking an inherited figure's
    .meta.json points against the new result JSON on a follow-up re-fold (#729)."""
    text = (_REPO_ROOT / ".claude/agents/analyzer.md").read_text()
    lines = text.splitlines()
    # Slice the Step 3.5 region: from its `### Step 3.5` heading up to (but not
    # including) the next `### ` H3 or `## ` H2.
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("### Step 3.5")),
        None,
    )
    assert start is not None, "analyzer.md is missing the `### Step 3.5` heading"
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith("### ") or lines[i].startswith("## ")
        ),
        len(lines),
    )
    step35 = "\n".join(lines[start:end])
    lowered = step35.lower()
    for token, label in (
        ("inherited", "the inherited-figure trigger"),
        (".meta.json", "the .meta.json sidecar the cross-check reads"),
        ("points", "the per-point `points` key the cross-check compares"),
    ):
        assert token in lowered, (
            f"analyzer.md Step 3.5 dropped {label} (token {token!r}); the #729 "
            "inherited-figure cross-check rule was weakened or removed."
        )
    assert "follow-up re-fold" in lowered or "follow-up re-folds" in lowered, (
        "analyzer.md Step 3.5 dropped the same-issue follow-up re-fold scoping "
        "(token 'follow-up re-fold'); the #729 cross-check rule was weakened."
    )


def _write_lessons_fixture(rules_dir, rule_names, indexed_names):
    """Write N fake rule files + a LESSONS.md indexing `indexed_names`."""
    rules_dir.mkdir(parents=True, exist_ok=True)
    for name in rule_names:
        (rules_dir / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    rows = "\n".join(f"- {n}.md — x." for n in indexed_names)
    (rules_dir / "LESSONS.md").write_text(f"# LESSONS\n\n## Rules\n\n{rows}\n", encoding="utf-8")


def _write_lessons_at_exact_bytes(rules_dir, total_bytes):
    """Write a valid one-rule LESSONS.md padded to EXACTLY `total_bytes` bytes.

    Pads with ASCII 'x' prose after the row (the pad never matches the row
    regex, so per-row caps are unaffected); asserts the realized byte count
    (the em-dash in the row is multibyte, so bytes != chars).
    """
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / "alpha.md").write_text("# alpha\n", encoding="utf-8")
    base = "# LESSONS\n\n## Rules\n\n- alpha.md — x.\n\n"
    pad = total_bytes - len(base.encode("utf-8")) - 1  # -1: trailing newline
    assert pad > 0, "total_bytes too small for the fixture skeleton"
    content = base + "x" * pad + "\n"
    assert len(content.encode("utf-8")) == total_bytes
    (rules_dir / "LESSONS.md").write_bytes(content.encode("utf-8"))


def _write_lessons_row(rules_dir, name, row_bytes):
    """Write one rule file + a LESSONS.md whose single row for `name` is
    EXACTLY `row_bytes` bytes (ASCII trigger padding; em-dash is 3 bytes)."""
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    prefix = f"- {name}.md — "
    pad = row_bytes - len(prefix.encode("utf-8"))
    assert pad > 0, "row_bytes too small for the row skeleton"
    row = prefix + "x" * pad
    assert len(row.encode("utf-8")) == row_bytes
    (rules_dir / "LESSONS.md").write_text(f"# LESSONS\n\n## Rules\n\n{row}\n", encoding="utf-8")


def test_check_lessons_index_fails_on_missing_row(tmp_path):
    rules = tmp_path / ".claude" / "rules"
    # rule 'gamma' exists but is NOT indexed -> FAIL (armed defaults: the
    # tiny unpadded fixture passes every size mode, isolating parity).
    _write_lessons_fixture(rules, ["alpha", "beta", "gamma"], ["alpha", "beta"])
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs, "expected a FAIL for the un-indexed rule 'gamma'"
    assert any("gamma" in e for e in errs)


def test_check_lessons_index_fails_on_stale_row(tmp_path):
    rules = tmp_path / ".claude" / "rules"
    # 'delta' is indexed but has no rule file -> FAIL
    _write_lessons_fixture(rules, ["alpha", "beta"], ["alpha", "beta", "delta"])
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("delta" in e for e in errs)


def test_check_lessons_index_passes_on_match(tmp_path):
    rules = tmp_path / ".claude" / "rules"
    _write_lessons_fixture(rules, ["alpha", "beta"], ["alpha", "beta"])
    assert check_lessons_index(repo_root=tmp_path) == []


def test_check_lessons_index_passes_on_live_repo():
    # Sanity: the real repo must PASS after this change lands.
    assert check_lessons_index() == []


def test_check_lessons_index_fails_when_index_exceeds_cap(tmp_path):
    # Leanness cap is mechanical — an index over _LESSONS_MAX_BYTES must FAIL.
    # non-row mode disabled: the prose padding is NON-ROW bytes by
    # construction, so the fixture must isolate the CAP failure mode
    # (the non-row budget has its own boundary test below).
    from workflow_lint import _LESSONS_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "alpha.md").write_text("# alpha\n", encoding="utf-8")
    rows = "- alpha.md — x.\n"
    # Pad with prose so the index breaches the byte cap regardless of its value.
    padding = "x" * (_LESSONS_MAX_BYTES + 100)
    (rules / "LESSONS.md").write_text(
        f"# LESSONS\n\n## Rules\n\n{rows}\n\n{padding}\n",
        encoding="utf-8",
    )
    errs = check_lessons_index(repo_root=tmp_path, nonrow_max_bytes=None)
    assert errs and any("leanness cap" in e for e in errs)
    # #1504: the cap FAIL names the largest rows as actionable trim targets.
    assert any("Largest rows:" in e and "alpha" in e for e in errs)


def test_check_lessons_index_fails_on_duplicate_row(tmp_path):
    # One 'alpha.md' rule file, but TWO 'alpha' index rows. A set-based
    # implementation would collapse the duplicate and silently PASS (the
    # missing/stale set-diffs both read empty); the Counter-based check must
    # FAIL because the contract is exactly one matching row per rule (#739 r2).
    rules = tmp_path / ".claude" / "rules"
    _write_lessons_fixture(rules, ["alpha"], ["alpha", "alpha"])
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs, "expected a FAIL for the duplicate 'alpha' index row"
    assert any(("duplicate" in e or "exactly one" in e) and "alpha" in e for e in errs)


def test_check_lessons_index_warns_in_warn_band(tmp_path):
    # The #992 early-warning band: an index strictly between _LESSONS_WARN_BYTES
    # and _LESSONS_MAX_BYTES emits one advisory WARN (warn_sink / stderr),
    # never a FAIL; the over-cap FAIL branch takes precedence over the WARN.
    # non-row mode disabled throughout: the band fixtures pad with NON-ROW
    # prose by design and must isolate the WARN-band mode.
    from workflow_lint import _LESSONS_MAX_BYTES, _LESSONS_WARN_BYTES

    # Pin the band constant itself (#992 plan latitude: 7000-7400, below cap).
    assert 7000 <= _LESSONS_WARN_BYTES <= 7400 < _LESSONS_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"

    # (1) Sub-warn-band fixture -> no FAIL, empty sink.
    sink: list[str] = []
    _write_lessons_fixture(rules, ["alpha"], ["alpha"])
    assert check_lessons_index(repo_root=tmp_path, warn_sink=sink, nonrow_max_bytes=None) == []
    assert sink == []

    # (2) EXACTLY at the threshold -> still no warn (the band is strictly-greater).
    sink = []
    _write_lessons_at_exact_bytes(rules, _LESSONS_WARN_BYTES)
    assert check_lessons_index(repo_root=tmp_path, warn_sink=sink, nonrow_max_bytes=None) == []
    assert sink == []

    # (3) One byte over the threshold -> no FAIL, exactly one warn-band message
    # naming the largest rows as trim targets (#1504).
    sink = []
    _write_lessons_at_exact_bytes(rules, _LESSONS_WARN_BYTES + 1)
    assert check_lessons_index(repo_root=tmp_path, warn_sink=sink, nonrow_max_bytes=None) == []
    assert len(sink) == 1 and "warn band" in sink[0]
    assert "Largest rows:" in sink[0] and "alpha" in sink[0]

    # (4) Over the cap -> the FAIL branch fires; no warn message rides along.
    sink = []
    _write_lessons_at_exact_bytes(rules, _LESSONS_MAX_BYTES + 100)
    errs = check_lessons_index(repo_root=tmp_path, warn_sink=sink, nonrow_max_bytes=None)
    assert errs and any("leanness cap" in e for e in errs)
    assert sink == []


def _write_lessons_nonrow(rules_dir, nonrow_target):
    """One valid alpha row + non-row padding sized so non-row bytes == target."""
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / "alpha.md").write_text("x", encoding="utf-8")
    row = "- alpha.md — x."
    base = f"# LESSONS\n\n## Rules\n\n{row}\n"
    row_bytes = len(row.encode("utf-8"))
    pad = nonrow_target - (len(base.encode("utf-8")) - row_bytes)
    assert pad >= 0, (nonrow_target, pad)
    (rules_dir / "LESSONS.md").write_bytes(base.encode("utf-8") + b"x" * pad)


def test_check_lessons_index_fails_on_nonrow_over_budget(tmp_path):
    # Strictly-greater boundary pair for the #1504 non-row scaffolding budget.
    from workflow_lint import _LESSONS_NONROW_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    _write_lessons_nonrow(rules, _LESSONS_NONROW_MAX_BYTES + 1)
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("_LESSONS_NONROW_MAX_BYTES" in e and "non-row" in e for e in errs)
    _write_lessons_nonrow(rules, _LESSONS_NONROW_MAX_BYTES)
    assert check_lessons_index(repo_root=tmp_path) == []


def test_check_lessons_index_nonrow_ignores_row_bytes(tmp_path):
    # THE retirement pin (#1504): rows at the per-row cap, ARMED defaults,
    # zero workflow_lint.py edits — row growth never needs a constant bump.
    from workflow_lint import _LESSONS_ROW_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    rows = []
    for name in ("alpha", "beta", "gamma"):
        (rules / f"{name}.md").write_text("x", encoding="utf-8")
        prefix = f"- {name}.md — "
        rows.append(prefix + "y" * (_LESSONS_ROW_MAX_BYTES - len(prefix.encode("utf-8"))))
    content = "# LESSONS\n\n## Rules\n\n" + "\n".join(rows) + "\n"
    (rules / "LESSONS.md").write_text(content, encoding="utf-8")
    assert check_lessons_index(repo_root=tmp_path) == []


def test_lessons_total_ratchet_retired():
    # Anti-resurrection pin (#1504): re-introducing the per-growth TOTAL
    # ratchet (the 4-incidents/48h conflict magnet) must consciously delete
    # this test.
    import inspect

    import workflow_lint as wl

    assert not hasattr(wl, "_LESSONS_RATCHET_BYTES")
    assert not hasattr(wl, "_LESSONS_RATCHET_MAX_HEADROOM_BYTES")
    params = inspect.signature(wl.check_lessons_index).parameters
    assert "ratchet_bytes" not in params
    assert "nonrow_max_bytes" in params


def test_lessons_index_bundled_in_no_flags_source_pin():
    """NON-VACUOUS no-flags bundling pin (#1504; the #1385 source-pin
    pattern): `check_lessons_index` must be dispatched by the BARE
    ``workflow_lint.py`` run. The ratchet retirement's deliberateness
    replacement (per-row caps + non-row budget + index parity) rests on
    this bundling firing on every gating path (guard hook + Step 10d
    pre-push lint), so its silent un-bundling must turn a test RED."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_lessons_index or no_flags:\s*\n"
        r"\s*errors\.extend\(check_lessons_index\(\)\)",
        src,
    ), "check_lessons_index is not dispatched on the no-flags branch"
    assert "or args.check_lessons_index" in src, (
        "--check-lessons-index is missing from the no_flags detection tuple"
    )


def test_check_lessons_index_fails_on_row_over_cap(tmp_path):
    # Per-row cap (#1269): one bloated row FAILs, NAMED, at addition time.
    from workflow_lint import _LESSONS_ROW_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"
    _write_lessons_row(rules, "alpha", _LESSONS_ROW_MAX_BYTES + 1)
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("'alpha'" in e and "per-row cap" in e for e in errs)

    # Strictly-greater boundary: a row at EXACTLY the cap passes.
    _write_lessons_row(rules, "alpha", _LESSONS_ROW_MAX_BYTES)
    assert check_lessons_index(repo_root=tmp_path) == []


def test_check_lessons_index_grandfather_row_over_its_cap_fails(tmp_path, monkeypatch):
    # A grandfathered row over ITS cap FAILs, naming BOTH remedies (trim the
    # row vs bump-with-hug the dict entry). Synthetic grandfather entry so
    # the test is decoupled from the live dict's churn.
    import workflow_lint

    monkeypatch.setattr(workflow_lint, "_LESSONS_ROW_GRANDFATHER_MAX_BYTES", {"alpha": 460})
    rules = tmp_path / ".claude" / "rules"
    _write_lessons_row(rules, "alpha", 461)
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any(
        "grandfather cap" in e
        and "trim the row" in e
        and "_LESSONS_ROW_GRANDFATHER_MAX_BYTES['alpha']" in e
        for e in errs
    )

    # Strictly-greater + exact-hug boundary: a row at EXACTLY the cap passes
    # (cap - actual == 0 <= headroom bound).
    _write_lessons_row(rules, "alpha", 460)
    assert check_lessons_index(repo_root=tmp_path) == []

    # Exact hug bound passes: cap - actual == the headroom bound exactly.
    from workflow_lint import _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES

    _write_lessons_row(rules, "alpha", 460 - _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES)
    assert check_lessons_index(repo_root=tmp_path) == []


def test_check_lessons_index_grandfather_hug_and_obsolete_entry_fail(tmp_path, monkeypatch):
    # Grandfather hygiene (#986 pattern): a cap more than the headroom bound
    # above the live row FAILs (loose/stale cap), and an entry whose row
    # dropped to <= the general row cap FAILs as obsolete (remove it).
    import workflow_lint
    from workflow_lint import (
        _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES,
        _LESSONS_ROW_MAX_BYTES,
    )

    monkeypatch.setattr(workflow_lint, "_LESSONS_ROW_GRANDFATHER_MAX_BYTES", {"alpha": 460})
    rules = tmp_path / ".claude" / "rules"

    # Hug FAIL: one byte past the exact-hug bound (row still over the
    # general cap, so the obsolete branch does not fire).
    _write_lessons_row(rules, "alpha", 460 - _LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES - 1)
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("max headroom" in e and "lower the cap" in e for e in errs)

    # Obsolete FAIL: the row now fits the general cap — remove the entry.
    _write_lessons_row(rules, "alpha", _LESSONS_ROW_MAX_BYTES)
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("no longer needs grandfathering" in e for e in errs)


def test_lessons_budget_constants_sane():
    # Live-tree config coherence (#1269/#1504): the constants must describe
    # the real LESSONS.md, and the production defaults must be ARMED.
    import inspect

    import workflow_lint as wl

    assert wl._LESSONS_WARN_BYTES < wl._LESSONS_MAX_BYTES
    assert 0 < wl._LESSONS_NONROW_MAX_BYTES < wl._LESSONS_MAX_BYTES
    # Defaults armed: a default flipped to None would disarm the non-row /
    # row caps fleet-wide while every explicit-kwarg test stayed green.
    params = inspect.signature(wl.check_lessons_index).parameters
    assert params["nonrow_max_bytes"].default == wl._LESSONS_NONROW_MAX_BYTES
    assert params["row_max_bytes"].default == wl._LESSONS_ROW_MAX_BYTES
    # Live-tree fit: the real index must satisfy the cap AND the non-row
    # budget, computed exactly as the check computes it (row bytes = the
    # _LESSONS_ROW_RE full-line matches; everything else is scaffolding).
    live = (wl._REPO_ROOT / ".claude" / "rules" / "LESSONS.md").read_bytes()
    assert len(live) <= wl._LESSONS_MAX_BYTES
    live_row_total = sum(
        len(m.group(0).encode("utf-8")) for m in wl._LESSONS_ROW_RE.finditer(live.decode("utf-8"))
    )
    live_nonrow = len(live) - live_row_total
    assert 0 <= live_nonrow <= wl._LESSONS_NONROW_MAX_BYTES, (live_nonrow, len(live))
    # Grandfather entries: each cap sits above the general row cap and hugs
    # its LIVE row (the synthetic-fixture tests cover the failure modes).
    rows = {m.group("name"): m.group(0) for m in wl._LESSONS_ROW_RE.finditer(live.decode("utf-8"))}
    for name, cap in wl._LESSONS_ROW_GRANDFATHER_MAX_BYTES.items():
        assert cap > wl._LESSONS_ROW_MAX_BYTES, name
        assert name in rows, f"grandfather entry '{name}' has no live index row"
        row_bytes = len(rows[name].encode("utf-8"))
        assert row_bytes <= cap, (name, row_bytes, cap)
        assert cap - row_bytes <= wl._LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES, (
            name,
            row_bytes,
            cap,
        )


def _write_rule_file(tmp_path, text, name="fixture-rule"):
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True, exist_ok=True)
    (rules / f"{name}.md").write_text(text, encoding="utf-8")


_VALID_RULE = '---\ndescription: "ok (paper: true) fixture"\npaths:\n  - "docs/**"\n---\n# body\n'


def test_rule_frontmatter_fails_on_unquoted_colon_description(tmp_path):
    # The live #1385 offender class: an unquoted `description:` containing
    # ': ' fails yaml.safe_load ("mapping values are not allowed here") and
    # the rule silently never on-demand-loads.
    _write_rule_file(tmp_path, '---\ndescription: proto (paper: true) x\npaths:\n  - "a/**"\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for the unquoted colon-bearing description"
    assert any("fixture-rule.md" in e and "not valid YAML" in e for e in errs)


def test_rule_frontmatter_fails_on_globs_key(tmp_path):
    # Valid YAML, but the stale `globs:` key the harness ignores — the check
    # names the `globs:` -> `paths:` rename.
    _write_rule_file(tmp_path, '---\ndescription: "ok"\nglobs:\n  - "a/**"\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for the stale globs: key"
    assert any("fixture-rule.md" in e and "globs" in e and "paths" in e for e in errs)


def test_rule_frontmatter_fails_on_missing_paths(tmp_path):
    _write_rule_file(tmp_path, '---\ndescription: "ok"\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for frontmatter with no paths: key"
    assert any("fixture-rule.md" in e and "no `paths:` key" in e for e in errs)


def test_rule_frontmatter_fails_on_non_list_paths(tmp_path):
    _write_rule_file(tmp_path, '---\ndescription: "ok"\npaths: "docs/**"\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for a scalar paths: value"
    assert any("fixture-rule.md" in e and "NON-EMPTY YAML list" in e for e in errs)


def test_rule_frontmatter_fails_on_empty_paths(tmp_path):
    _write_rule_file(tmp_path, '---\ndescription: "ok"\npaths: []\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for an empty paths: list"
    assert any("fixture-rule.md" in e and "NON-EMPTY YAML list" in e for e in errs)


def test_rule_frontmatter_fails_on_non_string_path_entry(tmp_path):
    _write_rule_file(tmp_path, '---\ndescription: "ok"\npaths:\n  - 3\n---\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for a non-string paths: entry"
    assert any("fixture-rule.md" in e and "non-empty strings" in e for e in errs)


def test_rule_frontmatter_fails_on_unterminated_block(tmp_path):
    _write_rule_file(tmp_path, '---\ndescription: "ok"\npaths:\n  - "a/**"\n# no closer\n')
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for an unterminated frontmatter block"
    assert any("fixture-rule.md" in e and "never closed" in e for e in errs)


def test_rule_frontmatter_fails_on_non_mapping(tmp_path):
    _write_rule_file(tmp_path, "---\n- a\n- b\n---\n")
    errs = check_rule_frontmatter_parses(repo_root=tmp_path)
    assert errs, "expected a FAIL for non-mapping frontmatter"
    assert any("fixture-rule.md" in e and "not a key: value mapping" in e for e in errs)


def test_rule_frontmatter_passes_on_no_frontmatter(tmp_path):
    # No leading '---' => always-on / LESSONS-indexed rule, exempt.
    _write_rule_file(tmp_path, "# title\nbody\n")
    assert check_rule_frontmatter_parses(repo_root=tmp_path) == []


def test_rule_frontmatter_passes_on_valid(tmp_path):
    # Pins the no-false-positive claim for legitimate forms: a quoted
    # colon-bearing description, extra-key tolerance (`name:`), and a
    # flow-style paths: list.
    _write_rule_file(tmp_path, _VALID_RULE)
    _write_rule_file(
        tmp_path,
        '---\nname: x\ndescription: "ok"\npaths:\n  - "a/**"\n---\n',
        name="extra-key-rule",
    )
    _write_rule_file(
        tmp_path,
        '---\ndescription: "ok"\npaths: ["a/**", "b/*.py"]\n---\n',
        name="flow-style-rule",
    )
    assert check_rule_frontmatter_parses(repo_root=tmp_path) == []


def test_rule_frontmatter_passes_on_live_repo():
    # The live-tree invariant that forces the 5 offender fixes to land in the
    # same diff as this check (mirrors test_check_lessons_index_passes_on_live_repo).
    assert check_rule_frontmatter_parses() == []


def test_rule_frontmatter_bundled_in_no_flags():
    """NON-VACUOUS no-flags bundling pin (#1385; the house source-pin shape
    of test_pipe_python_bundled_in_no_flags_source_pin): the check must be
    dispatched by the BARE ``workflow_lint.py`` run — a later refactor of
    the no_flags tuple / dispatch ladder must not silently unbundle it (the
    exact 'present but never fires' meta-class this check closes)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_rule_frontmatter_parses or no_flags:\s*\n"
        r"\s*errors\.extend\(check_rule_frontmatter_parses\(\)\)",
        src,
    ), "check_rule_frontmatter_parses is not dispatched on the no-flags branch"
    assert "or args.check_rule_frontmatter_parses" in src, (
        "--check-rule-frontmatter-parses is missing from the no_flags detection tuple"
    )


def test_compute_shape_review_lens_live_tree_passes() -> None:
    """The real .claude/agents tree carries the #806 lens in both files."""
    assert check_compute_shape_review_lens() == []


def test_compute_shape_review_lens_flags_missing(tmp_path) -> None:
    """A code-reviewer agent pair missing the lens FAILs the #806 check."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    # codex file has the lens (all FOUR #806+#875 tokens); the Claude file
    # does not → FAILs for the Claude file only, none for the codex file.
    (agents / "code-reviewer.md").write_text("# reviewer\nno lens here\n")
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nStep 0.67 Compute-shape-vs-dispatcher\nblocker tag compute-shape-mismatch\n"
        "work-conserving schedule sub-check\nanti-pattern (d) per-row compression\n"
    )
    errors = check_compute_shape_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for the code-reviewer.md missing the lens"
    # Key on the SUBJECT of each error — the file path before the first ': '
    # (the message body cross-references the sibling filename in prose, so a
    # naive substring search would collide). Every error must be ABOUT the
    # Claude file; none about the codex file (which carries all four tokens).
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("code-reviewer.md") for s in subjects), subjects
    assert any(s.endswith("/code-reviewer.md") for s in subjects), subjects
    assert all(not s.endswith("/codex-code-reviewer.md") for s in subjects), subjects


def test_compute_shape_review_lens_flags_both_files(tmp_path) -> None:
    """Both files missing the lens accumulate one FAIL per file (#806).

    Pins the per-file error-accumulation loop (not just the scoping asserted
    by ``test_compute_shape_review_lens_flags_missing``): each file is missing
    exactly ONE distinct required token (of the four #806+#875 tokens), so the
    loop emits exactly one error per file — two total, one subject per file. A
    regression that broke out of the loop after the first file, or that
    de-duplicated across files, would fail this.
    """
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    # Claude file carries 3 of 4 tokens (missing `work-conserving`); codex
    # file carries 3 of 4 (missing `compute-shape-mismatch`) → one distinct
    # missing token apiece → exactly one error per file.
    (agents / "code-reviewer.md").write_text(
        "# reviewer\nStep 0.67 Compute-shape-vs-dispatcher\n"
        "blocker tag compute-shape-mismatch\nanti-pattern (d) per-row compression\n"
    )
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nStep 0.67 Compute-shape-vs-dispatcher\n"
        "work-conserving schedule sub-check\nanti-pattern (d) per-row compression\n"
    )
    errors = check_compute_shape_review_lens(repo_root=tmp_path)
    assert len(errors) == 2, errors
    subjects = {e.split(": ", 1)[0] for e in errors}
    assert any(s.endswith("/code-reviewer.md") for s in subjects), subjects
    assert any(s.endswith("/codex-code-reviewer.md") for s in subjects), subjects


def test_compute_shape_review_lens_flags_missing_875_tokens(tmp_path) -> None:
    """Legacy-#806-tokens-only files FAIL on the two #875 tokens.

    Regression test for the #875 extension (work-conserving schedule sub-check
    + throughput anti-pattern (d)): both tmp files carry ONLY the two legacy
    #806 tokens, so the check must emit exactly 4 errors (2 per file), each
    naming `work-conserving` or `per-row compression`. Under the pre-#875
    two-token tuple this tree returned `[]`, so this test fails when the lint
    tuple lacks the #875 tokens and passes post-fix.
    """
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    legacy_only = (
        "# agent\nStep 0.67 Compute-shape-vs-dispatcher\nblocker tag compute-shape-mismatch\n"
    )
    (agents / "code-reviewer.md").write_text(legacy_only)
    (agents / "codex-code-reviewer.md").write_text(legacy_only)
    errors = check_compute_shape_review_lens(repo_root=tmp_path)
    assert len(errors) == 4, errors
    assert all("'work-conserving'" in e or "'per-row compression'" in e for e in errors), errors
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert sum(s.endswith("/code-reviewer.md") for s in subjects) == 2, subjects
    assert sum(s.endswith("/codex-code-reviewer.md") for s in subjects) == 2, subjects


def test_long_loop_restartability_review_lens_live_tree_passes() -> None:
    """The real tree carries the #823/#881 lens on all three surfaces."""
    assert check_long_loop_restartability_review_lens() == []


def _write_long_loop_conforming_tree(tmp_path) -> None:
    """Write all three #881 surfaces with their full per-file token sets."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n### Step 3.6: Long-loop restartability\nresume predicate\n"
    )
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nThe Step 3.6 Long-loop restartability rule VERBATIM\n"
        "persistence + resume predicate pair\nSteps 3, 3.5, 3.6, 3.7, 4.5\n"
    )
    (rules / "code-style.md").write_text(
        "# style\nIntra-phase grain for long loops\nresume predicate\n"
    )


def test_long_loop_restartability_review_lens_conforming_tmp_tree_passes(tmp_path) -> None:
    """A tmp tree carrying every per-file token returns no errors (#881)."""
    _write_long_loop_conforming_tree(tmp_path)
    assert check_long_loop_restartability_review_lens(repo_root=tmp_path) == []


def test_long_loop_restartability_review_lens_flags_missing_per_file(tmp_path) -> None:
    """Each surface missing one distinct token FAILs exactly once per file (#881).

    Strips a DIFFERENT required token per surface — the Claude file loses the
    `Long-loop restartability` heading, the codex file loses the inlined-rubric
    `3.5, 3.6, 3.7` enumeration (the executable-prompt pin the copy-list-only
    token check would false-PASS), the rules file loses `Intra-phase grain` —
    so the per-file token loop must emit exactly one error per file, each
    naming its missing token.
    """
    _write_long_loop_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    rules = tmp_path / ".claude" / "rules"
    (agents / "code-reviewer.md").write_text("# reviewer\nresume predicate\n")
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nThe Step 3.6 Long-loop restartability rule VERBATIM\n"
        "persistence + resume predicate pair\nSteps 3, 3.5, 3.7, 4.5\n"
    )
    (rules / "code-style.md").write_text("# style\nresume predicate\n")
    errors = check_long_loop_restartability_review_lens(repo_root=tmp_path)
    assert len(errors) == 3, errors
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert sum(s.endswith("/code-reviewer.md") for s in subjects) == 1, subjects
    assert sum(s.endswith("/codex-code-reviewer.md") for s in subjects) == 1, subjects
    assert sum(s.endswith("/code-style.md") for s in subjects) == 1, subjects
    assert any("'Long-loop restartability'" in e for e in errors), errors
    assert any("'3.5, 3.6, 3.7'" in e for e in errors), errors
    assert any("'Intra-phase grain'" in e for e in errors), errors


def test_hollow_gate_review_lens_live_tree_passes() -> None:
    """The real tree carries the #890 lens on all three surfaces."""
    assert check_hollow_verification_gate_review_lens() == []


def _write_hollow_gate_conforming_tree(tmp_path) -> None:
    """Write all three #890 surfaces with their full per-file assertions."""
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n**Hollow-verification-gate sub-check.** trace gate->dispatch\n"
        "**Blocker tags:** [`hollow-verification-gate` | `substantive`]\n"
    )
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nthe hollow-verification-gate sub-check (copy in full)\n"
        "**Blocker tags:** [`hollow-verification-gate` | `substantive`]\n"
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.67, 0.68, 3.6}}\n"
    )
    (agents / "efficiency-critic.md").write_text(
        "# eff\n4. **Hollow-verification-gate (Step 0.68 sub-check).**\n"
        "**Blocker tags:** [`hollow-verification-gate` | `substantive`]\n"
    )


def test_hollow_gate_review_lens_conforming_tmp_tree_passes(tmp_path) -> None:
    """A tmp tree carrying every per-file assertion returns no errors (#890)."""
    _write_hollow_gate_conforming_tree(tmp_path)
    assert check_hollow_verification_gate_review_lens(repo_root=tmp_path) == []


def test_hollow_gate_review_lens_flags_missing_per_file(tmp_path) -> None:
    """Each surface failing a DIFFERENT assertion FAILs exactly once per file.

    The Claude file loses the sub-check PROSE (keeps its Blocker-tags line),
    the codex file drops the tag from its Blocker-tags LINE (keeps prose +
    the 0.68 placeholder), the efficiency file loses its Blocker-tags line
    entirely — so the check emits exactly one error per file, one per
    assertion kind (prose token / tag-off-template-line / template-line-gone).
    """
    _write_hollow_gate_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "code-reviewer.md").write_text(
        "# reviewer\nno sub-check here\n"
        "**Blocker tags:** [`hollow-verification-gate` | `substantive`]\n"
    )
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nthe hollow-verification-gate sub-check (copy in full)\n"
        "**Blocker tags:** [`substantive`]\n"
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.67, 0.68, 3.6}}\n"
    )
    (agents / "efficiency-critic.md").write_text(
        "# eff\n4. **Hollow-verification-gate (Step 0.68 sub-check).**\n"
    )
    errors = check_hollow_verification_gate_review_lens(repo_root=tmp_path)
    assert len(errors) == 3, errors
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert sum(s.endswith("/code-reviewer.md") for s in subjects) == 1, subjects
    assert sum(s.endswith("/codex-code-reviewer.md") for s in subjects) == 1, subjects
    assert sum(s.endswith("/efficiency-critic.md") for s in subjects) == 1, subjects
    assert any("'Hollow-verification-gate sub-check'" in e for e in errors), errors
    assert any("dropped out of the verdict template" in e for e in errors), errors
    assert any("no line starts with" in e for e in errors), errors


def test_hollow_gate_review_lens_flags_missing_rubric_enumeration(tmp_path) -> None:
    """A codex placeholder line lacking '0.68' FAILs (the #606 class)."""
    _write_hollow_gate_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\nthe hollow-verification-gate sub-check (copy in full)\n"
        "**Blocker tags:** [`hollow-verification-gate` | `substantive`]\n"
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.67, 0.7, 3.6}}\n"
    )
    errors = check_hollow_verification_gate_review_lens(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("/codex-code-reviewer.md"), errors
    assert "0.68" in errors[0] and "INLINED RUBRIC" in errors[0], errors


def test_hollow_gate_review_lens_flags_missing_file(tmp_path) -> None:
    """A missing required surface file is itself an error (the #891 shape)."""
    _write_hollow_gate_conforming_tree(tmp_path)
    (tmp_path / ".claude" / "agents" / "efficiency-critic.md").unlink()
    errors = check_hollow_verification_gate_review_lens(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("/efficiency-critic.md"), errors
    assert "missing" in errors[0], errors


def test_hollow_gate_review_lens_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting the
    ``or no_flags`` ladder branch must fail this test (mutation-visible),
    closing the dead-tripwire gap where all direct-call tests stay green while
    the CLI never runs the check. Follows the
    ``test_vm_thread_cap_guidance_bundled_in_no_flags`` pattern (in-process
    ``main([])``, ``_REPO_ROOT`` monkeypatched; other bundled checks contribute
    unrelated errors on the minimal tree, so the assertion keys on the
    hollow-gate diagnostic + the offending file path)."""
    import workflow_lint as wl

    _write_hollow_gate_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "efficiency-critic.md").write_text(
        "# eff\n4. **Hollow-verification-gate (Step 0.68 sub-check).**\n"
        "**Blocker tags:** [`substantive`]\n"
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a violating tree:\n{err}"
    assert "hollow-verification-gate" in err and "efficiency-critic.md" in err, (
        f"the hollow-gate diagnostic (naming efficiency-critic.md) is missing "
        f"from the no-flags run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


def _write_smoke_arch_conforming_tree(tmp_path) -> None:
    """Write all three #822 surfaces in conforming shape under tmp_path.

    Tests then break exactly ONE surface each, so failures stay attributable
    (absence errors from the untouched surfaces would otherwise pollute
    counts).
    """
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n"
        "### Step 0.55: Smoke-architecture marker presence gate\n"
        "check for an `epm:smoke-architecture-check` events row.\n"
        "### Step 0.8\nnext section\n"
    )
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\n"
        '- "Step 0.55: Smoke-architecture marker presence gate" bullet naming\n'
        "  `epm:smoke-architecture-check`.\n"
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0, 0.5, 0.55, 0.6}}\n"
    )
    skill = tmp_path / ".claude" / "skills" / "issue"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "# issue skill\n"
        "**5c-bis. Mechanical strip** — a blocker naming\n"
        "`epm:smoke-architecture-check` gets the per-blocker sub-recipe.\n"
        "**5c-ter. Next step**\n"
    )


def test_smoke_architecture_review_lens_live_tree_passes() -> None:
    """The real tree carries the #822 presence gate on all three surfaces."""
    assert check_smoke_architecture_review_lens() == []


def test_smoke_architecture_review_lens_conforming_fixture_passes(tmp_path) -> None:
    """The synthetic conforming tree passes — validates the fixture itself."""
    _write_smoke_arch_conforming_tree(tmp_path)
    assert check_smoke_architecture_review_lens(repo_root=tmp_path) == []


def test_smoke_architecture_review_lens_flags_missing_step_section(tmp_path) -> None:
    """code-reviewer.md without the '### Step 0.55' section FAILs (#822)."""
    _write_smoke_arch_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "code-reviewer.md").write_text("# reviewer\nno presence gate here\n")
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for code-reviewer.md missing Step 0.55"
    # Key on the SUBJECT of each error — the file path before the first ': '
    # (message bodies cross-reference sibling filenames in prose, so a naive
    # substring search would collide).
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/code-reviewer.md") for s in subjects), subjects
    assert all(not s.endswith("/codex-code-reviewer.md") for s in subjects), subjects


def test_smoke_architecture_review_lens_flags_marker_missing_from_section(tmp_path) -> None:
    """A Step 0.55 section whose body drops the marker name FAILs (#822).

    The region is the section body up to the next '### ' heading — the marker
    appearing elsewhere in the file must NOT satisfy the check.
    """
    _write_smoke_arch_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n"
        "### Step 0.55: Smoke-architecture marker presence gate\n"
        "body without the marker name.\n"
        "### Step 0.8\nmentions `epm:smoke-architecture-check` outside the region\n"
    )
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for the marker-less Step 0.55 body"
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/code-reviewer.md") for s in subjects), subjects


def test_smoke_architecture_review_lens_flags_codex_tokens(tmp_path) -> None:
    """codex-code-reviewer.md missing bullet heading + marker FAILs twice (#822)."""
    _write_smoke_arch_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\n{{INLINED RUBRIC FROM code-reviewer.md Steps 0, 0.5, 0.55, 0.6}}\n"
    )
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    # Both required copy-list tokens are missing → one error per token.
    assert len(errors) == 2, errors
    subjects = {e.split(": ", 1)[0] for e in errors}
    assert subjects and all(s.endswith("/codex-code-reviewer.md") for s in subjects), subjects


def test_smoke_architecture_review_lens_flags_marker_outside_codex_bullet(tmp_path) -> None:
    """A codex copy-list bullet stripped of the marker FAILs even when the
    marker survives elsewhere in the file (#822, round 2).

    The bullet region runs from the heading token to the next line-start
    '- "' bullet; a marker mention in a later '**Blocker tags:**' line must
    NOT satisfy the check (the file-global drift case the round-1 check
    missed).
    """
    _write_smoke_arch_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\n"
        '- "Step 0.55: Smoke-architecture marker presence gate" bullet with\n'
        "  the marker name stripped from the bullet body.\n"
        '- "Step 0.6: End-to-end smoke gate" next bullet.\n'
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0, 0.5, 0.55, 0.6}}\n"
        "**Blocker tags:** `marker-shape` blockers name "
        "`epm:smoke-architecture-check` here, outside the bullet.\n"
    )
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for the marker-less Step 0.55 copy-list bullet"
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/codex-code-reviewer.md") for s in subjects), subjects
    assert any("copy-list bullet" in e for e in errors), errors


def test_smoke_architecture_review_lens_flags_rubric_placeholder(tmp_path) -> None:
    """The '{{INLINED RUBRIC' placeholder line without '0.55' FAILs (#822)."""
    _write_smoke_arch_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex\n"
        '- "Step 0.55: Smoke-architecture marker presence gate" bullet naming\n'
        "  `epm:smoke-architecture-check`.\n"
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0, 0.5, 0.6}}\n"
    )
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for the 0.55-less rubric placeholder line"
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/codex-code-reviewer.md") for s in subjects), subjects
    assert any("0.55" in e for e in errors), errors


def test_smoke_architecture_review_lens_flags_skill_region(tmp_path) -> None:
    """A 5c-bis region without the marker sub-recipe FAILs (#822).

    The marker appearing OUTSIDE the '**5c-bis.' → '**5c-ter.' region must not
    satisfy the check.
    """
    _write_smoke_arch_conforming_tree(tmp_path)
    skill = tmp_path / ".claude" / "skills" / "issue"
    (skill / "SKILL.md").write_text(
        "# issue skill\n"
        "`epm:smoke-architecture-check` mentioned before the region.\n"
        "**5c-bis. Mechanical strip** — no sub-recipe here.\n"
        "**5c-ter. Next step**\n"
    )
    errors = check_smoke_architecture_review_lens(repo_root=tmp_path)
    assert errors, "expected a FAIL for the sub-recipe-less 5c-bis region"
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/SKILL.md") for s in subjects), subjects


# ---------------------------------------------------------------------------
# ``check_smoke_output_hygiene`` (#842): the smoke output-path hygiene rule
# ("Smoke outputs never overwrite committed artifacts") must sit INSIDE the
# load-bearing region of each of its three surfaces — region-aware +
# whitespace-normalized. Incident #722: smoke runs clobbered committed
# eval_results//figures/ artifacts three times.
# ---------------------------------------------------------------------------

_SMOKE_HYGIENE_SURFACES = (
    ".claude/agents/experiment-implementer.md",
    ".claude/agents/code-reviewer.md",
    ".claude/skills/issue/SKILL.md",
)


def _write_smoke_hygiene_conforming_tree(tmp_path) -> None:
    """Write all three #842 surfaces in conforming shape under tmp_path.

    Tests then break exactly ONE surface each, so failures stay attributable
    (absence errors from the untouched surfaces would otherwise pollute
    counts).
    """
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "experiment-implementer.md").write_text(
        "# implementer\n"
        "\n"
        "3. **End-to-end smoke run PER PHASE.** For EACH distinct entrypoint\n"
        "   run a tiny slice and record the digest.\n"
        "\n"
        "   **Smoke outputs never overwrite committed artifacts.** Divert\n"
        "   smoke output to a scratch dir, or restore-after-smoke and confirm\n"
        "   `git status --porcelain -- eval_results/ figures/` is empty.\n"
        "4. **Self-review against plan.** Walk the plan.\n"
    )
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n"
        "\n"
        "### Step 0.6: End-to-end smoke gate (`type:experiment` only)\n"
        "\n"
        "The smoke gate body.\n"
        "\n"
        '**Smoke output-path hygiene ("Smoke outputs never overwrite committed artifacts").**\n'
        "Clobber is a Critical tagged `substantive`; reviewer-self runs\n"
        "restore what their own commands touched.\n"
        "\n"
        "### Step 0.65: Raw-completions upload wiring gate\n"
        "\n"
        "next section\n"
    )
    skill = tmp_path / ".claude" / "skills" / "issue"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "# issue skill\n"
        "\n"
        "**End-to-end smoke gate (experiment tasks).** A code-review PASS\n"
        "needs a per-phase smoke on a tiny real slice.\n"
        "Smoke outputs never overwrite committed artifacts — the disposition\n"
        "is stated per the implementer contract.\n"
        "\n"
        "**5b. Read both markers.**\n"
    )


def test_smoke_output_hygiene_live_tree_passes() -> None:
    """The real tree carries the #842 hygiene rule on all three surfaces."""
    assert check_smoke_output_hygiene() == []


def test_smoke_output_hygiene_conforming_fixture_passes(tmp_path) -> None:
    """The synthetic conforming tree passes — validates the fixture itself."""
    _write_smoke_hygiene_conforming_tree(tmp_path)
    assert check_smoke_output_hygiene(repo_root=tmp_path) == []


_SMOKE_HYGIENE_ANCHORLESS = {
    # Region heading present, anchor ABSENT from the region — but present
    # elsewhere in the same file, so whole-file substring matching would
    # false-green. Pins the region-awareness property.
    ".claude/agents/experiment-implementer.md": (
        "# implementer\n"
        "\n"
        "3. **End-to-end smoke run PER PHASE.** For EACH distinct entrypoint\n"
        "   run a tiny slice and record the digest.\n"
        "4. **Self-review against plan.** Smoke outputs never overwrite\n"
        "   committed artifacts is cross-referenced here, OUTSIDE item 3.\n"
    ),
    ".claude/agents/code-reviewer.md": (
        "# reviewer\n"
        "\n"
        "### Step 0.6: End-to-end smoke gate (`type:experiment` only)\n"
        "\n"
        "The smoke gate body, hygiene rule dropped.\n"
        "\n"
        "### Step 4: Run / Verify Tests\n"
        "\n"
        "Smoke outputs never overwrite committed artifacts — mentioned\n"
        "outside the Step 0.6 region the Codex twin inlines.\n"
    ),
    ".claude/skills/issue/SKILL.md": (
        "# issue skill\n"
        "\n"
        "**End-to-end smoke gate (experiment tasks).** A code-review PASS\n"
        "needs a per-phase smoke; hygiene sentence dropped.\n"
        "\n"
        "**5b. Read both markers.** Smoke outputs never overwrite committed\n"
        "artifacts — mentioned after the smoke-gate paragraph ends.\n"
    ),
}


@pytest.mark.parametrize("surface", _SMOKE_HYGIENE_SURFACES)
def test_smoke_output_hygiene_fails_on_missing_anchor(tmp_path, surface) -> None:
    """Dropping the anchor from any ONE surface's region FAILs, naming that
    file (#842) — even when the anchor phrase survives ELSEWHERE in the same
    file (region-awareness: a leftover cross-reference must not false-green,
    and the code-reviewer copy specifically must stay inside Step 0.6, the
    only step the Codex twin's inlined rubric carries)."""
    _write_smoke_hygiene_conforming_tree(tmp_path)
    (tmp_path / surface).write_text(_SMOKE_HYGIENE_ANCHORLESS[surface])
    errors = check_smoke_output_hygiene(repo_root=tmp_path)
    assert errors, f"expected a FAIL for {surface} missing the anchor"
    subjects = [e.split(": ", 1)[0] for e in errors]
    fname = surface.rsplit("/", 1)[-1]
    assert all(s.endswith(f"/{fname}") for s in subjects), (surface, errors)
    assert any("absent from" in e for e in errors), errors


@pytest.mark.parametrize("surface", _SMOKE_HYGIENE_SURFACES)
def test_smoke_output_hygiene_fails_on_missing_file(tmp_path, surface) -> None:
    """A surface file absent from the tree FAILs with the missing-file branch
    naming that path (#842)."""
    _write_smoke_hygiene_conforming_tree(tmp_path)
    (tmp_path / surface).unlink()
    errors = check_smoke_output_hygiene(repo_root=tmp_path)
    assert errors, f"expected a FAIL for the absent {surface}"
    fname = surface.rsplit("/", 1)[-1]
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith(f"/{fname}") for s in subjects), (surface, errors)
    assert any("missing" in e for e in errors), errors


def test_smoke_output_hygiene_fails_on_missing_region_heading(tmp_path) -> None:
    """A surface whose region heading was restructured away FAILs LOUD (#842
    property (c)) — never a vacuous pass, even if the anchor survives
    somewhere in the file."""
    _write_smoke_hygiene_conforming_tree(tmp_path)
    agents = tmp_path / ".claude" / "agents"
    (agents / "code-reviewer.md").write_text(
        "# reviewer\n"
        "\n"
        "### Step 0.7: A renamed step\n"
        "\n"
        "Smoke outputs never overwrite committed artifacts — the anchor\n"
        "survives but its Step 0.6 region heading is gone.\n"
    )
    errors = check_smoke_output_hygiene(repo_root=tmp_path)
    assert errors, "expected a FAIL for the missing Step 0.6 region heading"
    subjects = [e.split(": ", 1)[0] for e in errors]
    assert all(s.endswith("/code-reviewer.md") for s in subjects), errors
    assert any("region heading" in e for e in errors), errors


def test_smoke_output_hygiene_hard_wrapped_anchor_passes(tmp_path) -> None:
    """An anchor hard-wrapped across two lines INSIDE the correct region still
    PASSes — pins the whitespace-normalized matching so an innocent prose
    reflow cannot spuriously FAIL the fleet's default run (#842 property (a))."""
    _write_smoke_hygiene_conforming_tree(tmp_path)
    skill = tmp_path / ".claude" / "skills" / "issue"
    (skill / "SKILL.md").write_text(
        "# issue skill\n"
        "\n"
        "**End-to-end smoke gate (experiment tasks).** A code-review PASS\n"
        "needs a per-phase smoke on a tiny real slice. Smoke outputs never\n"
        "overwrite committed artifacts — the disposition is stated per the\n"
        "implementer contract.\n"
        "\n"
        "**5b. Read both markers.**\n"
    )
    assert check_smoke_output_hygiene(repo_root=tmp_path) == []


def test_smoke_output_hygiene_wired_into_default_run(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags CLI-path REGISTRATION test (#842): the default run must
    exercise ``check_smoke_output_hygiene`` — a check that exists only behind
    its ``--check-smoke-output-hygiene`` flag while never being bundled into
    the ``no_flags`` dispatch would leave every other acceptance command
    green (the same bundling gap the #712 §4f wiring test pinned).

    Doctors a minimal tree missing the anchor on one surface, points the
    lint module's ``_REPO_ROOT`` at it, and invokes ``main([])`` in-process:
    the run must exit non-zero with the #842 diagnostic in stderr. Other
    bundled checks contribute unrelated errors on the minimal tree (missing
    LESSONS.md etc.) — the assertion keys on the #842 error string, so those
    are harmless.
    """
    import workflow_lint as wl

    _write_smoke_hygiene_conforming_tree(tmp_path)
    (tmp_path / ".claude/skills/issue/SKILL.md").write_text(
        _SMOKE_HYGIENE_ANCHORLESS[".claude/skills/issue/SKILL.md"]
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an anchor-less tree:\n{err}"
    assert "#842" in err, (
        f"the #842 smoke-output-hygiene diagnostic is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


# --- #891 shared-VM thread-cap guidance-pin tests ---------------------------

_VM_CAP_PREFIX = (
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8"
    " MALLOC_ARENA_MAX=2"
)

_VM_CAP_FLOORS = {
    ".claude/skills/issue/SKILL.md": 1,
    ".claude/agents/experiment-implementer.md": 2,
    ".claude/rules/code-style.md": 3,
    ".claude/rules/analyzer-section-reference.md": 1,
}


def _write_vm_cap_fixture(root: Path, counts: dict[str, int]) -> None:
    """Write the four #891 pinned guidance files, each carrying ``counts[rel]``
    template-instance stand-ins of the thread-cap prefix. A rel with count 0
    is written WITHOUT the prefix; a rel absent from ``counts`` is not written
    at all (the missing-file case)."""
    for rel, n in counts.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {rel}", "rationale prose uses the shorthand OMP/MKL/OPENBLAS/NUMEXPR=8 only."]
        lines += [
            f"template {i}: `setsid nohup env {_VM_CAP_PREFIX} <cmd> < /dev/null >> <log> 2>&1`"
            for i in range(n)
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_vm_thread_cap_guidance_live_tree_passes() -> None:
    """The real tree carries the #891 prefix at (or above) every count floor."""
    assert check_vm_thread_cap_guidance() == []


def test_vm_thread_cap_guidance_flags_missing_prefix(tmp_path) -> None:
    """All four files present; one lacks the prefix entirely -> exactly one
    error naming it (the other three satisfy their count floors)."""
    counts = dict(_VM_CAP_FLOORS)
    counts[".claude/rules/analyzer-section-reference.md"] = 0
    _write_vm_cap_fixture(tmp_path, counts)
    errors = check_vm_thread_cap_guidance(repo_root=tmp_path)
    assert len(errors) == 1, errors
    subject = errors[0].split(": ", 1)[0]
    assert subject.endswith("analyzer-section-reference.md"), errors
    assert "0 occurrence(s)" in errors[0], errors


def test_vm_thread_cap_guidance_flags_below_count_floor(tmp_path) -> None:
    """A file with SOME occurrences but fewer than its floor (1 of the
    required 3 in code-style.md) FAILs — the template-strip case a bare
    presence check would miss."""
    counts = dict(_VM_CAP_FLOORS)
    counts[".claude/rules/code-style.md"] = 1
    _write_vm_cap_fixture(tmp_path, counts)
    errors = check_vm_thread_cap_guidance(repo_root=tmp_path)
    assert len(errors) == 1, errors
    subject = errors[0].split(": ", 1)[0]
    assert subject.endswith("code-style.md"), errors
    assert "expected >= 3" in errors[0], errors


def test_vm_thread_cap_guidance_flags_missing_file(tmp_path) -> None:
    """A missing guidance file is itself an error (#891)."""
    counts = dict(_VM_CAP_FLOORS)
    counts.pop(".claude/agents/experiment-implementer.md")
    _write_vm_cap_fixture(tmp_path, counts)
    errors = check_vm_thread_cap_guidance(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("experiment-implementer.md"), errors
    assert "missing" in errors[0], errors


def test_vm_thread_cap_guidance_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #891 check — deleting
    its ``or no_flags`` branch must fail this test (mutation-visible), closing
    the dead-tripwire gap where all direct-call tests stay green while the CLI
    never runs the check. Follows the
    ``test_smoke_output_hygiene_wired_into_default_run`` pattern: one pinned
    file below its floor, ``_REPO_ROOT`` monkeypatched to the fixture,
    ``main([])`` in-process. Other bundled checks contribute unrelated errors
    on the minimal tree, so the assertion keys on the #891 diagnostic + the
    offending file path."""
    import workflow_lint as wl

    counts = dict(_VM_CAP_FLOORS)
    counts[".claude/rules/code-style.md"] = 1
    _write_vm_cap_fixture(tmp_path, counts)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a below-floor tree:\n{err}"
    assert "#891" in err and "code-style.md" in err, (
        f"the #891 vm-thread-cap diagnostic (naming code-style.md) is missing "
        f"from the no-flags default run's stderr — the check is not bundled "
        f"into no_flags:\n{err}"
    )


# --- #1154 marker-recipe snippet-pin tests -----------------------------------

# Minimal conforming fixture tree: the two pinned doc files carry one verbatim
# snippet per registered pin (the bandstop snippet deliberately WRAPS across a
# line break — the reflow proof for the whitespace-normalized matching — and
# both docs carry the bare-`※` 63680 decoy that must never be captured); the
# four src files carry the constant lines from the live tree, including the
# `marker_tail_tokens` decoy field the sft pattern must not match.
_MARKER_RECIPE_FIXTURE_FILES: dict[str, str] = {
    "docs/marker_training_recipe.md": (
        "# marker training recipe (fixture)\n"
        "` ※` (Qwen-2.5-7B token id 83399). DV = on-policy log P(marker).\n"
        "| Marker | ` ※` id 83399 (assert encoding) | single rare token |\n"
        "Constraint: marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`).\n"
        "| Loss mask | marker-only (`MarkerOnlyDataCollator(tail_tokens=0)`) | keeps R |\n"
        "7. Run a pre-sweep anchor smoke. Confirm source ΔG ∈ [5, 12]\n"
        "   nat AND bystanders below the argmax ceiling.\n"
        "| #906 | completed | render-exact gate (4/200 rows dropped, reject floor 0.10) |\n"
        "Avoid bare `※` id 63680 (wrong token).\n"
    ),
    ".claude/rules/marker-training-recipe.md": (
        "# marker-training-recipe rule (fixture)\n"
        "Loss via `MarkerOnlyDataCollator(tail_tokens=0)`, response frozen.\n"
        "- Fail-loud above a rejection-fraction floor (0.10).\n"
        "> Stop when source log P over base ∈ [5, 12] nat (gate on bystanders).\n"
        '` ※` id 83399 only (assert `encode(" ※") == [83399]`). Avoid bare `※` id 63680.\n'
    ),
    "src/explore_persona_space/artifacts/recipe.py": "MARKER_TOKEN_ID = 83399\n",
    "src/explore_persona_space/train/sft.py": (
        "class MarkerOnlyDataCollator:\n"
        "    def __init__(\n"
        "        self,\n"
        "        tail_tokens: int = 0,\n"
        "    ) -> None:\n"
        "        pass\n"
        "\n"
        "\n"
        "class TrainLoraConfig:\n"
        "    marker_tail_tokens: int = 0\n"
    ),
    "src/explore_persona_space/artifacts/organisms.py": "MIX_MAX_REJECT_FRAC = 0.10\n",
    "src/explore_persona_space/eval/callbacks.py": (
        "class MarkerBandStopCallback:\n"
        "    def __init__(\n"
        "        self,\n"
        "        low_nats: float = 5.0,\n"
        "        high_nats: float = 12.0,\n"
        "    ) -> None:\n"
        "        pass\n"
    ),
}


def _write_marker_recipe_fixture(
    root: Path,
    overrides: dict[str, str] | None = None,
    omit: tuple[str, ...] = (),
) -> None:
    """Write the #1154 pinned doc + src files at the registry's exact rel-paths
    under ``root``. ``overrides`` swap in mutated file contents (seeded drift);
    rel-paths in ``omit`` are not written at all (the missing-file case)."""
    contents = dict(_MARKER_RECIPE_FIXTURE_FILES)
    contents.update(overrides or {})
    for rel, text in contents.items():
        if rel in omit:
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def test_marker_recipe_snippets_live_tree_passes() -> None:
    """The real tree's pinned doc snippets agree with the code constants
    (launch invariant: 0 false positives; the live drift gate going forward)."""
    assert check_marker_recipe_snippets() == []


def test_marker_recipe_snippets_fixture_tree_passes(tmp_path) -> None:
    """The conforming fixture tree passes — pins the fixture itself so every
    mutation test below fails for its seeded drift, not fixture rot."""
    _write_marker_recipe_fixture(tmp_path)
    assert check_marker_recipe_snippets(repo_root=tmp_path) == []


def test_marker_recipe_snippets_flags_code_drift(tmp_path) -> None:
    """Mutating a bound code constant (MARKER_TOKEN_ID 83399 -> 99999) FAILs
    BOTH token-id pins, each naming the pin label and both values."""
    _write_marker_recipe_fixture(
        tmp_path,
        overrides={"src/explore_persona_space/artifacts/recipe.py": "MARKER_TOKEN_ID = 99999\n"},
    )
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 2, errors
    for label in ("marker-token-id", "rule-marker-token-id"):
        matching = [e for e in errors if f"pin '{label}'" in e]
        assert len(matching) == 1, errors
        assert "'83399'" in matching[0] and "'99999'" in matching[0], errors


def test_marker_recipe_snippets_flags_doc_drift(tmp_path) -> None:
    """The doc citing a stale value (reject floor 0.12 vs code 0.10) FAILs with
    exactly one error whose subject is the doc and which names both values."""
    drifted = _MARKER_RECIPE_FIXTURE_FILES["docs/marker_training_recipe.md"].replace(
        "reject floor 0.10", "reject floor 0.12"
    )
    _write_marker_recipe_fixture(tmp_path, overrides={"docs/marker_training_recipe.md": drifted})
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 1, errors
    subject = errors[0].split(": ", 1)[0]
    assert subject.endswith("docs/marker_training_recipe.md"), errors
    assert "'0.12'" in errors[0] and "'0.10'" in errors[0], errors


def test_marker_recipe_snippets_flags_missing_doc_snippet(tmp_path) -> None:
    """Rephrasing a pinned doc sentence away from its pattern FAILs loud
    ('doc snippet not found' — the rot alarm), naming the pin."""
    rephrased = _MARKER_RECIPE_FIXTURE_FILES["docs/marker_training_recipe.md"].replace(
        "reject floor 0.10", "rejection cutoff 0.10"
    )
    _write_marker_recipe_fixture(tmp_path, overrides={"docs/marker_training_recipe.md": rephrased})
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "doc snippet not found" in errors[0], errors
    assert "pin 'mix-reject-floor'" in errors[0], errors


def test_marker_recipe_snippets_flags_missing_code_symbol(tmp_path) -> None:
    """Removing a bound symbol from its source file FAILs every pin citing it
    ('code constant ... not found' — the rename/move alarm)."""
    _write_marker_recipe_fixture(
        tmp_path,
        overrides={"src/explore_persona_space/artifacts/organisms.py": "OTHER_CONST = 1\n"},
    )
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 2, errors  # the docs pin + the rule pin both cite organisms.py
    for e in errors:
        assert "not found" in e and "MIX_MAX_REJECT_FRAC" in e, errors


def test_marker_recipe_snippets_flags_ambiguous_src(tmp_path) -> None:
    """A second, conflicting `tail_tokens: int = 5,` signature in sft.py makes
    the collator pins ambiguous -> FAIL (the doc citation genuinely became
    ambiguous; the registry pattern must be tightened)."""
    ambiguous = _MARKER_RECIPE_FIXTURE_FILES["src/explore_persona_space/train/sft.py"] + (
        "\n"
        "\n"
        "class OtherCollator:\n"
        "    def __init__(\n"
        "        self,\n"
        "        tail_tokens: int = 5,\n"
        "    ) -> None:\n"
        "        pass\n"
    )
    _write_marker_recipe_fixture(
        tmp_path, overrides={"src/explore_persona_space/train/sft.py": ambiguous}
    )
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 2, errors  # the docs pin + the rule pin both cite sft.py
    for e in errors:
        assert "ambiguous" in e and "tail_tokens" in e, errors


def test_marker_recipe_snippets_flags_missing_file(tmp_path) -> None:
    """A missing pinned doc file is itself an error — once per pin bound to it
    (5 rule-file pins)."""
    _write_marker_recipe_fixture(tmp_path, omit=(".claude/rules/marker-training-recipe.md",))
    errors = check_marker_recipe_snippets(repo_root=tmp_path)
    assert len(errors) == 5, errors
    for e in errors:
        assert "missing" in e, errors
        assert e.split(": ", 1)[0].endswith("marker-training-recipe.md"), errors


def test_marker_recipe_snippets_does_not_capture_wrong_token_id() -> None:
    """On the LIVE tree, the token-id pins capture only 83399 — never the
    bare-`※` wrong-token id 63680 (docs line ~196 / rule line ~236 mention it
    with a backtick, not a space, before ※ — the space anchor skips it)."""
    import workflow_lint as wl

    for pin in _MARKER_RECIPE_PINS:
        if "token-id" not in pin.label:
            continue
        doc_text = (wl._REPO_ROOT / pin.doc_rel).read_text(encoding="utf-8")
        captures = re.findall(pin.doc_pattern, re.sub(r"\s+", " ", doc_text))
        assert captures, f"pin '{pin.label}': no live doc matches"
        assert "63680" not in captures, (pin.label, captures)
        assert set(captures) == {"83399"}, (pin.label, captures)


def test_marker_recipe_pins_have_one_capture_group() -> None:
    """Registry invariant: every pin's doc_pattern AND src_pattern compile with
    exactly ONE capture group (findall then returns bare value strings)."""
    for pin in _MARKER_RECIPE_PINS:
        assert re.compile(pin.doc_pattern).groups == 1, pin.label
        assert re.compile(pin.src_pattern).groups == 1, pin.label


def test_marker_recipe_values_equal_float_forms() -> None:
    """Value comparison is float-based when both sides parse ('5' == '5.0' —
    the doc band vs the callbacks.py float defaults), else exact-string."""
    assert _values_equal("5", "5.0") is True
    assert _values_equal("0.10", "0.1") is True
    assert _values_equal("83399", "83399") is True
    assert _values_equal("83399", "99999") is False
    assert _values_equal("abc", "abc") is True
    assert _values_equal("abc", "abd") is False


def test_marker_recipe_snippets_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #1154 check — deleting
    its ``or no_flags`` branch must fail this test (mutation-visible), closing
    the dead-tripwire gap where all direct-call tests stay green while the CLI
    never runs the check. Same mechanism as
    ``test_vm_thread_cap_guidance_bundled_in_no_flags``: a seeded code drift,
    ``_REPO_ROOT`` monkeypatched to the fixture, ``main([])`` in-process.
    Other bundled checks contribute unrelated errors on the minimal tree, so
    the assertion keys on the #1154 diagnostic + the offending doc path."""
    import workflow_lint as wl

    _write_marker_recipe_fixture(
        tmp_path,
        overrides={"src/explore_persona_space/artifacts/recipe.py": "MARKER_TOKEN_ID = 99999\n"},
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a drifted tree:\n{err}"
    assert "#1154" in err and "marker_training_recipe.md" in err, (
        f"the #1154 marker-recipe diagnostic (naming marker_training_recipe.md) "
        f"is missing from the no-flags default run's stderr — the check is not "
        f"bundled into no_flags:\n{err}"
    )


# --- #963 stale-label disposition-clause tests -------------------------------

# Conforming fixture: deliberately re-wrapped at a DIFFERENT column than the
# live SKILL.md, with FOUR of the five required tokens split mid-phrase across
# a line break (the reflow proof for the whitespace-normalized matching), and
# a span-end DECOY sentence AFTER the `\n\n` terminator that matches the
# negative regex ("On None, skip ...") — which must NOT trip the check
# (mechanically pins the paragraph-scoped extraction).
_STALE_LABEL_CONFORMING = (
    "# issue skill\n"
    "\n"
    "**Stale-label disposition rule (mechanical evidence only).** Run\n"
    "`task_workflow.followup_retro_close_evidence(events, label)` before executing\n"
    "a dispatched label. This check is a GHOST-label filter, NOT an\n"
    "execution gate. A None return\n"
    "means NO prior-run evidence exists and for a fresh never-run label\n"
    "the label\n"
    "EXECUTES as the dispatched round. The\n"
    "skip-and-surface disposition applies ONLY when the orchestrator suspects\n"
    "the label already ran.\n"
    "\n"
    "**Next.** On None, skip the label.\n"
)

_STALE_LABEL_EXECUTE_TOKEN = "the label EXECUTES as the dispatched round"


def _write_stale_label_tree(tmp_path, body: str) -> None:
    """Write ``.claude/skills/issue/SKILL.md`` under ``tmp_path`` with ``body``."""
    skill = tmp_path / ".claude" / "skills" / "issue"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(body, encoding="utf-8")


def test_stale_label_disposition_clause_live_tree_passes() -> None:
    """The real SKILL.md carries the #894/#763 paragraph with all five tokens
    and no unconditional skip-on-None coupling (pins Assumption 4)."""
    assert check_stale_label_disposition_clause() == []


def test_stale_label_disposition_clause_conforming_tmp_tree_passes(tmp_path) -> None:
    """A re-wrapped but token-identical paragraph PASSes (normalization works),
    and the span-end DECOY ('On None, skip the label.' AFTER the blank-line
    terminator) does not trip the negative regex (paragraph scoping works)."""
    _write_stale_label_tree(tmp_path, _STALE_LABEL_CONFORMING)
    assert check_stale_label_disposition_clause(repo_root=tmp_path) == []


def test_stale_label_disposition_clause_flags_missing_execute_clause(tmp_path) -> None:
    """Deleting the fresh-label-execute clause -> exactly one error naming
    that token (the task's primary regression target)."""
    body = _STALE_LABEL_CONFORMING.replace("EXECUTES as the dispatched round", "runs normally")
    assert body != _STALE_LABEL_CONFORMING
    _write_stale_label_tree(tmp_path, body)
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert repr(_STALE_LABEL_EXECUTE_TOKEN) in errors[0], errors


def test_stale_label_disposition_clause_flags_unconditional_skip_on_none(tmp_path) -> None:
    """A paragraph regaining 'On None return, skip the label ...' INSIDE the
    span FAILs via the negative regex. All five positive tokens are kept
    present so the test isolates the regex: ``len(errors) == 1`` is asserted
    mechanically, not incidentally."""
    body = _STALE_LABEL_CONFORMING.replace(
        "the label already ran.\n",
        "the label already ran. On None return, skip the label and surface it.\n",
    )
    assert body != _STALE_LABEL_CONFORMING
    _write_stale_label_tree(tmp_path, body)
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "'On None ... skip'" in errors[0], errors


def test_stale_label_disposition_clause_flags_duplicate_anchor(tmp_path) -> None:
    """A SECOND copy of the bold anchor -> exactly one duplicate-anchor error
    (MF2: span identity is load-bearing for the negative assertion — a stale
    duplicate could satisfy the token scan while the operative paragraph
    regresses)."""
    body = (
        _STALE_LABEL_CONFORMING
        + "\n**Stale-label disposition rule (stale duplicate).** Old copy.\n"
    )
    _write_stale_label_tree(tmp_path, body)
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "UNIQUE" in errors[0], errors
    assert "2 bold anchors" in errors[0], errors


def test_stale_label_disposition_clause_flags_split_paragraph(tmp_path) -> None:
    """A blank line inserted mid-paragraph (before the execute clause)
    truncates the span at the first blank line and FAILs the downstream
    tokens (pins Assumption 3's intended truncation behavior)."""
    body = _STALE_LABEL_CONFORMING.replace("\nthe label\nEXECUTES", "\n\nthe label\nEXECUTES")
    assert body != _STALE_LABEL_CONFORMING
    _write_stale_label_tree(tmp_path, body)
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert errors, "expected missing-token FAILs on a split paragraph"
    assert all("missing token" in e for e in errors), errors
    assert any(repr(_STALE_LABEL_EXECUTE_TOKEN) in e for e in errors), errors


def test_stale_label_disposition_clause_flags_missing_paragraph(tmp_path) -> None:
    """SKILL.md present but anchor absent -> exactly one error naming the
    bold anchor."""
    _write_stale_label_tree(tmp_path, "# issue skill\n\nNo stale-label paragraph here.\n")
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "missing the bold anchor" in errors[0], errors
    assert repr("**Stale-label disposition rule") in errors[0], errors


def test_stale_label_disposition_clause_flags_missing_file(tmp_path) -> None:
    """An empty tmp tree (no SKILL.md at all) -> a missing-file error."""
    errors = check_stale_label_disposition_clause(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "missing" in errors[0], errors


def test_stale_label_disposition_clause_paragraph_at_eof_passes(tmp_path) -> None:
    """A conforming paragraph that is the LAST content of the file — no
    blank-line terminator after it — still PASSes: pins the ``end == -1``
    span fallback (the span extends to EOF when ``text.find("\\n\\n", start)``
    misses)."""
    body = _STALE_LABEL_CONFORMING.split("\n\n**Next.")[0]
    assert body != _STALE_LABEL_CONFORMING
    # Precondition for exercising the fallback: no blank line anywhere at or
    # after the anchor, so the span-end search MUST return -1.
    assert "\n\n" not in body[body.find("**Stale-label disposition rule") :]
    _write_stale_label_tree(tmp_path, body)
    assert check_stale_label_disposition_clause(repo_root=tmp_path) == []


def test_stale_label_disposition_clause_wired_into_default_run(tmp_path, capsys, monkeypatch):
    """The no-flags CLI-path REGISTRATION test (MF1): the default run must
    exercise ``check_stale_label_disposition_clause`` — deleting the dispatch
    branch (``if args.check_stale_label_disposition or no_flags:``) or its
    ``or no_flags`` disjunct must fail this test (mutation-visible), closing
    the dead-tripwire gap where all direct-call tests stay green while the
    CLI never runs the check. NOTE: this test canNOT pin the
    ``or args.check_stale_label_disposition`` membership in the ``no_flags``
    tuple — ``main([])`` passes no flags, so ``no_flags`` computes True with
    or without that line; the tuple membership is pinned by
    ``test_stale_label_disposition_clause_dedicated_flag_isolated`` below.
    Follows the ``test_smoke_output_hygiene_wired_into_default_run`` /
    ``test_vm_thread_cap_guidance_bundled_in_no_flags`` house pattern:
    doctored non-conforming tree (execute clause deleted), ``_REPO_ROOT``
    monkeypatched to the fixture, ``main([])`` in-process. Other bundled
    checks contribute unrelated errors on the minimal tree, so the assertion
    keys on the #963 diagnostic string."""
    import workflow_lint as wl

    body = _STALE_LABEL_CONFORMING.replace("EXECUTES as the dispatched round", "runs normally")
    assert body != _STALE_LABEL_CONFORMING
    _write_stale_label_tree(tmp_path, body)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a non-conforming tree:\n{err}"
    assert "#963" in err, (
        f"the #963 stale-label-disposition diagnostic is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


def test_stale_label_disposition_clause_dedicated_flag_isolated(tmp_path, capsys, monkeypatch):
    """The dedicated ``--check-stale-label-disposition`` flag runs ONLY the
    stale-label check (``no_flags`` computes False): on a minimal tree where
    the stale-label paragraph CONFORMS but the full default bundle FAILs
    (other bundled checks miss their files), the dedicated-flag invocation
    exits 0. Mutation-visibility — the leg the ``main([])`` wiring test above
    cannot pin: deleting ``or args.check_stale_label_disposition`` from the
    ``no_flags`` tuple makes the dedicated-flag invocation compute
    ``no_flags`` True and run the FULL bundle on the failing minimal tree ->
    rc != 0 -> this test FAILs. (Verified empirically on 2026-07-04 by
    stripping that tuple line: this test fails, the wiring test stays green.)
    """
    import workflow_lint as wl

    _write_stale_label_tree(tmp_path, _STALE_LABEL_CONFORMING)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    # Precondition: the FULL default bundle FAILs on this minimal tree, so a
    # no_flags mis-computation below is observable as rc != 0.
    assert wl.main([]) != 0, "precondition: the default bundle PASSed on the minimal tree"
    capsys.readouterr()  # discard the precondition run's output
    rc = wl.main(["--check-stale-label-disposition"])
    err = capsys.readouterr().err
    assert rc == 0, (
        f"--check-stale-label-disposition ran more than the (conforming) stale-label "
        f"check — no_flags mis-computed True, i.e. the flag's membership in the "
        f"no_flags tuple in workflow_lint.main() is missing:\n{err}"
    )


# --- #1181 crash-fix-relaunch contract pin tests ------------------------------

_CRASH_FIX_SURFACES = (
    ".claude/agents/experimenter.md",
    ".claude/rules/crash-fix-rounds.md",
    ".claude/skills/issue/SKILL.md",
)

_CRASH_FIX_ANCHORS = {
    ".claude/agents/experimenter.md": "**Crash-fix relaunch (brief carries `fix_sha=`):**",
    ".claude/rules/crash-fix-rounds.md": "The fresh `epm:run-launched` note ALSO records",
    ".claude/skills/issue/SKILL.md": "*`code`-row relaunch contract (#779):*",
}

# Design intent baked into the fixture (#1181 plan §4.5): (i) tokens hard-wrap
# mid-phrase (normalization proof); (ii) the experimenter.md span ends at
# ``\n3. `` with NO blank line (proves the ``\n\d+\. `` terminator) and a
# 'resolves EMPTY' DECOY sits after that span (proves paragraph scoping of the
# negative regex); (iii) the crash-fix-rounds fixture carries the healthy
# 'resolves EMPTY / to the fresh path' trio BEFORE its anchor (scoping again,
# plus a tripwire if a future edit widens the regex to file scope).
_CRASH_FIX_CONFORMING: dict[str, str] = {
    ".claude/agents/experimenter.md": (
        "# experimenter\n"
        "\n"
        "2. **Verify HEAD.** Standard pre-launch sync.\n"
        "\n"
        "   **Crash-fix relaunch (brief carries `fix_sha=`):** additionally run\n"
        "   `git merge-base --is-ancestor <fix_sha> HEAD` on the pod (ANY\n"
        "   non-zero exit = fix absent — do NOT launch) and execute the\n"
        "   brief's stale-checkpoint disposition before launch, confirming\n"
        "   the resume glob resolves as the disposition requires (empty /\n"
        "   the fresh path / exactly the RETAINED expected paths).\n"
        "   On a MooseFS-backed pod (`/workspace` lane), ALSO run the MooseFS\n"
        "   content read of every fix-touched path — `git hash-object -- <f>`\n"
        "   vs `git rev-parse HEAD:<f>` (#1112).\n"
        "3. **Run preflight.**\n"
        "\n"
        "Decoy AFTER the span: the glob resolves EMPTY unconditionally.\n"
    ),
    ".claude/rules/crash-fix-rounds.md": (
        "# crash-fix rounds\n"
        "\n"
        "2. **Stale-checkpoint disposition (element 5).** check it\n"
        "   resolves EMPTY / to the fresh path / to exactly the RETAINED\n"
        "   expected paths (for a `retain` declaration).\n"
        "\n"
        "The fresh `epm:run-launched` note ALSO records `fix_sha=<sha>` and the\n"
        "executed disposition (note-token convention, same class as `pid=`).\n"
        "The `code`-row respawn BRIEF the orchestrator composes for the\n"
        "experimenter carries both (`fix_sha=` +\n"
        "the element-5 disposition verbatim). EXEMPT: `infra`-row experimenter\n"
        "respawns (no code fix).\n"
    ),
    ".claude/skills/issue/SKILL.md": (
        "# issue skill\n"
        "\n"
        "Step 7 routing table lives here.\n"
        "\n"
        "   *`code`-row relaunch contract (#779):* the post-review relaunch — the\n"
        "   Step 6 experimenter respawn (brief carries `fix_sha=` + the element-5\n"
        "   stale-artifact disposition, copied from the implementer's fix-engaged\n"
        "   declaration) — enforces BOTH\n"
        "   before dispatch: the fix-commit ancestry probe and the declared\n"
        "   disposition.\n"
        "\n"
        "**Zombie-GPU stall recovery brief.** Unrelated paragraph.\n"
    ),
}

# The four load-bearing pins (#1181 plan § deviations; MooseFS added by #1594):
# the disposition trio, the disposition-conditional confirm, the fix_sha=
# note-token duty, and the MooseFS content-read duty (#1112).
_CRASH_FIX_TRIO_TOKEN = "empty / the fresh path / exactly the RETAINED expected paths"
_CRASH_FIX_CONFIRM_TOKEN = "confirming the resume glob resolves as the disposition requires"
_CRASH_FIX_SHA_TOKEN = "records `fix_sha=<sha>` and the executed disposition"
_CRASH_FIX_MOOSEFS_TOKEN = "MooseFS content read"


def _write_crash_fix_tree(tmp_path, bodies: dict[str, str] | None = None) -> None:
    """Write the three #1181 contract surfaces under ``tmp_path``."""
    for rel, body in (bodies or _CRASH_FIX_CONFORMING).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def test_crash_fix_relaunch_contract_live_tree_passes() -> None:
    """The real tree carries the #1081 contract on all three surfaces, with
    unique anchors, every required token, and no unconditional 'resolves
    EMPTY' coupling (pins plan #1181 Assumptions 1-5)."""
    assert check_crash_fix_relaunch_contract() == []


def test_crash_fix_relaunch_contract_conforming_tmp_tree_passes(tmp_path) -> None:
    """The synthetic conforming tree passes — validates the fixture itself:
    hard-wrapped tokens (whitespace normalization), the experimenter.md span
    ending at ``\\n3. `` with no blank line (the ``\\n\\d+\\. `` terminator),
    the post-span 'resolves EMPTY' decoy, and the pre-anchor healthy trio
    (paragraph scoping in both directions)."""
    _write_crash_fix_tree(tmp_path)
    assert check_crash_fix_relaunch_contract(repo_root=tmp_path) == []


def test_crash_fix_relaunch_contract_in_span_healthy_trio_passes(tmp_path) -> None:
    """A healthy disposition-conditional trio ('resolves EMPTY / to the fresh
    path / ...') INSIDE an anchored span PASSes — exercises the negative
    regex's lookahead exemption ``(?!\\s*/)``, which the conforming fixture
    alone leaves untested (both of its healthy trios sit outside the spans)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    rel = ".claude/rules/crash-fix-rounds.md"
    bodies[rel] = bodies[rel].replace(
        "respawns (no code fix).\n",
        "respawns (no code fix). The relaunch then checks the glob\n"
        "resolves EMPTY / to the fresh path / to exactly the RETAINED\n"
        "expected paths before posting.\n",
    )
    assert bodies[rel] != _CRASH_FIX_CONFORMING[rel]
    _write_crash_fix_tree(tmp_path, bodies)
    assert check_crash_fix_relaunch_contract(repo_root=tmp_path) == []


@pytest.mark.parametrize("surface", _CRASH_FIX_SURFACES)
def test_crash_fix_relaunch_contract_fails_on_missing_anchor(tmp_path, surface) -> None:
    """Deleting/renaming any ONE surface's anchor FAILs, naming that file and
    #1081 (an anchor rename requires a deliberate lint update)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    bodies[surface] = bodies[surface].replace(_CRASH_FIX_ANCHORS[surface], "**Renamed.**")
    assert bodies[surface] != _CRASH_FIX_CONFORMING[surface]
    _write_crash_fix_tree(tmp_path, bodies)
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert len(errors) == 1, errors
    fname = surface.rsplit("/", 1)[-1]
    assert errors[0].split(": ", 1)[0].endswith(f"/{fname}"), errors
    assert "missing the anchor" in errors[0] and "#1081" in errors[0], errors


def test_crash_fix_relaunch_contract_fails_on_duplicate_anchor(tmp_path) -> None:
    """A SECOND copy of the experimenter.md anchor -> exactly one
    duplicate-anchor error (span identity is load-bearing: a stale duplicate
    could satisfy the token scan while the operative paragraph regresses)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    rel = ".claude/agents/experimenter.md"
    bodies[rel] += "\n**Crash-fix relaunch (brief carries `fix_sha=`):** stale copy.\n"
    _write_crash_fix_tree(tmp_path, bodies)
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "UNIQUE" in errors[0] and "2 anchors" in errors[0], errors


@pytest.mark.parametrize(
    ("surface", "old", "new", "token"),
    [
        pytest.param(
            ".claude/agents/experimenter.md",
            "(empty /\n   the fresh path / exactly the RETAINED expected paths)",
            "(as declared)",
            _CRASH_FIX_TRIO_TOKEN,
            id="disposition-trio",
        ),
        pytest.param(
            ".claude/agents/experimenter.md",
            "confirming\n   the resume glob resolves as the disposition requires",
            "checking\n   the resume glob looks right",
            _CRASH_FIX_CONFIRM_TOKEN,
            id="conditional-confirm",
        ),
        pytest.param(
            ".claude/rules/crash-fix-rounds.md",
            "records `fix_sha=<sha>` and the\nexecuted disposition",
            "records the executed\ndisposition",
            _CRASH_FIX_SHA_TOKEN,
            id="fix-sha-note-token",
        ),
        pytest.param(
            ".claude/agents/experimenter.md",
            "ALSO run the MooseFS\n   content read of every fix-touched path",
            "ALSO run a served-bytes\n   check of every fix-touched path",
            _CRASH_FIX_MOOSEFS_TOKEN,
            id="moosefs-content-read",
        ),
    ],
)
def test_crash_fix_relaunch_contract_fails_on_missing_load_bearing_token(
    tmp_path, surface, old, new, token
) -> None:
    """Deleting any of the FOUR load-bearing pins from its fixture FAILs with
    a missing-token error naming that token — mutation-visibility for the
    surfaces table: an edit that drops one of these tokens from
    ``_CRASH_FIX_CONTRACT_SURFACES`` makes the corresponding case pass on the
    mutated fixture, so this test FAILs (presence asserted, not exact count)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    assert old in bodies[surface], f"fixture drift: {old!r} not found in {surface}"
    bodies[surface] = bodies[surface].replace(old, new)
    _write_crash_fix_tree(tmp_path, bodies)
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert errors, f"expected a missing-token FAIL for {token!r}"
    assert any(repr(token) in e and "missing token" in e for e in errors), errors


def test_crash_fix_relaunch_contract_fails_on_unconditional_empty_regression(tmp_path) -> None:
    """Regressing the D3 confirm back to the unconditional 'resolves EMPTY'
    wording (the #1081 round-2 blocker retain-disposition-d3-empty-glob)
    FAILs with BOTH error classes present — the missing-token error for the
    trio AND the negative-regex error. Presence of both classes is asserted,
    not an exact error count (the mutation also drops the confirm token)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    rel = ".claude/agents/experimenter.md"
    bodies[rel] = bodies[rel].replace(
        "the resume glob resolves as the disposition requires (empty /\n"
        "   the fresh path / exactly the RETAINED expected paths)",
        "the resume glob resolves EMPTY before launch",
    )
    assert bodies[rel] != _CRASH_FIX_CONFORMING[rel]
    _write_crash_fix_tree(tmp_path, bodies)
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert any(repr(_CRASH_FIX_TRIO_TOKEN) in e and "missing token" in e for e in errors), errors
    assert any("'resolves EMPTY'" in e for e in errors), errors


def test_crash_fix_relaunch_contract_fails_on_missing_enforces_both_token(tmp_path) -> None:
    """Dropping the 'enforces BOTH before dispatch: ...' sentence from the
    SKILL.md fixture FAILs with a missing-token error (the orchestrator-side
    enforcement duty)."""
    bodies = dict(_CRASH_FIX_CONFORMING)
    rel = ".claude/skills/issue/SKILL.md"
    bodies[rel] = bodies[rel].replace(
        " — enforces BOTH\n"
        "   before dispatch: the fix-commit ancestry probe and the declared\n"
        "   disposition.",
        ".",
    )
    assert bodies[rel] != _CRASH_FIX_CONFORMING[rel]
    _write_crash_fix_tree(tmp_path, bodies)
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "missing token" in errors[0] and "enforces BOTH before dispatch" in errors[0], errors


@pytest.mark.parametrize("surface", _CRASH_FIX_SURFACES)
def test_crash_fix_relaunch_contract_fails_on_missing_file(tmp_path, surface) -> None:
    """A surface file absent from the tree FAILs with the missing-file branch
    naming that path."""
    _write_crash_fix_tree(tmp_path)
    (tmp_path / surface).unlink()
    errors = check_crash_fix_relaunch_contract(repo_root=tmp_path)
    assert len(errors) == 1, errors
    fname = surface.rsplit("/", 1)[-1]
    assert errors[0].split(": ", 1)[0].endswith(f"/{fname}"), errors
    assert "missing" in errors[0], errors


def test_crash_fix_relaunch_contract_wired_into_default_run(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags CLI-path REGISTRATION test: the default run must exercise
    ``check_crash_fix_relaunch_contract`` — deleting the dispatch branch
    (``if args.check_crash_fix_relaunch_contract or no_flags:``) or its
    ``or no_flags`` disjunct must fail this test. Doctors a tree missing the
    experimenter.md anchor, points ``_REPO_ROOT`` at it, and invokes
    ``main([])`` in-process: rc != 0 with the #1081 diagnostic in stderr.
    Other bundled checks contribute unrelated errors on the minimal tree — the
    assertion keys on the #1081 error string. (The ``no_flags`` tuple
    membership is pinned by ``..._dedicated_flag_isolated`` below, per the
    stale-label precedent.)"""
    import workflow_lint as wl

    bodies = dict(_CRASH_FIX_CONFORMING)
    rel = ".claude/agents/experimenter.md"
    bodies[rel] = bodies[rel].replace(_CRASH_FIX_ANCHORS[rel], "**Renamed.**")
    _write_crash_fix_tree(tmp_path, bodies)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an anchor-less tree:\n{err}"
    assert "#1081" in err, (
        f"the #1081 crash-fix-relaunch diagnostic is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


def test_crash_fix_relaunch_contract_dedicated_flag_isolated(tmp_path, capsys, monkeypatch) -> None:
    """The dedicated ``--check-crash-fix-relaunch-contract`` flag runs ONLY
    this check (``no_flags`` computes False): on a minimal tree where the
    three contract surfaces CONFORM but the full default bundle FAILs (other
    bundled checks miss their files), the dedicated-flag invocation exits 0 —
    mutation-visibility for the ``or args.check_crash_fix_relaunch_contract``
    membership in the ``no_flags`` tuple (the leg the ``main([])`` wiring
    test above cannot pin, per the stale-label precedent)."""
    import workflow_lint as wl

    _write_crash_fix_tree(tmp_path)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    # Precondition: the FULL default bundle FAILs on this minimal tree, so a
    # no_flags mis-computation below is observable as rc != 0.
    assert wl.main([]) != 0, "precondition: the default bundle PASSed on the minimal tree"
    capsys.readouterr()  # discard the precondition run's output
    rc = wl.main(["--check-crash-fix-relaunch-contract"])
    err = capsys.readouterr().err
    assert rc == 0, (
        f"--check-crash-fix-relaunch-contract ran more than the (conforming) "
        f"contract check — no_flags mis-computed True, i.e. the flag's membership "
        f"in the no_flags tuple in workflow_lint.main() is missing:\n{err}"
    )


# --- #1153 awk elision-program parity tests ----------------------------------

# A program with the ``f=!f`` anchor and no single quotes (matches the live
# program's shape at the time of writing; the check compares homes against
# EACH OTHER, so tests only need SOME shared program).
_AWK_ELISION_TEST_PROGRAM = (
    "/^```/{f=!f; next} f{next} /^<details/{d=1} d{if(/<\\/details>/)d=0; next} "
    "/^>/{next} {print} END{if(f||d) exit 3}"
)

_AWK_HOME_SKILL = ".claude/skills/issue/SKILL.md"
_AWK_HOME_ANALYZER = ".claude/rules/analyzer-section-reference.md"


def _write_awk_home(root: Path, rel: str, body: str) -> None:
    """Write ``rel`` under ``root`` with ``body`` (parents created)."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _awk_skill_body(program: str = _AWK_ELISION_TEST_PROGRAM) -> str:
    """SKILL.md-shaped home: the program inside a FENCED bash block, 3-space
    indent, with its own input/output continuation line."""
    return (
        "# issue skill\n\nStep 9a-humanize ban gate:\n\n"
        "```bash\n"
        f"   awk '{program}' \\\n"
        "     body.md > elided.md\n"
        "```\n"
    )


def _awk_analyzer_body(
    program: str = _AWK_ELISION_TEST_PROGRAM, anchor_line: str | None = None
) -> str:
    """analyzer-section-reference.md-shaped home: the program in a 4-space
    INDENTED block with DIFFERENT surroundings/paths than the skill home.
    ``anchor_line`` replaces the whole program line (malformed-line cases)."""
    line = anchor_line if anchor_line is not None else f"    awk '{program}' \\"
    return f"# analyzer section reference\n\nStep 4.5:\n\n{line}\n        draft.md > out.md\n"


def test_awk_elision_parity_live_tree_passes() -> None:
    """The real tree carries ONE anchor line per home, 2 quotes each, with
    byte-identical extracted programs (#1153)."""
    assert check_awk_elision_parity() == []


def test_awk_elision_parity_conforming_tmp_tree_passes(tmp_path) -> None:
    """Same program, DIFFERENT surroundings — one fenced ```bash block, one
    4-space-indented block, different in/out paths on the continuation
    lines — PASSes: pins that the check tolerates the homes' real
    formatting differences and compares the quoted PROGRAM only."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body())
    assert check_awk_elision_parity(repo_root=tmp_path) == []


def test_awk_elision_parity_flags_drift(tmp_path) -> None:
    """One home's program mutated (END clause dropped) -> exactly one error
    naming BOTH paths + the edit-both-homes remediation."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    drifted = _AWK_ELISION_TEST_PROGRAM.replace(" END{if(f||d) exit 3}", "")
    assert drifted != _AWK_ELISION_TEST_PROGRAM
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body(program=drifted))
    errors = check_awk_elision_parity(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert _AWK_HOME_SKILL in errors[0] and _AWK_HOME_ANALYZER in errors[0], errors
    assert "identically" in errors[0], errors


def test_awk_elision_parity_flags_missing_file(tmp_path) -> None:
    """A missing home is itself an error — a moved/deleted copy must not
    silently pass (#1153)."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    errors = check_awk_elision_parity(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("analyzer-section-reference.md"), errors
    assert "missing" in errors[0], errors


def test_awk_elision_parity_flags_zero_anchor(tmp_path) -> None:
    """A home present but with NO anchor line (program removed) FAILs."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, "# analyzer section reference\n\nno program\n")
    errors = check_awk_elision_parity(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("analyzer-section-reference.md"), errors
    assert "found 0" in errors[0], errors


def test_awk_elision_parity_flags_duplicate_anchor(tmp_path) -> None:
    """TWO anchor lines in one home FAIL — span identity is load-bearing
    (which copy would the parity read?)."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body() + _awk_analyzer_body())
    errors = check_awk_elision_parity(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "found 2" in errors[0], errors


@pytest.mark.parametrize(
    ("anchor_line", "expected_token"),
    [
        pytest.param(
            "    awk '/^```/{f=!f; next} f{next} \\",
            "expected exactly 2",
            id="reflow-mid-program-one-quote",
        ),
        pytest.param(
            "    awk '/{f=!f}/'\\''x' \\",
            "expected exactly 2",
            id="gained-quote-escape",
        ),
        pytest.param(
            "    awk '/{f=!f}/' | awk '{print}' \\",
            "expected exactly 2",
            id="second-quoted-span",
        ),
        pytest.param(
            "    the 'f=!f' toggle (no awk program) \\",
            "could not extract",
            id="two-quotes-no-awk-span",
        ),
    ],
)
def test_awk_elision_parity_flags_malformed_anchor_line(
    tmp_path, anchor_line: str, expected_token: str
) -> None:
    """A malformed anchor line FAILs loudly per home: a mid-program reflow
    (1 quote), a gained shell quote-escape (5 quotes — the truncation
    false-PASS window the exactly-2-quotes assert closes), a second quoted
    span on the line (4 quotes), and a 2-quote line with no ``awk '...'``
    span (extraction finds 0)."""
    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body(anchor_line=anchor_line))
    errors = check_awk_elision_parity(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].split(": ", 1)[0].endswith("analyzer-section-reference.md"), errors
    assert expected_token in errors[0], errors


def test_awk_elision_parity_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #1153 check —
    deleting its dispatch branch (``if args.check_awk_elision_parity or
    no_flags:``) or its ``or no_flags`` disjunct must fail this test
    (mutation-visible). House pattern:
    ``test_vm_thread_cap_guidance_bundled_in_no_flags`` — drifted tree,
    ``_REPO_ROOT`` monkeypatched to the fixture, ``main([])`` in-process;
    other bundled checks contribute unrelated errors on the minimal tree, so
    the assertion keys on the #1153 drift diagnostic."""
    import workflow_lint as wl

    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    drifted = _AWK_ELISION_TEST_PROGRAM.replace(" END{if(f||d) exit 3}", "")
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body(program=drifted))
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a drifted tree:\n{err}"
    assert "#1153" in err and "identically" in err, (
        f"the #1153 awk-elision-parity drift diagnostic is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


def test_awk_elision_parity_dedicated_flag_isolated(tmp_path, capsys, monkeypatch) -> None:
    """The dedicated ``--check-awk-elision-parity`` flag runs ONLY this check
    (``no_flags`` computes False): on a minimal tree where the two awk homes
    CONFORM but the full default bundle FAILs (other bundled checks miss
    their files), the dedicated-flag invocation exits 0 — pins the
    ``or args.check_awk_elision_parity`` membership in the ``no_flags``
    tuple, the leg the ``main([])`` wiring test above cannot pin (house
    pattern: ``test_stale_label_disposition_clause_dedicated_flag_isolated``)."""
    import workflow_lint as wl

    _write_awk_home(tmp_path, _AWK_HOME_SKILL, _awk_skill_body())
    _write_awk_home(tmp_path, _AWK_HOME_ANALYZER, _awk_analyzer_body())
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    # Precondition: the FULL default bundle FAILs on this minimal tree, so a
    # no_flags mis-computation below is observable as rc != 0.
    assert wl.main([]) != 0, "precondition: the default bundle PASSed on the minimal tree"
    capsys.readouterr()  # discard the precondition run's output
    rc = wl.main(["--check-awk-elision-parity"])
    err = capsys.readouterr().err
    assert rc == 0, (
        f"--check-awk-elision-parity ran more than the (conforming) awk-elision "
        f"check — no_flags mis-computed True, i.e. the flag's membership in the "
        f"no_flags tuple in workflow_lint.main() is missing:\n{err}"
    )


# ---------------------------------------------------------------------------
# --check-section-reference-pointers (#1159): every grain-level section of an
# <agent>-{section,lens}-reference.md rule file must stay pointer-reachable
# ('§ <exact heading>') from its owning agent spec.
# ---------------------------------------------------------------------------


def _write_section_ref_tree(tmp_path, ref_name, ref_body, agent_name=None, agent_body=""):
    """Write a minimal tmp tree: one reference rule file (+ optional agent spec)."""
    rules = tmp_path / ".claude" / "rules"
    agents = tmp_path / ".claude" / "agents"
    rules.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)
    (rules / ref_name).write_text(ref_body, encoding="utf-8")
    if agent_name is not None:
        (agents / f"{agent_name}.md").write_text(agent_body, encoding="utf-8")
    return tmp_path


def test_section_ref_pointer_coverage_passes_on_live_repo():
    """The live-trees-pass invariant at merge time — zero pointer backfills owed.

    ALSO asserts the scanned reference-file set contains the 4 known files, so
    a suffix-tuple typo cannot pass vacuously via an empty scan set."""
    from workflow_lint import _REPO_ROOT as lint_repo_root
    from workflow_lint import _SECTION_REFERENCE_SUFFIXES

    assert check_section_reference_pointer_coverage() == []
    scanned = {
        p.name
        for p in (lint_repo_root / ".claude" / "rules").glob("*.md")
        if p.name.endswith(_SECTION_REFERENCE_SUFFIXES)
    }
    expected = {
        "analyzer-section-reference.md",
        "critic-lens-reference.md",
        "planner-section-reference.md",
        "clean-result-critic-lens-reference.md",
    }
    assert expected <= scanned, f"scan set lost known reference files: {expected - scanned}"


def test_section_ref_pointer_coverage_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting the
    ``or no_flags`` ladder branch must fail this test (mutation-visible; house
    pattern: ``test_hollow_gate_review_lens_bundled_in_no_flags``). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on this check's diagnostic + the offending file name."""
    import workflow_lint as wl

    _write_section_ref_tree(
        tmp_path,
        "foo-lens-reference.md",
        "# Title\n\n### Lens 1 — Alpha\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\nno pointer here\n",
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on a violating tree:\n{err}"
    assert "foo-lens-reference.md" in err and "pointer-reachable" in err, (
        f"the pointer-coverage diagnostic (naming foo-lens-reference.md) is missing "
        f"from the no-flags run's stderr — the check is not bundled into no_flags:\n{err}"
    )


def test_section_ref_pointer_coverage_fails_on_missing_pointer(tmp_path):
    """An unpointed grain heading FAILs, naming file, heading, and owning spec."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title\n\n## Step 9\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\nprose without any pointer\n",
    )
    errs = check_section_reference_pointer_coverage(repo_root=tmp_path)
    assert len(errs) == 1, errs
    assert "foo-section-reference.md" in errs[0]
    assert "Step 9" in errs[0]
    assert ".claude/agents/foo.md" in errs[0]


def test_section_ref_pointer_coverage_passes_on_wrapped_pointer(tmp_path):
    """A pointer line-wrapped mid-heading in the spec passes (whitespace norm)."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title\n\n## Step 9 — the long heading name\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\nFull text: § Step 9 — the long\nheading name (grep heading).\n",
    )
    assert check_section_reference_pointer_coverage(repo_root=tmp_path) == []


def test_section_ref_pointer_coverage_skips_fenced_pseudo_headings(tmp_path):
    """A '## fake' inside a ```bash fence needs no pointer (fence-aware)."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title\n\n## Real\n\n```bash\n## fake\necho fenced\n```\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\n§ Real\n",
    )
    assert check_section_reference_pointer_coverage(repo_root=tmp_path) == []


def test_section_ref_pointer_coverage_h3_grain_when_no_h2(tmp_path):
    """With zero H2s the grain is H3 (critic-lens-reference shape): H3s are
    checked, and a missing H3 pointer FAILs."""
    _write_section_ref_tree(
        tmp_path,
        "foo-lens-reference.md",
        "# Title\n\n### Lens 1 — Alpha\n\nbody\n\n### Lens 2 — Beta\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\n§ Lens 1 — Alpha\n",
    )
    errs = check_section_reference_pointer_coverage(repo_root=tmp_path)
    assert len(errs) == 1, errs
    assert "Lens 2 — Beta" in errs[0]


def test_section_ref_pointer_coverage_h2_grain_wins_when_mixed(tmp_path):
    """A file with H2s AND H3s is H2-grain: only H2s require pointers (the
    documented grain-mixing drift path — H3s drop from coverage)."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title\n\n## Big section\n\n### sub-detail without pointer\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\n§ Big section\n",
    )
    assert check_section_reference_pointer_coverage(repo_root=tmp_path) == []


def test_section_ref_pointer_coverage_fails_on_orphan_reference(tmp_path):
    """A suffix-matched rule file with no .claude/agents/<agent>.md FAILs."""
    _write_section_ref_tree(
        tmp_path,
        "foo-lens-reference.md",
        "# Title\n\n### Lens 1 — Alpha\n\nbody\n",
        agent_name=None,
    )
    # agents/ dir must exist for the orphan case to be about the FILE, not the dir
    (tmp_path / ".claude" / "agents").mkdir(parents=True, exist_ok=True)
    errs = check_section_reference_pointer_coverage(repo_root=tmp_path)
    assert len(errs) == 1, errs
    assert "orphan" in errs[0] and ".claude/agents/foo.md" in errs[0]


def test_section_ref_pointer_coverage_fails_on_headingless_reference(tmp_path):
    """A suffix-matched file with an H1 only is malformed (zero grain headings)."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title only\n\nprose, no sections\n",
        agent_name="foo",
        agent_body="# foo\n",
    )
    errs = check_section_reference_pointer_coverage(repo_root=tmp_path)
    assert len(errs) == 1, errs
    assert "malformed" in errs[0]


def test_section_ref_pointer_coverage_requires_section_sigil(tmp_path):
    """Heading text present in the spec WITHOUT the '§ ' prefix still FAILs —
    a prose mention of a section name is not pointer coverage."""
    _write_section_ref_tree(
        tmp_path,
        "foo-section-reference.md",
        "# Title\n\n## Step 9\n\nbody\n",
        agent_name="foo",
        agent_body="# foo\nsee Step 9 in the reference file\n",
    )
    errs = check_section_reference_pointer_coverage(repo_root=tmp_path)
    assert len(errs) == 1, errs
    assert "Step 9" in errs[0]


def test_section_ref_pointer_coverage_ignores_non_suffixed_rules(tmp_path):
    """A rules file NOT ending in a reference suffix is out of the scan set."""
    _write_section_ref_tree(
        tmp_path,
        "foo-review.md",
        "# Title\n\n## Unpointed section\n\nbody\n",
        agent_name=None,
    )
    assert check_section_reference_pointer_coverage(repo_root=tmp_path) == []


# ---------------------------------------------------------------------------
# --check-git-recipes-root-guard (#1176): execute the LIVE repo-root branch
# guard (scripts/guard_repo_root_branch.sh) against every bash-fenced git
# recipe in the workflow docs. All gated-git literals below are Python STRING
# DATA (test fixtures written via tmp_path) — the hook gates Bash TOOL calls,
# not file contents.
# ---------------------------------------------------------------------------

_RG_BLOCKED_CMD = "git checkout -b __wl_rg_test_branch__"


def _write_rg_skill(tmp_path: Path, slug: str, body: str) -> Path:
    """Write a synthetic ``.claude/skills/<slug>/SKILL.md`` under ``tmp_path``."""
    p = tmp_path / ".claude" / "skills" / slug / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_check_git_recipes_root_guard_flags_blocked_recipe(tmp_path):
    """A bash fence whose whole-block feed the live hook BLOCKS (exit 2) is
    exactly one error naming the file + the fence OPENER line."""
    _write_rg_skill(tmp_path, "x", f"Intro line\n```bash\n{_RG_BLOCKED_CMD}\n```\n")
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:2:" in errors[0], errors
    assert "BLOCKED" in errors[0], errors
    # remediation names both paths: fix the recipe, or the exemption sentinel
    assert "allow-root-guard-block" in errors[0], errors


def test_check_git_recipes_root_guard_passes_waived_recipe(tmp_path):
    """A per-clause worktree-qualified destructive form passes the hook."""
    _write_rg_skill(
        tmp_path,
        "x",
        'Recover inside the worktree:\n```bash\ngit -C "$WT" reset --hard origin/main\n```\n',
    )
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []


def test_check_git_recipes_root_guard_multiline_construct_passes(tmp_path):
    """Whole-block feed: a for-loop + heredoc + comment construct with only
    ``git -C`` forms passes — per-line feeding would shred the loop and
    false-positive on the inert heredoc body / comment line."""
    body = (
        "Recovery recipe:\n"
        "```bash\n"
        "for f in a b; do\n"
        '  git -C "$WT" reset --hard origin/main\n'
        "done\n"
        "# a comment line mentioning git switch is inert as pasted\n"
        "cat <<'EOF' > /tmp/wl_rg_note.txt\n"
        "plain note text\n"
        "EOF\n"
        'git -C "$WT" status\n'
        "```\n"
    )
    _write_rg_skill(tmp_path, "x", body)
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []


def test_check_git_recipes_root_guard_skips_exempt_fence(tmp_path):
    """A sentinel with a NON-EMPTY reason on the immediately-preceding
    non-blank line waives the fence; an EMPTY-reason sentinel does NOT."""
    _write_rg_skill(
        tmp_path,
        "exempt",
        "<!-- workflow-lint: allow-root-guard-block: deliberate anti-pattern example -->\n"
        f"```bash\n{_RG_BLOCKED_CMD}\n```\n",
    )
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []

    _write_rg_skill(
        tmp_path,
        "empty_reason",
        f"<!-- workflow-lint: allow-root-guard-block: -->\n```bash\n{_RG_BLOCKED_CMD}\n```\n",
    )
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "empty_reason" in errors[0], errors


def test_check_git_recipes_root_guard_ignores_non_bash_fences(tmp_path):
    """Only bash/sh/shell-tagged fences are executable recipes; a
    python-tagged fence carrying a blocked literal is never fed to the hook."""
    _write_rg_skill(
        tmp_path,
        "x",
        f'```python\ncmd = "{_RG_BLOCKED_CMD}"\n```\n',
    )
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []


def test_check_git_recipes_root_guard_selftest_fails_loud(tmp_path):
    """A missing hook and a fail-OPEN hook (exit 0 on the blocked probe —
    the jq-missing fail-soft shape) each produce ONE loud error, never a
    silent pass."""
    _write_rg_skill(tmp_path, "x", f"```bash\n{_RG_BLOCKED_CMD}\n```\n")
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=tmp_path / "missing.sh")
    assert len(errors) == 1, errors
    assert "missing" in errors[0], errors

    fail_open = tmp_path / "fail_open.sh"
    fail_open.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=fail_open)
    assert len(errors) == 1, errors
    assert "self-test" in errors[0], errors


def test_check_git_recipes_root_guard_selftest_fail_closed(tmp_path):
    """The complement cell: a fail-CLOSED stub (exit 2 on EVERYTHING,
    including the benign probe) is also ONE loud self-test error."""
    _write_rg_skill(tmp_path, "x", "```bash\necho benign\n```\n")
    fail_closed = tmp_path / "fail_closed.sh"
    fail_closed.write_text("#!/usr/bin/env bash\nexit 2\n", encoding="utf-8")
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=fail_closed)
    assert len(errors) == 1, errors
    assert "self-test" in errors[0], errors


def test_check_git_recipes_root_guard_nested_fence_recovers(tmp_path):
    """Replicates the nested-fence shape formerly at weekly/SKILL.md:196-204
    (skill retired 2026-08-05; this fixture preserves the shape)
    (outer ```markdown fence containing an inner ```diff fence) FOLLOWED by
    a blocked bash fence: the parity-toggle parser must recover and flag the
    blocked fence at its CORRECT opener line — the naive empty-tag-closer
    rule desyncs here and silently hides the bash fence (a false negative
    the live-tree test cannot see)."""
    body = (
        "```markdown\n"  # 1  open (markdown)
        "1. **Target:** x\n"  # 2
        "   ```diff\n"  # 3  same-token fence line -> CLOSES the outer fence
        "   - old\n"  # 4  prose (outside any fence)
        "   + new\n"  # 5
        "   ```\n"  # 6  opens an untagged fence
        "```\n"  # 7  closes it
        "\n"  # 8
        "```bash\n"  # 9  the blocked fence the parser must still see
        f"{_RG_BLOCKED_CMD}\n"  # 10
        "```\n"  # 11
    )
    _write_rg_skill(tmp_path, "x", body)
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:9:" in errors[0], errors


def test_check_git_recipes_root_guard_unterminated_fence_scanned(tmp_path):
    """A blocked bash fence with NO closer at EOF is still scanned (fail
    toward checking — previously only a docstring claim)."""
    _write_rg_skill(tmp_path, "x", f"```bash\n{_RG_BLOCKED_CMD}\n")
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:1:" in errors[0], errors


def test_check_git_recipes_root_guard_known_miss_residuals(tmp_path):
    """The two archived #1047 shapes are DISCLOSED residuals, not covered:
    (a) an inline code span in a prose bullet carrying a blocked command;
    (b) a ``#``-commented blocked command inside a bash fence (the hook's
    comment-tail strip correctly allows the block-as-pasted). Both -> 0
    hits, and the check's docstring must name both residual shapes so the
    disclosure is durable."""
    body = (
        f"- Then run `{_RG_BLOCKED_CMD}` at the repo root.\n"
        "\n"
        "```bash\n"
        f"# {_RG_BLOCKED_CMD}   <- run this uncommented\n"
        "echo done\n"
        "```\n"
    )
    _write_rg_skill(tmp_path, "x", body)
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []
    doc = check_git_recipes_root_guard.__doc__ or ""
    assert "inline-code recipes" in doc, "docstring must name the prose inline-code residual"
    assert "commented instruction lines" in doc, (
        "docstring must name the commented-instruction-line residual"
    )
    # the other two named residuals ride along
    assert "untagged" in doc, "docstring must name the untagged-fence residual"
    assert "placeholder-substitution" in doc, (
        "docstring must name the placeholder-substitution false-PASS direction"
    )


def test_check_git_recipes_root_guard_tilde_fence(tmp_path):
    """``~~~bash`` fences are scanned too; a sentinel line with TRAILING
    WHITESPACE after the comment closer still waives."""
    _write_rg_skill(tmp_path, "tilde", f"~~~bash\n{_RG_BLOCKED_CMD}\n~~~\n")
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:1:" in errors[0], errors

    _write_rg_skill(
        tmp_path,
        "tilde",
        "<!-- workflow-lint: allow-root-guard-block: deliberate example -->   \n"
        f"~~~bash\n{_RG_BLOCKED_CMD}\n~~~\n",
    )
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []


def test_check_git_recipes_root_guard_live_tree_passes():
    """The real post-disposition tree PASSES (the ``test_live_trees_pass``
    invariant): locks in the #1176 dispositions — the refactor/SKILL.md
    recipe fix and the gotchas.md pod-side exemption sentinel — and fails
    loud if a future doc edit adds a recipe the live hook blocks."""
    assert check_git_recipes_root_guard() == []


def test_workflow_lint_check_git_recipes_root_guard_cli_exits_zero():
    """CLI flag smoke: ``--check-git-recipes-root-guard`` exits 0 on the
    committed tree. Bundle membership (no_flags) is auto-covered by
    ``test_workflow_lint_default_exits_zero``."""
    result = _run("--check-git-recipes-root-guard")
    assert result.returncode == 0, (
        f"workflow_lint --check-git-recipes-root-guard failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _make_detached_repo(tmp_path: Path) -> Path:
    """git init an on-main repo, one empty commit, then detach HEAD —
    the git state a concurrent shared-root ``pull --rebase`` exposes
    (#1506: the hook's on-main gate reads it as off-main)."""
    repo = tmp_path / "ambient_detached"
    scrub = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    scrub["GIT_CONFIG_GLOBAL"] = "/dev/null"
    scrub["GIT_CONFIG_NOSYSTEM"] = "1"
    common: dict = dict(check=True, capture_output=True, text=True, env=scrub)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], **common)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=workflow-lint-test",
            "-c",
            "user.email=workflow-lint-test@localhost",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "detached-head probe fixture",
        ],
        **common,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach"], **common)
    return repo


def test_check_git_recipes_root_guard_selftest_deterministic_under_ambient_detach(
    tmp_path, monkeypatch
):
    """Regression #1545 (incident #1506): with ambient GIT_DIR/GIT_WORK_TREE
    pointing at a DETACHED-HEAD repo — simulating, via env injection, the
    off-main git state a concurrent shared-root rebase exposes, WITHOUT
    touching the real root — the check stays deterministic: self-test
    passes, a benign git-bearing fence lints clean. Pre-fix, the ambient
    env leaked into the hook subprocess, the on-main gate read off-main,
    the blocked probe returned rc 0, and the self-test failed loud (the
    13-failure #1506 shape). Complementary pin:
    ``test_root_guard_hook_fails_open_off_main_fixture`` asserts the
    hook-side mechanism (off-main fail-open + GIT_DIR sensitivity); this
    test asserts the check-side neutralization — the pair covers
    complementary hook-stops-honoring-GIT_DIR states, so neither is
    redundant with the other."""
    ambient = _make_detached_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(ambient / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(ambient))
    _write_rg_skill(tmp_path, "x", '```bash\ngit -C "$WT" status\n```\n')
    assert check_git_recipes_root_guard(repo_root=tmp_path) == []


def test_run_root_guard_pins_fixture_over_ambient_git_env(tmp_path, monkeypatch):
    """Unit: ambient GIT_* scrubbed + fixture pinned — the blocked
    self-test probe reads rc 2 against the LIVE hook even with a
    detached-HEAD GIT_DIR in the ambient env."""
    from workflow_lint import _build_root_guard_fixture

    ambient = _make_detached_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(ambient / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(ambient))
    fixture = tmp_path / "fixture"
    _build_root_guard_fixture(fixture)
    rc, _ = _run_root_guard(_ROOT_GUARD_HOOK, _ROOT_GUARD_SELFTEST_BLOCKED, fixture)
    assert rc == 2


def test_root_guard_hook_fails_open_off_main_fixture(tmp_path):
    """Mechanism pin (the committed reproduce-first demonstration): the
    LIVE hook's on-main gate (``guard_repo_root_branch.sh``'s
    ``rev-parse --abbrev-ref HEAD`` + ``[ "$cur" = main ] || exit 0``
    gate) fail-opens off main — the blocked probe reads rc 0 when the
    pinned git state is a detached-HEAD repo. This is the #1506 flap
    mechanism; it ALSO fails loud if a future hook edit stops honoring
    GIT_DIR (which would silently break this whole isolation).
    Complementary pin:
    ``test_check_git_recipes_root_guard_selftest_deterministic_under_ambient_detach``
    asserts the check-side neutralization; this test pins the hook-side
    mechanism — the pair covers complementary
    hook-stops-honoring-GIT_DIR states, so neither is redundant with the
    other."""
    detached = _make_detached_repo(tmp_path)
    rc, _ = _run_root_guard(_ROOT_GUARD_HOOK, _ROOT_GUARD_SELFTEST_BLOCKED, detached)
    assert rc == 0


def test_check_git_recipes_root_guard_fixture_build_failure_fails_loud(tmp_path, monkeypatch):
    """Must-Fix (Statistics critic, plan v3): a fixture-build failure must
    surface as exactly ONE loud lint error — never a silent [] that
    no-ops the whole check (the silent-under-enforcement class #1545
    exists to close; a later refactor converting the except arm into a
    silent return [] would otherwise leave all other tests green while
    the check no-ops, _live_tree_passes passing vacuously)."""
    import workflow_lint as wl

    def _boom(tmp):
        raise OSError("forced fixture-build failure (test)")

    monkeypatch.setattr(wl, "_build_root_guard_fixture", _boom)
    _write_rg_skill(tmp_path, "x", '```bash\ngit -C "$WT" status\n```\n')
    errors = check_git_recipes_root_guard(repo_root=tmp_path)
    assert len(errors) == 1 and "fixture build failed" in errors[0], errors


def _write_rg_selftest_aware_hook(hook: Path, fence_branch: str) -> None:
    """Write a fake hook that passes BOTH self-test probes (blocked-probe
    token -> exit 2, benign-probe token -> exit 0; the stdin-distinguishable
    tokens of ``_ROOT_GUARD_SELFTEST_BLOCKED`` / ``_ROOT_GUARD_SELFTEST_BENIGN``)
    and runs ``fence_branch`` (one-line bash) for every other stdin — i.e.
    for fence probes. Hooks run as ``["bash", path]``; no exec bit needed."""
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "input=$(cat)\n"
        'case "$input" in\n'
        "  *__workflow_lint_root_guard_selftest__*) exit 2;;\n"
        '  *"root-guard selftest"*) exit 0;;\n'
        f"  *) {fence_branch};;\n"
        "esac\n",
        encoding="utf-8",
    )


def test_check_git_recipes_root_guard_unexpected_rc_warns_not_fails(tmp_path):
    """A hook returning a persistent unexpected rc (3) on fence probes —
    while passing both self-test probes — yields NO FAIL and ONE WARN
    naming the fence + rc, after exactly one retry (the fence branch runs
    exactly twice; #1610: a transient rc must not flip a clean gate)."""
    _write_rg_skill(tmp_path, "x", "```bash\ngit status\n```\n")
    counter = tmp_path / "fence_invocations.txt"
    hook = tmp_path / "hook.sh"
    _write_rg_selftest_aware_hook(hook, f'echo x >> "{counter}"; exit 3')
    warns: list[str] = []
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=hook, warn_sink=warns)
    assert errors == [], errors
    assert len(warns) == 1, warns
    assert "rc=3" in warns[0] and "SKILL.md:1" in warns[0], warns
    # retried exactly once: the fence branch was invoked EXACTLY 2 times
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 2


def test_check_git_recipes_root_guard_unexpected_rc_stderr_fallback(tmp_path, capsys):
    """The production path (no ``warn_sink``) routes the persistent
    unexpected-rc WARN to stderr with the ``WARN: `` prefix — never into
    the returned FAIL list."""
    _write_rg_skill(tmp_path, "x", "```bash\ngit status\n```\n")
    hook = tmp_path / "hook.sh"
    _write_rg_selftest_aware_hook(hook, "exit 3")
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=hook)
    assert errors == [], errors
    err = capsys.readouterr().err
    assert "WARN: " in err and "rc=3" in err, err


def test_check_git_recipes_root_guard_transient_rc_retry_recovers(tmp_path):
    """A hook flaky on the FIRST fence invocation (rc 3) and clean (rc 0)
    on the retry: the verdict follows the retry — no FAIL, no WARN.
    Single fence per fixture => no cross-thread marker race."""
    _write_rg_skill(tmp_path, "x", "```bash\ngit status\n```\n")
    marker = tmp_path / "first_fence_probe_done"
    hook = tmp_path / "hook.sh"
    _write_rg_selftest_aware_hook(
        hook, f'if [ ! -e "{marker}" ]; then touch "{marker}"; exit 3; fi; exit 0'
    )
    warns: list[str] = []
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=hook, warn_sink=warns)
    assert errors == [], errors
    assert warns == [], warns


def test_check_git_recipes_root_guard_transient_rc_retry_blocked(tmp_path):
    """Same flaky shape but the retry returns rc 2: the retry verdict is a
    normal BLOCKED FAIL (the retry never launders a real block), no WARN."""
    _write_rg_skill(tmp_path, "x", "```bash\ngit status\n```\n")
    marker = tmp_path / "first_fence_probe_done"
    hook = tmp_path / "hook.sh"
    _write_rg_selftest_aware_hook(
        hook, f'if [ ! -e "{marker}" ]; then touch "{marker}"; exit 3; fi; exit 2'
    )
    warns: list[str] = []
    errors = check_git_recipes_root_guard(repo_root=tmp_path, hook_path=hook, warn_sink=warns)
    assert len(errors) == 1, errors
    assert "BLOCKED" in errors[0], errors
    assert warns == [], warns


# ---------------------------------------------------------------------------
# --check-bare-commit-pathspec (#1648): a fenced `git commit` recipe with no
# ` -- <pathspec>` tail sweeps the whole staged index at the always-concurrent
# shared repo root (incident 7dbde267f1; #1630 fixed /daily per-file). All
# git-commit literals below are Python STRING DATA (fixtures written via
# tmp_path) — the check scans only workflow-surface .md files, so neither
# this test file nor workflow_lint.py can self-flag.
# ---------------------------------------------------------------------------


def test_check_bare_commit_pathspec_flags_bare_commit(tmp_path):
    """A bash fence with a pathspec-less `git commit -m` is exactly one error
    naming file:line (fence opener + offset) + the remediation paths."""
    _write_rg_skill(tmp_path, "x", '```bash\ngit add logs/x.md\ngit commit -m "x"\n```\n')
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:3:" in errors[0], errors
    assert "-- <pathspec>" in errors[0], errors
    # remediation names both paths: append a pathspec, or the waiver sentinel
    assert "allow-bare-commit-block" in errors[0], errors


def test_check_bare_commit_pathspec_allows_exempt_forms(tmp_path):
    """Every structural exemption in one fence passes; the sharp-edge
    NEGATIVES (per-invocation -C, flag-less xargs, xargs after the commit)
    each still flag exactly once."""
    body = (
        "```bash\n"
        'git commit -m "x" -- logs/x.md\n'
        'git -C "$WT" commit -m "x"\n'
        'xargs -r -a f.txt git commit -m "x" --\n'
        'git commit --dry-run -m "x"\n'
        '# git commit -m "x"\n'
        "```\n"
    )
    _write_rg_skill(tmp_path, "x", body)
    assert check_bare_commit_pathspec(repo_root=tmp_path) == []

    # (i) per-invocation -C: the FIRST invocation's -C (and its pathspec)
    # must not waive the SECOND, bare invocation on the same line.
    _write_rg_skill(
        tmp_path, "x", '```bash\ngit -C "$WT" commit -m "a" -- x && git commit -m "b"\n```\n'
    )
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors

    # (ii) flag-less xargs runs the command ONCE on empty input
    # (whole-index sweep) -> NOT exempt without -r/--no-run-if-empty.
    _write_rg_skill(tmp_path, "x", '```bash\nxargs -a f.txt git commit -m "x" --\n```\n')
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors

    # (iii) xargs AFTER the commit never waives it (prefix-bounded search).
    _write_rg_skill(
        tmp_path, "x", '```bash\ngit commit -m "x" && xargs -r -a f.txt git rm --\n```\n'
    )
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors


def test_check_bare_commit_pathspec_multiline_quoted_message(tmp_path):
    """Quote-balance joining classifies multi-line `-m "..."` messages on the
    full command; backslash continuations join per bash semantics; errors
    report the FIRST line of the joined logical line."""
    # (a) -C form + (b) the xargs form with a multi-line message and a
    # trailing ` --` (the issue/SKILL.md additive-checkout shape) -> clean.
    body = (
        "```bash\n"
        'git -C "$WT" commit -m "line1\n'
        'line2"\n'
        'xargs -r -a f.txt git commit -m "issue: line1\n'
        "line2\n"
        'line3" --\n'
        "```\n"
    )
    _write_rg_skill(tmp_path, "x", body)
    assert check_bare_commit_pathspec(repo_root=tmp_path) == []

    # (c) a BARE commit with a multi-line message and no pathspec still
    # flags (quote-joining must not hide bare commits), at the first line.
    _write_rg_skill(tmp_path, "x", '```bash\ngit commit -m "line1\nline2"\n```\n')
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:2:" in errors[0], errors

    # (d) backslash continuation: a compliant commit whose ` -- <paths>`
    # sits on the continuation line is clean ...
    _write_rg_skill(tmp_path, "x", '```bash\ngit commit -m "x" \\\n  -- logs/x.md\n```\n')
    assert check_bare_commit_pathspec(repo_root=tmp_path) == []

    # ... and a bare backslash-continued commit flags at the FIRST line of
    # the joined logical (pins the continuation join + line arithmetic).
    _write_rg_skill(tmp_path, "x", '```bash\ngit commit \\\n  -m "x"\necho done\n```\n')
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:2:" in errors[0], errors


def test_check_bare_commit_pathspec_fence_sentinel_waives(tmp_path):
    """A sentinel with a NON-EMPTY reason on the immediately-preceding
    non-blank line waives the fence; an EMPTY-reason sentinel does NOT
    (mirrors the root-guard empty-reason discipline)."""
    _write_rg_skill(
        tmp_path,
        "x",
        "<!-- workflow-lint: allow-bare-commit-block: deliberate anti-pattern example -->\n"
        '```bash\ngit commit -m "x"\n```\n',
    )
    assert check_bare_commit_pathspec(repo_root=tmp_path) == []

    _write_rg_skill(
        tmp_path,
        "x",
        '<!-- workflow-lint: allow-bare-commit-block: -->\n```bash\ngit commit -m "x"\n```\n',
    )
    errors = check_bare_commit_pathspec(repo_root=tmp_path)
    assert len(errors) == 1, errors


def test_check_bare_commit_pathspec_ignores_non_bash_fences(tmp_path):
    """Only bash/sh/shell-tagged fences are scanned: an untagged fence, a
    python-tagged fence, and prose inline code never flag — and the check's
    docstring must NAME the residual classes so the disclosure is durable
    (root-guard known_miss_residuals style)."""
    body = (
        '- Then run `git commit -m "x"` at the repo root.\n'
        "\n"
        "```\n"
        'git commit -m "x"\n'
        "```\n"
        "\n"
        "```python\n"
        "cmd = 'git commit -m x'\n"
        "```\n"
    )
    _write_rg_skill(tmp_path, "x", body)
    assert check_bare_commit_pathspec(repo_root=tmp_path) == []
    doc = check_bare_commit_pathspec.__doc__ or ""
    assert "untagged" in doc, "docstring must name the untagged-fence residual"
    assert "inline-code" in doc, "docstring must name the prose inline-code residual"
    assert "heredoc" in doc, "docstring must name the heredoc-body residual"
    assert "scripts/**" in doc, "docstring must name the shell-script out-of-scope residual"


def test_check_bare_commit_pathspec_live_tree_passes():
    """The real post-disposition tree PASSES (the ``test_live_trees_pass``
    invariant): locks in the #1648 dispositions — the ideation / weekly /
    issue-v2 pathspec fixes — and fails loud on any future bare-commit
    recipe landing in the workflow docs."""
    assert check_bare_commit_pathspec() == []


def test_check_bare_commit_pathspec_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #1648 check —
    deleting its ``or no_flags`` branch must fail this test
    (mutation-visible), closing the present-but-never-runs class. Follows
    the ``test_vm_thread_cap_guidance_bundled_in_no_flags`` house pattern:
    one offender fixture, ``_REPO_ROOT`` monkeypatched, ``main([])``
    in-process. Other bundled checks contribute unrelated errors on the
    minimal tree, so the assertion keys on the #1648 diagnostic + the
    offending file path."""
    import workflow_lint as wl

    _write_rg_skill(tmp_path, "x", '```bash\ngit add logs/x.md\ngit commit -m "x"\n```\n')
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offender tree:\n{err}"
    assert "allow-bare-commit-block" in err and "skills/x/SKILL.md" in err, (
        f"the #1648 bare-commit diagnostic (naming the fixture skill) is missing "
        f"from the no-flags default run's stderr — the check is not bundled "
        f"into no_flags:\n{err}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_skill_bang_backtick`` (incident class #1243/#1266:
# a bang directly against a backtick in preprocessor-loaded skill markdown
# makes Claude Code execute the following text as inline shell AT SKILL LOAD
# — commit 90af0ce2d9 introduced two such prose spans in
# .claude/skills/issue/SKILL.md and every /issue session boot died until
# hotfix f75e1b4c13 reworded them). Fixtures build the hazardous adjacency
# at RUNTIME (concatenation with chr(96)) so this test file never contains
# it in source — the live-tree invariant below scans this repo's own
# .claude/ markdown, and the check has no waiver by design.
# ---------------------------------------------------------------------------

_TICK = chr(96)  # backtick — never written literally next to a bang here
_BANG_TICK = "!" + _TICK


def _write_bang_md(root: Path, rel: str, text: str) -> Path:
    """Write a fixture markdown file under ``root/rel``, creating parents."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_skill_bang_backtick_fail_incident_prose_span(tmp_path):
    """The verbatim 90af0ce2d9 incident line shape — a code span whose
    content ends in a bang against the CLOSING backtick — is flagged with a
    path:lineno-prefixed error (and the offending line is never echoed)."""
    line = (
        "the guard exits nonzero ("
        + _TICK
        + "grep -q pattern file"
        + _BANG_TICK
        + ") before dispatch"
    )
    _write_bang_md(tmp_path, "skills/issue/SKILL.md", "# heading\n\n" + line + "\n")
    errors = check_skill_bang_backtick(claude_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "SKILL.md:3" in errors[0]
    assert "grep -q pattern" not in errors[0], "error string must not echo the line"


def test_check_skill_bang_backtick_fail_inside_fenced_block(tmp_path):
    """No fenced-block exemption: the same adjacency inside a fenced bash
    block is flagged (the preprocessor is not verified to ignore fences)."""
    fence = _TICK * 3
    body = (
        "# doc\n\n"
        + fence
        + "bash\n"
        + "echo start "
        + _BANG_TICK
        + "whoami"
        + _TICK
        + " end\n"
        + fence
        + "\n"
    )
    _write_bang_md(tmp_path, "skills/s/SKILL.md", body)
    errors = check_skill_bang_backtick(claude_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "SKILL.md:4" in errors[0]


def test_check_skill_bang_backtick_fail_agents_and_commands_roots(tmp_path):
    """Root coverage: hits under agents/ and commands/ are both flagged, and
    an absent skills/ root does not crash (the exists-guard)."""
    _write_bang_md(tmp_path, "agents/foo.md", "x " + _BANG_TICK + "cmd" + _TICK + " y\n")
    _write_bang_md(tmp_path, "commands/bar.md", "z " + _BANG_TICK + "cmd" + _TICK + "\n")
    # skills/ deliberately absent — must not crash
    errors = check_skill_bang_backtick(claude_dir=tmp_path)
    assert len(errors) == 2, f"expected two errors, got: {errors}"
    joined = "\n".join(errors)
    assert "foo.md:1" in joined
    assert "bar.md:1" in joined


def test_check_skill_bang_backtick_pass_dollar_bang(tmp_path):
    """The '$!' shell-pid prose shape (the 3 live SKILL.md instances) is
    carved out by the lookbehind — empirically inert across healthy boots."""
    line = "capture the pid (" + _TICK + "echo $" + _BANG_TICK + ") after launch"
    _write_bang_md(tmp_path, "skills/issue/SKILL.md", line + "\n")
    assert check_skill_bang_backtick(claude_dir=tmp_path) == []


def test_check_skill_bang_backtick_pass_bang_space_and_bare_bang(tmp_path):
    """Negatives: the f75e1b4c13 reworded shape (bang, space, then more span
    content) and a bang at end-of-line with a backtick only on the NEXT line
    (per-line scan: the characters must be byte-adjacent) both pass."""
    body = (
        "use "
        + _TICK
        + "if ! grep -q pattern"
        + _TICK
        + " to test\n"
        + "watch out!\n"
        + _TICK
        + "code"
        + _TICK
        + "\n"
    )
    _write_bang_md(tmp_path, "skills/s.md", body)
    assert check_skill_bang_backtick(claude_dir=tmp_path) == []


def test_check_skill_bang_backtick_repo_tree_is_clean():
    """Durability pin: the committed .claude/{skills,agents,commands}
    markdown must carry no non-dollar bang-against-backtick spans — the
    skill preprocessor executes such a span as inline shell at load, and
    the #1243 incident killed every /issue session boot until hotfix
    f75e1b4c13 reworded the two offending spans."""
    errors = check_skill_bang_backtick()
    assert errors == [], (
        ".claude/{skills,agents,commands} markdown carries bang-against-"
        "backtick inline-exec spans (#1243/#1266 session-killer class); "
        "reword them — insert a space before the backtick or write "
        "'bang-backtick' in prose (no waiver exists by design):\n" + "\n".join(errors)
    )


def test_workflow_lint_check_skill_bang_backtick_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-skill-bang-backtick")
    assert result.returncode == 0, (
        f"workflow_lint --check-skill-bang-backtick failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_skill_bang_backtick_bundled_in_no_flags():
    """NON-VACUOUS no-flags bundling pin (the
    ``test_pipe_python_bundled_in_no_flags_source_pin`` shape):
    ``check_skill_bang_backtick`` must be dispatched by the BARE
    ``workflow_lint.py`` run. Without this pin every other test in this
    suite stays green with the two wiring lines absent (fixtures call the
    function directly; ``_run`` exits 0 vacuously; the live-tree test
    bypasses ``main()``), silently disarming the Step 10d protection."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_skill_bang_backtick or no_flags:\s*\n"
        r"\s*errors\.extend\(check_skill_bang_backtick\(\)\)",
        src,
    ), "check_skill_bang_backtick is not dispatched on the no-flags branch"
    assert "or args.check_skill_bang_backtick" in src, (
        "--check-skill-bang-backtick is missing from the no_flags detection tuple"
    )


# ---------------------------------------------------------------------------
# Unit tests for ``check_agents_note_argv_verdict`` (#1743/#1785): no agent
# spec under .claude/agents/ may prescribe posting a verdict/marker body
# through an argv-prose ``--note`` command substitution (the #1722/#1756
# incident family; the fix channel is ``post-marker --file``). The fixture
# files below carry the REAL banned literals verbatim — tests/ sits outside
# every scan root of the check, so the fixtures never trip it on the
# committed tree.
# ---------------------------------------------------------------------------


def test_check_agents_note_argv_verdict_fail_p1(tmp_path):
    """FAIL — P1, the #1743 acceptance-grep pattern: a --note body opened
    as a command substitution around cat."""
    (tmp_path / "some-agent.md").write_text(
        "Post the verdict via:\n"
        '`task.py post-marker <N> epm:code-review --note "$(cat /tmp/v.md)"`\n'
    )
    errors = check_agents_note_argv_verdict(agents_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "some-agent.md:2" in errors[0]
    assert "#1743" in errors[0]
    assert "--file" in errors[0]


def test_check_agents_note_argv_verdict_fail_p2(tmp_path):
    """FAIL — P2, the #1743 reviewer's broader variant: any --note opening
    directly into a command substitution (not just cat)."""
    (tmp_path / "agent.md").write_text(
        'run `task.py post-marker 7 epm:results --note "$(uv run python gen.py)"`\n'
    )
    errors = check_agents_note_argv_verdict(agents_dir=tmp_path)
    assert len(errors) == 1, f"expected exactly one error, got: {errors}"
    assert "agent.md:1" in errors[0]


def test_check_agents_note_argv_verdict_pass_clean_spec(tmp_path):
    """PASS — a clean spec prescribing the sanctioned --file channel."""
    (tmp_path / "agent.md").write_text(
        "## Reporting\n\nCompose the report body with the Write tool, then post it via\n"
        "`uv run python scripts/task.py post-marker <N> epm:results --file /tmp/r.md`.\n"
    )
    assert check_agents_note_argv_verdict(agents_dir=tmp_path) == []


def test_check_agents_note_argv_verdict_pass_note_without_substitution(tmp_path):
    """PASS — the sanctioned variable form: command substitutions resolved
    into a shell variable FIRST, then the variable passed as the note (no
    substitution opener adjacent to the note token). The check must never
    be "fixed" into flagging this shape."""
    (tmp_path / "agent.md").write_text(
        'Resolve every substitution into a variable first, e.g. NOTE="round 3 done",\n'
        'then run `task.py post-marker 5 epm:progress --note "$NOTE"`.\n'
    )
    assert check_agents_note_argv_verdict(agents_dir=tmp_path) == []


def test_check_agents_note_argv_verdict_live_tree_clean():
    """The committed .claude/agents/ tree must carry no argv-prose --note
    verdict-post prescription — the #1743 acceptance grep as a standing
    regression tripwire (production default, no kwarg; baseline verified
    clean 2026-07-29)."""
    errors = check_agents_note_argv_verdict()
    assert errors == [], (
        "agent specs prescribe the #1743-banned argv-prose --note "
        "verdict post (use the post-marker --file channel):\n" + "\n".join(errors)
    )


def test_check_agents_note_argv_verdict_bundled_in_no_flags():
    """NON-VACUOUS no-flags bundling pin (the
    ``test_pipe_python_bundled_in_no_flags_source_pin`` shape):
    ``check_agents_note_argv_verdict`` must be dispatched by the BARE
    ``workflow_lint.py`` run — source-inspection assert on the dispatch
    branch + the no_flags detection-tuple membership (the #1385/#1648
    silent-unbundling class)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_agents_note_argv_verdict or no_flags:\s*\n"
        r"\s*errors\.extend\(check_agents_note_argv_verdict\(\)\)",
        src,
    ), "check_agents_note_argv_verdict is not dispatched on the no-flags branch"
    assert "or args.check_agents_note_argv_verdict" in src, (
        "--check-agents-note-argv-verdict is missing from the no_flags detection tuple"
    )


# ---------------------------------------------------------------------------
# --check-live-hf-retry-routing (#1547): bare live HF call sites
# ---------------------------------------------------------------------------


def _hf_routing_root(tmp_path: Path) -> Path:
    """A minimal tmp repo root carrying the two scanned scope roots."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "explore_persona_space").mkdir(parents=True)
    return tmp_path


def test_check_live_hf_retry_routing_bare_call_flags(tmp_path):
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    errors = check_live_hf_retry_routing(repo_root=root)
    assert len(errors) == 1, errors
    assert "scripts/new_tool.py:3" in errors[0]
    assert "live-hf-retry-routing" in errors[0]


def test_check_live_hf_retry_routing_wrapped_call_passes(tmp_path):
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from explore_persona_space.orchestrate.hub import retry_transient\n"
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return retry_transient(\n"
        "        lambda: hf_hub_download('org/repo', 'a.bin'),\n"
        "        what='hf_hub_download(org/repo/a.bin)',\n"
        "    )\n",
        encoding="utf-8",
    )
    assert check_live_hf_retry_routing(repo_root=root) == []


def test_check_live_hf_retry_routing_multiline_call_with_trailing_what_passes(tmp_path):
    """The wrap opens 1 line above the call; the `what=` descriptor kwarg sits
    BELOW the +/-3 window and matches the call regex textually — the
    `what=`-skip (wrap-idiom descriptor) keeps it from flagging."""
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from explore_persona_space.orchestrate.hub import retry_transient\n"
        "from huggingface_hub import hf_hub_download\n"
        "def fetch(rel):\n"
        "    return retry_transient(\n"
        "        lambda: hf_hub_download(\n"
        "            'org/repo',\n"
        "            rel,\n"
        "            repo_type='dataset',\n"
        "            revision='main',\n"
        "        ),\n"
        "        what=f'hf_hub_download(org/repo/{rel})',\n"
        "    )\n",
        encoding="utf-8",
    )
    assert check_live_hf_retry_routing(repo_root=root) == []


def test_check_live_hf_retry_routing_no_retry_waiver_passes(tmp_path):
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def probe():\n"
        "    # NO_RETRY: deliberate existence probe; caller handles the error\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n"
        "def probe2():\n"
        "    return hf_hub_download('org/repo', 'b.bin')  # NO_RETRY: fail-fast by design\n",
        encoding="utf-8",
    )
    assert check_live_hf_retry_routing(repo_root=root) == []


def test_check_live_hf_retry_routing_adjacent_wrap_does_not_launder_bare_sibling(tmp_path):
    """#1547 review directive 2: two calls in one window, one wrapped one bare
    — the bare one still flags (the wrap anchors to the CALL'S OWN
    expression via the open-paren balance, never mere window proximity)."""
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from explore_persona_space.orchestrate.hub import retry_transient\n"
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    a = retry_transient(lambda: hf_hub_download('org/r', 'a'), what='x')\n"
        "    b = hf_hub_download('org/r', 'b')\n"
        "    return a, b\n",
        encoding="utf-8",
    )
    errors = check_live_hf_retry_routing(repo_root=root)
    assert len(errors) == 1, errors
    assert "scripts/new_tool.py:5" in errors[0]


def test_check_live_hf_retry_routing_frozen_snapshot_skipped(tmp_path):
    from workflow_lint import HF_ROUTING_FROZEN_SNAPSHOT, check_live_hf_retry_routing

    frozen_member = "scripts/eval_issue562_panel.py"
    assert frozen_member in HF_ROUTING_FROZEN_SNAPSHOT
    root = _hf_routing_root(tmp_path)
    (root / frozen_member).write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    assert check_live_hf_retry_routing(repo_root=root) == []


def test_check_live_hf_retry_routing_pattern_string_and_generated_code_files_skipped(tmp_path):
    """#1547 review directive 3: the check's own file + verify_plan.py
    (pattern strings) and backends/gcp.py (generated pod-side heredoc
    strings with their own bounded retry) are constant-excluded."""
    from workflow_lint import check_live_hf_retry_routing

    root = _hf_routing_root(tmp_path)
    bare = "x = api.upload_folder(folder_path='d', path_in_repo='p')\n"
    (root / "scripts" / "workflow_lint.py").write_text(bare, encoding="utf-8")
    (root / "scripts" / "verify_plan.py").write_text(bare, encoding="utf-8")
    (root / "src" / "explore_persona_space" / "backends").mkdir(parents=True)
    (root / "src" / "explore_persona_space" / "backends" / "gcp.py").write_text(
        bare, encoding="utf-8"
    )
    assert check_live_hf_retry_routing(repo_root=root) == []


def test_check_live_hf_retry_routing_live_tree_passes():
    """Acceptance criterion 1 (#1547): the committed tree carries ZERO bare
    live HF call sites — the leg-A routing left nothing un-routed, and the
    frozen snapshot covers exactly the historical set. A NEW bare call in
    live code fails this test (the JUDGE_PIN test_live_trees_pass idiom)."""
    from workflow_lint import check_live_hf_retry_routing

    assert check_live_hf_retry_routing() == []


def test_workflow_lint_check_live_hf_retry_routing_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-live-hf-retry-routing")
    assert result.returncode == 0, (
        f"workflow_lint --check-live-hf-retry-routing failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_check_live_hf_retry_routing_registered_in_no_flags_default_run():
    """The check is bundled into the no-flags default run (#1547): the flag
    is in the no_flags detection tuple AND dispatched on the no-flags
    branch."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_live_hf_retry_routing or no_flags:\s*\n"
        r"\s*errors\.extend\(check_live_hf_retry_routing\(\)\)",
        src,
    ), "check_live_hf_retry_routing is not dispatched on the no-flags branch"
    assert "or args.check_live_hf_retry_routing" in src, (
        "--check-live-hf-retry-routing is missing from the no_flags detection tuple"
    )


# ---------------------------------------------------------------------------
# --regen-hf-routing-snapshot (#1568): paste-ready snapshot regeneration
# ---------------------------------------------------------------------------


def test_regen_hf_routing_snapshot_lists_bare_skips_wrapped_and_waived(tmp_path, capsys):
    """The regen walker applies the same bare/wrapped/waived predicate as the
    check. The tmp root is NON-GIT by construction — pinned explicitly below,
    naming the gate-tree constraint (the Step 10d `git archive | tar -x`
    landing tree has no .git; #1568 hard constraint 1)."""
    from workflow_lint import regen_hf_routing_snapshot

    root = _hf_routing_root(tmp_path)
    assert not (root / ".git").exists()
    (root / "scripts" / "bare.py").write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    (root / "scripts" / "wrapped.py").write_text(
        "from explore_persona_space.orchestrate.hub import retry_transient\n"
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return retry_transient(\n"
        "        lambda: hf_hub_download('org/repo', 'a.bin'),\n"
        "        what='hf_hub_download(org/repo/a.bin)',\n"
        "    )\n",
        encoding="utf-8",
    )
    (root / "scripts" / "waived.py").write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def probe():\n"
        "    # NO_RETRY: deliberate existence probe\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    assert regen_hf_routing_snapshot(repo_root=root) == 0
    out = capsys.readouterr().out
    entries = re.findall(r'^        "([^"]+)",$', out, flags=re.MULTILINE)
    assert entries == ["scripts/bare.py"], entries


def test_regen_hf_routing_snapshot_is_snapshot_blind(tmp_path, capsys):
    """A bare call AT a frozen-snapshot member path is exempt for the CHECK
    but enumerated by regen — the snapshot is re-derived, never inherited."""
    from workflow_lint import (
        HF_ROUTING_FROZEN_SNAPSHOT,
        check_live_hf_retry_routing,
        regen_hf_routing_snapshot,
    )

    frozen_member = "scripts/eval_issue562_panel.py"
    assert frozen_member in HF_ROUTING_FROZEN_SNAPSHOT
    root = _hf_routing_root(tmp_path)
    (root / frozen_member).write_text(
        "from huggingface_hub import hf_hub_download\n"
        "def fetch():\n"
        "    return hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    assert check_live_hf_retry_routing(repo_root=root) == []
    assert regen_hf_routing_snapshot(repo_root=root) == 0
    out = capsys.readouterr().out
    assert f'        "{frozen_member}",\n' in out


def test_regen_hf_routing_snapshot_output_paste_round_trip(tmp_path, capsys):
    """stdout is byte-compatible with the source shape: the exact header
    line, sorted quoted entries, `    }` + `)` closers — and the set body
    `ast.literal_eval`s back to the offender set (paste-replace safe)."""
    import ast

    from workflow_lint import regen_hf_routing_snapshot

    root = _hf_routing_root(tmp_path)
    for name in ("b_tool.py", "a_tool.py"):
        (root / "scripts" / name).write_text(
            "from huggingface_hub import hf_hub_download\n"
            "x = hf_hub_download('org/repo', 'a.bin')\n",
            encoding="utf-8",
        )
    assert regen_hf_routing_snapshot(repo_root=root) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "HF_ROUTING_FROZEN_SNAPSHOT: frozenset[str] = frozenset("
    assert lines[1] == "    {"
    assert lines[-2] == "    }"
    assert lines[-1] == ")"
    parsed = ast.literal_eval("\n".join(lines[1:-1]))
    assert parsed == {"scripts/a_tool.py", "scripts/b_tool.py"}
    entries = re.findall(r'^        "([^"]+)",$', out, flags=re.MULTILINE)
    assert entries == sorted(entries)


def test_regen_hf_routing_snapshot_diff_summary_on_stderr(tmp_path, capsys):
    """stderr carries the +/- summary vs the compiled-in constant: the tmp
    offender is a `+` line, and (on a tmp root) every compiled-in member
    reads as a `-` line — the delta the round's code review must see."""
    from workflow_lint import HF_ROUTING_FROZEN_SNAPSHOT, regen_hf_routing_snapshot

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_offender.py").write_text(
        "from huggingface_hub import hf_hub_download\nx = hf_hub_download('org/repo', 'a.bin')\n",
        encoding="utf-8",
    )
    assert regen_hf_routing_snapshot(repo_root=root) == 0
    err = capsys.readouterr().err
    assert f"+1 added, -{len(HF_ROUTING_FROZEN_SNAPSHOT)} removed" in err
    assert "# + scripts/new_offender.py" in err
    assert err.count("# - ") == len(HF_ROUTING_FROZEN_SNAPSHOT)


def test_regen_hf_routing_snapshot_live_tree_subset_of_current():
    """Walker-parity pin: on the LIVE tree the regen offender set is a
    subset of the compiled-in snapshot (the check is green, so no offender
    outside the snapshot exists). Divergence diagnosis is in the assert
    message."""
    from workflow_lint import (
        _REPO_ROOT,
        HF_ROUTING_FROZEN_SNAPSHOT,
        _hf_routing_file_errors,
        _hf_routing_scan_files,
    )

    offenders = {
        rel for py, rel in _hf_routing_scan_files(_REPO_ROOT) if _hf_routing_file_errors(py, rel)
    }
    unexpected = sorted(offenders - HF_ROUTING_FROZEN_SNAPSHOT)
    assert not unexpected, (
        f"regen enumerates offenders outside HF_ROUTING_FROZEN_SNAPSHOT: {unexpected}. "
        "If any of these is workflow_lint.py / verify_plan.py / backends/gcp.py, the "
        "regen walker DIVERGED from the check walker (dropped a constant exclusion — "
        "code bug). Otherwise a genuinely-new bare-call offender landed on main after "
        "this branch forked — NOT a walker bug: rebase onto origin/main and re-run "
        "(that red IS the staleness signal working; #1568 plan §6 kill criterion 3)."
    )


def test_workflow_lint_regen_hf_routing_snapshot_cli():
    """rc=0 and stdout is EXACTLY the paste-ready literal: the early dispatch
    means no `workflow_lint:` check-output line ever contaminates stdout
    (the bundle never ran)."""
    result = _run("--regen-hf-routing-snapshot")
    assert result.returncode == 0, (
        f"workflow_lint --regen-hf-routing-snapshot failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.startswith("HF_ROUTING_FROZEN_SNAPSHOT: frozenset[str] = frozenset(")
    assert "workflow_lint:" not in result.stdout


def test_regen_flag_early_dispatch_and_not_in_no_flags_tuple():
    """Source pins (#1568): (a) the regen dispatch early-returns BEFORE the
    no_flags computation; (b) the flag is NOT in the no_flags detection
    tuple (the early return makes membership dead code — pin it OUT, the
    inverse of the check's registration test); (c) the check's FAIL message
    names the recovery flag (the message-hint durability pin)."""
    src = _LINT.read_text(encoding="utf-8")
    dispatch_at = src.index("if args.regen_hf_routing_snapshot:")
    no_flags_at = src.index("no_flags = not (")
    assert dispatch_at < no_flags_at, "regen dispatch must precede the no_flags computation"
    tuple_match = re.search(r"no_flags = not \(.*?\n    \)", src, flags=re.DOTALL)
    assert tuple_match is not None, "no_flags detection tuple not found in workflow_lint.py"
    assert "regen_hf_routing_snapshot" not in tuple_match.group(0), (
        "--regen-hf-routing-snapshot must NOT be in the no-flags detection tuple"
    )
    fn_start = src.index("def _hf_routing_file_errors(")
    fn_end = src.index("def check_live_hf_retry_routing(", fn_start)
    assert "--regen-hf-routing-snapshot" in src[fn_start:fn_end], (
        "the live-hf-retry-routing FAIL message must name --regen-hf-routing-snapshot"
    )


# ---------------------------------------------------------------------------
# --check-bare-list-repo-files (#1624): data-repo full-listing wedge
# ---------------------------------------------------------------------------


def test_check_bare_list_repo_files_flags_attribute_call(tmp_path):
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import HfApi\n"
        "def fetch():\n"
        "    return HfApi().list_repo_files('org/repo')\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors
    assert "[bare-list-repo-files]" in errors[0]
    assert "scripts/new_tool.py:3" in errors[0]
    assert "list_hf_files_under_path" in errors[0]


def test_check_bare_list_repo_files_flags_imported_name_alias(tmp_path):
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import list_repo_files as lrf\n"
        "def fetch():\n"
        "    return lrf('org/repo')\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors
    assert "scripts/new_tool.py:3" in errors[0]


def test_check_bare_list_repo_files_flags_imported_name_unaliased(tmp_path):
    """The plain (un-aliased) `from huggingface_hub import list_repo_files`
    + bare `list_repo_files(...)` form — the asname-aware Name leg's
    default branch."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import list_repo_files\n"
        "def fetch():\n"
        "    return list_repo_files('org/repo')\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors
    assert "scripts/new_tool.py:3" in errors[0]


def test_check_bare_list_repo_files_src_scope_covered(tmp_path):
    """The scope delta vs --check-hub-verify-retry (#1202, scripts/ only):
    src/explore_persona_space/ offenders ARE flagged."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "src" / "explore_persona_space" / "new_mod.py").write_text(
        "from huggingface_hub import HfApi\nx = HfApi().list_repo_files('org/repo')\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors
    assert "src/explore_persona_space/new_mod.py:2" in errors[0]


def test_check_bare_list_repo_files_waiver_passes(tmp_path):
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import HfApi\n"
        "def fetch():\n"
        "    # LIST_REPO_FILES_EXEMPT: small model repo, full listing is the point\n"
        "    return HfApi().list_repo_files('org/small-repo')\n",
        encoding="utf-8",
    )
    assert check_bare_list_repo_files(repo_root=root) == []


def test_check_bare_list_repo_files_waiver_on_hit_line_passes(tmp_path):
    """Waiver placement on the HIT line itself (the plan tests only the
    preceding-line placement; the helper honors both)."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import HfApi\n"
        "x = HfApi().list_repo_files('org/r')  # LIST_REPO_FILES_EXEMPT: tiny repo by design\n",
        encoding="utf-8",
    )
    assert check_bare_list_repo_files(repo_root=root) == []


def test_check_bare_list_repo_files_short_waiver_reason_still_flags(tmp_path):
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "new_tool.py").write_text(
        "from huggingface_hub import HfApi\n"
        "# LIST_REPO_FILES_EXEMPT: ok\n"
        "x = HfApi().list_repo_files('org/r')\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors


def test_check_bare_list_repo_files_frozen_snapshot_skipped(tmp_path):
    from workflow_lint import LIST_REPO_FILES_FROZEN_SNAPSHOT, check_bare_list_repo_files

    frozen_member = "scripts/dispatch_neg_geometry_504.py"
    assert frozen_member in LIST_REPO_FILES_FROZEN_SNAPSHOT
    root = _hf_routing_root(tmp_path)
    (root / frozen_member).write_text(
        "from huggingface_hub import HfApi\nx = HfApi().list_repo_files('org/repo')\n",
        encoding="utf-8",
    )
    assert check_bare_list_repo_files(repo_root=root) == []


def test_check_bare_list_repo_files_prose_mentions_never_flag(tmp_path):
    """The AST-invisibility guarantee: docstring / comment / f-string
    mentions are ast.Constant nodes and can never match — no
    pattern-string exclusion constants exist for this check by design."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "prose_only.py").write_text(
        '"""Docstring naming list_repo_files( for reference."""\n'
        "# comment naming list_repo_files( too\n"
        "def msg(repo):\n"
        "    return f'HF list_repo_files({repo}) timed out'\n",
        encoding="utf-8",
    )
    assert check_bare_list_repo_files(repo_root=root) == []


def test_check_bare_list_repo_files_monkeypatch_targets_exempt_load_save_flagged(tmp_path):
    """Inherited #1482/#1561 semantics: Store/Del patch/restore targets are
    exempt; the Load-ctx bare-reference SAVE still flags (a saved alias
    later called still wedges) and takes the waiver."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "patcher.py").write_text(
        "from huggingface_hub import HfApi\n"
        "def patch(fake):\n"
        "    orig = HfApi.list_repo_files\n"
        "    HfApi.list_repo_files = fake\n"
        "    return orig\n",
        encoding="utf-8",
    )
    errors = check_bare_list_repo_files(repo_root=root)
    assert len(errors) == 1, errors
    assert "scripts/patcher.py:3" in errors[0]


def test_check_bare_list_repo_files_scoped_helpers_invisible(tmp_path):
    """The scoped recipes are structurally invisible (different attr/name
    strings): list_repo_files_complete / list_hf_files_under_path /
    api.list_repo_tree never match the narrowed target set."""
    from workflow_lint import check_bare_list_repo_files

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "scoped.py").write_text(
        "from explore_persona_space.orchestrate.hub import (\n"
        "    list_hf_files_under_path,\n"
        "    list_repo_files_complete,\n"
        ")\n"
        "def fetch(api):\n"
        "    a = list_repo_files_complete(api, 'org/repo', path_in_repo='x')\n"
        "    b = list_hf_files_under_path(api, 'org/repo', 'prefix')\n"
        "    c = api.list_repo_tree('org/repo', path_in_repo='p', recursive=True)\n"
        "    d = api.file_exists('org/repo', 'p/file.json')\n"
        "    return a, b, c, d\n",
        encoding="utf-8",
    )
    assert check_bare_list_repo_files(repo_root=root) == []


def test_check_bare_list_repo_files_live_tree_passes():
    """Acceptance criterion 1 (#1624): the committed tree carries ZERO bare
    list_repo_files sites outside the frozen snapshot (the
    test_live_trees_pass idiom). A NEW bare call in scripts/ or src/ fails
    this test."""
    from workflow_lint import check_bare_list_repo_files

    assert check_bare_list_repo_files() == []


def test_check_bare_list_repo_files_registered_in_no_flags_default_run():
    """The check is bundled into the no-flags default run (#1624): the flag
    is in the no_flags detection tuple AND dispatched on the no-flags
    branch."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_bare_list_repo_files or no_flags:\s*\n"
        r"\s*errors\.extend\(check_bare_list_repo_files\(\)\)",
        src,
    ), "check_bare_list_repo_files is not dispatched on the no-flags branch"
    assert "or args.check_bare_list_repo_files" in src, (
        "--check-bare-list-repo-files is missing from the no_flags detection tuple"
    )


def test_workflow_lint_check_bare_list_repo_files_cli_exits_zero():
    """The dedicated flag must exist and pass on the committed tree."""
    result = _run("--check-bare-list-repo-files")
    assert result.returncode == 0, (
        f"workflow_lint --check-bare-list-repo-files failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_regen_list_repo_files_snapshot_lists_offenders_and_diff(tmp_path, capsys):
    """The regen walker applies the same bare/waived predicate as the check
    (snapshot-blind), prints the paste-ready literal on stdout, and the +/-
    diff summary vs the compiled-in constant on stderr (#1568 idiom)."""
    from workflow_lint import LIST_REPO_FILES_FROZEN_SNAPSHOT, regen_list_repo_files_snapshot

    root = _hf_routing_root(tmp_path)
    (root / "scripts" / "bare.py").write_text(
        "from huggingface_hub import HfApi\nx = HfApi().list_repo_files('org/repo')\n",
        encoding="utf-8",
    )
    (root / "scripts" / "waived.py").write_text(
        "from huggingface_hub import HfApi\n"
        "# LIST_REPO_FILES_EXEMPT: small repo, full listing deliberate\n"
        "x = HfApi().list_repo_files('org/small')\n",
        encoding="utf-8",
    )
    (root / "scripts" / "scoped.py").write_text(
        "def fetch(api):\n    return api.list_repo_tree('org/repo', path_in_repo='p')\n",
        encoding="utf-8",
    )
    assert regen_list_repo_files_snapshot(repo_root=root) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "LIST_REPO_FILES_FROZEN_SNAPSHOT: frozenset[str] = frozenset("
    assert lines[1] == "    {"
    assert lines[-2] == "    }"
    assert lines[-1] == ")"
    entries = re.findall(r'^        "([^"]+)",$', captured.out, flags=re.MULTILINE)
    assert entries == ["scripts/bare.py"], entries
    assert f"+1 added, -{len(LIST_REPO_FILES_FROZEN_SNAPSHOT)} removed" in captured.err
    assert "# + scripts/bare.py" in captured.err


def test_hub_verify_bare_hits_default_targets_unchanged():
    """Parametrization regression (#1624 plan §4e test 14): with no
    `targets` kwarg the walker still matches the full
    HUB_VERIFY_BARE_TARGETS set (protects check_hub_verify_retry), and the
    narrowed call matches list_repo_files only."""
    import ast

    from workflow_lint import _hub_verify_bare_hits

    tree = ast.parse(
        "from huggingface_hub import file_exists\n"
        "def f(api):\n"
        "    a = api.list_repo_tree('org/r', path_in_repo='p')\n"
        "    b = file_exists('org/r', 'x')\n"
        "    c = api.list_repo_files('org/r')\n"
    )
    default_patterns = sorted(p for _, p in _hub_verify_bare_hits(tree))
    assert default_patterns == [".list_repo_files(", ".list_repo_tree(", "file_exists("]
    narrowed = _hub_verify_bare_hits(tree, targets=frozenset({"list_repo_files"}))
    assert narrowed == [(5, ".list_repo_files(")]


def test_check_inline_round_duty_mirror_symbol_and_argparse_pin() -> None:
    """#1713 lost-update regression pin. Fails if a future whole-file
    snapshot of scripts/workflow_lint.py silently drops the
    check_inline_round_duty_mirror function OR unregisters the flag.

    Complements tests/test_workflow_lint_inline_round_duty_mirror.py by
    firing at import + argparse layer, so a delete of THAT test file
    still leaves this pin.
    """
    import contextlib
    import io

    from workflow_lint import check_inline_round_duty_mirror, main

    assert callable(check_inline_round_duty_mirror)

    buf = io.StringIO()
    rc = None
    with contextlib.redirect_stdout(buf):
        try:
            main(["--help"])
        except SystemExit as e:
            rc = e.code
    help_text = buf.getvalue()
    assert "--check-inline-round-duty-mirror" in help_text, (
        "flag missing from argparse; #1701 restoration incomplete"
    )
    assert rc == 0
