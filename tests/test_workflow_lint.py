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
    BATCH_JUDGE_LEGACY_ALLOWLIST,
    _iter_ask_target_files,
    _other_worktree_prefix,
    check_agent_model_pins,
    check_asks,
    check_autonomous_asks,
    check_batch_judge_client,
    check_compute_shape_review_lens,
    check_dispatcher_cvd_pin,
    check_gate_ids_unique,
    check_heredoc_dotenv,
    check_lessons_index,
    check_long_loop_restartability_review_lens,
    check_marker_registry,
    check_no_literal_round_marker_versions,
    check_no_workflow_improver_spawn,
    check_pipe_python,
    check_script_references,
    check_skill_references,
    check_smoke_architecture_review_lens,
    check_smoke_output_hygiene,
    check_stale_label_disposition_clause,
    check_upload_as_file,
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


def test_workflow_lint_pipe_python_bundled_in_no_flags():
    """`check_pipe_python` is wired into the no-flags default run (bundled,
    same policy as `check_heredoc_dotenv`): a bare `workflow_lint.py`
    invocation exercises it. The committed tree is clean, so the no-flags
    run exits 0 — and a planted offender in a tmp scripts dir would be
    caught by the function test above; here we assert the bundling holds
    by confirming the flag is among the no-flags checks via a clean exit."""
    result = _run()
    assert result.returncode == 0, (
        f"workflow_lint (no flags) failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
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
    rows = "\n".join(
        f"- **{n}** ([`.claude/rules/{n}.md`]({n}.md)) — fires when: x." for n in indexed_names
    )
    (rules_dir / "LESSONS.md").write_text(f"# LESSONS\n\n## Rules\n\n{rows}\n", encoding="utf-8")


def test_check_lessons_index_fails_on_missing_row(tmp_path):
    rules = tmp_path / ".claude" / "rules"
    # rule 'gamma' exists but is NOT indexed -> FAIL
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
    from workflow_lint import _LESSONS_MAX_BYTES

    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "alpha.md").write_text("# alpha\n", encoding="utf-8")
    rows = "- **alpha** ([`.claude/rules/alpha.md`](alpha.md)) — fires when: x.\n"
    # Pad with prose so the index breaches the byte cap regardless of its value.
    padding = "x" * (_LESSONS_MAX_BYTES + 100)
    (rules / "LESSONS.md").write_text(
        f"# LESSONS\n\n## Rules\n\n{rows}\n\n{padding}\n",
        encoding="utf-8",
    )
    errs = check_lessons_index(repo_root=tmp_path)
    assert errs and any("leanness cap" in e for e in errs)


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

_VM_CAP_PREFIX = "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8"

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
