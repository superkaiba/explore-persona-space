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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    _iter_ask_target_files,
    _other_worktree_prefix,
    check_asks,
    check_autonomous_asks,
    check_script_references,
    check_wandb_required,
)

from explore_persona_space.workflow import load_workflow_yaml  # noqa: E402


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
