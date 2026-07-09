"""Tests for the workflow-fix dedup + recursion-guard predicates (#678).

Workflow-surface fixes are filed as ``kind: infra`` tasks and implemented by a
background ``/issue <N> --auto`` session (the ``workflow-improver`` subagent
auto-spawn was retired by #678). Two pieces of orchestrator logic gate that
routing, both backed by read-only predicates in ``task_workflow``:

- **DEDUP** — ``is_open_workflow_fix_task(target_file, fingerprint)`` so the
  SAME bug on the SAME file is not double-filed, while a DISTINCT bug on the
  same hot file (different fingerprint) STILL files its own task and gets its
  own plan review (the A1 grain fix — the load-bearing #678 change);
  cross-channel since #1180 — ``daily-fix:``-titled /daily filings count.
- **RECURSION GUARD** — ``is_workflow_fix_session(N)`` so a workflow-fix
  session never auto-files MORE workflow-fix tasks for its own findings (the
  "unbounded fan-out" failure mode).

These tests exercise the real ``create_task`` ``--body-file``-verbatim
round-trip (NOT mocks) so the body ``## Provenance`` grep fallback is genuinely
covered. The ``fake_repo`` fixture mirrors ``tests/test_task_workflow.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ─── Fake-repo fixture (mirrors tests/test_task_workflow.py) ────────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo with task_workflow's resolvers rebound to it."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    return tmp_path, tw


# A realistic body-file template — the ``## Provenance`` block carries the
# ``workflow_fix_target:`` + ``fingerprint:`` lines VERBATIM (as the orchestrator
# fills them at file-time), exercising the grep fallback.
def _wf_fix_body(target_file: str, fingerprint: str, proposed_change: str) -> str:
    return (
        "## Goal\n\n"
        f"{proposed_change}\n\n"
        "## Scope / surfaces\n\n"
        f"- Primary target: `{target_file}`\n\n"
        "## Provenance\n\n"
        f"- workflow_fix_target: {target_file}\n"
        f"- fingerprint: {fingerprint}\n"
    )


def _file_wf_fix_task(tw, target_file, fingerprint, *, proposed_change="Fix the gate.", tags=None):
    """File a kind:infra workflow-fix task at proposed (a non-terminal status)."""
    if tags is None:
        tags = ["wf-fix", f"wf-fix-fp:{fingerprint}"]
    return tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title=f"workflow-fix: {proposed_change[:50]}",
            body=_wf_fix_body(target_file, fingerprint, proposed_change),
            tags=tags,
        )
    )


# ─── Fingerprint stability + distinctness ──────────────────────────────────


def test_fingerprint_stable_across_reformatting():
    """Reformatting a candidate's prose (case / whitespace / trailing punct)
    yields the SAME fingerprint — so a re-raise of the same bug is deduped."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    assert tw.wf_fix_fingerprint("fix the gate", "the gate is wrong") == fp
    assert tw.wf_fix_fingerprint("  Fix  the   gate!  ", "  THE GATE IS WRONG  ") == fp


def test_fingerprint_differs_for_distinct_bugs():
    """Two distinct (proposed_change, bug_observed) pairs -> different fps."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    fp_a = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    fp_b = tw.wf_fix_fingerprint("Re-point the marker docs.", "The marker doc is stale")
    assert fp_a != fp_b


# ─── Dedup predicate (A1 grain: target_file + fingerprint) ─────────────────


def test_open_same_file_same_fp_is_duplicate(fake_repo):
    """SAME file + SAME fingerprint at a non-terminal status -> deduped (id returned)."""
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    tid = _file_wf_fix_task(tw, target, fp)
    assert tw.is_open_workflow_fix_task(target, fp) == tid


def test_open_same_file_DIFFERENT_fp_is_NOT_duplicate(fake_repo):
    """A1 regression guard: SAME file, DIFFERENT fingerprint -> NOT a duplicate.

    A distinct bug on the same hot file must file its own task and get its own
    plan review. v1's file-only dedup would have silently dropped it.
    """
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp_existing = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    fp_other = tw.wf_fix_fingerprint("Re-point the marker docs.", "The marker doc is stale")
    _file_wf_fix_task(tw, target, fp_existing)
    assert tw.is_open_workflow_fix_task(target, fp_other) is None


def test_file_only_match_when_fingerprint_none(fake_repo):
    """With fingerprint=None, any open fix on the file matches (coarse mode)."""
    _, tw = fake_repo
    target = "CLAUDE.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    tid = _file_wf_fix_task(tw, target, fp)
    assert tw.is_open_workflow_fix_task(target, None) == tid
    # A different file is never matched.
    assert tw.is_open_workflow_fix_task("scripts/workflow_lint.py", None) is None


def test_terminal_status_does_not_block(fake_repo):
    """A completed (and an archived) workflow-fix task on the same (file, fp)
    does NOT block a re-raise — a closed fix is not an open duplicate."""
    _, tw = fake_repo
    target = ".claude/rules/workflow-fix-on-bug.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")

    completed = _file_wf_fix_task(tw, target, fp)
    tw.set_status(completed, "completed")
    assert tw.is_open_workflow_fix_task(target, fp) is None

    archived = _file_wf_fix_task(tw, target, fp)
    tw.set_status(archived, "archived")
    assert tw.is_open_workflow_fix_task(target, fp) is None


def test_fingerprint_tag_alone_matches_without_provenance_fp_line(fake_repo):
    """The fingerprint half of the key is satisfied by the wf-fix-fp tag even if
    the body carries no `fingerprint:` Provenance line (tag is the primary key)."""
    _, tw = fake_repo
    target = ".claude/workflow.yaml"
    fp = tw.wf_fix_fingerprint("Add a marker.", "Marker missing")
    # Body has the workflow_fix_target line but NO `fingerprint:` line; the tag carries it.
    body = f"## Goal\n\nAdd a marker.\n\n## Provenance\n\n- workflow_fix_target: {target}\n"
    tid = tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="workflow-fix: Add a marker",
            body=body,
            tags=["wf-fix", f"wf-fix-fp:{fp}"],
        )
    )
    assert tw.is_open_workflow_fix_task(target, fp) == tid


def test_non_infra_task_is_not_a_workflow_fix(fake_repo):
    """A kind:experiment task with a workflow-fix-shaped title/body is ignored."""
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title=f"workflow-fix: nope {target}",
            body=_wf_fix_body(target, fp, "Fix the gate."),
            tags=[f"wf-fix-fp:{fp}"],
        )
    )
    assert tw.is_open_workflow_fix_task(target, fp) is None


# ─── Cross-channel dedup: daily-fix: titles (#1180) ─────────────────────────


def test_daily_fix_titled_open_task_is_duplicate(fake_repo):
    """#1180: a /daily route-2 filing (title `daily-fix:`, same Provenance +
    fp tag) IS visible to the orchestrator-channel dedup — fine AND coarse mode."""
    _, tw = fake_repo
    target = "scripts/daily_drive_filings.py"
    fp = tw.wf_fix_fingerprint("Widen the title prefix.", "Predicate blind to daily-fix titles")
    tid = tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="daily-fix: dedup predicate blind to daily-fix titles",
            body=_wf_fix_body(target, fp, "Widen the title prefix."),
            tags=["wf-fix", f"wf-fix-fp:{fp}", "daily-auto-filed"],
        )
    )
    assert tw.is_open_workflow_fix_task(target, fp) == tid
    assert tw.is_open_workflow_fix_task(target, None) == tid


def test_daily_fix_same_file_DIFFERENT_fp_is_NOT_duplicate(fake_repo):
    """The widening must not weaken the A1 grain across channels: a daily-filed
    task on the same file with a DIFFERENT fingerprint is NOT a duplicate."""
    _, tw = fake_repo
    target = "scripts/daily_drive_filings.py"
    fp_existing = tw.wf_fix_fingerprint("Widen the title prefix.", "Predicate blind to daily-fix")
    fp_other = tw.wf_fix_fingerprint("A different daily bug.", "Something else broke")
    tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="daily-fix: dedup predicate blind to daily-fix titles",
            body=_wf_fix_body(target, fp_existing, "Widen the title prefix."),
            tags=["wf-fix", f"wf-fix-fp:{fp_existing}"],
        )
    )
    assert tw.is_open_workflow_fix_task(target, fp_other) is None


def test_unlisted_title_prefix_is_still_ignored(fake_repo):
    """The predicate stays prefix-bound: an infra task with a wf-fix-shaped
    body + tags but a title outside WF_FIX_TITLE_PREFIXES does not match."""
    _, tw = fake_repo
    target = "scripts/daily_drive_filings.py"
    fp = tw.wf_fix_fingerprint("Widen the title prefix.", "Predicate blind to daily-fix")
    tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="infra: not a wf-fix filing",
            body=_wf_fix_body(target, fp, "Widen the title prefix."),
            tags=["wf-fix", f"wf-fix-fp:{fp}"],
        )
    )
    assert tw.is_open_workflow_fix_task(target, fp) is None


# ─── create_task body-verbatim round-trip (grep fallback) ──────────────────


def test_create_task_writes_provenance_line_verbatim(fake_repo):
    """create_task writes the body VERBATIM, so the `workflow_fix_target:`
    Provenance line is present + grep-findable (the dedup fallback)."""
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    tid = _file_wf_fix_task(tw, target, fp)
    body_text = (tw.find_task_path(tid) / "body.md").read_text()
    assert f"workflow_fix_target: {target}" in body_text
    assert f"fingerprint: {fp}" in body_text


# ─── Recursion-guard predicate (the "unbounded fan-out" invariant) ─────────


def test_is_workflow_fix_session_true_on_provenance_line(fake_repo):
    """A task whose body carries `workflow_fix_target:` IS a workflow-fix
    session; a plain infra task is NOT."""
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    wf_tid = _file_wf_fix_task(tw, target, fp)
    assert tw.is_workflow_fix_session(wf_tid) is True

    plain = tw.create_task(
        tw.NewTaskRequest(kind="infra", title="plain infra task", body="## Goal\n\nDo a thing.\n")
    )
    assert tw.is_workflow_fix_session(plain) is False


def test_is_workflow_fix_session_true_on_driver_injected_body(fake_repo):
    """#1173 cross-predicate pin: a body shaped by the daily filing driver's
    ``ensure_wf_fix_provenance`` injection satisfies the REAL predicates —
    ``is_workflow_fix_session`` (recursion guard) AND, once the title prefix +
    tags the dedup predicate requires are applied (mirroring _file_wf_fix_task),
    ``is_open_workflow_fix_task`` — not just a test-invented substring."""
    _, tw = fake_repo
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from daily_drive_filings import ensure_wf_fix_provenance

    target = ".claude/skills/daily/SKILL.md"
    fp = tw.wf_fix_fingerprint("change text for fix-a", "bug text for fix-a")
    body, changed = ensure_wf_fix_provenance("## Goal\n\nx\n", target, fp)
    assert changed is True
    tid = tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="workflow-fix: driver-injected provenance",
            body=body,
            tags=["wf-fix", f"wf-fix-fp:{fp}"],
        )
    )
    assert tw.is_workflow_fix_session(tid) is True
    assert tw.is_open_workflow_fix_task(target, fp) == tid


# ─── Primary dedup key: title prefix round-trips through the registry ───────


def test_title_prefix_roundtrips_view_json(fake_repo):
    """After create_task(title="workflow-fix: X"), the title surfaces through the
    REGISTRY snapshot + frontmatter (the primary, view --json-visible dedup key)."""
    _, tw = fake_repo
    target = ".claude/skills/issue/SKILL.md"
    fp = tw.wf_fix_fingerprint("Fix the gate.", "The gate is wrong")
    tid = _file_wf_fix_task(tw, target, fp, proposed_change="Fix the gate")

    # Registry snapshot (the dashboard list-view + dedup pre-filter read it).
    reg = tw._load_registry()
    assert reg["tasks"][str(tid)]["title"].startswith("workflow-fix:")
    # Frontmatter on body.md agrees.
    fm, _ = tw._read_body(tw.find_task_path(tid) / "body.md")
    assert fm["title"].startswith("workflow-fix:")
