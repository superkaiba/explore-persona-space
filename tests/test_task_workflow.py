"""Tests for explore_persona_space.task_workflow.

Each test runs in a temporary directory that is set up as a fake repo
(git init + minimal layout). Git commits are NOT skipped — we want the
end-to-end behavior (git mv during set_status, etc.) under test — but
auto-push is disabled by leaving TASK_PY_AUTO_PUSH unset.
"""

# ruff: noqa: E501
# The fixture body strings below include long lines that mirror real
# clean-result content (Why-this-experiment Application/Decision lines
# carry ≥40 chars of substance and tend to exceed 100 cols). Reflowing
# them would change the markdown structure under test.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ─── Fake-repo fixture ─────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up tmp_path as a git repo and rebind task_workflow's globals to
    point at it. Returns the repo root path.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    # initial empty commit so HEAD exists
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    # Import lazily so we can monkeypatch module-level paths
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import importlib

    import explore_persona_space.task_workflow as tw

    importlib.reload(tw)

    monkeypatch.setattr(tw, "REPO", tmp_path)
    monkeypatch.setattr(tw, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tw, "REGISTRY_PATH", tmp_path / "tasks" / "REGISTRY.json")
    # Per-test lock dir to avoid cross-talk
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    return tmp_path, tw


def _git_log_count(repo: Path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


# ─── Smoke: import the module ──────────────────────────────────────────────


def test_module_imports():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    assert "proposed" in tw.STATUSES
    assert "completed" in tw.STATUSES
    assert tw.PARK_STATUS == "awaiting_promotion"


# ─── Frontmatter parsing ──────────────────────────────────────────────────


def test_frontmatter_roundtrip(fake_repo):
    _, tw = fake_repo
    text = "---\ntitle: Foo\nkind: experiment\ntags:\n  - a\n  - b\n---\nbody here\n"
    fm, body = tw._split_frontmatter(text)
    assert fm["title"] == "Foo"
    assert fm["kind"] == "experiment"
    assert fm["tags"] == ["a", "b"]
    assert body == "body here\n"
    rebuilt = tw._join_frontmatter(fm, body)
    fm2, body2 = tw._split_frontmatter(rebuilt)
    assert fm2 == fm
    assert body2 == body


def test_frontmatter_missing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    fm, body = tw._split_frontmatter("hello no frontmatter\n")
    assert fm == {}
    assert body == "hello no frontmatter\n"


# ─── create_task ──────────────────────────────────────────────────────────


def test_create_task_basic(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="My first task", body="Goal: do X")
    )
    assert new_id == 1
    task_dir = repo / "tasks" / "proposed" / "1"
    assert task_dir.is_dir()
    assert (task_dir / "body.md").exists()
    assert (task_dir / "events.jsonl").exists()
    assert (task_dir / "comments.jsonl").exists()
    assert (task_dir / "artifacts").is_dir()
    assert (task_dir / "plans").is_dir()
    # Frontmatter populated
    fm, body = tw._split_frontmatter((task_dir / "body.md").read_text())
    assert fm["title"] == "My first task"
    assert fm["kind"] == "experiment"
    assert "Goal: do X" in body
    # Registry updated
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["highest_id"] == 1
    assert reg["tasks"]["1"]["path"] == "tasks/proposed/1"
    # Created event present
    events = tw.list_events(1)
    assert events[0]["kind"] == "epm:created"


def test_create_task_increments_id(fake_repo):
    _, tw = fake_repo
    a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    c = tw.create_task(tw.NewTaskRequest(kind="experiment", title="C"))
    assert (a, b, c) == (1, 2, 3)


def test_create_task_with_parent(fake_repo):
    _, tw = fake_repo
    parent = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Parent"))
    child = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Child", parent_id=parent))
    task = tw.get_task(child)
    assert task["frontmatter"]["parent_id"] == parent


# ─── Status transitions ──────────────────────────────────────────────────


def test_set_status_moves_folder(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    old = repo / "tasks" / "proposed" / str(new_id)
    assert old.is_dir()
    tw.set_status(new_id, "running")
    new = repo / "tasks" / "running" / str(new_id)
    assert not old.exists()
    assert new.is_dir()
    # Registry updated
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["path"] == f"tasks/running/{new_id}"
    # Status-changed event posted
    events = tw.list_events(new_id)
    assert events[-1]["kind"] == "epm:status-changed"
    assert events[-1]["from"] == "proposed"
    assert events[-1]["to"] == "running"


def test_set_status_invalid_raises(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.set_status(new_id, "not-a-status")


def test_set_status_idempotent(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    n_before = len(tw.list_events(new_id))
    tw.set_status(new_id, "proposed")  # no-op
    n_after = len(tw.list_events(new_id))
    assert n_after == n_before  # no new event when already there


# ─── post_event ──────────────────────────────────────────────────────────


def test_post_event_appends(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan", by="planner", note="plan v1 written")
    tw.post_event(new_id, "epm:plan-approved", by="user")
    events = tw.list_events(new_id)
    assert [e["kind"] for e in events] == ["epm:created", "epm:plan", "epm:plan-approved"]
    assert events[1]["note"] == "plan v1 written"


def test_post_event_oversize_note_raises(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.post_event(new_id, "epm:huge", note="x" * (tw.EVENT_NOTE_MAX + 1))


def test_latest_event(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan")
    tw.post_event(new_id, "epm:run-launched")
    latest = tw.latest_event(new_id)
    assert latest["kind"] == "epm:run-launched"
    # Filter by prefix
    plan_only = tw.latest_event(new_id, prefix="epm:plan")
    assert plan_only["kind"] == "epm:plan"


# ─── Body / title / tags ────────────────────────────────────────────────


def test_set_body_preserves_frontmatter(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    tw.set_body(new_id, "new body")
    fm, body = tw._split_frontmatter(
        (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    )
    assert fm["title"] == "X"
    assert body == "new body\n"


def test_set_body_snapshot_creates_original(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    tw.set_body(new_id, "clean-result content", snapshot_original=True)
    orig = repo / "tasks" / "proposed" / str(new_id) / "original-body.md"
    assert orig.exists()
    assert "old body" in orig.read_text()


def test_set_title_updates_registry(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Old"))
    tw.set_title(new_id, "New title")
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["title"] == "New title"


def test_add_remove_tag(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.add_tag(new_id, "qwen-7b")
    tw.add_tag(new_id, "qwen-7b")  # dedup
    tw.add_tag(new_id, "lang-inv")
    task = tw.get_task(new_id)
    assert task["frontmatter"]["tags"] == ["qwen-7b", "lang-inv"]
    tw.remove_tag(new_id, "qwen-7b")
    task = tw.get_task(new_id)
    assert task["frontmatter"]["tags"] == ["lang-inv"]


# ─── Plans ────────────────────────────────────────────────────────────────


def test_new_plan_version_versions_and_symlinks(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    v1 = tw.new_plan_version(new_id, "plan v1 content")
    v2 = tw.new_plan_version(new_id, "plan v2 content")
    v3 = tw.new_plan_version(new_id, "plan v3 content")
    assert (v1, v2, v3) == (1, 2, 3)
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    assert (plans_dir / "v1.md").read_text().strip() == "plan v1 content"
    assert (plans_dir / "v3.md").read_text().strip() == "plan v3 content"
    # Symlink points to latest
    assert (plans_dir / "plan.md").is_symlink()
    assert (plans_dir / "plan.md").resolve() == (plans_dir / "v3.md").resolve()


# ─── Promotion ───────────────────────────────────────────────────────────


def test_promote_requires_awaiting_promotion(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(RuntimeError):
        tw.promote(new_id, "useful")


def test_promote_useful(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.set_status(new_id, "awaiting_promotion")
    new_path = tw.promote(new_id, "useful")
    assert "completed" in str(new_path)
    task = tw.get_task(new_id)
    assert task["status"] == "completed"
    assert task["frontmatter"]["classification"] == "useful"
    # Both epm:promoted and epm:status-changed (to completed) appended
    kinds = [e["kind"] for e in tw.list_events(new_id)]
    assert "epm:promoted" in kinds
    assert kinds[-1] == "epm:status-changed"


def test_promote_invalid_verdict(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.set_status(new_id, "awaiting_promotion")
    with pytest.raises(ValueError):
        tw.promote(new_id, "maybe")


# ─── Queries ──────────────────────────────────────────────────────────────


def test_list_by_status(fake_repo):
    _, tw = fake_repo
    a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    tw.set_status(a, "running")
    rows = tw.list_by_status("proposed")
    assert {r["id"] for r in rows} == {b}
    rows = tw.list_by_status("running")
    assert {r["id"] for r in rows} == {a}


def test_find_task_path(fake_repo):
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    path = tw.find_task_path(new_id)
    assert path == repo / "tasks" / "proposed" / str(new_id)


def test_find_task_path_missing(fake_repo):
    _, tw = fake_repo
    with pytest.raises(FileNotFoundError):
        tw.find_task_path(99999)


# ─── Audit ────────────────────────────────────────────────────────────────


def test_audit_clean(fake_repo):
    _, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="B"))
    assert tw.audit() == []


def test_audit_detects_orphan_dir(fake_repo):
    repo, tw = fake_repo
    tw.create_task(tw.NewTaskRequest(kind="experiment", title="A"))
    # Create a task folder on disk WITHOUT registering
    orphan = repo / "tasks" / "proposed" / "9999"
    orphan.mkdir(parents=True)
    (orphan / "body.md").write_text("---\ntitle: orphan\n---\n")
    problems = tw.audit()
    assert any("9999" in p for p in problems)


# ─── Comments ────────────────────────────────────────────────────────────


def test_append_comment_sequential_ids(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    c1 = tw.append_comment(new_id, author="mentor", kind="question", body="why X?")
    c2 = tw.append_comment(
        new_id, author="claude", kind="answer", body="because Y", in_reply_to=c1["id"]
    )
    assert c1["id"] == "c001"
    assert c2["id"] == "c002"
    assert c2["in_reply_to"] == "c001"


def test_append_comment_unknown_kind(fake_repo):
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    with pytest.raises(ValueError):
        tw.append_comment(new_id, author="x", kind="unknown-kind", body="...")


# ─── Locking — multiple ops serialize without breaking ──────────────────


def test_back_to_back_mutations_commit_cleanly(fake_repo):
    repo, tw = fake_repo
    n_commits_before = _git_log_count(repo)
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    tw.post_event(new_id, "epm:plan", by="planner")
    tw.set_status(new_id, "planning")
    tw.set_title(new_id, "X renamed")
    tw.set_status(new_id, "awaiting_promotion")
    tw.promote(new_id, "useful")
    # Should have N commits (one per op) on top of the initial commit
    n_commits_after = _git_log_count(repo)
    assert n_commits_after > n_commits_before
    # Final state consistent
    assert tw.audit() == []
    final = tw.get_task(new_id)
    assert final["status"] == "completed"
    assert final["frontmatter"]["classification"] == "useful"


# ─── task.py migrate-body subcommand ──────────────────────────────────────


def _move_to_awaiting(tw, task_id: int) -> None:
    """Helper to push a task through to awaiting_promotion."""
    tw.set_status(task_id, "awaiting_promotion")


# Minimal canonical PASS body — useful as a fixture target. Every check
# (title, four H2s, TL;DR labels, hero image, caption, confidence, repro
# subgroups + URL + sentinel scrub, cherry-picked, qual-data link) is
# satisfied.
CANONICAL_PASS_BODY = """\
# Toy clean-result body (LOW confidence)

## Why this experiment

- **Application:** predict — the fixture stands in for a forecasting experiment about model behavior.
- **Decision this changes:** whether the migration tooling can recognize a fully-conformant body and treat it as PASS.
- **Expected outcome + branches:** classify_body returns PASS (the migration is a no-op) or returns something else (the migration logic has regressed).
- **What gets cut if we run this:** nothing — this is a smoke-test fixture, not a real experiment.

## TL;DR

- **Motivation:** I wanted a smoke-test fixture.
- **What I ran:** I wrote a minimal markdown body and ran verify_task_body.
- **Results:** The fixture passes all twelve checks.
- **Next steps:** Use this fixture in migration tests.

## Figure

![Hero figure placeholder](figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

## Details

The full Details section explaining what was done and how.

Confidence: LOW — based on toy data only, not a real experiment so does not generalize.

## Reproducibility

**Artifacts:** n/a

**Compute:** n/a

**Code:** n/a
"""


# Conformant-but-failing fixture: four-H2 shape, but Reproducibility is
# missing its three boldface subgroup labels and uses H3 instead.
CONFORMANT_FAILING_H3_REPRO_BODY = """\
# Conformant-failing body using H3 repro subgroups (LOW confidence)

## Why this experiment

- **Application:** predict — fixture stands in for a forecasting experiment about model behavior.
- **Decision this changes:** whether the H3-Repro remediation patch successfully promotes labels and re-PASSes the verifier.
- **Expected outcome + branches:** remediation flips Artifacts/Compute/Code to bold and the body PASSes, or the body stays FAIL and the test surfaces a real regression.
- **What gets cut if we run this:** nothing — this is a smoke-test fixture, not a real experiment.

## TL;DR

- **Motivation:** toy motivation.
- **What I ran:** toy run description goes here.
- **Results:** toy results paragraph explaining what we saw.
- **Next steps:** none in particular.

## Figure

![Hero figure placeholder](figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

## Details

Confidence: LOW — based on toy data only, not generalizable, no real experiment.

## Reproducibility

### Artifacts

| field | value |
|---|---|
| Model | n/a |

### Compute

| field | value |
|---|---|
| Hours | 0 |

### Code

| field | value |
|---|---|
| Script | n/a |
"""


# v4-legacy fixture: <details open><summary>## H2</summary> wrappers around
# TL;DR / Summary / Details / Source issues. No H1, no Figure, no Repro.
V4_LEGACY_BODY = """\
<details open>
<summary>

## TL;DR

</summary>

- Toy bullet one without label.
- Toy bullet two.

</details>

<details open>
<summary>

## Summary

</summary>

- **Motivation:** ...
- **Experiment:** ...
- **Results:** ...
- **Confidence: LOW** — toy.

</details>

## Details

Body of details here. Confidence: LOW — toy fixture; the conversion test only
exercises shape changes, not content surgery.

<details open>
<summary>

## Source issues

</summary>

Refs go here.

</details>
"""


def _make_task_at_awaiting(
    tw,
    *,
    title: str,
    body: str,
    task_id_hint: int | None = None,
    application: str | None = "predict",
) -> int:
    """Create a task and push it to awaiting_promotion. Returns the id.

    Post-create, patches ``application: <application>`` into the task's
    frontmatter so check #12 (Why-this-experiment gate) reads a real
    value. The default ``"predict"`` matches the body fixtures' Application
    line. Pass ``application=None`` to leave the frontmatter untouched
    (e.g. to exercise the legacy-sentinel skip path).
    """
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title=title, body=body))
    if task_id_hint is not None:
        assert new_id == task_id_hint, f"id drift: got {new_id}, expected {task_id_hint}"
    if application is not None:
        body_path = tw.find_task_path(new_id) / "body.md"
        fm, body_only = tw._read_body(body_path)
        fm["application"] = application
        tw._write_body(body_path, fm, body_only)
    tw.set_status(new_id, "awaiting_promotion")
    return new_id


def test_migrate_body_classify_pass(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    # CANONICAL_PASS_BODY is a body string — classify_body needs the real
    # frontmatter (with `application:`) to fully PASS the eleven body
    # checks plus check #12 (Why-this-experiment gate).
    assert classify_body(CANONICAL_PASS_BODY, fm={"application": "predict"}) == BodyClass.PASS


def test_migrate_body_classify_v4_legacy(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    assert classify_body(V4_LEGACY_BODY) == BodyClass.V4_LEGACY


def test_migrate_body_classify_conformant_failing(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    assert classify_body(CONFORMANT_FAILING_H3_REPRO_BODY) == BodyClass.CONFORMANT_FAILING


def test_migrate_body_classify_legacy_html(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    legacy = "<!-- legacy-sagan-card -->\n<section><h1>foo</h1></section>\n"
    assert classify_body(legacy) == BodyClass.LEGACY_HTML


def test_migrate_body_conformant_failing_remediation(fake_repo):
    """A four-H2 body with H3 Repro subgroups gets the labels promoted to
    bold and ends up passing verify_task_body.
    """
    _, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(
        tw, title="Conformant-failing", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )

    result = migrate_one(new_id, apply=True)
    assert result.classification == BodyClass.CONFORMANT_FAILING
    assert result.verify_after == "PASS"
    assert not result.needs_user

    # Body now has bold subgroup labels, no more H3 Artifacts/Compute/Code.
    task = tw.get_task(new_id)
    body = task["body"]
    assert "**Artifacts:**" in body
    assert "**Compute:**" in body
    assert "**Code:**" in body
    assert "### Artifacts" not in body
    assert "### Compute" not in body
    assert "### Code" not in body


def test_migrate_body_dry_run_does_not_write(fake_repo):
    """`--dry-run` (apply=False) must not modify body.md or commit."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(
        tw, title="Dry-run check", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )

    body_path = repo / "tasks" / "awaiting_promotion" / str(new_id) / "body.md"
    before_text = body_path.read_text()
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=False)
    assert result.verify_after.startswith("DRY-")
    assert body_path.read_text() == before_text
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_idempotency(fake_repo):
    """Applying the patch twice produces zero git diff after the second apply."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(tw, title="Idempotency", body=CONFORMANT_FAILING_H3_REPRO_BODY)

    result1 = migrate_one(new_id, apply=True)
    assert result1.verify_after == "PASS"
    n_commits_after_first = _git_log_count(repo)

    # Second apply should be a no-op.
    result2 = migrate_one(new_id, apply=True)
    # The body should already PASS (classified as PASS, no further actions).
    from explore_persona_space.task_workflow_migrate import BodyClass

    assert result2.classification == BodyClass.PASS
    assert _git_log_count(repo) == n_commits_after_first


def test_migrate_body_v4_to_new_strips_details_wrappers(fake_repo):
    """v4-legacy bodies that fail post-patch (e.g. missing TL;DR labels) are
    LEFT UNCHANGED (per plan §3 Phase E step 5), but the shape conversion is
    visible in dry-run output via the action log.
    """
    _, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import (
        BodyClass,
        convert_v4_to_target,
        migrate_one,
    )

    new_id = _make_task_at_awaiting(tw, title="v4 fixture (LOW confidence)", body=V4_LEGACY_BODY)

    # Drive the conversion through migrate_one in dry-run mode.
    result = migrate_one(new_id, apply=False)
    assert result.classification == BodyClass.V4_LEGACY
    # The body almost certainly still fails after mechanical conversion
    # (TL;DR labels are missing, no Figure image). Plan §3 Phase E step 5 says
    # leave the body alone in that case.
    assert result.needs_user

    # But the standalone converter SHOULD have done its mechanical work —
    # exercise it directly to confirm the strip + inject logic.
    converted, actions = convert_v4_to_target(V4_LEGACY_BODY, title="v4 fixture (LOW confidence)")
    assert "<details open>" not in converted
    assert "## Figure" in converted
    assert "## Reproducibility" in converted
    assert "# v4 fixture (LOW confidence)" in converted
    # Action log surfaces every patch step.
    assert any("toggle wrapper" in a for a in actions)
    assert any("H1" in a or "title" in a for a in actions)
    assert any("Reproducibility" in a for a in actions)


def test_migrate_body_v4_legacy_unchanged_when_post_patch_still_fails(fake_repo):
    """Per plan: if mechanical patch insufficient, body is reverted to original."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(tw, title="Untouched-fail", body=V4_LEGACY_BODY)

    body_path = repo / "tasks" / "awaiting_promotion" / str(new_id) / "body.md"
    before_text = body_path.read_text()
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=True)
    assert result.needs_user
    assert result.verify_after == "FAIL"
    # Body is unchanged on disk, no extra commits.
    assert body_path.read_text() == before_text
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_pass_body_is_noop(fake_repo):
    """A PASS body produces no actions and no commits."""
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(tw, title="Already PASS", body=CANONICAL_PASS_BODY)
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=True)
    assert result.classification == BodyClass.PASS
    assert result.verify_after == "PASS"
    assert _git_log_count(repo) == n_commits_before


def test_migrate_body_remediate_repro_subgroups_idempotent(fake_repo):
    """remediate_repro_subgroups is idempotent — re-running on already-promoted
    labels produces no change. Covers the H3-promotion case AND the
    `**Label.**` (period) punctuation-fix case.
    """
    _, _ = fake_repo
    from explore_persona_space.task_workflow_migrate import remediate_repro_subgroups

    # Case (a): H3 headings
    body_h3 = (
        "## Reproducibility\n\n### Artifacts\n\nfoo\n\n### Compute\n\nbar\n\n### Code\n\nbaz\n"
    )
    out1, _actions1 = remediate_repro_subgroups(body_h3)
    out2, actions2 = remediate_repro_subgroups(out1)
    assert "**Artifacts:**" in out1
    assert "**Compute:**" in out1
    assert "**Code:**" in out1
    assert actions2 == []
    assert out1 == out2

    # Case (b): `**Label.**` punctuation
    body_dot = (
        "## Reproducibility\n\n**Artifacts.**\n\nfoo\n\n**Compute.**\n\nbar\n\n**Code.**\n\nbaz\n"
    )
    out3, _actions3 = remediate_repro_subgroups(body_dot)
    out4, actions4 = remediate_repro_subgroups(out3)
    assert "**Artifacts:**" in out3
    assert "**Artifacts.**" not in out3
    assert actions4 == []
    assert out3 == out4


def test_migrate_body_report_classification(fake_repo):
    """`task.py migrate-body --report` enumerates every awaiting_promotion task."""
    _, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import (
        BodyClass,
        list_awaiting_promotion_ids,
        migrate_one,
    )

    # Three bodies in three classes.
    a = _make_task_at_awaiting(tw, title="A (LOW confidence)", body=CANONICAL_PASS_BODY)
    b = _make_task_at_awaiting(
        tw, title="B (LOW confidence)", body=CONFORMANT_FAILING_H3_REPRO_BODY
    )
    c = _make_task_at_awaiting(tw, title="C (LOW confidence)", body=V4_LEGACY_BODY)

    ids = list_awaiting_promotion_ids()
    assert set(ids) >= {a, b, c}

    classes = {tid: migrate_one(tid, apply=False).classification for tid in (a, b, c)}
    assert classes[a] == BodyClass.PASS
    assert classes[b] == BodyClass.CONFORMANT_FAILING
    assert classes[c] == BodyClass.V4_LEGACY
