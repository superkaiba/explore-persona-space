"""Tests for explore_persona_space.task_workflow.

Each test runs in a temporary directory that is set up as a fake repo
(git init + minimal layout). Git commits are NOT skipped — we want the
end-to-end behavior (git mv during set_status, etc.) under test — but
auto-push is disabled by leaving TASK_PY_AUTO_PUSH unset.
"""

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
    """Set up tmp_path as a git repo and rebind task_workflow's resolver to
    point at it. Returns the repo root path.

    The 2026-05-25 worktree-staleness fix replaced module-level
    ``REPO``/``TASKS_DIR``/``REGISTRY_PATH`` constants with the function
    accessors ``repo_root()`` / ``tasks_dir()`` / ``registry_path()``
    (with a PEP-562 attribute shim for backward compatibility — see
    ``task_workflow.py`` header). Tests now monkeypatch the FUNCTIONS,
    not the attributes, so every in-module call site picks up the tmp
    repo. The branch guard inside the real ``repo_root()`` would
    otherwise refuse to resolve from a non-``main`` development branch.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    # initial empty commit so HEAD exists
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    # Drop any cached resolution from a prior test so our overrides win.
    tw.invalidate_cache()

    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
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
    # Same-issue follow-up rounds hold this status (un-phantomed 2026-06-10);
    # it is neither terminal nor the park status.
    assert "followups_running" in tw.STATUSES
    assert "followups_running" not in tw.TERMINAL_STATUSES
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


def test_set_status_commits_both_sides_of_move(fake_repo):
    """Regression: ``set_status`` must commit BOTH the source-path deletion
    AND the destination-path addition of its ``git mv``, so the index is
    clean afterward. Otherwise the source-path deletion lingers as a
    staged change and gets swept into the next unrelated ``git commit``.

    Incident: 2026-05-24, tasks 382/383 source-side deletions in
    ``tasks/proposed/`` were left staged by ``set_status proposed →
    planning`` and got swept into commit 49e49f4a (an unrelated
    ``.claude/agents/planner.md`` edit), under a misleading commit
    message.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Move-me"))
    tw.set_status(new_id, "planning")

    # After set_status, the index must be CLEAN — no orphan staged
    # deletion for the source path.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert diff_cached == "", (
        f"set_status left orphan staged changes in the index: {diff_cached!r}. "
        f"The source-side deletion of `git mv` was not included in the commit."
    )

    # And the HEAD commit must record BOTH sides of the move.
    show = subprocess.run(
        ["git", "show", "HEAD", "--name-status", "--format="],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    # Expect a delete row (D) for the source and either A (added) or R
    # (renamed) for the destination.
    deleted = [line for line in show if line.startswith("D\t")]
    added_or_renamed = [line for line in show if line.startswith(("A\t", "R"))]
    assert deleted, f"set_status commit missing source deletion: {show}"
    assert added_or_renamed, f"set_status commit missing destination addition: {show}"


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


def test_post_event_default_version_auto_increments_per_kind(fake_repo):
    """Omitted version = max(existing for this kind)+1, per kind (#480).

    Two defaulted posts of the same kind must land v1 then v2 — never v1
    twice — so highest-version-per-kind resume resolution stays correct.
    A second kind starts independently at v1.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    first = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    second = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    other_kind = tw.post_event(new_id, "epm:interpretation", by="analyzer")
    assert first["version"] == 1
    assert second["version"] == 2
    assert other_kind["version"] == 1


def test_post_event_explicit_version_wins_and_seeds_default(fake_repo):
    """An explicit version is respected verbatim (even if lower than the
    current max), and a later defaulted post resumes from the true max —
    mirroring new_plan_version's max+1 (not count+1) semantics.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    explicit = tw.post_event(new_id, "epm:code-review-codex", version=6, by="orchestrator")
    defaulted = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    lower_explicit = tw.post_event(new_id, "epm:code-review-codex", version=3, by="orchestrator")
    after_lower = tw.post_event(new_id, "epm:code-review-codex", by="orchestrator")
    assert explicit["version"] == 6
    assert defaulted["version"] == 7
    assert lower_explicit["version"] == 3
    assert after_lower["version"] == 8


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


# ─── set_body: duplicate-frontmatter strip ─────────────────────────────────
#
# Regression: task #389 (2026-05-26) — the analyzer wrote draft body files
# carrying frontmatter and passed them through `task.py set-body`; the
# canonical frontmatter prepended on top of the caller's frontmatter, and
# body.md ended up with TWO `---...---` blocks. The dashboard parsed the
# first as the header card and rendered the second as literal YAML at the
# top of the visible body. `set_body()` now strips leading frontmatter
# from the new-body content before write, idempotently.


def _count_frontmatter_blocks(text: str) -> int:
    """Count consecutive leading `---\\n...\\n---\\n` blocks in `text`."""
    count = 0
    rest = text
    while rest.startswith("---\n"):
        end = rest.find("\n---\n", 4)
        if end == -1:
            break
        count += 1
        rest = rest[end + len("\n---\n") :]
    return count


def test_set_body_strips_leading_frontmatter_in_input(fake_repo):
    """A caller passing `---\\n...\\n---\\n<body>` produces exactly ONE
    frontmatter block in body.md — the canonical one — not two stacked.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old body"))
    body_with_fm = (
        "---\n"
        "title: A stale title from the caller\n"
        "kind: something_else\n"
        "made_up_field: caller noise\n"
        "---\n"
        "# Real H1 (HIGH confidence)\n\nReal body content here.\n"
    )
    tw.set_body(new_id, body_with_fm)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    fm, body = tw._split_frontmatter(written)
    # Canonical frontmatter is preserved (the original task title `"X"`),
    # NOT replaced by the caller's "A stale title from the caller".
    assert fm["title"] == "X"
    assert "made_up_field" not in fm
    # Body region starts at the H1, not at a stray `---` line.
    assert body.lstrip().startswith("# Real H1")


def test_set_body_no_frontmatter_unchanged(fake_repo):
    """A caller passing plain body content (no leading `---`) still works —
    the strip is a no-op and only the canonical frontmatter is prepended.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old"))
    plain = "# Real H1 (HIGH confidence)\n\nPlain body, no frontmatter.\n"
    tw.set_body(new_id, plain)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    _, body = tw._split_frontmatter(written)
    assert body.lstrip().startswith("# Real H1")


def test_set_body_strips_multiple_stacked_frontmatter_blocks(fake_repo):
    """Pathological: caller passes content with two stacked frontmatter
    blocks. `set_body` strips ALL of them, leaving exactly one (the
    canonical) frontmatter block in body.md.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X", body="old"))
    pathological = (
        "---\nfirst: block\n---\n---\nsecond: block\n---\n# H1 (HIGH confidence)\n\nBody text.\n"
    )
    tw.set_body(new_id, pathological)
    written = (repo / "tasks" / "proposed" / str(new_id) / "body.md").read_text()
    assert _count_frontmatter_blocks(written) == 1
    _, body = tw._split_frontmatter(written)
    assert body.lstrip().startswith("# H1")
    assert "first: block" not in written
    assert "second: block" not in written


def test_set_body_strip_is_idempotent(fake_repo, monkeypatch: pytest.MonkeyPatch):
    """Calling `set_body` twice with the same content (once with leading
    frontmatter, once with the same content already stripped) produces
    byte-identical body.md.
    """
    repo, tw = fake_repo
    # Freeze the timestamp source: the two create_task calls below each
    # write `created_at` into frontmatter, so without this they can
    # straddle a second boundary and spuriously break the byte-equality
    # assert (observed flake 2026-06-10). The test's intent — strip
    # idempotency of set_body CONTENT — is unaffected.
    monkeypatch.setattr(tw, "_utcnow_iso", lambda: "2026-01-01T00:00:00Z")
    id_a = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Same", body="old"))
    id_b = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Same", body="old"))
    with_fm = "---\nstale: stuff\n---\n# H1 (HIGH confidence)\n\nIdentical body content here.\n"
    stripped = "# H1 (HIGH confidence)\n\nIdentical body content here.\n"
    tw.set_body(id_a, with_fm)
    tw.set_body(id_b, stripped)
    text_a = (repo / "tasks" / "proposed" / str(id_a) / "body.md").read_text()
    text_b = (repo / "tasks" / "proposed" / str(id_b) / "body.md").read_text()
    # Only the title frontmatter field differs (Same vs Same — actually
    # identical), so the files MUST be byte-identical modulo task id (no
    # id appears in body.md). They should match exactly.
    assert text_a == text_b


def test_strip_leading_frontmatter_blocks_unit():
    """Direct unit test on the private helper — covers the no-frontmatter,
    one-block, two-block, and malformed-block cases."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.task_workflow import _strip_leading_frontmatter_blocks as strip

    assert strip("plain body\n") == "plain body\n"
    assert strip("# H1\n\nbody\n") == "# H1\n\nbody\n"
    assert strip("---\nfoo: bar\n---\nbody\n") == "body\n"
    # Stacked blocks
    assert strip("---\na: 1\n---\n---\nb: 2\n---\nbody\n") == "body\n"
    # Malformed leading block (no closing `---`) is left alone
    assert strip("---\nfoo: bar\nno closing\n# H1\n") == "---\nfoo: bar\nno closing\n# H1\n"
    # Leading blank lines after stripping are dropped
    assert strip("---\nfoo: bar\n---\n\n\n# H1\n") == "# H1\n"
    # Idempotence: stripping an already-stripped string is a no-op
    once = strip("---\nfoo: bar\n---\nbody\n")
    twice = strip(once)
    assert once == twice


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


def test_new_plan_version_skips_gap_uses_max_plus_one(fake_repo):
    """Regression: with a numbering gap (e.g. v1,v2,v3,v4,v6 — no v5,
    because a draft lived only in /tmp and was never registered), the
    next plan MUST be v7 — NOT v6, which would silently overwrite the
    highest existing plan. Closes the task #524 incident: the count-based
    resolver (``len(existing)+1``) computed v6 over an existing v6 and
    destroyed it without warning. Source of truth is now
    ``max(existing v<N>) + 1``.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    # Pre-stage a gapped set of plan files (no v5).
    for n in (1, 2, 3, 4, 6):
        (plans_dir / f"v{n}.md").write_text(f"plan v{n} content\n")
    v6_original = (plans_dir / "v6.md").read_text()

    next_v = tw.new_plan_version(new_id, "plan v7 content")

    # MUST advance past the highest existing version, not fill the gap and
    # MUST NOT overwrite v6.
    assert next_v == 7, f"expected v7 (max+1), got v{next_v}"
    assert (plans_dir / "v7.md").read_text().strip() == "plan v7 content"
    assert (plans_dir / "v6.md").read_text() == v6_original, (
        "v6.md was overwritten — the count-based resolver bug has regressed"
    )
    # v5 stays absent — we don't backfill gaps.
    assert not (plans_dir / "v5.md").exists()
    # Symlink points to v7.
    assert (plans_dir / "plan.md").resolve() == (plans_dir / "v7.md").resolve()


def test_new_plan_version_refuses_to_overwrite_existing_target(
    fake_repo, monkeypatch: pytest.MonkeyPatch
):
    """Belt-and-suspenders: the resolver derives ``next_v = max(existing) + 1``
    inside ``_locked()`` and writes immediately after — so under normal
    operation the computed target file CANNOT pre-exist. The explicit
    ``target.exists()`` guard fires only if something external creates
    the file between the glob and the write (a process holding no lock,
    a filesystem race, manual staging during the critical section). The
    guard is cheap and documents the invariant. To exercise it we simulate
    that race by wrapping the lock so a sentinel file appears at the
    computed slot after the glob but before the write.
    """
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))
    plans_dir = repo / "tasks" / "proposed" / str(new_id) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / "v1.md").write_text("plan v1 content\n")
    sentinel = "PRE_STAGED_SHOULD_NOT_BE_OVERWRITTEN\n"

    # Race simulation: replace Path.write_text so the first call (the
    # resolver's write to v2.md) finds v2.md already present. Note the
    # resolver writes v2.md FIRST, then the symlink — so we intercept on
    # the first call only and re-raise via the resolver's own guard.
    real_glob = type(plans_dir).glob

    def racing_glob(self, pattern):
        result = list(real_glob(self, pattern))
        # Inject the racing pre-existing file BEFORE write_text runs.
        if self == plans_dir and pattern == "v*.md":
            (plans_dir / "v2.md").write_text(sentinel)
        return iter(result)

    monkeypatch.setattr(type(plans_dir), "glob", racing_glob)

    with pytest.raises(RuntimeError, match=r"refusing to overwrite.*v2\.md"):
        tw.new_plan_version(new_id, "plan v2 fresh content")

    # The racing pre-existing file is preserved untouched.
    assert (plans_dir / "v2.md").read_text() == sentinel


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


def test_commit_does_not_sweep_unrelated_staged_files(fake_repo):
    """Regression: ``_git_commit`` must commit ONLY the paths it was asked to,
    even when other files are staged in the index by a parallel agent.

    Prior behavior used bare ``git commit -m <msg>``, which captures the entire
    index. A parallel workflow-improver agent (or user) with staged work would
    have those changes silently swept into a task.py marker commit and
    re-attributed under an unrelated task's message. Fix is ``commit --only --
    <paths>`` plus narrowing the early-return ``diff --cached --quiet`` check
    to the same paths.
    """
    repo, tw = fake_repo

    # Simulate a parallel agent's uncommitted, staged work.
    unrelated_a = repo / "unrelated_agent_work_a.txt"
    unrelated_a.write_text("agent A scratch\n")
    unrelated_b = repo / ".claude" / "unrelated_agent_work_b.md"
    unrelated_b.parent.mkdir(parents=True, exist_ok=True)
    unrelated_b.write_text("agent B scratch\n")
    subprocess.run(
        ["git", "add", "unrelated_agent_work_a.txt", ".claude/unrelated_agent_work_b.md"],
        cwd=repo,
        check=True,
    )

    n_commits_before = _git_log_count(repo)

    # Run a task.py operation that triggers _git_commit.
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # task.py committed exactly once.
    assert _git_log_count(repo) == n_commits_before + 1

    # The commit's changed-file list must NOT mention the unrelated staged files.
    show = subprocess.run(
        ["git", "show", "HEAD", "--name-only", "--format="],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    show = [s for s in show if s.strip()]
    assert "unrelated_agent_work_a.txt" not in show, (
        f"task.py commit swept in an unrelated staged file. Files in HEAD: {show}"
    )
    assert ".claude/unrelated_agent_work_b.md" not in show, (
        f"task.py commit swept in an unrelated staged file. Files in HEAD: {show}"
    )
    # Every committed path should live under tasks/.
    assert all(s.startswith("tasks/") for s in show), (
        f"task.py commit reached outside tasks/. Files in HEAD: {show}"
    )

    # The unrelated files must still be staged and unchanged in the working tree.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "unrelated_agent_work_a.txt" in diff_cached
    assert ".claude/unrelated_agent_work_b.md" in diff_cached
    assert new_id is not None


def test_commit_early_return_ignores_unrelated_staged_files(fake_repo):
    """Regression: when the paths task.py wants to commit are already at the
    committed state, ``_git_commit`` must early-return — even if OTHER files
    are staged in the index. Prior bare ``diff --cached --quiet`` would see
    the unrelated staged work, miss the early-return, and create a phantom
    commit (re-committing the same task state under a new SHA).
    """
    repo, tw = fake_repo

    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="X"))

    # Stage unrelated work AFTER creating the task; the next set_status to the
    # current status should be idempotent and produce no new commit, but the
    # unrelated staged work would have tricked the old early-return check.
    unrelated = repo / "scratch.txt"
    unrelated.write_text("scratch\n")
    subprocess.run(["git", "add", "scratch.txt"], cwd=repo, check=True)

    n_commits_before = _git_log_count(repo)

    # Idempotent set_status — task is already in 'proposed'.
    tw.set_status(new_id, "proposed")

    # No new commit should have been created.
    assert _git_log_count(repo) == n_commits_before, (
        "task.py created a phantom commit when target paths were unchanged but "
        "unrelated files were staged."
    )
    # The unrelated file is still staged.
    diff_cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "scratch.txt" in diff_cached


# ─── task.py migrate-body subcommand ──────────────────────────────────────


def _move_to_awaiting(tw, task_id: int) -> None:
    """Helper to push a task through to awaiting_promotion."""
    tw.set_status(task_id, "awaiting_promotion")


# Minimal canonical PASS body — useful as a fixture target. Every check
# (title, the three required H2s of the 2-content-section spec in order,
# TL;DR Motivation opener, hero image inline under TL;DR, confidence
# sentence, repro subgroups + URL + sentinel scrub, cherry-picked,
# qual-data link) is satisfied. Non-v2 (no `<!-- clean-result-v2 -->`
# sentinel), so the body Confidence sentence is still required and the
# nested-TL;DR-shape rule is skipped. The `## Goal` H2 sits AFTER
# `## Reproducibility` — extra H2s are tolerated only there (stray-H2
# rule, verify check 2).
CANONICAL_PASS_BODY = """\
# Toy clean-result body (LOW confidence)

## Human TL;DR

A plain-English first-pass take: this toy fixture exercises the fully-conformant
clean-result shape end to end and passes every verifier check.

## TL;DR

- **Motivation:** I wanted a smoke-test fixture.
- **What I ran:** I wrote a minimal markdown body and ran verify_task_body.
- **Results:** The fixture passes every check.
- **Next steps:** Use this fixture in migration tests.

![Hero figure placeholder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

## Reproducibility

**Artifacts:** n/a

**Compute:** n/a

**Code:** n/a

Confidence: LOW — based on toy data only, not a real experiment so does not generalize.

## Goal

Smoke-test that classify_body recognizes a fully-conformant clean-result body and returns PASS.
"""


# Conformant-but-failing fixture: current required-H2 shape (Human TL;DR /
# TL;DR / Reproducibility), but Reproducibility is missing its three
# boldface subgroup labels and uses H3 instead — the one defect the
# `remediate_repro_subgroups` patch fixes mechanically.
CONFORMANT_FAILING_H3_REPRO_BODY = """\
# Conformant-failing body using H3 repro subgroups (LOW confidence)

## Human TL;DR

A plain-English first-pass take: this fixture is conformant except for the H3
Reproducibility subgroup headings, which the remediation patch promotes to bold.

## TL;DR

- **Motivation:** toy motivation.
- **What I ran:** toy run description goes here.
- **Results:** toy results paragraph explaining what we saw.
- **Next steps:** none in particular.

![Hero figure placeholder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_X/hero.png)

*Hero figure showing the toy data points and the regression line and bootstrap envelope.*

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

Confidence: LOW — based on toy data only, not generalizable, no real experiment.
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
) -> int:
    """Create a task and push it to awaiting_promotion. Returns the id.

    The Goal-of-experiment soft check WARNs on missing `goal:` frontmatter
    but never FAILs, so the helper no longer needs to inject `application:`
    or `goal:` for the migrate-body fixtures to classify correctly.
    """
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title=title, body=body))
    if task_id_hint is not None:
        assert new_id == task_id_hint, f"id drift: got {new_id}, expected {task_id_hint}"
    tw.set_status(new_id, "awaiting_promotion")
    return new_id


def test_migrate_body_classify_pass(fake_repo):
    from explore_persona_space.task_workflow_migrate import BodyClass, classify_body

    # CANONICAL_PASS_BODY exercises the fully-conformant body shape under
    # the 2-content-section spec (2026-W22, task #454): Human TL;DR /
    # TL;DR / Reproducibility in order, hero image inline under TL;DR,
    # `## Goal` H2 after Reproducibility (extra H2s tolerated only
    # there), and an absolute figure URL.
    assert classify_body(CANONICAL_PASS_BODY, fm={}) == BodyClass.PASS


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
    """A current-spec-shaped body with H3 Repro subgroups gets the labels
    promoted to bold and ends up passing verify_task_body.
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


def test_migrate_body_v4_legacy_routes_to_needs_user(fake_repo):
    """V4_LEGACY bodies are classified but NOT converted — `migrate_one`
    routes them straight to `needs_user` with a retirement reason. The old
    `convert_v4_to_target` chain targeted the retired four-H2 shape (its
    output always hard-FAILed the verifier's stray-H2 check under the
    2-content-section spec), so the converter was removed (2026-06-09).
    """
    _, tw = fake_repo
    import explore_persona_space.task_workflow_migrate as migrate_mod
    from explore_persona_space.task_workflow_migrate import BodyClass, migrate_one

    new_id = _make_task_at_awaiting(tw, title="v4 fixture (LOW confidence)", body=V4_LEGACY_BODY)

    result = migrate_one(new_id, apply=False)
    assert result.classification == BodyClass.V4_LEGACY
    assert result.needs_user
    assert "auto-conversion was retired" in result.needs_user_reason
    assert "SPEC.md" in result.needs_user_reason
    # No conversion is attempted: the action log is empty.
    assert result.actions == []
    # The retired converter and its helpers are gone from the module.
    assert not hasattr(migrate_mod, "convert_v4_to_target")
    assert not hasattr(migrate_mod, "strip_v4_details_wrappers")


def test_migrate_body_v4_legacy_unchanged_on_apply(fake_repo):
    """`--apply` on a V4_LEGACY body is a guaranteed no-op: needs_user,
    body unchanged on disk, no commits (converter retired 2026-06-09).
    """
    repo, tw = fake_repo
    from explore_persona_space.task_workflow_migrate import migrate_one

    new_id = _make_task_at_awaiting(tw, title="Untouched-fail", body=V4_LEGACY_BODY)

    body_path = repo / "tasks" / "awaiting_promotion" / str(new_id) / "body.md"
    before_text = body_path.read_text()
    n_commits_before = _git_log_count(repo)

    result = migrate_one(new_id, apply=True)
    assert result.needs_user
    # verify_after mirrors verify_before — the body was never touched.
    assert result.verify_after == "FAIL"
    assert result.verify_before == "FAIL"
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


# ─── set_goal — canonical Goal-of-the-experiment field ───────────────────


def test_set_goal_writes_frontmatter_and_h2(fake_repo):
    """set_goal updates frontmatter `goal:` AND injects a `## Goal` H2."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="Goal test", body="# Goal test\n\nbody here\n")
    )
    changed = tw.set_goal(new_id, "Measure persona collapse under fine-tuning", by="user")
    assert changed is True
    task = tw.get_task(new_id)
    assert task["frontmatter"]["goal"] == "Measure persona collapse under fine-tuning"
    assert "## Goal" in task["body"]
    assert "Measure persona collapse under fine-tuning" in task["body"]
    # Pre-existing "body here" content is preserved below the Goal block.
    assert "body here" in task["body"]


def test_set_goal_emits_marker(fake_repo):
    """set_goal posts a single epm:goal-updated v1 marker carrying from/to/by."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Goal marker test"))
    tw.set_goal(new_id, "First goal sentence", by="user")
    markers = [e for e in tw.list_events(new_id) if e["kind"] == "epm:goal-updated"]
    assert len(markers) == 1
    m = markers[0]
    assert m["version"] == 1
    assert m["by"] == "user"
    assert m["from"] is None  # no prior goal
    assert m["to"] == "First goal sentence"


def test_set_goal_idempotent_no_op(fake_repo):
    """Re-applying the same goal is a no-op: no new marker, no commit."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Idem"))
    tw.set_goal(new_id, "Sticky goal", by="user")
    n_events_after_first = len(tw.list_events(new_id))
    n_commits_after_first = _git_log_count(repo)
    # Same goal value -> changed=False, no marker, no commit
    changed = tw.set_goal(new_id, "Sticky goal", by="user")
    assert changed is False
    assert len(tw.list_events(new_id)) == n_events_after_first
    assert _git_log_count(repo) == n_commits_after_first


def test_set_goal_refinement_emits_second_marker(fake_repo):
    """Changing the goal emits a second marker with the prior `from:` value."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Refine"))
    tw.set_goal(new_id, "Initial goal", by="user")
    tw.set_goal(new_id, "Refined goal", by="clarifier", reason="clarifier sharpening")
    markers = [e for e in tw.list_events(new_id) if e["kind"] == "epm:goal-updated"]
    assert len(markers) == 2
    assert markers[0]["from"] is None and markers[0]["to"] == "Initial goal"
    assert markers[1]["from"] == "Initial goal" and markers[1]["to"] == "Refined goal"
    assert markers[1]["by"] == "clarifier"
    assert markers[1]["reason"] == "clarifier sharpening"


def test_set_goal_rejects_empty(fake_repo):
    """Empty / whitespace-only goal raises ValueError."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Empty goal"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "   ", by="user")
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "", by="user")


def test_set_goal_rejects_invalid_by(fake_repo):
    """`by` must be one of user|clarifier|planner."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="By"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "g", by="critic")  # critics are explicitly forbidden
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "g", by="analyzer")


def test_set_goal_normalizes_multiline_whitespace(fake_repo):
    """set_goal collapses internal whitespace so a multi-line input becomes
    a single sentence in BOTH the frontmatter scalar and the body H2 block.

    Regression: bare `.strip()` only trimmed edges, so newlines / tabs /
    runs of spaces survived. A multi-paragraph Goal then became an
    orphan-paragraph trap because `_inject_or_replace_goal_h2` only
    refreshes the first paragraph after `## Goal`, leaving stale text in
    the body on the next refinement. Reviewer flag M1.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Multiline goal"))

    multi = "First sentence\n\nSecond paragraph that should NOT exist\nthird line"
    tw.set_goal(new_id, multi, by="user")

    # Frontmatter is a single-line scalar.
    fm, body = tw._read_body(tw.find_task_path(new_id) / "body.md")
    expected = "First sentence Second paragraph that should NOT exist third line"
    assert fm["goal"] == expected, f"frontmatter goal not normalized: {fm['goal']!r}"

    # Body's ## Goal block has the header, a blank line, then exactly one
    # non-empty line carrying the normalized goal — no orphan paragraphs.
    lines = body.splitlines()
    goal_idx = lines.index("## Goal")
    assert lines[goal_idx + 1] == "", f"missing blank after ## Goal: {lines[goal_idx + 1]!r}"
    assert lines[goal_idx + 2] == expected, f"goal body not normalized: {lines[goal_idx + 2]!r}"
    # The next line is either blank (separator before the next section) or the
    # end of the body — but it must NOT be more goal-text-paragraph content.
    if goal_idx + 3 < len(lines):
        assert lines[goal_idx + 3] == "" or lines[goal_idx + 3].startswith(("#", "<")), (
            f"orphan content after goal: {lines[goal_idx + 3]!r}"
        )

    # Refining replaces cleanly — no orphan paragraphs left from the multi-line.
    tw.set_goal(new_id, "Refined goal", by="planner")
    _, body2 = tw._read_body(tw.find_task_path(new_id) / "body.md")
    assert "Second paragraph that should NOT exist" not in body2
    assert "third line" not in body2
    assert "Refined goal" in body2


def test_set_goal_normalizes_tabs_and_extra_spaces(fake_repo):
    """set_goal collapses tabs and runs of internal spaces too."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Whitespace"))

    tw.set_goal(new_id, "  foo\tbar    baz   ", by="user")

    fm, _ = tw._read_body(tw.find_task_path(new_id) / "body.md")
    assert fm["goal"] == "foo bar baz"


def test_set_goal_rejects_whitespace_only_multiline(fake_repo):
    """A goal that is empty AFTER normalization (e.g. only newlines and
    spaces) still raises ValueError — the normalization must not allow
    blank goals to slip through.
    """
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Blank"))
    with pytest.raises(ValueError):
        tw.set_goal(new_id, "\n\n  \t  \n", by="user")


def test_create_task_with_goal_kwarg_for_experiment(fake_repo):
    """NewTaskRequest.goal is honored when kind=experiment at creation time."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title="Created with goal",
            body="# Created with goal\n",
            goal="Initial goal at creation",
        )
    )
    task = tw.get_task(new_id)
    assert task["frontmatter"]["goal"] == "Initial goal at creation"
    assert "## Goal" in task["body"]


def test_create_task_with_goal_kwarg_ignored_for_infra(fake_repo):
    """NewTaskRequest.goal is silently ignored when kind != experiment."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="infra",
            title="Infra task with stray goal",
            body="# infra\n",
            goal="this should be ignored",
        )
    )
    task = tw.get_task(new_id)
    assert task["frontmatter"].get("goal") is None
    assert "## Goal" not in task["body"]


def test_get_goal_returns_current_value(fake_repo):
    """get_goal returns the on-disk frontmatter goal (or None)."""
    _, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Get goal"))
    assert tw.get_goal(new_id) is None
    tw.set_goal(new_id, "Visible via get_goal", by="user")
    assert tw.get_goal(new_id) == "Visible via get_goal"


def test_set_goal_preserves_body_after_goal_block(fake_repo):
    """Refining the Goal must NOT swallow content following the Goal section."""
    _, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(
            kind="experiment",
            title="Preserve",
            body="# Preserve\n\nfirst pre-goal paragraph\n",
            goal="G1",
        )
    )
    tw.set_goal(new_id, "G2 refined", by="planner")
    task = tw.get_task(new_id)
    assert "first pre-goal paragraph" in task["body"]
    assert "G2 refined" in task["body"]
    # Old goal text must NOT linger after refinement.
    assert "G1" not in task["body"]


def test_registry_denormalizes_goal(fake_repo):
    """REGISTRY.json entries pick up the `goal` field for cheap querying."""
    repo, tw = fake_repo
    new_id = tw.create_task(
        tw.NewTaskRequest(kind="experiment", title="Reg goal", goal="Registry-visible goal")
    )
    reg = json.loads((repo / "tasks" / "REGISTRY.json").read_text())
    assert reg["tasks"][str(new_id)]["goal"] == "Registry-visible goal"


# ─── Binding concerns (concerns.jsonl) ────────────────────────────────────


_GOOD_RATIONALE = (
    "The probe-position confound only affects the secondary stratification; "
    "the primary contrast survives. Documenting in Methodology corrections."
)


@pytest.fixture
def concerns_task(fake_repo):
    """Create a clean task and yield (repo, tw, task_id) for concerns tests."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="experiment", title="Concerns under test"))
    return repo, tw, new_id


def test_raise_concern_appends_to_concerns_jsonl(concerns_task):
    """First raise writes one row to concerns.jsonl with the expected fields,
    and mirrors a `epm:concern-raised` event to events.jsonl."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "probe-position-undefined",
        severity="CONCERN",
        summary="Probe position is undefined for the trigger-conditional contrast.",
        raised_by="code-reviewer",
        raised_at_round=1,
        evidence="src/foo.py:42",
    )
    concerns_path = tw.find_task_path(tid) / "concerns.jsonl"
    assert concerns_path.exists()
    rows = [json.loads(line) for line in concerns_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "raised"
    assert row["concern_id"] == "probe-position-undefined"
    assert row["severity"] == "CONCERN"
    assert row["raised_by"] == "code-reviewer"
    assert row["raised_at_round"] == 1
    assert row["evidence"] == "src/foo.py:42"
    # Mirror event posted.
    kinds = [e["kind"] for e in tw.list_events(tid)]
    assert "epm:concern-raised" in kinds


def test_raise_concern_rejects_bad_concern_id(concerns_task):
    """Concern IDs must be lowercase kebab-case, 2-80 chars, alphanum-start."""
    _, tw, tid = concerns_task
    for bad in ("UpperCase", "with_underscore", "with space", "x", "-leading-dash", ""):
        with pytest.raises(ValueError, match="concern_id"):
            tw.raise_concern(
                tid,
                bad,
                severity="CONCERN",
                summary="bad id",
                raised_by="critic",
                raised_at_round=1,
            )
    # Borderline 80-char alphanumeric+hyphen passes.
    eighty = "a" + "-".join(["b"] * 39)  # length 79
    assert len(eighty) <= 80
    tw.raise_concern(
        tid,
        eighty,
        severity="NIT",
        summary="borderline length",
        raised_by="critic",
        raised_at_round=1,
    )


def test_raise_concern_idempotent_same_round_same_severity(concerns_task):
    """Re-raising at the SAME round with the SAME severity is a no-op."""
    _, tw, tid = concerns_task
    first = tw.raise_concern(
        tid,
        "n2-seeds-uninterpretable",
        severity="CONCERN",
        summary="N=2 seeds gives essentially no statistical power.",
        raised_by="critic",
        raised_at_round=1,
    )
    second = tw.raise_concern(
        tid,
        "n2-seeds-uninterpretable",
        severity="CONCERN",
        summary="N=2 seeds gives essentially no statistical power.",
        raised_by="critic",
        raised_at_round=1,
    )
    # Second call returns the existing event payload (timestamps match).
    assert first["ts"] == second["ts"]
    # Only one row written to concerns.jsonl + one mirror event.
    concerns_path = tw.find_task_path(tid) / "concerns.jsonl"
    rows = [json.loads(line) for line in concerns_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    mirror_count = sum(1 for e in tw.list_events(tid) if e["kind"] == "epm:concern-raised")
    assert mirror_count == 1


def test_address_then_reraise_records_verified_open(concerns_task):
    """Re-raising AFTER an `addressed` event becomes a `verified-open` event,
    not a fresh `raised` event. This is the key cross-round visibility
    mechanism that makes concerns binding across stages."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "missing-mlm-control",
        severity="CONCERN",
        summary="No MLM baseline control.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    tw.address_concern(
        tid,
        "missing-mlm-control",
        addressed_by="implementer",
        addressed_at_round=1,
    )
    tw.raise_concern(
        tid,
        "missing-mlm-control",
        severity="CONCERN",
        summary="Still no MLM control after claimed fix.",
        raised_by="code-reviewer",
        raised_at_round=2,
    )
    events = tw.list_concerns(tid)
    assert [e["event"] for e in events] == ["raised", "addressed", "verified-open"]
    assert events[-1]["raised_at_round"] == 2


def test_list_concerns_open_only_filters_addressed_and_deferred(concerns_task):
    """`open_only=True` returns only concerns whose LATEST event is `raised`
    or `verified-open`. Addressed and deferred concerns drop out."""
    _, tw, tid = concerns_task
    # A: raised, then addressed — should NOT be open.
    tw.raise_concern(
        tid, "a-fixed", severity="CONCERN", summary="A", raised_by="r", raised_at_round=1
    )
    tw.address_concern(tid, "a-fixed", addressed_by="i", addressed_at_round=1)
    # B: raised, then deferred — should NOT be open.
    tw.raise_concern(
        tid, "b-deferred", severity="CONCERN", summary="B", raised_by="r", raised_at_round=1
    )
    tw.defer_concern(tid, "b-deferred", by="user", rationale=_GOOD_RATIONALE)
    # C: raised, addressed, re-raised (verified-open) — SHOULD be open.
    tw.raise_concern(
        tid, "c-reraised", severity="CONCERN", summary="C", raised_by="r", raised_at_round=1
    )
    tw.address_concern(tid, "c-reraised", addressed_by="i", addressed_at_round=1)
    tw.raise_concern(
        tid,
        "c-reraised",
        severity="CONCERN",
        summary="C still open",
        raised_by="r",
        raised_at_round=2,
    )
    # D: raised, never touched — SHOULD be open.
    tw.raise_concern(tid, "d-raw", severity="NIT", summary="D", raised_by="r", raised_at_round=1)
    open_rows = tw.list_concerns(tid, open_only=True)
    open_ids = {r["concern_id"] for r in open_rows}
    assert open_ids == {"c-reraised", "d-raw"}


def test_defer_concern_requires_by_user(concerns_task):
    """Library function rejects --by other than user/reconciler (defense
    in depth — CLI also rejects)."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "c1-rejected", severity="CONCERN", summary="first", raised_by="r1", raised_at_round=1
    )
    with pytest.raises(ValueError, match="user-only"):
        tw.defer_concern(tid, "c1-rejected", by="implementer", rationale=_GOOD_RATIONALE)
    with pytest.raises(ValueError, match="user-only"):
        tw.defer_concern(tid, "c1-rejected", by="critic", rationale=_GOOD_RATIONALE)
    # 'user' and 'reconciler' both succeed.
    tw.defer_concern(tid, "c1-rejected", by="user", rationale=_GOOD_RATIONALE)
    tw.raise_concern(
        tid,
        "c2-reconciler",
        severity="CONCERN",
        summary="second",
        raised_by="r1",
        raised_at_round=1,
    )
    tw.defer_concern(tid, "c2-reconciler", by="reconciler", rationale=_GOOD_RATIONALE)


def test_defer_concern_rejects_blocker(concerns_task):
    """BLOCKER concerns cannot be user-deferred — strict gate."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "critical-bug",
        severity="BLOCKER",
        summary="This will corrupt data.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    with pytest.raises(ValueError, match="BLOCKER"):
        tw.defer_concern(tid, "critical-bug", by="user", rationale=_GOOD_RATIONALE)


def test_defer_concern_rejects_short_rationale(concerns_task):
    """Rationale floor is 40 chars after strip."""
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "rationale-test", severity="CONCERN", summary="r", raised_by="r1", raised_at_round=1
    )
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "rationale-test", by="user", rationale="too short")
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "rationale-test", by="user", rationale="a" * 39)
    # Exactly 40 succeeds (non-boilerplate).
    tw.defer_concern(tid, "rationale-test", by="user", rationale="X" * 40)


def test_defer_concern_rejects_boilerplate_rationale(concerns_task):
    """Boilerplate phrases like 'user accepted', 'lgtm', 'wontfix' are
    rejected by the normalization-based validator (casefold + whitespace
    collapse).

    All known boilerplate phrases are short (<40 chars), so under the
    full ``defer_concern`` chain the length floor fires first. We validate
    the boilerplate path directly via the underlying validator helper so
    the blocklist's mechanical coverage is exercised regardless of
    length-rule ordering.
    """
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid, "boilerplate-test", severity="CONCERN", summary="b", raised_by="r1", raised_at_round=1
    )
    # Whatever phrase fails length first via defer_concern.
    with pytest.raises(ValueError, match="≥"):
        tw.defer_concern(tid, "boilerplate-test", by="user", rationale="user accepted")
    # Direct validator call exercises the blocklist branch.
    boilerplate_phrases = [
        "user accepted",
        "User Accepted",  # casefold-equivalent
        "ok",
        "LGTM",
        "wontfix",
        "Won't Fix",  # whitespace + case normalization
        "  user   said   ok  ",  # internal-whitespace collapse
    ]
    for phrase in boilerplate_phrases:
        # Pad each phrase with leading/trailing whitespace ≥40 chars to
        # try to bypass the length floor; the validator should still
        # reject because the NORMALIZED form matches the blocklist.
        # Note: collapsing whitespace inside ALSO collapses; the only
        # way a long-padded phrase survives the length check is if the
        # padding is leading/trailing — and `strip()` then takes us back
        # to a short phrase. So we test the blocklist via the validator
        # directly (it's called pre-length elsewhere in the code path
        # via raise/address; defer's chain runs length first).
        with pytest.raises(ValueError):
            tw._validate_deferral_rationale(phrase)
    # Sanity: a non-boilerplate phrase passes the validator (still subject
    # to the length floor, which we test below).
    tw._validate_deferral_rationale(_GOOD_RATIONALE)


def test_address_unknown_concern_raises(concerns_task):
    """`address_concern` refuses to address a concern that was never raised
    — prevents orphaned audit-log entries."""
    _, tw, tid = concerns_task
    with pytest.raises(ValueError, match="never been raised"):
        tw.address_concern(tid, "phantom", addressed_by="implementer", addressed_at_round=1)


def test_defer_unknown_concern_raises(concerns_task):
    """`defer_concern` refuses to defer a concern that was never raised."""
    _, tw, tid = concerns_task
    with pytest.raises(ValueError, match="never been raised"):
        tw.defer_concern(tid, "phantom", by="user", rationale=_GOOD_RATIONALE)


def test_concerns_follow_task_on_status_move(concerns_task):
    """`concerns.jsonl` lives inside `tasks/<status>/<N>/`, so `set_status`'s
    `git mv` of the task folder carries it along automatically.

    This is the key persistence property — concerns raised by the
    code-reviewer at status:code_reviewing survive into status:running,
    status:interpreting, status:reviewing, and status:awaiting_promotion
    without any explicit migration step.
    """
    _, tw, tid = concerns_task
    tw.raise_concern(
        tid,
        "trigger-conditional-contrast-missing",
        severity="CONCERN",
        summary="Plan v1.2 named this as Scenario B verdict criterion.",
        raised_by="code-reviewer",
        raised_at_round=1,
    )
    src_dir = tw.find_task_path(tid)
    src_concerns = src_dir / "concerns.jsonl"
    src_rows = src_concerns.read_text()
    # Move through several statuses; concerns.jsonl must come along each time.
    for status in ("planning", "approved", "running", "interpreting", "awaiting_promotion"):
        tw.set_status(tid, status)
        cur_dir = tw.find_task_path(tid)
        cur_concerns = cur_dir / "concerns.jsonl"
        assert cur_concerns.exists(), f"concerns.jsonl missing after move to {status}"
        assert cur_concerns.read_text() == src_rows, (
            f"concerns.jsonl content drifted after move to {status}"
        )


def test_raise_concern_holds_flock_and_commits(concerns_task):
    """Every raise/address/defer creates exactly ONE git commit (matches the
    existing `_git_commit` per-mutation contract). Concerns + mirror event
    land in the SAME commit so an `events.jsonl` reader and a
    `concerns.jsonl` reader never see a half-applied update."""
    repo, tw, tid = concerns_task
    before = _git_log_count(repo)
    tw.raise_concern(
        tid, "flock-test", severity="CONCERN", summary="ft", raised_by="r1", raised_at_round=1
    )
    after_raise = _git_log_count(repo)
    assert after_raise == before + 1
    tw.address_concern(tid, "flock-test", addressed_by="impl", addressed_at_round=1)
    after_address = _git_log_count(repo)
    assert after_address == after_raise + 1
    # Commit must include BOTH files.
    out = subprocess.run(
        ["git", "show", "--name-only", "--pretty=", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The task is at proposed/ — verify both files in commit.
    assert "concerns.jsonl" in out
    assert "events.jsonl" in out


def test_raise_concern_ordering_preserved(concerns_task):
    """concerns.jsonl preserves write order — append-only, no reordering."""
    _, tw, tid = concerns_task
    ids = ["alpha-id", "beta-id", "gamma-id", "delta-id"]
    for i, cid in enumerate(ids, start=1):
        tw.raise_concern(
            tid,
            cid,
            severity="NIT",
            summary=f"#{i}",
            raised_by="r",
            raised_at_round=1,
        )
    rows = tw.list_concerns(tid)
    assert [r["concern_id"] for r in rows] == ids


def test_cli_handlers_raise_address_defer_list_roundtrip(concerns_task, capsys):
    """End-to-end roundtrip for the CLI handler functions wired in
    ``scripts/task.py``.

    The CLI is exercised at the handler-function layer (not via
    ``subprocess.run``) because ``task_workflow.repo_root()`` branch-guards
    to ``main`` and resolves via ``git rev-parse`` from the module path.
    A subprocess would bypass the test's ``fake_repo`` monkeypatch and
    target the real repo (when on ``main``) or auto-route to a managed
    main-pinned worktree (when on a feature branch), so the CLI write
    would land in a directory that does not contain the fixture's task.
    The library-level path here uses the same handler functions called
    by ``main()`` and gives equivalent coverage of argument plumbing,
    JSON output formatting, and exit-code behaviour — without the
    cross-process resolver mismatch documented in the
    ``feedback_branch_guard_blocks_subprocess`` workflow-improver note.
    """
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import task as task_cli  # type: ignore[import-not-found]

    _repo, _tw, tid = concerns_task

    def _ns(**kwargs):
        return argparse.Namespace(**kwargs)

    # raise via CLI handler
    task_cli.cmd_raise_concern(
        _ns(
            number=tid,
            concern_id="cli-test-concern",
            severity="CONCERN",
            summary="A concern raised via the CLI for end-to-end coverage.",
            by="code-reviewer",
            round=1,
            evidence=None,
        )
    )
    capsys.readouterr()  # drain the raise payload

    # list-concerns --open-only --json shows the raised event
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=True, json=True))
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["concern_id"] == "cli-test-concern"
    assert rows[0]["event"] == "raised"

    # address via CLI handler
    task_cli.cmd_address_concern(
        _ns(
            number=tid,
            concern_id="cli-test-concern",
            by="implementer",
            round=1,
            summary=None,
        )
    )
    capsys.readouterr()  # drain the address payload

    # list-concerns --open-only --json now returns empty
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=True, json=True))
    assert json.loads(capsys.readouterr().out) == []

    # full list shows both events in order
    task_cli.cmd_list_concerns(_ns(number=tid, open_only=False, json=True))
    rows = json.loads(capsys.readouterr().out)
    assert [r["event"] for r in rows] == ["raised", "addressed"]

    # defer with --by other than 'user' or 'reconciler' is rejected. The CLI
    # layer raises SystemExit with the "user-only" message; the library
    # layer additionally defends in depth (ValueError). Either is acceptable
    # — both signal that automation may not defer concerns.
    with pytest.raises((SystemExit, ValueError)) as excinfo:
        task_cli.cmd_defer_concern(
            _ns(
                number=tid,
                concern_id="cli-test-concern",
                by="implementer",
                rationale=_GOOD_RATIONALE,
                round=1,
            )
        )
    assert "user-only" in str(excinfo.value).lower() or "user" in str(excinfo.value).lower()
