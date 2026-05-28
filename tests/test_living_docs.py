"""Tests for ``scripts/living_docs.py``.

Every test runs against a TEMP fixture: a tmp dir set up as a fake repo
(``git init`` + minimal layout) with a synthetic ``docs/open_questions.md``
carrying a handful of ``<!-- q:* -->`` anchored questions + State
trailers, plus a few fake task folders. The real ``docs/`` and ``tasks/``
are never touched — the ``task_workflow`` resolver is monkeypatched at the
function level (same pattern as ``tests/test_task_workflow.py``) and
``living_docs`` is handed an injected :class:`LivingDocsPaths`.

Coverage:
- ``link()`` writes ``relates_to`` onto body.md frontmatter AND appends
  ``#N`` to the matching question's State evidence list; stubs a missing id.
- ``check()`` PASSes on a consistent fixture and FAILs (nonzero) on each
  injected drift class (missing back-link, dangling evidence, uncovered
  completed result, stale State date).
- ``apply()`` applies a confirmed patch (replacement + append) and lands a
  single git commit with a prepended changelog line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ─── Imports of the modules under test ─────────────────────────────────────

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import living_docs as ld  # noqa: E402

import explore_persona_space.task_workflow as tw  # noqa: E402

TODAY = ld._today()


# ─── Fixture: fake repo + living docs + tasks ──────────────────────────────


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_log_count(repo: Path) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def _make_task(repo: Path, task_id: int, status: str, *, body: str) -> None:
    """Create tasks/<status>/<id>/body.md and register it (minimal)."""
    import json

    task_dir = repo / "tasks" / status / str(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "body.md").write_text(body)
    (task_dir / "events.jsonl").touch()
    (task_dir / "comments.jsonl").touch()
    reg_path = repo / "tasks" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"highest_id": 0, "tasks": {}}
    fm, _ = tw._split_frontmatter(body)
    reg["tasks"][str(task_id)] = {
        "path": f"tasks/{status}/{task_id}",
        "title": fm.get("title", ""),
        "kind": fm.get("kind", "experiment"),
        "status": status,
        "has_clean_result": bool(fm.get("has_clean_result", False)),
    }
    reg["highest_id"] = max(reg["highest_id"], task_id)
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


_OPEN_QUESTIONS_TEMPLATE = f"""# Open Questions — Test fixture

Some intro prose.

---

## Thread A — geometry

**A1. What predicts marker implantability?** <!-- q:a1 -->
Three predictors failed. Either a non-geometric predictor exists or it is uniform.
> **State:** {ld.MATURITY_BUDDING} budding · MODERATE · updated 2026-05-20 · evidence: #207

**A2. Are persona vectors the same object?** <!-- q:a2 -->
Cosine 0.5-0.65; overlapping neighborhood, not the same direction.
> **State:** {ld.MATURITY_SEEDLING} seedling · LOW · updated 2026-05-20 · evidence: #208

---

## Thread B — leakage

**B1. Does the gradient generalize?** <!-- q:b1 -->
Open question prose.
> **State:** {ld.MATURITY_EVERGREEN} evergreen · HIGH · updated 2026-05-20 · evidence: #207, #208
"""


def _task_body(title: str, *, relates_to: list[str] | None, has_clean_result: bool, **fm) -> str:
    """Build a task body.md string with the given frontmatter."""
    lines = ["---", f"title: {title}", "kind: experiment"]
    if relates_to is not None:
        lines.append("relates_to:")
        for q in relates_to:
            lines.append(f"  - {q}")
    lines.append(f"has_clean_result: {str(has_clean_result).lower()}")
    for k, v in fm.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append(f"# {title}")
    lines.append("")
    return "\n".join(lines) + "\n"


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a fake repo with living docs + tasks; return (repo, paths)."""
    repo = tmp_path
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@test.test")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "commit.gpgsign", "false")

    # Living docs.
    (repo / "docs").mkdir()
    (repo / "docs" / "open_questions.md").write_text(_OPEN_QUESTIONS_TEMPLATE)
    (repo / "docs" / "papers.md").write_text("# Papers\n\n- placeholder\n")

    # Tasks: #207 (consistent w/ a1+b1), #208 (consistent w/ a2+b1).
    _make_task(
        repo,
        207,
        "completed",
        body=_task_body(
            "Task 207",
            relates_to=["a1", "b1"],
            has_clean_result=True,
            promoted_at="2026-05-19T00:00:00Z",
        ),
    )
    _make_task(
        repo,
        208,
        "completed",
        body=_task_body(
            "Task 208",
            relates_to=["a2", "b1"],
            has_clean_result=True,
            promoted_at="2026-05-19T00:00:00Z",
        ),
    )

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init fixture")

    # Rebind task_workflow's resolver + lock to the tmp repo.
    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: repo)
    monkeypatch.setattr(tw, "tasks_dir", lambda: repo / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: repo / "tasks" / "REGISTRY.json")
    lock_dir = repo / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")

    paths = ld.LivingDocsPaths(
        repo_root=repo,
        open_questions=repo / "docs" / "open_questions.md",
        papers=repo / "docs" / "papers.md",
        lock_path=lock_dir / "lock",
    )
    return repo, paths


# ─── link() ────────────────────────────────────────────────────────────────


def test_link_writes_relates_to_and_evidence(fixture):
    repo, paths = fixture
    # New task #300 linked to a1 + a2.
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    result = ld.link(300, ["a1", "A2"], paths=paths)

    assert result["task_id"] == 300
    assert result["relates_to"] == ["a1", "a2"]
    assert result["stubbed"] == []

    # body.md frontmatter carries relates_to (flat list, lowercased).
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "300" / "body.md")
    assert fm["relates_to"] == ["a1", "a2"]

    # open_questions.md evidence lists now include #300.
    text = (repo / "docs" / "open_questions.md").read_text()
    questions = ld._collect_question_evidence(text)
    assert 300 in questions["a1"]["evidence"]
    assert 300 in questions["a2"]["evidence"]
    # b1 untouched.
    assert 300 not in questions["b1"]["evidence"]


def test_link_is_idempotent_on_evidence(fixture):
    repo, paths = fixture
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    ld.link(300, ["a1"], paths=paths)
    ld.link(300, ["a1"], paths=paths)  # second call must not duplicate

    text = (repo / "docs" / "open_questions.md").read_text()
    questions = ld._collect_question_evidence(text)
    assert questions["a1"]["evidence"].count(300) == 1
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "300" / "body.md")
    assert fm["relates_to"] == ["a1"]


def test_link_stubs_missing_question(fixture):
    repo, paths = fixture
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    result = ld.link(300, ["znew"], paths=paths)
    assert result["stubbed"] == ["znew"]

    text = (repo / "docs" / "open_questions.md").read_text()
    assert "<!-- q:znew -->" in text
    questions = ld._collect_question_evidence(text)
    assert questions["znew"]["has_state"] is True
    assert 300 in questions["znew"]["evidence"]


# ─── check() ───────────────────────────────────────────────────────────────


def test_check_passes_on_consistent_fixture(fixture):
    _repo, paths = fixture
    report = ld.check(paths=paths)
    assert report.ok, report.render()


def test_check_fails_on_missing_backlink(fixture):
    """Question evidence lists #207 for a2's neighbor but #207 doesn't relate_to it."""
    repo, paths = fixture
    # Inject: add #207 to a2's evidence without touching #207's relates_to.
    text = (repo / "docs" / "open_questions.md").read_text()
    text = ld._add_evidence_to_question(text, "a2", 207)
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("a2" in p and "#207" in p for p in report.problems)


def test_check_fails_on_dangling_evidence(fixture):
    """A question lists a task id that does not exist."""
    repo, paths = fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text = ld._add_evidence_to_question(text, "a1", 9999)
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("#9999" in p and "no such task" in p for p in report.problems)


def test_check_fails_on_uncovered_completed_result(fixture):
    """A completed has_clean_result task in no question's evidence."""
    repo, paths = fixture
    _make_task(
        repo,
        400,
        "completed",
        body=_task_body(
            "Task 400",
            relates_to=None,
            has_clean_result=True,
            promoted_at="2026-05-19T00:00:00Z",
        ),
    )
    report = ld.check(paths=paths)
    assert not report.ok
    assert any("#400" in p and "no question" in p for p in report.problems)


def test_check_fails_on_stale_state_date(fixture):
    """A linked result promoted after the question's State date is flagged."""
    repo, paths = fixture
    # #207 is linked to a1 (State date 2026-05-20). Promote it later.
    body = _task_body(
        "Task 207",
        relates_to=["a1", "b1"],
        has_clean_result=True,
        promoted_at="2026-05-25T00:00:00Z",
    )
    (repo / "tasks" / "completed" / "207" / "body.md").write_text(body)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("a1" in p and "stale" in p for p in report.problems)


def test_check_fails_on_missing_state_trailer(fixture):
    """An anchored question without a State trailer is structurally flagged."""
    repo, paths = fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text += "\n\n**C1. No state here.** <!-- q:c1 -->\nJust prose, no trailer.\n"
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("c1" in p and "State" in p for p in report.problems)


# ─── apply() ───────────────────────────────────────────────────────────────


def test_apply_replacement_and_changelog_single_commit(fixture):
    repo, paths = fixture
    before = _git_log_count(repo)

    old_state = (
        f"> **State:** {ld.MATURITY_BUDDING} budding · MODERATE · "
        f"updated 2026-05-20 · evidence: #207"
    )
    new_state = (
        f"> **State:** {ld.MATURITY_EVERGREEN} evergreen · HIGH · "
        f"updated {TODAY} · evidence: #207, #300"
    )
    patch = ld.DocPatch(
        changelog_line="bumped A1 to HIGH after #300",
        open_questions_replacements=[(old_state, new_state)],
    )
    touched = ld.apply(300, patch, paths=paths)
    assert paths.open_questions in touched

    text = (repo / "docs" / "open_questions.md").read_text()
    assert new_state in text
    assert old_state not in text
    # Changelog prepended with today's date.
    assert f"**{TODAY}** — bumped A1 to HIGH after #300" in text
    assert ld._CHANGELOG_BEGIN in text

    # Exactly one new commit.
    assert _git_log_count(repo) == before + 1


def test_apply_append_block(fixture):
    repo, paths = fixture
    new_q = (
        "**D1. A brand new question.** <!-- q:d1 -->\n"
        "Prose.\n"
        f"> **State:** {ld.MATURITY_SEEDLING} seedling · LOW · updated {TODAY} · evidence: #300"
    )
    patch = ld.DocPatch(
        changelog_line="added D1",
        open_questions_appends=[new_q],
    )
    ld.apply(300, patch, paths=paths)
    text = (repo / "docs" / "open_questions.md").read_text()
    assert "<!-- q:d1 -->" in text
    questions = ld._collect_question_evidence(text)
    assert 300 in questions["d1"]["evidence"]


def test_apply_touches_papers(fixture):
    repo, paths = fixture
    patch = ld.DocPatch(
        changelog_line="noted a new paper",
        papers_appends=["- New Paper 2026 — relevant to A1."],
    )
    touched = ld.apply(300, patch, paths=paths)
    assert paths.papers in touched
    assert "New Paper 2026" in (repo / "docs" / "papers.md").read_text()


def test_apply_missing_target_raises(fixture):
    _repo, paths = fixture
    patch = ld.DocPatch(
        changelog_line="bad patch",
        open_questions_replacements=[("THIS TEXT DOES NOT EXIST", "x")],
    )
    with pytest.raises(ValueError, match="not found"):
        ld.apply(300, patch, paths=paths)


def test_apply_requires_changelog_line():
    with pytest.raises(ValueError, match="changelog_line"):
        ld.DocPatch.from_dict({"open_questions_appends": ["x"]})


# ─── DocPatch.from_dict ────────────────────────────────────────────────────


def test_docpatch_from_dict_coerces_pairs():
    patch = ld.DocPatch.from_dict(
        {
            "changelog_line": "cl",
            "open_questions_replacements": [["a", "b"]],
            "papers_replacements": [["c", "d"]],
        }
    )
    assert patch.open_questions_replacements == [("a", "b")]
    assert patch.papers_replacements == [("c", "d")]
    assert patch.touches_open_questions()
    assert patch.touches_papers()


# ─── backfill_reverse() ──────────────────────────────────────────────────────


def test_backfill_reverse_writes_missing_reverse_links(fixture):
    repo, paths = fixture
    # #300 is referenced in a question's evidence (doc side) but carries NO
    # relates_to yet; #999 is a dangling evidence id with no task on disk.
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    text = ld._read(paths.open_questions)
    text = ld._add_evidence_to_question(text, "a1", 300)
    text = ld._add_evidence_to_question(text, "a1", 999)
    ld._write_atomic(paths.open_questions, text)

    # The missing back-link is real drift before the backfill.
    assert not ld.check(paths=paths).ok

    result = ld.backfill_reverse(paths=paths)

    assert (300, ["a1"]) in result["changed"]
    assert 999 in result["missing"]
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "300" / "body.md")
    assert fm["relates_to"] == ["a1"]

    # Idempotent: a second run leaves #300 untouched.
    again = ld.backfill_reverse(paths=paths)
    assert all(tid != 300 for tid, _ in again["changed"])
    assert 300 in again["unchanged"]


def test_backfill_reverse_dry_run_does_not_write(fixture):
    repo, paths = fixture
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    text = ld._add_evidence_to_question(ld._read(paths.open_questions), "a1", 300)
    ld._write_atomic(paths.open_questions, text)

    result = ld.backfill_reverse(paths=paths, dry_run=True)

    assert result["dry_run"] is True
    assert (300, ["a1"]) in result["changed"]
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "300" / "body.md")
    assert not fm.get("relates_to")  # nothing written
