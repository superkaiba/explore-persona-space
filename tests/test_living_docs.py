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


# ─── mark_unmapped() ─────────────────────────────────────────────────────────


def test_check_exempts_intentionally_unmapped(fixture):
    repo, paths = fixture
    # A completed clean-result with no relates_to and no evidence anywhere —
    # this is real coverage drift until it is deliberately exempted.
    _make_task(
        repo,
        400,
        "completed",
        body=_task_body("Task 400", relates_to=None, has_clean_result=True),
    )
    assert not ld.check(paths=paths).ok  # #400 uncovered -> drift

    result = ld.mark_unmapped(400, "no open question fits", paths=paths)
    assert result["living_docs_unmapped"] == "no open question fits"

    rep = ld.check(paths=paths)
    assert rep.ok  # #400 now exempt; no other drift in the fixture
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "400" / "body.md")
    assert fm["living_docs_unmapped"] == "no open question fits"


# ─── Belief-format Evidence carrier (live docs/open_questions.md format) ───
#
# Every question in docs/open_questions.md as of 2026-05-29 uses the
# `> **Belief:** ... **Confidence:** LOW. **Evidence:** #N, #M.` carrier
# instead of the State trailer. These tests fix the pin so the script
# keeps working when the live doc is edited the way it already is.


_BELIEF_FIXTURE_TEMPLATE = """# Open Questions — Belief-format fixture

---

## Thread X

**X1. Does evidence appended single-line work?** <!-- q:x1 -->
Inline prose.
> **Belief:** Working hypothesis. **Confidence:** MODERATE. **Evidence:** #207, #208.

**X2. Does the multi-line Belief carrier work?** <!-- q:x2 -->
> **Belief:** Open; long-running prose that wraps before Confidence.
> *Next: run the followup.*
> **Confidence:** LOW. **Evidence:** #208.

**X3. Does an empty-evidence carrier accept first link?** <!-- q:x3 -->
> **Belief:** Untested. **Confidence:** LOW. **Evidence:** none in-house yet.

**X4. Does a parenthetical-annotated evidence carrier accept appends?** <!-- q:x4 -->
> **Belief:** Mixed. **Confidence:** LOW. **Evidence:** #207 (older), #208 (newer).

**X5. Sentinel + parenthetical aside (live q:identity-what-is-behavior shape).** <!-- q:x5 -->
> **Belief:** Open. **Confidence:** LOW. **Evidence:** none in-house yet (groundwork in #207).
"""


@pytest.fixture
def belief_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Belief-format variant of `fixture` — questions use the live
    docs/open_questions.md carrier (Belief / Confidence / Evidence)
    instead of the State trailer. Tasks #207 and #208 are reconciled
    against the fixture so check() PASSes out of the box.
    """
    repo = tmp_path
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@test.test")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "docs").mkdir()
    (repo / "docs" / "open_questions.md").write_text(_BELIEF_FIXTURE_TEMPLATE)
    (repo / "docs" / "papers.md").write_text("# Papers\n\n- placeholder\n")

    # #207 → x1, x4, x5; #208 → x1, x2, x4 — mirrors the fixture above
    # so check() is clean before any test mutation. (x5's evidence
    # parenthetical names #207, so #207 must back-link to x5 for the
    # bidirectional check to pass.)
    _make_task(
        repo,
        207,
        "completed",
        body=_task_body(
            "Task 207",
            relates_to=["x1", "x4", "x5"],
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
            relates_to=["x1", "x2", "x4"],
            has_clean_result=True,
            promoted_at="2026-05-19T00:00:00Z",
        ),
    )

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init belief fixture")

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


def test_collect_evidence_parses_belief_carrier(belief_fixture):
    """`_collect_question_evidence` reports carrier=belief and parses #N refs."""
    _repo, paths = belief_fixture
    questions = ld._collect_question_evidence(ld._read(paths.open_questions))
    # All five are recognised, all carry the Belief carrier, none has_state.
    for qid in ("x1", "x2", "x3", "x4", "x5"):
        assert questions[qid]["carrier"] == "belief", qid
        assert questions[qid]["has_state"] is False
        assert questions[qid]["date"] is None
    assert questions["x1"]["evidence"] == [207, 208]
    assert questions["x2"]["evidence"] == [208]
    assert questions["x3"]["evidence"] == []
    # Parenthetical annotations don't confuse the #N parser.
    assert questions["x4"]["evidence"] == [207, 208]
    # `none in-house yet (... #207)` still parses #207 from the
    # parenthetical (used for bidirectional drift detection).
    assert questions["x5"]["evidence"] == [207]


def test_link_appends_to_belief_carrier_single_line(belief_fixture):
    """link() against a single-line Belief-format question appends #N before the period."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        300,
        "completed",
        body=_task_body("Task 300", relates_to=None, has_clean_result=True),
    )
    result = ld.link(300, ["x1"], paths=paths)
    assert result["stubbed"] == []

    text = (repo / "docs" / "open_questions.md").read_text()
    # The whole line shape is preserved; #300 lands at the end of the value,
    # before the terminating period.
    assert (
        "> **Belief:** Working hypothesis. **Confidence:** MODERATE. "
        "**Evidence:** #207, #208, #300." in text
    )
    questions = ld._collect_question_evidence(text)
    assert questions["x1"]["evidence"] == [207, 208, 300]


def test_link_appends_to_belief_carrier_multi_line(belief_fixture):
    """link() finds the Evidence on a LATER blockquote line of the same section."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        301,
        "completed",
        body=_task_body("Task 301", relates_to=None, has_clean_result=True),
    )
    ld.link(301, ["x2"], paths=paths)
    text = (repo / "docs" / "open_questions.md").read_text()
    assert "> **Confidence:** LOW. **Evidence:** #208, #301." in text


def test_link_replaces_empty_belief_evidence(belief_fixture):
    """`none in-house yet` is REPLACED with the first #N (not appended to)."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        302,
        "completed",
        body=_task_body("Task 302", relates_to=None, has_clean_result=True),
    )
    ld.link(302, ["x3"], paths=paths)
    text = (repo / "docs" / "open_questions.md").read_text()
    # Whole `none in-house yet` chunk is replaced; doc reads cleanly.
    assert "> **Belief:** Untested. **Confidence:** LOW. **Evidence:** #302." in text
    assert "none in-house yet, #302" not in text


def test_link_is_idempotent_on_belief_carrier(belief_fixture):
    """link()ing the same task twice against a Belief carrier is a no-op."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        303,
        "completed",
        body=_task_body("Task 303", relates_to=None, has_clean_result=True),
    )
    ld.link(303, ["x1"], paths=paths)
    ld.link(303, ["x1"], paths=paths)

    text = (repo / "docs" / "open_questions.md").read_text()
    questions = ld._collect_question_evidence(text)
    assert questions["x1"]["evidence"].count(303) == 1


def test_link_appends_to_belief_carrier_with_annotations(belief_fixture):
    """link() preserves parenthetical annotations on existing #N refs."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        304,
        "completed",
        body=_task_body("Task 304", relates_to=None, has_clean_result=True),
    )
    ld.link(304, ["x4"], paths=paths)
    text = (repo / "docs" / "open_questions.md").read_text()
    assert (
        "> **Belief:** Mixed. **Confidence:** LOW. "
        "**Evidence:** #207 (older), #208 (newer), #304." in text
    )


def test_check_clean_on_belief_fixture(belief_fixture):
    """check() does NOT report the Belief format as structural drift."""
    _repo, paths = belief_fixture
    report = ld.check(paths=paths)
    assert report.ok, report.render()


def test_check_flags_section_with_no_carrier(belief_fixture):
    """A question with neither State trailer nor Belief Evidence line FAILs."""
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text += "\n\n**Y1. No carrier.** <!-- q:y1 -->\nJust prose.\n"
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("y1" in p and "no" in p.lower() for p in report.problems)


def test_check_fails_on_dangling_evidence_belief_carrier(belief_fixture):
    """A Belief-carrier question listing a non-existent task id is flagged."""
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text = ld._add_evidence_to_question(text, "x1", 9999)
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("#9999" in p and "no such task" in p for p in report.problems)


def test_check_fails_on_missing_backlink_belief_carrier(belief_fixture):
    """Belief-carrier evidence → relates_to bidirectional drift is caught."""
    repo, paths = belief_fixture
    # Inject: add #208 to x4's evidence WITHOUT updating #208's relates_to.
    # (x4's existing evidence is [207, 208 from annotation]... actually 208
    # IS already in x4 from the fixture annotations [207 (older), 208 (newer)],
    # so we need a different lever — add #207 to x2 (currently [208]).
    text = (repo / "docs" / "open_questions.md").read_text()
    text = ld._add_evidence_to_question(text, "x2", 207)
    (repo / "docs" / "open_questions.md").write_text(text)

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("x2" in p and "#207" in p for p in report.problems)


def test_stub_question_round_trips(belief_fixture):
    """A freshly-stubbed question (State format) accepts a second link()."""
    repo, paths = belief_fixture
    _make_task(
        repo,
        305,
        "completed",
        body=_task_body("Task 305", relates_to=None, has_clean_result=True),
    )
    _make_task(
        repo,
        306,
        "completed",
        body=_task_body("Task 306", relates_to=None, has_clean_result=True),
    )
    # First link stubs the question.
    r1 = ld.link(305, ["znew"], paths=paths)
    assert r1["stubbed"] == ["znew"]
    # Second link appends to the State carrier inside the stub.
    r2 = ld.link(306, ["znew"], paths=paths)
    assert r2["stubbed"] == []

    text = (repo / "docs" / "open_questions.md").read_text()
    questions = ld._collect_question_evidence(text)
    assert questions["znew"]["carrier"] == "state"
    assert sorted(questions["znew"]["evidence"]) == [305, 306]


def test_append_to_belief_evidence_line_unit():
    """Direct unit test for the appender (no fixture / fs round-trip)."""
    line = "> **Belief:** Foo. **Confidence:** LOW. **Evidence:** #1, #2."
    out = ld._append_to_belief_evidence_line(line, 3, q_id="x1")
    assert out == ("> **Belief:** Foo. **Confidence:** LOW. **Evidence:** #1, #2, #3.")

    # No-period form (Belief Evidence line ending without trailing dot).
    line2 = "> **Evidence:** #1, #2"
    out2 = ld._append_to_belief_evidence_line(line2, 3, q_id="x1")
    assert out2 == "> **Evidence:** #1, #2, #3"

    # Empty evidence value: replaced, not appended.
    line3 = "> **Belief:** Untested. **Confidence:** LOW. **Evidence:** none in-house yet."
    out3 = ld._append_to_belief_evidence_line(line3, 7, q_id="x1")
    assert out3 == ("> **Belief:** Untested. **Confidence:** LOW. **Evidence:** #7.")

    # Idempotent: re-adding an existing id is a no-op.
    line4 = "> **Evidence:** #1, #2, #3."
    out4 = ld._append_to_belief_evidence_line(line4, 2, q_id="x1")
    assert out4 == line4

    # All sentinel phrases (case-insensitive, trailing-period-tolerant) →
    # REPLACE. None of these has any prose worth preserving.
    for sentinel in (
        "none in-house yet",
        "none in-house yet.",
        "None in-house yet",
        "none yet",
        "tbd",
        "TBD.",
        "none",
        "None",
    ):
        line_s = f"> **Evidence:** {sentinel}"
        out_s = ld._append_to_belief_evidence_line(line_s, 42, q_id="x")
        # tail-period preserved (the regex captures it on the original line)
        assert out_s.endswith("#42") or out_s.endswith("#42."), out_s
        assert sentinel not in out_s, (
            f"sentinel '{sentinel}' should have been REPLACED, got: {out_s}"
        )


# ─── Regression: parenthetical-aside Evidence (the bug in the code review) ─
#
# The live `docs/open_questions.md` line for q:identity-what-is-behavior
# carries:
#     > **Evidence:** none in-house yet (definitional groundwork tracked in #428).
# Pre-fix `_append_to_belief_evidence_line` gated the REPLACE path on
# "no #N refs parsed from the value", so #428 (inside the parenthetical)
# falsely triggered the APPEND path AFTER `none in-house yet` produced
# the awkward `none in-house yet (... #428), #N.`. The fix gates REPLACE
# on a sentinel-string match instead, keeping the parenthetical and
# appending #N at the end. These tests pin that behavior.


def test_append_preserves_parenthetical_aside_around_existing_ref():
    """Live q:identity-what-is-behavior shape — #N appends, prose preserved."""
    line = (
        "> **Belief:** Open — no settled definition. **Confidence:** LOW. "
        "**Evidence:** none in-house yet (definitional groundwork tracked in #428)."
    )
    out = ld._append_to_belief_evidence_line(line, 999, q_id="identity-what-is-behavior")
    expected = (
        "> **Belief:** Open — no settled definition. **Confidence:** LOW. "
        "**Evidence:** none in-house yet (definitional groundwork tracked in #428), #999."
    )
    assert out == expected
    # The aside must still be present verbatim.
    assert "(definitional groundwork tracked in #428)" in out
    # And #428 must NOT appear a second time (no double-counting).
    assert out.count("#428") == 1


def test_append_idempotent_on_id_inside_parenthetical():
    """If the linked #N is already inside the parenthetical aside, it's a no-op."""
    line = "> **Evidence:** none in-house yet (definitional groundwork tracked in #428)."
    out = ld._append_to_belief_evidence_line(line, 428, q_id="identity-what-is-behavior")
    assert out == line


def test_link_against_x5_appends_does_not_replace(belief_fixture):
    """End-to-end: linking a fresh task to x5 (sentinel + parenthetical-#207)
    APPENDs `, #N` rather than wiping the aside via the empty-value REPLACE path.
    """
    repo, paths = belief_fixture
    _make_task(
        repo,
        310,
        "completed",
        body=_task_body("Task 310", relates_to=None, has_clean_result=True),
    )
    ld.link(310, ["x5"], paths=paths)
    text = (repo / "docs" / "open_questions.md").read_text()
    assert (
        "> **Belief:** Open. **Confidence:** LOW. **Evidence:** "
        "none in-house yet (groundwork in #207), #310." in text
    )
    # The aside is still present — the bug would have produced
    # `**Evidence:** #310.` and dropped the (groundwork in #207) prose.
    assert "(groundwork in #207)" in text

    # Side-effect parity: the task frontmatter relates_to AND the doc
    # text both reflect the link (the reviewer flagged frontmatter ↔ doc
    # drift as a side-effect of the bug).
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "310" / "body.md")
    assert fm["relates_to"] == ["x5"]
    questions = ld._collect_question_evidence(text)
    assert 310 in questions["x5"]["evidence"]


def test_link_does_not_infer_parenthetical_id_for_backfill(belief_fixture):
    """backfill_reverse infers relates_to from the parsed #N set in the
    Evidence value — which DOES include #N inside parentheticals. This
    is desired (#207 in x5's parenthetical IS evidence that #207 relates
    to x5, and the fixture's #207 relates_to confirms this). This test
    pins that the bidirectional check stays clean on the as-shipped
    fixture (no drift introduced by parenthetical-#N parsing).
    """
    _repo, paths = belief_fixture
    # The fixture wires #207 → x5 explicitly to mirror the parenthetical
    # citation. check() must PASS — i.e., the parenthetical #207 is not
    # treated as dangling, and #207's relates_to back-link is satisfied.
    report = ld.check(paths=paths)
    assert report.ok, report.render()
