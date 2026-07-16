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


# ─── check() degrades on registry/filesystem inconsistency (#725) ──────────


def test_check_degrades_on_drifted_registry_path(fixture, caplog):
    """Mode A (post-#898 stale-entry envelope): a registry entry points at a
    missing dir while the task exists at exactly ONE on-disk status.

    ``find_task_path`` no longer raises — it falls back to the on-disk path
    with a logged drift WARNING (READ path; the registry is never rewritten
    here — repair happens on the task's next registry-writing mutation or
    ``task.py audit --repair --apply``). ``check()`` therefore reports NO
    drift line and the check axes run against the FULL index; the drift is
    still surfaced, via the WARNING.
    """
    import json
    import logging

    repo, paths = fixture
    # Drift #207's registry path to a nonexistent dir. The on-disk
    # tasks/completed/207/ still exists; only the registry POINTS somewhere
    # else — exactly find_task_path's 1-hit stale-entry fallback branch.
    reg_path = repo / "tasks" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text())
    reg["tasks"]["207"]["path"] = "tasks/approved/207"  # nonexistent
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    tw.invalidate_cache()  # drop the cached registry so the new path is read

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.task_workflow"):
        report = ld.check(paths=paths)

    # (1) No drift line — the fallback resolved #207 to tasks/completed/207.
    assert not any("registry/filesystem inconsistent" in p for p in report.problems)

    # (2) Index intact: #207 still contributes relates_to + coverage, so the
    # otherwise-clean fixture stays clean end to end.
    assert report.ok, report.problems

    # (3) The drift IS surfaced, via the find_task_path WARNING naming the
    # task, the stale registry path, and the on-disk path. Both indexers
    # resolve #207, so assert >=1 matching record — the count is an
    # implementation detail, never pinned.
    drift_warnings = [
        r
        for r in caplog.records
        if "REGISTRY says" in r.getMessage() and "found on disk at" in r.getMessage()
    ]
    assert drift_warnings, caplog.text
    assert any(
        "tasks/approved/207" in r.getMessage() and "tasks/completed/207" in r.getMessage()
        for r in drift_warnings
    ), [r.getMessage() for r in drift_warnings]


def test_check_degrades_on_missing_body_md(fixture):
    """Mode B: a registry entry's dir exists but body.md is missing.

    ``tw._read_body → path.read_text()`` raises ``FileNotFoundError``;
    ``check()`` must surface the drifted task as a problem (NOT raise), AND
    the surviving check axes must still run.
    """
    repo, paths = fixture
    # Delete #207's body.md while keeping the registry entry intact AND the
    # directory present — find_task_path returns successfully (dir exists);
    # the subsequent read_text raises.
    body = repo / "tasks" / "completed" / "207" / "body.md"
    body.unlink()

    # (1) Must not raise.
    report = ld.check(paths=paths)

    # (2) Drift finding present, names #207.
    drift_lines = [p for p in report.problems if "registry/filesystem inconsistent" in p]
    assert any("#207" in p for p in drift_lines), report.problems
    assert len(drift_lines) == 1, drift_lines

    # (3) Surviving axes still ran: same bidirectional flag as Mode A — #207's
    # drop from the relates index has the same downstream effect whether it
    # was Mode A or Mode B.
    nondrift = [p for p in report.problems if "registry/filesystem inconsistent" not in p]
    assert any("207" in p and "a1" in p for p in nondrift), nondrift
    assert not report.ok


def test_check_degrades_on_dir_missing_everywhere(fixture):
    """Mode C: the registry entry is intact but the task dir is gone from
    EVERY status folder — ``find_task_path``'s 0-hit branch still raises
    ``FileNotFoundError``, and ``check()`` must degrade + flag (the #725
    contract survives the #898 envelope for genuinely-unresolvable entries).
    """
    import shutil

    repo, paths = fixture
    shutil.rmtree(repo / "tasks" / "completed" / "207")

    # (1) Must not raise.
    report = ld.check(paths=paths)

    # (2) Drift finding present, names #207; #208 stays well-formed → the
    # two indexers' findings dedup to exactly one drift line.
    drift_lines = [p for p in report.problems if "registry/filesystem inconsistent" in p]
    assert any("#207" in p for p in drift_lines), report.problems
    assert len(drift_lines) == 1, drift_lines

    # (3) Surviving axes still ran: #207's drop from the relates index leaves
    # a1's evidence backlink dangling — same downstream effect as Mode B.
    nondrift = [p for p in report.problems if "registry/filesystem inconsistent" not in p]
    assert any("207" in p and "a1" in p for p in nondrift), nondrift
    assert not report.ok


def test_check_degrades_on_multi_status_stale_entry(fixture):
    """Mode D (≥2-hit): the registry entry is stale AND the task exists at
    TWO on-disk statuses — ``find_task_path`` raises ``StaleTaskPathError``
    (a ``FileNotFoundError`` subclass), and ``check()`` degrades + flags
    exactly as for any unresolvable entry.
    """
    import json
    import shutil

    repo, paths = fixture
    shutil.copytree(repo / "tasks" / "completed" / "207", repo / "tasks" / "running" / "207")
    reg_path = repo / "tasks" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text())
    reg["tasks"]["207"]["path"] = "tasks/approved/207"  # nonexistent
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    tw.invalidate_cache()

    report = ld.check(paths=paths)  # (1) no raise
    drift_lines = [p for p in report.problems if "registry/filesystem inconsistent" in p]
    assert any("#207" in p for p in drift_lines), report.problems  # (2) flagged
    assert not report.ok


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


def test_app_anchor_is_carrier_exempt(belief_fixture):
    """Application anchors (app1..app6, app-<slug>) are a render-only class:
    carrier='app', zero evidence edges (inline #N in their free-text Status
    prose are NOT parsed), and check() does not flag them for a missing
    carrier. The dashboard's TS parser ports this contract, so pin it."""
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text += (
        "\n\n## Applications\n\n"
        "- **App 9 — Test application** (gloss). **Status: idea.** "
        "Seeds: prior work #207, depends on #208. <!-- q:app9 -->\n"
    )
    (repo / "docs" / "open_questions.md").write_text(text)

    questions = ld._collect_question_evidence(ld._read(paths.open_questions))
    assert questions["app9"]["carrier"] == "app"
    assert questions["app9"]["evidence"] == []  # inline #207/#208 are NOT edges
    assert questions["app9"]["has_state"] is False

    report = ld.check(paths=paths)
    assert not any("app9" in p for p in report.problems), report.problems


def test_bare_app_anchor_is_not_carrier_exempt(belief_fixture):
    """The app-exemption regex must NOT match the bare id 'app' — a `q:app`
    anchor with no carrier is a real missing-carrier error, not a render-only
    Application node (guards _APP_ANCHOR_RE against an over-broad match)."""
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text += "\n\n**Bare app, no carrier.** <!-- q:app -->\nJust prose.\n"
    (repo / "docs" / "open_questions.md").write_text(text)

    questions = ld._collect_question_evidence(ld._read(paths.open_questions))
    assert questions["app"]["carrier"] == "none"

    report = ld.check(paths=paths)
    assert not report.ok
    assert any("app" in p and "no" in p.lower() for p in report.problems)


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


# ─── Regression: q: / q- prefix tolerance (issue #468 fallout) ─────────────
#
# The /issue Step 0c-link gate and the SKILL.md examples use the
# ``q:beh-b-to-bprime`` / ``q-app5`` prefixed forms (mirroring the
# anchor syntax). Pre-fix link() lowercased the input but kept the
# ``q:`` prefix, so _find_anchor_line never matched (anchors capture the
# id WITHOUT the prefix) — every call stubbed a junk ``q:<id>``
# question. _normalize_qid strips the prefix at the input boundary so
# prefixed and bare inputs produce byte-identical state.


def test_normalize_qid_strips_q_colon_prefix():
    """`q:foo` and `foo` normalize to the same bare id (`foo`)."""
    assert ld._normalize_qid("q:foo") == "foo"
    assert ld._normalize_qid("Q:Foo") == "foo"
    assert ld._normalize_qid("q-Foo") == "foo"
    assert ld._normalize_qid("foo") == "foo"
    assert ld._normalize_qid("  q:beh-b-to-bprime  ") == "beh-b-to-bprime"
    # Only one prefix is stripped — `q:q:foo` becomes `q:foo`. This is
    # intentional: a bare id should never legitimately START with a
    # second `q:` prefix, and stripping more would mask user typos.
    assert ld._normalize_qid("q:q:foo") == "q:foo"


def test_normalize_qid_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        ld._normalize_qid("q:")
    with pytest.raises(ValueError, match="non-empty"):
        ld._normalize_qid("   ")


def test_link_accepts_prefixed_and_bare_qids_identically(belief_fixture):
    """`link <N> q:x1` and `link <N> x1` produce byte-identical state.

    Pre-fix bug: the prefixed form failed _find_anchor_line (anchors
    capture `x1`, not `q:x1`), then `link` falsely stubbed `q:x1` —
    creating a malformed `<!-- q:q:x1 -->` anchor (the stub composed
    its own anchor by re-prefixing the input). This pins the
    prefix-tolerant boundary.
    """
    repo, paths = belief_fixture
    # Snapshot the doc + fixture state pre-link so we can compare.
    initial_doc = (repo / "docs" / "open_questions.md").read_text()

    _make_task(
        repo,
        400,
        "completed",
        body=_task_body("Task 400", relates_to=None, has_clean_result=True),
    )
    r_prefix = ld.link(400, ["q:x1"], paths=paths)
    doc_prefix = (repo / "docs" / "open_questions.md").read_text()
    fm_prefix, _ = tw._read_body(repo / "tasks" / "completed" / "400" / "body.md")

    # Reset doc + fixture; relink with bare id from the same starting state.
    (repo / "docs" / "open_questions.md").write_text(initial_doc)
    _make_task(
        repo,
        401,
        "completed",
        body=_task_body("Task 401", relates_to=None, has_clean_result=True),
    )
    r_bare = ld.link(401, ["x1"], paths=paths)
    doc_bare = (repo / "docs" / "open_questions.md").read_text()
    fm_bare, _ = tw._read_body(repo / "tasks" / "completed" / "401" / "body.md")

    # Same shape (modulo the task id). No `q:q:` anchor anywhere.
    assert r_prefix["relates_to"] == ["x1"]
    assert r_bare["relates_to"] == ["x1"]
    assert r_prefix["stubbed"] == []
    assert r_bare["stubbed"] == []
    assert fm_prefix["relates_to"] == ["x1"]
    assert fm_bare["relates_to"] == ["x1"]
    assert "<!-- q:q:x1 -->" not in doc_prefix
    assert "<!-- q:q:x1 -->" not in doc_bare

    # The Evidence lines only differ by the appended task id.
    assert "**Evidence:** #207, #208, #400." in doc_prefix
    assert "**Evidence:** #207, #208, #401." in doc_bare


# ─── Regression: atomicity — partial-apply on mid-list failure (issue #468) ─
#
# Pre-fix order: write+commit body.md FIRST, then iterate doc edits. A
# raise on the second id (e.g. "no carrier") left body.md committed with
# the merged relates_to while the doc was untouched, and re-running the
# same call kept re-appending the (already-merged) ids to body.md. Fix:
# validate the full doc edit in memory FIRST, write body.md + doc + one
# commit atomically.


def test_link_partial_failure_writes_nothing(belief_fixture):
    """If ONE id is unresolvable, body.md AND the doc are untouched.

    Setup: append an anchored question with NEITHER a State trailer nor
    a Belief Evidence line (no parseable carrier) — pre-fix this raised
    AFTER body.md had already been committed with the merged relates_to.
    Post-fix the raise fires before any write.
    """
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    # `q:y1` has an anchor but no carrier — unresolvable for evidence append.
    text += "\n\n**Y1. No carrier here.** <!-- q:y1 -->\nJust prose, no carrier.\n"
    (repo / "docs" / "open_questions.md").write_text(text)
    pre_doc = (repo / "docs" / "open_questions.md").read_text()

    _make_task(
        repo,
        410,
        "completed",
        body=_task_body("Task 410", relates_to=None, has_clean_result=True),
    )
    pre_fm, _ = tw._read_body(repo / "tasks" / "completed" / "410" / "body.md")
    assert not pre_fm.get("relates_to")

    # x1 is fine; y1 raises. The whole call must abort with NO writes.
    with pytest.raises(ValueError, match="no evidence carrier"):
        ld.link(410, ["x1", "y1"], paths=paths)

    # body.md frontmatter unchanged — no relates_to written.
    post_fm, _ = tw._read_body(repo / "tasks" / "completed" / "410" / "body.md")
    assert not post_fm.get("relates_to"), (
        f"body.md should be untouched on partial failure, got {post_fm.get('relates_to')!r}"
    )
    # Doc unchanged — x1's Evidence line still has the pre-link refs only.
    post_doc = (repo / "docs" / "open_questions.md").read_text()
    assert post_doc == pre_doc, "doc should be byte-identical on partial failure"


def test_link_partial_failure_does_not_accumulate_relates_to_on_retry(belief_fixture):
    """Two failed link() calls leave body.md relates_to empty, not double-merged.

    Pre-fix bug: the first call committed `relates_to: [x1, y1]` before
    failing on y1's carrier; the second call re-merged onto that list,
    accumulating duplicates. With atomic validate-first, each failed
    call is a no-op, so relates_to stays clean.
    """
    repo, paths = belief_fixture
    text = (repo / "docs" / "open_questions.md").read_text()
    text += "\n\n**Y1. No carrier.** <!-- q:y1 -->\nJust prose.\n"
    (repo / "docs" / "open_questions.md").write_text(text)

    _make_task(
        repo,
        411,
        "completed",
        body=_task_body("Task 411", relates_to=None, has_clean_result=True),
    )
    with pytest.raises(ValueError, match="no evidence carrier"):
        ld.link(411, ["x1", "y1"], paths=paths)
    with pytest.raises(ValueError, match="no evidence carrier"):
        ld.link(411, ["x1", "y1"], paths=paths)

    fm, _ = tw._read_body(repo / "tasks" / "completed" / "411" / "body.md")
    assert not fm.get("relates_to"), fm.get("relates_to")


# ─── Regression: App anchor linking (issue #468) ───────────────────────────
#
# Application anchors (`<!-- q:app1 -->` through `q:app<n>`, plus
# `q:app-<slug>`) legitimately have no Belief/State/Evidence carrier —
# they carry a free-text `**Status:**` bullet under the `## Applications`
# H2. Pre-fix link() hard-raised on them (no carrier found), so an App
# id could never be linked. Fix: relates_to-only link, doc text
# untouched. `_check_bidirectional` already treats app anchors as
# carrier-exempt, so the relates_to-only edge is consistent — not silent
# drift.


def _add_app_anchor(repo: Path, slug: str, *, status: str = "idea") -> None:
    """Append a minimal `## Applications` block carrying a `q:<slug>` anchor."""
    text = (repo / "docs" / "open_questions.md").read_text()
    text += (
        f"\n\n## Applications\n\n"
        f"- **App — {slug}** (gloss). **Status: {status}.** "
        f"Notes. <!-- q:{slug} -->\n"
    )
    (repo / "docs" / "open_questions.md").write_text(text)


def test_link_to_existing_app_anchor_writes_relates_to_only(belief_fixture):
    """An existing App anchor accepts the link without touching the doc."""
    repo, paths = belief_fixture
    _add_app_anchor(repo, "app99")
    pre_doc = (repo / "docs" / "open_questions.md").read_text()

    _make_task(
        repo,
        420,
        "completed",
        body=_task_body("Task 420", relates_to=None, has_clean_result=True),
    )
    result = ld.link(420, ["app99"], paths=paths)
    assert result["relates_to"] == ["app99"]
    assert result["stubbed"] == []

    fm, _ = tw._read_body(repo / "tasks" / "completed" / "420" / "body.md")
    assert fm["relates_to"] == ["app99"]
    # Doc must be byte-identical — App anchors carry no parseable carrier
    # and their relates_to-only link is recorded by check() as exempt.
    assert (repo / "docs" / "open_questions.md").read_text() == pre_doc


def test_link_prefixed_app_id_normalizes_and_succeeds(belief_fixture):
    """`q:app99` and `app99` resolve to the same existing Application anchor."""
    repo, paths = belief_fixture
    _add_app_anchor(repo, "app99")

    _make_task(
        repo,
        421,
        "completed",
        body=_task_body("Task 421", relates_to=None, has_clean_result=True),
    )
    result = ld.link(421, ["q:app99"], paths=paths)
    assert result["relates_to"] == ["app99"]
    assert result["stubbed"] == []
    fm, _ = tw._read_body(repo / "tasks" / "completed" / "421" / "body.md")
    assert fm["relates_to"] == ["app99"]


def test_link_mixed_question_and_app_ids_is_atomic_and_consistent(belief_fixture):
    """A mixed call (x1 + app99) leaves the doc consistent: x1 evidence
    updated, App anchor untouched, and `check()` passes — relates_to
    carries BOTH ids, and the bidirectional check exempts the App edge.
    """
    repo, paths = belief_fixture
    _add_app_anchor(repo, "app99")

    _make_task(
        repo,
        422,
        "completed",
        body=_task_body("Task 422", relates_to=None, has_clean_result=True),
    )
    result = ld.link(422, ["x1", "app99"], paths=paths)
    assert sorted(result["relates_to"]) == ["app99", "x1"]
    assert result["stubbed"] == []

    fm, _ = tw._read_body(repo / "tasks" / "completed" / "422" / "body.md")
    assert sorted(fm["relates_to"]) == ["app99", "x1"]

    text = (repo / "docs" / "open_questions.md").read_text()
    # x1 got the new ref; the App bullet is unchanged.
    assert "**Evidence:** #207, #208, #422." in text
    assert "<!-- q:app99 -->" in text
    # check() is clean: the App edge is carrier-exempt.
    report = ld.check(paths=paths)
    assert report.ok, report.render()


def test_link_to_app_anchor_does_not_double_apply(belief_fixture):
    """Linking the same task to the same App id twice is a no-op."""
    repo, paths = belief_fixture
    _add_app_anchor(repo, "app99")

    _make_task(
        repo,
        423,
        "completed",
        body=_task_body("Task 423", relates_to=None, has_clean_result=True),
    )
    ld.link(423, ["app99"], paths=paths)
    ld.link(423, ["app99"], paths=paths)

    fm, _ = tw._read_body(repo / "tasks" / "completed" / "423" / "body.md")
    assert fm["relates_to"] == ["app99"]  # not [app99, app99]


def test_plan_doc_edit_app_anchor_is_pure_noop_on_doc(belief_fixture):
    """The pure planner returns byte-identical text for an existing App anchor."""
    repo, _paths = belief_fixture
    _add_app_anchor(repo, "app99")
    text = (repo / "docs" / "open_questions.md").read_text()
    new_text, stubbed = ld._plan_doc_edit(text, ["app99"], 999)
    assert new_text == text
    assert stubbed == []
