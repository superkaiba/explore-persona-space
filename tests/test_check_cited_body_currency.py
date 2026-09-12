"""Tests for scripts/check_cited_body_currency.py — the #2384 pre-persist gate.

Twelve cases per plan #2384 §3.5: verdict semantics (STALE/CLEAN), the
fail-soft contract (unresolvable id / git failure / internal crash all exit
0 — exit 3 is reachable ONLY from a positively-established stale citation),
extraction filters (self-reference, code fences, URL-adjacency, the
slash-separated lineage-list idiom, the id cap), the lost-DRAFT_START
oldest-breadcrumb re-derivation, and the SKILL.md gate-prose durability pin.

Fixture: the fake-repo pattern of tests/test_task_workflow_list_children.py
(git-init tmp_path, task_workflow resolvers rebound); commits carry
deterministic GIT_AUTHOR_DATE/GIT_COMMITTER_DATE so last-commit-unix reads
are exact. The helper is loaded by path so the WORKTREE copy under test is
the one exercised.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "check_cited_body_currency", REPO_ROOT / "scripts" / "check_cited_body_currency.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_cited_body_currency", mod)
    spec.loader.exec_module(mod)
    return sys.modules["check_cited_body_currency"]


ccb = _load_helper()

# Deterministic timeline (unix seconds). REF is the draft-start reference;
# BEFORE/AFTER bracket it.
T_INIT = 1_699_000_000
BEFORE = 1_700_050_000
REF = 1_700_100_000
AFTER = 1_700_200_000


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> str:
    """Run git in ``repo``, raising on failure; returns stdout (most callers
    ignore it, the rename-chase tests read shas from it)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _commit(repo: Path, paths: list[Path], msg: str, unix: int) -> None:
    """Stage ``paths`` and commit with a pinned author+committer date."""
    stamp = f"{unix} +0000"
    _git(repo, "add", "--", *[str(p) for p in paths])
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        msg,
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )


def _iso(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_task(repo: Path, issue: int, status: str = "completed") -> Path:
    folder = repo / "tasks" / status / str(issue)
    folder.mkdir(parents=True)
    (folder / "body.md").write_text(f"---\nid: {issue}\n---\n\n# Task {issue}\n")
    return folder


def _write_events(folder: Path, dispatch_unixes: list[int]) -> None:
    rows = [
        {
            "ts": _iso(u),
            "kind": "epm:progress",
            "note": f"planner-dispatch round={i + 1} plan_version=v{i + 1}",
        }
        for i, u in enumerate(dispatch_unixes)
    ]
    (folder / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """git-init tmp_path and rebind task_workflow's resolvers to it."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    _git(tmp_path, "config", "user.email", "test@test.test")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / ".gitkeep").write_text("")
    _commit(tmp_path, [tmp_path / ".gitkeep"], "init", T_INIT)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    (tmp_path / "tasks").mkdir()
    return tmp_path


def _run(plan_text: str, tmp_path: Path, argv_tail: list[str], capsys) -> tuple[int, str, str]:
    plan = tmp_path / "draft.md"
    plan.write_text(plan_text)
    rc = ccb.main(["--plan-file", str(plan), *argv_tail])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# ── 1-2: verdict semantics ──────────────────────────────────────────────────


def test_stale_cited_body_detected(fake_repo, capsys):
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)
    rc, out, err = _run(
        "Plan grounds on the #9991 result.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 3
    assert f"CITED-BODY-CURRENCY: STALE ids=9991 checked=1 since={REF}" in out
    # The stale-detail block (stderr) names the cited body so the planner can
    # re-ground without re-deriving the window.
    assert "stale cited body #9991" in err


def test_clean_when_cited_body_predates_reference(fake_repo, capsys):
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    rc, out, _ = _run(
        "Plan grounds on the #9991 result.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 0
    assert f"CITED-BODY-CURRENCY: CLEAN checked=1 since={REF}" in out


# ── 3-5: fail-soft contract (exit 3 ONLY from positive staleness) ───────────


def test_failsoft_on_unresolvable_id(fake_repo, capsys):
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    rc, out, _ = _run(
        "Grounds on #9991 and the never-filed #9999.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 0
    assert f"CLEAN checked=1 since={REF} unresolved=1" in out


def test_failsoft_on_git_failure(tmp_path, monkeypatch, capsys):
    """Tasks tree with NO git repo: resolve_repo_root raises -> the one
    top-level handler prints UNKNOWN and returns 0 (never exit 3)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    folder = tmp_path / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text("# Task 9991\n")
    plan = tmp_path / "draft.md"
    plan.write_text("Grounds on #9991.")
    rc = ccb.main(["--issue", "8000", "--since-unix", str(REF), "--plan-file", str(plan)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CITED-BODY-CURRENCY: UNKNOWN reason=RuntimeError: git repo unresolvable" in out


def test_failsoft_on_internal_crash(fake_repo, monkeypatch, capsys):
    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    # Patch the extractor `check()` actually calls. Round 2 added the
    # cap-disclosure variant and made it the callee; `extract_cited_ids` is now
    # a thin wrapper, so patching THAT would no longer reach the code path and
    # the fail-soft contract would go untested.
    monkeypatch.setattr(ccb, "extract_cited_ids_with_total", _boom)
    rc, out, _ = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 0
    assert "CITED-BODY-CURRENCY: UNKNOWN reason=RuntimeError: boom" in out


# ── 6-9, 11: extraction filters ─────────────────────────────────────────────


def test_self_reference_excluded(fake_repo, capsys):
    rc, out, _ = _run(
        "This task (#8000) cites nothing else.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 0
    assert f"CLEAN checked=0 since={REF}" in out


def test_code_fence_refs_ignored(fake_repo, capsys):
    # 9991's body is committed AFTER the reference, so any extraction leak
    # from the fence / indented block would flip the verdict to STALE.
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)
    plan = (
        "# Plan\n\nProse without citations.\n\n"
        "```bash\nuv run python scripts/foo.py --parent 9991  # see #9991\n```\n\n"
        "    indented code referencing #9991\n\nDone.\n"
    )
    assert ccb.extract_cited_ids(plan, self_issue=8000) == []
    rc, out, _ = _run(plan, fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys)
    assert rc == 0
    assert f"CLEAN checked=0 since={REF}" in out


def test_url_adjacent_refs_ignored(fake_repo, capsys):
    # The drop class is `\w` ONLY (plan #2384 §2.2): word-adjacent forms like
    # `issues#9991` / `plan.md#9991` are anchors, not citations.
    ids = ccb.extract_cited_ids(
        "See issues#9991 and plan.md#9991 but #9992 is a real citation.", self_issue=1
    )
    assert ids == [9992]
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)
    rc, out, _ = _run(
        "See issues#9991 and plan.md#9991 anchors only.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 0
    assert f"CLEAN checked=0 since={REF}" in out


def test_slash_separated_refs_extracted(fake_repo, capsys):
    # `#884/#1045/#1134`-style lineage lists are ~10% of real citations; a
    # widened `[/\w.-]` drop class would lose every non-first member.
    assert ccb.extract_cited_ids("Lineage #9991/#9992/#9993.", self_issue=1) == [
        9991,
        9992,
        9993,
    ]
    for issue in (9991, 9992, 9993):
        folder = _make_task(fake_repo, issue)
        _commit(fake_repo, [folder / "body.md"], f"persist #{issue} body", BEFORE)
    rc, out, _ = _run(
        "Lineage #9991/#9992/#9993.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(REF)],
        capsys,
    )
    assert rc == 0
    assert f"CLEAN checked=3 since={REF}" in out


def test_id_cap_enforced():
    text = " ".join(f"#{100 + i}" for i in range(60))
    ids = ccb.extract_cited_ids(text, self_issue=1)
    assert len(ids) == ccb._MAX_CITED_IDS == 40
    assert ids == [100 + i for i in range(40)]


# ── 10: lost DRAFT_START re-derivation ──────────────────────────────────────


@pytest.mark.parametrize("since_argv", [["--since-unix", ""], []])
def test_lost_draft_start_rederives_from_oldest_breadcrumb(fake_repo, capsys, since_argv):
    """Empty AND absent --since-unix both re-derive from the OLDEST
    planner-dispatch breadcrumb: the cited body's correction (t2) sits
    between the old breadcrumb (t1) and the new one (t3), so STALE proves
    the oldest was used (the newest would certify the window CLEAN)."""
    t1, t2, t3 = 1_700_000_500, 1_700_050_000, 1_700_150_000
    own = _make_task(fake_repo, 8000, status="running")
    _write_events(own, [t1, t3])
    cited = _make_task(fake_repo, 9991)
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", t2)
    rc, out, _ = _run(
        "Plan grounds on the #9991 result.", fake_repo, ["--issue", "8000", *since_argv], capsys
    )
    assert rc == 3
    assert f"CITED-BODY-CURRENCY: STALE ids=9991 checked=1 since={t1}" in out


# ── 12: SKILL.md gate-prose durability pin ──────────────────────────────────


def test_skillmd_names_cited_body_gate():
    """The adversarial-planner SKILL.md carries the gate section + the
    two->three pre-persist-duties correction, and 04-step-2.md carries the
    matching summary — the prose the whole gate hangs off (#2384 §3.2/§3.3)."""
    skill = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    assert "Cited-body currency gate" in skill
    assert "scripts/check_cited_body_currency.py" in skill
    assert "two pre-persist duties" not in skill
    assert "three pre-persist duties" in skill
    step2 = (REPO_ROOT / ".claude" / "skills" / "issue" / "steps" / "04-step-2.md").read_text()
    assert "**Cited-body currency gate:**" in step2
    assert "scripts/check_cited_body_currency.py" in step2


# ── ROUND-2 regressions (#2384 code-review v1 blockers 2, 5, 7, 8, 9, 11, 12) ─


def test_repo_root_pairs_with_the_worktree_holding_tasks_dir(tmp_path, monkeypatch, capsys):
    """Blocker 2 (Critical). ``task_workflow`` may route reads through the
    managed ``_task-main-pin`` LINKED worktree, so ``tasks_dir()`` resolves
    inside it while ``dirname(--git-common-dir)`` names the PRIMARY checkout.

    The pre-fix pairing ran ``git -C <primary> log -- <pin path>``: git exits
    0 with EMPTY output for a path in another working tree, so every id
    counted ``git_failed`` and the BLOCKING leg silently stopped blocking.
    Asserts (a) the root is the linked worktree's own toplevel and (b) a
    genuinely stale body is still caught through it (exit 3)."""
    primary = tmp_path / "primary"
    primary.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=primary, check=True)
    _git(primary, "config", "user.email", "test@test.test")
    _git(primary, "config", "user.name", "test")
    _git(primary, "config", "commit.gpgsign", "false")
    (primary / ".gitkeep").write_text("")
    _commit(primary, [primary / ".gitkeep"], "init", T_INIT)

    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-q", "--detach", str(linked), "HEAD")

    folder = linked / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text("# Task 9991\n")
    _commit(linked, [folder / "body.md"], "correct #9991 body", AFTER)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "tasks_dir", lambda: linked / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: linked / "tasks" / "REGISTRY.json")

    # (a) the root is the tree that HOLDS tasks_dir(), not the primary checkout.
    assert ccb.resolve_repo_root().resolve() == linked.resolve()
    assert ccb.resolve_repo_root().resolve() != primary.resolve()

    # (b) and the blocking leg still blocks through it.
    plan = tmp_path / "draft.md"
    plan.write_text("Grounds on the #9991 result.")
    rc = ccb.main(["--issue", "8000", "--since-unix", str(REF), "--plan-file", str(plan)])
    out = capsys.readouterr().out
    assert rc == 3, out
    assert "STALE ids=9991 checked=1" in out


def test_repo_root_pairing_regression_would_have_failed_pre_fix(tmp_path, monkeypatch):
    """Blocker 2, mutation proof: re-running the OLD
    ``dirname(--git-common-dir)`` construction against the same linked
    worktree yields the PRIMARY checkout — the wrong tree — so the fixture
    above genuinely discriminates the fix from the defect."""
    primary = tmp_path / "primary"
    primary.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=primary, check=True)
    _git(primary, "config", "user.email", "test@test.test")
    _git(primary, "config", "user.name", "test")
    _git(primary, "config", "commit.gpgsign", "false")
    (primary / ".gitkeep").write_text("")
    _commit(primary, [primary / ".gitkeep"], "init", T_INIT)
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-q", "--detach", str(linked), "HEAD")
    tasks = linked / "tasks"
    tasks.mkdir()

    old = Path(
        ccb._git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=tasks
        ).splitlines()[-1]
    ).parent
    new = Path(ccb._git(["rev-parse", "--show-toplevel"], cwd=tasks).splitlines()[-1])
    assert old.resolve() == primary.resolve()
    assert new.resolve() == linked.resolve()
    assert old.resolve() != new.resolve()


@pytest.mark.parametrize(
    ("label", "plan_text", "expected"),
    [
        # A tilde run must not close a backtick fence.
        ("mixed_delimiters", "#111\n```\n#222\n~~~\n#333\n```\n#444", [111, 444]),
        # An inner ``` must not close a ```` block.
        ("longer_opener", "#111\n````\n#222\n```\n#333\n````\n#444", [111, 444]),
        # 4-space indent => indented CODE, not a fence: the block never opens,
        # so #222 stays visible prose.
        ("indented_is_not_a_fence", "#111\n    ```\n#222", [111, 222]),
        # 3-space indent IS a valid fence opener.
        ("three_space_indent_is_a_fence", "#111\n   ```\n#222\n   ```\n#333", [111, 333]),
        # An unclosed fence swallows the rest of the document (CommonMark).
        ("unclosed_fence", "#111\n```\n#222", [111]),
        # A closing fence may not carry an info string.
        ("close_needs_bare_delim", "#111\n```\n#222\n``` js\n#333\n```\n#444", [111, 444]),
    ],
)
def test_fence_semantics_are_delimiter_aware(label, plan_text, expected):
    """Blocker 5 (Major). The old toggle fired on any stripped line starting
    with three backticks/tildes, so a mismatched delimiter, a shorter inner
    run, or an INDENTED code line inverted in/out for the whole rest of the
    document — silently dropping real citations or harvesting ids out of
    command examples."""
    assert ccb.extract_cited_ids(plan_text, self_issue=1) == expected, label


@pytest.mark.parametrize("bad", ["-1", "0", "999999999", "99999999999"])
def test_implausible_reference_is_unknown_never_a_verdict(fake_repo, capsys, bad):
    """Blocker 7 (Major). A non-positive / far-past / far-future reference
    silently INVERTS the gate — too far past marks every body stale, too far
    future certifies every body clean. Both are UNKNOWN + exit 0, never a
    verdict."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)
    rc, out, _ = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", bad], capsys
    )
    assert rc == 0
    assert "UNKNOWN reason=implausible draft-start reference" in out
    assert "STALE" not in out and "CLEAN" not in out


def test_plausible_reference_still_verdicts(fake_repo, capsys):
    """Blocker 7 companion: the plausibility fence must not swallow the
    ordinary case — a reference just inside the window still verdicts."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)
    rc, out, _ = _run(
        "Grounds on #9991.",
        fake_repo,
        ["--issue", "8000", "--since-unix", str(ccb._MIN_PLAUSIBLE_REF_UNIX)],
        capsys,
    )
    assert rc == 3
    assert "STALE ids=9991" in out


def test_status_move_is_labelled_rename_only(fake_repo, capsys):
    """Blocker 8 (Major). A ``git mv`` between status folders is #2384 §6's
    DOMINANT false-positive channel, and the label is its whole mitigation.

    Pre-fix the label was decided from the diff text, but a path-limited
    ``git diff <oldest>^..HEAD -- <new path>`` cannot show a rename at all:
    at ``<oldest>^`` nothing existed at that path, so the body rendered as
    ADDED lines and the "zero changed lines" predicate was unreachable — the
    label never fired, and the planner got a full-file diff of noise."""
    folder = _make_task(fake_repo, 9991, status="running")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    dest = fake_repo / "tasks" / "completed" / "9991"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # `git mv` stages both sides itself; re-adding the vanished source path
    # would exit 128.
    _git(fake_repo, "mv", str(folder), str(dest))
    stamp = f"{AFTER} +0000"
    _git(
        fake_repo,
        "commit",
        "-q",
        "-m",
        "task #9991: running -> completed",
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out
    assert "stale cited body #9991" in err
    assert "[rename-only]" in err
    # The full-file "addition" diff is suppressed under the label.
    assert "+# Task 9991" not in err


def test_content_edit_is_not_labelled_rename_only(fake_repo, capsys):
    """Blocker 8 companion / mutation guard: a real content correction must
    NOT pick up the rename-only label (which would tell the planner to ignore
    exactly the change the gate exists to surface)."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text("---\nid: 9991\n---\n\n# Task 9991\n\nCorrected result.\n")
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out
    assert "[rename-only]" not in err
    assert "Corrected result." in err


def _status_move(repo: Path, src: Path, dest: Path, unix: int, msg: str) -> None:
    """``git mv src dest`` committed at ``unix``. ``git mv`` stages both sides
    itself, so the vanished source path must not be re-added (exit 128)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "mv", str(src), str(dest))
    stamp = f"{unix} +0000"
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        msg,
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )


# ── ROUND-3 blocker: rename-following history ───────────────────────────────


def test_correction_then_status_move_reaches_the_operator(fake_repo, capsys):
    """Round-3 BLOCKER, on the reconciler's three-commit fixture: persist the
    body, CORRECT it at the OLD path, then ``git mv`` the task folder.

    That is the ordinary correct-then-``set-status`` sequence. Pre-fix, the
    path-limited ``git log`` TRUNCATED at the rename and
    returned only the ``R100`` move, so (a) the window log hid the correction
    and (b) ``classify_window`` saw an all-``R100`` window, labelled it
    ``rename-only``, and SUPPRESSED the diff. Since
    ``.claude/skills/adversarial-planner/SKILL.md`` makes that detail block the
    entire input to the operator's disposition, the operator saw a status-move
    label and an empty diff, recorded the sanctioned "plan text unaffected"
    disposition, and persisted a plan quoting a stale cited body — incident
    #2378 reproduced, now carrying a record asserting it was checked.

    Asserting the LABEL alone is insufficient: the suppression is the harm, so
    the corrected TEXT must be shown to reach the rendered detail."""
    folder = _make_task(fake_repo, 9991, status="interpreting")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text(
        "---\nid: 9991\n---\n\n# Task 9991\n\nCORRECTED: the sign is NEGATIVE.\n"
    )
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)
    _status_move(
        fake_repo,
        folder,
        fake_repo / "tasks" / "completed" / "9991",
        AFTER,
        "task #9991: interpreting -> completed",
    )

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out
    # (a) the label must NOT claim this window is a bare status move ...
    assert "[rename-only]" not in err
    # (b) ... so the corrected body reaches the operator's detail block ...
    assert "CORRECTED: the sign is NEGATIVE." in err
    # (c) ... and the window log carries the correction commit, not just the
    # move (the reconciler's "observed but not raised": a fix that follows
    # renames only in classify_window still shows an incomplete log).
    assert "correct #9991 result" in err
    assert "interpreting -> completed" in err


def test_multi_status_move_window_is_still_rename_only(fake_repo, capsys):
    """Mutation guard on the blocker fix. The cheap way to stop the mislabel
    is to let the pre-rename commits report NO status (the pre-fix per-commit
    probe keyed on the CURRENT path returned ``None`` for them), which breaks
    ``all(... == "R100")`` — but that also kills the label for a task moved
    through SEVERAL status folders with no edits, i.e. exactly the dominant
    false-positive channel #2384 §6 built the label to mitigate.

    Following renames in the shared history gives each move its own ``R100``,
    so a pure multi-move window keeps the label."""
    folder = _make_task(fake_repo, 9991, status="running")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    mid = fake_repo / "tasks" / "reviewing" / "9991"
    _status_move(fake_repo, folder, mid, AFTER, "task #9991: running -> reviewing")
    _status_move(
        fake_repo,
        mid,
        fake_repo / "tasks" / "completed" / "9991",
        AFTER,
        "task #9991: reviewing -> completed",
    )

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out
    assert "[rename-only]" in err
    assert "+# Task 9991" not in err  # the body diff stays suppressed
    # Both moves are in the window — the log is complete even when suppressed.
    assert "running -> reviewing" in err
    assert "reviewing -> completed" in err


@pytest.mark.parametrize(
    ("leaf", "why"),
    [
        ("naïve-9991", "non-ASCII: git QUOTES it without -z"),
        ("ta\tb-9991", "embedded tab: the old split('\\t') parse lost the path"),
    ],
)
def test_history_parses_quoting_hostile_paths(fake_repo, leaf, why):
    """Round-3 item 5. The round-2 probe read
    ``git show --format= --name-status -M <sha>`` WITHOUT ``-z`` and compared
    ``split('\\t')[-1]`` against the path — so a QUOTED (non-ASCII) or
    tab-bearing path never matched, the status came back ``None``, and the
    ``rename-only`` label silently went missing. ``-z`` disables quoting and
    NUL-delimits the records, so both shapes parse exactly."""
    src = fake_repo / "tasks" / "running" / leaf
    src.mkdir(parents=True)
    (src / "body.md").write_text("original\n")
    _commit(fake_repo, [src / "body.md"], "persist body", BEFORE)
    dest = fake_repo / "tasks" / "completed" / leaf
    _status_move(fake_repo, src, dest, AFTER, "status move")

    old_rel = f"tasks/running/{leaf}/body.md"
    new_rel = f"tasks/completed/{leaf}/body.md"

    # The parse itself: the raw path must survive the NUL-delimited read.
    move_sha = _git(fake_repo, "rev-parse", "HEAD").strip()
    entries = ccb.batch_name_status([move_sha], repo_root=fake_repo)
    assert ccb.commit_path_entry(entries[move_sha], new_rel) == ("R100", old_rel), why

    # ... and reach the window through the ordinary consumer. The persist
    # commit predates REF, so the chase into the old path finds nothing in
    # window and correctly stops — a CONCLUSIVE end (genuine in-window
    # exhaustion): one R100 row carrying both endpoints.
    rows, conclusive = ccb._window_history(new_rel, REF, repo_root=fake_repo)
    assert conclusive, why
    assert [r.status for r in rows] == ["R100"], why
    assert rows[0].path == new_rel and rows[0].prev_path == old_rel, why

    # The harm the parse defect caused, end to end: the advisory label.
    _, label = ccb.body_diff_since(new_rel, REF, repo_root=fake_repo)
    assert label == "rename-only", why


# ── ROUND-4 blocker: inconclusive / spoofed-bound chase ends ────────────────
#
# Reconciler ruling (epm:review-reconcile, round 3): an incomplete or
# wrongly-bounded rename chase must never certify a partial all-R100 window
# as `rename-only` — the label suppresses the corrected-body diff and steers
# the operator to record "plan text unaffected" over a stale citation (the
# #2378 harm). ONE mechanism, two parts: (a) the chase decision keys on the
# oldest IN-WINDOW row, not on segment truncation; (b) `rename-only` is
# emitted ONLY on a conclusive chase end.


def test_prior_incarnation_destination_does_not_hide_the_correction(fake_repo, capsys):
    """Round-4 BLOCKER, instance (iii) — the DECISIVE one: the ordinary
    follow-up re-park / reopen lane. The cited task moves OUT of a status
    folder BEFORE the draft starts, is corrected mid-draft, then moves BACK
    mid-draft.

    ``git log -- <final path>`` then holds the PRE-draft move-out on the SAME
    path, so the pre-fix ``len(window) < len(seg)`` bound check read
    "conclusive", the chase never followed the in-window ``R100``'s source,
    and the correction vanished from log AND diff under a bare
    ``rename-only`` label. The bound heuristic is INVALID here: the file
    ARRIVED in-window, so pre-window commits on the destination path belong
    to a PRIOR INCARNATION and bound nothing — the chase must follow the
    source regardless of the segment bound."""
    folder = _make_task(fake_repo, 9991, status="awaiting_promotion")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    out_folder = fake_repo / "tasks" / "followups_running" / "9991"
    _status_move(
        fake_repo,
        folder,
        out_folder,
        BEFORE + 100,
        "task #9991: awaiting_promotion -> followups_running",
    )
    # [draft starts at REF] ... the correction lands at the moved-out path ...
    (out_folder / "body.md").write_text(
        "---\nid: 9991\n---\n\n# Task 9991\n\nCORRECTED: the sign is NEGATIVE.\n"
    )
    _commit(fake_repo, [out_folder / "body.md"], "fold follow-up correction", AFTER)
    # ... and the task re-parks at the ORIGINAL path, still mid-draft.
    _status_move(
        fake_repo,
        out_folder,
        folder,
        AFTER + 100,
        "task #9991: followups_running -> awaiting_promotion",
    )

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out
    assert "[rename-only]" not in err
    assert "CORRECTED: the sign is NEGATIVE." in err
    assert "fold follow-up correction" in err  # the window log carries it too

    # Control (same fixture, SINCE before the move-out): the whole loop is
    # in-window, the bound plays no part, and the correction stays visible —
    # the defect is specifically the straddle.
    text, label = ccb.body_diff_since(
        "tasks/awaiting_promotion/9991/body.md", BEFORE + 50, repo_root=fake_repo
    )
    assert label != "rename-only"
    assert "CORRECTED: the sign is NEGATIVE." in text


def test_hop_cap_exhaustion_is_inconclusive_never_rename_only(fake_repo, capsys):
    """Round-4 BLOCKER, instance (i): a correction followed by
    ``_MAX_RENAME_CHASE_HOPS + 2`` in-window status moves exhausts the chase
    with the correction still unreached. Pre-fix the gathered window was
    all-``R100`` and earned a bare ``rename-only`` — suppressing the diff on
    exactly the window whose completeness the cap had just truncated. The cap
    is a LATENCY policy: on exhaustion the label is withheld and the window
    stays visible."""
    n_moves = ccb._MAX_RENAME_CHASE_HOPS + 2
    paths = [fake_repo / "tasks" / f"s{i}" / "9991" for i in range(n_moves + 1)]
    paths[0].mkdir(parents=True)
    (paths[0] / "body.md").write_text("original\n")
    _commit(fake_repo, [paths[0] / "body.md"], "persist body", BEFORE)
    (paths[0] / "body.md").write_text("CORRECTED: the sign is NEGATIVE.\n")
    _commit(fake_repo, [paths[0] / "body.md"], "correct body", AFTER)
    for i in range(n_moves):
        _status_move(
            fake_repo, paths[i], paths[i + 1], AFTER + 10 * (i + 1), f"move s{i} -> s{i + 1}"
        )

    rel = f"tasks/s{n_moves}/9991/body.md"
    text, label = ccb.body_diff_since(rel, REF, repo_root=fake_repo)
    # Round-5 tightening (`inconclusive-fixtures-do-not-pin-label-or-diff`):
    # `label != "rename-only"` + truthy `text` let a mutant emitting a
    # DIFFERENT label string ("chase-truncated") while suppressing the diff
    # pass the whole file — an inconclusive end must yield NO label at all,
    # and the gathered diff must actually reach the operator-facing text.
    assert label is None
    assert "diff --git" in text  # the gathered window's diff stays visible, unsuppressed
    assert "inconclusive" in capsys.readouterr().err  # the truncation is disclosed


def test_midchase_log_failure_is_inconclusive_never_rename_only(fake_repo, monkeypatch, capsys):
    """Round-4 BLOCKER, instance (ii): the round-2 incident fixture (persist,
    correct at the OLD path, ``git mv``) with the SECOND path-limited
    ``git log`` — the chase's hop into the rename source — forced to fail.

    Pre-fix the failure surfaced as an EMPTY segment, the loop broke with a
    one-row all-``R100`` window, and the label read ``rename-only`` with one
    generic git-failure stderr line and nothing marking the label
    untrustworthy. A failed probe cannot establish where the window ends, so
    the label is withheld and the window stays visible."""
    folder = _make_task(fake_repo, 9991, status="interpreting")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text("CORRECTED: the sign is NEGATIVE.\n")
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)
    _status_move(
        fake_repo,
        folder,
        fake_repo / "tasks" / "completed" / "9991",
        AFTER + 100,
        "task #9991: interpreting -> completed",
    )

    real_git = ccb._git
    seen = {"log_segments": 0}

    def flaky_git(args, *, cwd):
        if args[0] == "log" and args[1] == "--format=%H%x09%ct%x09%s":
            seen["log_segments"] += 1
            if seen["log_segments"] == 2:
                return None  # what _git returns on rc!=0 / timeout
        return real_git(args, cwd=cwd)

    monkeypatch.setattr(ccb, "_git", flaky_git)
    text, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert seen["log_segments"] >= 2  # the chase's source hop was actually reached
    # Round-5 tightening (`inconclusive-fixtures-do-not-pin-label-or-diff`):
    # same rationale as the hop-cap pin — no label of any kind over a
    # failed-probe window, and the rename diff must reach the shown text.
    assert label is None
    assert "diff --git" in text  # the one-row window log + rename diff stay visible
    assert "inconclusive" in capsys.readouterr().err  # the failure is disclosed


_FM_BODY = (
    "---\nid: 9991\nclassification: pending\n---\n\n# Task 9991\n\n"
    + "\n".join(f"padding line {i}" for i in range(8))
    + "\n\nRESULT: the sign is POSITIVE.\n"
)


def test_hop_cap_exhaustion_withholds_frontmatter_label(fake_repo, capsys):
    """Round-5 fix (ledger `inconclusive-frontmatter-label-overrides-conservative-end`,
    variant: hop-cap exhaustion). A buried in-window content correction, more
    status moves than the chase may cross, then a frontmatter-only edit at
    the final path. Round 4 gated only ``rename-only`` on conclusiveness, so
    the truncated diff — whose only VISIBLE changed lines are the
    ``classification`` flip — still earned ``frontmatter-only`` over a window
    whose omitted portion held the correction, while ``_chase_note``
    simultaneously printed "advisory label withheld". Inconclusive ⇒ NO
    label of any kind; the gathered diff stays visible."""
    n_moves = ccb._MAX_RENAME_CHASE_HOPS + 2
    paths = [fake_repo / "tasks" / f"s{i}" / "9991" for i in range(n_moves + 1)]
    paths[0].mkdir(parents=True)
    (paths[0] / "body.md").write_text(_FM_BODY)
    _commit(fake_repo, [paths[0] / "body.md"], "persist body", BEFORE)
    (paths[0] / "body.md").write_text(
        _FM_BODY.replace("RESULT: the sign is POSITIVE.", "CORRECTED: the sign is NEGATIVE.")
    )
    _commit(fake_repo, [paths[0] / "body.md"], "correct body", AFTER)
    for i in range(n_moves):
        _status_move(
            fake_repo, paths[i], paths[i + 1], AFTER + 10 * (i + 1), f"move s{i} -> s{i + 1}"
        )
    final = paths[n_moves] / "body.md"
    final.write_text(final.read_text().replace("classification: pending", "classification: useful"))
    _commit(fake_repo, [final], "user promotion sweep", AFTER + 10 * (n_moves + 5))

    rel = f"tasks/s{n_moves}/9991/body.md"
    text, label = ccb.body_diff_since(rel, REF, repo_root=fake_repo)
    assert label is None  # NOT "frontmatter-only": a truncated diff certifies nothing
    assert "diff --git" in text  # the gathered diff reaches the operator-facing text
    assert "CORRECTED: the sign is NEGATIVE." not in text  # the harm: correction IS omitted
    assert "inconclusive" in capsys.readouterr().err  # the truncation is disclosed


def test_midchase_log_failure_withholds_frontmatter_label(fake_repo, monkeypatch, capsys):
    """Round-5 fix (same ledger row, variant: mid-chase git failure). Persist,
    content-correct at the OLD path, ``git mv``, then a frontmatter-only edit
    at the NEW path; the chase's source hop is forced to fail. The one-hop
    diff shows only the rename + the ``classification`` flip (the correction
    sits below the failed hop), so round 4 still read ``frontmatter-only``
    over the truncated window. Inconclusive ⇒ NO label; diff stays visible."""
    folder = fake_repo / "tasks" / "interpreting" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text(_FM_BODY)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text(
        _FM_BODY.replace("RESULT: the sign is POSITIVE.", "CORRECTED: the sign is NEGATIVE.")
    )
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)
    dest = fake_repo / "tasks" / "completed" / "9991"
    _status_move(fake_repo, folder, dest, AFTER + 100, "task #9991: interpreting -> completed")
    final = dest / "body.md"
    final.write_text(final.read_text().replace("classification: pending", "classification: useful"))
    _commit(fake_repo, [final], "user promotion sweep", AFTER + 200)

    real_git = ccb._git
    seen = {"log_segments": 0}

    def flaky_git(args, *, cwd):
        if args[0] == "log" and args[1] == "--format=%H%x09%ct%x09%s":
            seen["log_segments"] += 1
            if seen["log_segments"] == 2:
                return None  # what _git returns on rc!=0 / timeout
        return real_git(args, cwd=cwd)

    monkeypatch.setattr(ccb, "_git", flaky_git)
    text, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert seen["log_segments"] >= 2  # the chase's source hop was actually reached
    assert label is None  # NOT "frontmatter-only" over a failed-probe window
    assert "diff --git" in text  # the rename + frontmatter diff stays visible
    assert "CORRECTED: the sign is NEGATIVE." not in text  # the harm: correction IS omitted
    assert "inconclusive" in capsys.readouterr().err  # the failure is disclosed


# ── #2654: frontmatter-only is decided from endpoint BODIES, not diff shape ─


def test_conclusive_word_prefixed_content_correction_is_unlabeled(fake_repo):
    """#2654 acceptance 1 (the pre-fix pin). A CONCLUSIVE window whose only
    change is a ``Word:``-shaped BODY line must NOT read ``frontmatter-only``.

    Pre-fix, the label came from a SHAPE test on changed diff lines:
    ``-RESULT: the sign is POSITIVE.`` and ``+CORRECTED: the sign is
    NEGATIVE.`` both matched ``^[+-][A-Za-z_][A-Za-z0-9_-]*:``, so
    ``all(...)`` held and the window read ``frontmatter-only`` — a genuine
    content correction labeled as a metadata edit, nudging the operator
    toward the "plan text unaffected" disposition (#2384 §6). Post-fix the
    label is decided by comparing the frontmatter-STRIPPED endpoint bodies,
    which differ here."""
    folder = fake_repo / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text(_FM_BODY)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text(
        _FM_BODY.replace("RESULT: the sign is POSITIVE.", "CORRECTED: the sign is NEGATIVE.")
    )
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)

    text, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert label is None  # NOT "frontmatter-only": the BODY changed
    assert "CORRECTED: the sign is NEGATIVE." in text  # the diff reaches the operator


def test_conclusive_frontmatter_only_edit_is_labeled(fake_repo):
    """#2654 acceptance 2 (the do-not-fix-by-deletion positive control, and
    the suite's FIRST positive ``frontmatter-only`` assertion). A genuine
    frontmatter-only edit — the user promotion sweep's ``classification``
    flip — still earns the label under the endpoint-body comparison: the
    frontmatter halves differ while the stripped bodies are byte-equal."""
    folder = fake_repo / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text(_FM_BODY)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text(
        _FM_BODY.replace("classification: pending", "classification: useful")
    )
    _commit(fake_repo, [folder / "body.md"], "user promotion sweep", AFTER)

    _, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert label == "frontmatter-only"


def test_body_without_frontmatter_is_unlabeled(fake_repo):
    """#2654 ``split_frontmatter`` contract guard. A body with NO leading
    ``---`` block whose only change is a ``Word:``-shaped line yields NO
    label. Without the both-endpoints-split requirement this file's empty
    "frontmatter" would compare equal on both sides and the body comparison
    would be the only guard; ``split_frontmatter`` returning ``None`` (never
    ``('', text)``) keeps the label structurally unreachable here."""
    folder = fake_repo / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    (folder / "body.md").write_text("# Task 9991\n\nRESULT: the sign is POSITIVE.\n")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    (folder / "body.md").write_text("# Task 9991\n\nCORRECTED: the sign is NEGATIVE.\n")
    _commit(fake_repo, [folder / "body.md"], "correct #9991 result", AFTER)

    _, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert label is None


def test_conclusive_all_r100_window_skips_endpoint_fetch(fake_repo, monkeypatch):
    """#2654 pin on the ``need_text`` gate's rename-only conjunct: a
    conclusive pure-status-move window still reads ``rename-only`` and the
    two endpoint ``git show`` content reads never run (the plan-§2 cost
    sentence: no wasted reads on a window the rename-only branch decides
    first)."""
    folder = _make_task(fake_repo, 9991, status="running")
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    _status_move(
        fake_repo,
        folder,
        fake_repo / "tasks" / "completed" / "9991",
        AFTER,
        "task #9991: running -> completed",
    )

    real_git_raw = ccb._git_raw
    seen = {"content_shows": 0}

    def counting_git_raw(args, *, cwd):
        # A content read is exactly `git show <rev>:<path>` — two args, no
        # flags; the batched `--name-status` probe also spells "show" but
        # always carries flag arguments. Counted on `_git_raw` (#2654 round
        # 2): the single subprocess site EVERY read routes through — the
        # endpoint content reads go via `_git_text_lossless`, so a wrapper
        # patching only `_git` would no longer see them.
        if args[0] == "show" and len(args) == 2:
            seen["content_shows"] += 1
        return real_git_raw(args, cwd=cwd)

    monkeypatch.setattr(ccb, "_git_raw", counting_git_raw)
    _, label = ccb.body_diff_since("tasks/completed/9991/body.md", REF, repo_root=fake_repo)
    assert label == "rename-only"
    assert seen["content_shows"] == 0  # the gate skipped both endpoint reads


# ── #2654 round 2: the endpoint reads are BYTE-faithful (concern
#    `lossy-decode-endpoint-read`). Fixture helpers first: the exotic bytes
#    must survive INTO the blob, or these tests pass vacuously. ─────────────


def _commit_verbatim(repo: Path, paths: list[Path], msg: str, unix: int) -> None:
    """Stage + commit with git's EOL clean filter defeated
    (``-c core.autocrlf=false`` on the ``add`` AND the ``commit`` —
    conversion happens at staging time). The #2654 round-2 fixtures must
    land their exotic bytes (a lone CR, an invalid UTF-8 byte) in the BLOB
    verbatim; the repo under test sets ``core.autocrlf = input``, which
    would otherwise clean a CR at commit and make the tests silently
    vacuous. Pairs with the fixture repo's ``* -text`` ``.gitattributes``."""
    stamp = f"{unix} +0000"
    _git(repo, "-c", "core.autocrlf=false", "add", "--", *[str(p) for p in paths])
    _git(
        repo,
        "-c",
        "core.autocrlf=false",
        "commit",
        "-q",
        "-m",
        msg,
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )


def _blob_bytes(repo: Path, rel: str) -> bytes:
    """The committed blob at ``HEAD:<rel>`` as raw BYTES (no text decode) —
    the fixture-is-real readback assertion of #2654 round 2: a future
    EOL-config change turns these tests into loud failures here rather than
    silent passes downstream."""
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{rel}"], check=True, capture_output=True
    ).stdout


def test_conclusive_body_byte_change_invisible_to_lossy_decode_is_unlabeled(fake_repo):
    """#2654 round-2 pin (concern ``lossy-decode-endpoint-read``, leg 1:
    universal-newline translation). A body whose only content change is a
    lone ``\\r`` becoming ``\\n``, committed TOGETHER WITH a genuine
    frontmatter edit, must NOT read ``frontmatter-only``: the body BYTES
    changed.

    Pre-fix the endpoint reads decoded through ``subprocess.run(text=True)``
    — universal-newline translation maps both ``\\r\\n`` and a lone ``\\r``
    to ``\\n`` — so the two byte-DISTINCT bodies compared EQUAL as strings
    and the label fired on a real body change. Post-fix the reads route
    through ``_git_raw`` bytes + a ``surrogateescape`` decode with no
    newline translation, so string equality implies byte equality. (A test
    forcing an endpoint-read FAILURE patches ``_git_raw`` or
    ``_git_text_lossless`` — patching ``_git`` no longer intercepts a
    content read.)

    Fixture-is-real guards, each load-bearing: the CR is written in BINARY
    mode (``write_bytes`` — ``write_text`` applies newline translation and
    would defeat the fixture before git ever sees it), committed with the
    EOL filters defeated (``_commit_verbatim`` + ``* -text``), and the
    committed blob is read BACK with a ``\\r`` presence assert."""
    (fake_repo / ".gitattributes").write_bytes(b"* -text\n")
    folder = fake_repo / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    rel = "tasks/completed/9991/body.md"
    old = _FM_BODY.replace("RESULT: the sign is POSITIVE.", "RESULT: A\rB").encode("utf-8")
    (folder / "body.md").write_bytes(old)
    _commit_verbatim(
        fake_repo, [fake_repo / ".gitattributes", folder / "body.md"], "persist #9991 body", BEFORE
    )
    assert b"A\rB" in _blob_bytes(fake_repo, rel), "fixture vacuous: git cleaned the lone CR"

    new = (
        _FM_BODY.replace("RESULT: the sign is POSITIVE.", "RESULT: A\nB")
        .replace("classification: pending", "classification: useful")
        .encode("utf-8")
    )
    (folder / "body.md").write_bytes(new)
    _commit_verbatim(fake_repo, [folder / "body.md"], "CR->LF + promotion sweep", AFTER)
    assert b"A\rB" not in _blob_bytes(fake_repo, rel)  # the byte change really landed
    assert b"A\nB" in _blob_bytes(fake_repo, rel)

    _, label = ccb.body_diff_since(rel, REF, repo_root=fake_repo)
    assert label is None  # the BODY bytes changed; "frontmatter-only" would be false


def test_conclusive_invalid_utf8_body_change_is_unlabeled(fake_repo):
    """#2654 round-2 pin (concern ``lossy-decode-endpoint-read``, leg 2:
    many-to-one replacement decode). Two bodies differing only in an
    INVALID UTF-8 byte (``0xff`` vs ``0xfe``), plus a genuine frontmatter
    edit, must NOT read ``frontmatter-only``.

    Pre-fix ``errors="replace"`` collapsed BOTH invalid bytes to the same
    U+FFFD, the bodies compared equal, and the label fired on a real body
    change; under ``surrogateescape`` they decode to DISTINCT lone
    surrogates. Same binary-write + EOL-filter-defeat + blob-readback
    fixture guards as the CR test above; an endpoint-read-failure fake
    would patch ``_git_raw``, not ``_git``."""
    (fake_repo / ".gitattributes").write_bytes(b"* -text\n")
    folder = fake_repo / "tasks" / "completed" / "9991"
    folder.mkdir(parents=True)
    rel = "tasks/completed/9991/body.md"
    old = _FM_BODY.encode("utf-8").replace(
        b"RESULT: the sign is POSITIVE.", b"RESULT: raw byte \xff."
    )
    (folder / "body.md").write_bytes(old)
    _commit_verbatim(
        fake_repo, [fake_repo / ".gitattributes", folder / "body.md"], "persist #9991 body", BEFORE
    )
    assert b"\xff" in _blob_bytes(fake_repo, rel), "fixture vacuous: 0xff not in the blob"

    new = (
        _FM_BODY.replace("classification: pending", "classification: useful")
        .encode("utf-8")
        .replace(b"RESULT: the sign is POSITIVE.", b"RESULT: raw byte \xfe.")
    )
    (folder / "body.md").write_bytes(new)
    _commit_verbatim(fake_repo, [folder / "body.md"], "byte swap + promotion sweep", AFTER)
    assert b"\xfe" in _blob_bytes(fake_repo, rel), "fixture vacuous: 0xfe not in the blob"

    _, label = ccb.body_diff_since(rel, REF, repo_root=fake_repo)
    assert label is None  # the BODY bytes changed; "frontmatter-only" would be false


@pytest.mark.parametrize(
    ("case", "text", "expected"),
    [
        (
            "normal block",
            "---\nid: 1\n---\nbody\n",
            ("---\nid: 1\n---\n", "body\n"),
        ),
        (
            "dots-closed block (YAML end-of-document)",
            "---\nid: 1\n...\nbody\n",
            ("---\nid: 1\n...\n", "body\n"),
        ),
        (
            "opener with trailing spaces still splits",
            "---  \nid: 1\n---\nbody\n",
            ("---  \nid: 1\n---\n", "body\n"),
        ),
        (
            "CRLF line endings still split",
            "---\r\nid: 1\r\n---\r\nbody\r\n",
            ("---\r\nid: 1\r\n---\r\n", "body\r\n"),
        ),
        (
            "dots on line 1 is NOT an opener (a closer token is not an opener)",
            "...\nid: 1\n---\nbody\n",
            None,
        ),
        ("no leading delimiter", "# Title\n---\nbody\n", None),
        ("unterminated block has no trustworthy body boundary", "---\nid: 1\nbody\n", None),
        ("empty string", "", None),
    ],
)
def test_split_frontmatter(case, text, expected):
    """#2654 REQUIRED unit cases: the opener and closer are deliberately
    asymmetric (the opener admits only ``---``, the closer ``---`` or
    ``...``; both tolerate trailing whitespace via ``\\s*$``), and that
    asymmetry is pinned as intended rather than inferred."""
    assert ccb.split_frontmatter(text) == expected, case


def test_cap_truncation_is_disclosed(fake_repo, capsys):
    """Blocker 9 (Minor, raised by BOTH reviewers). A bare ``CLEAN checked=40``
    over a 55-citation plan reads as full coverage while 15 citations went
    unexamined."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    total = ccb._MAX_CITED_IDS + 15
    # #9991 leads so it lands INSIDE the cap and the run reaches a real
    # verdict; the rest is unresolvable filler that only drives the count.
    plan_text = "#9991 " + " ".join(f"#{9000 + i}" for i in range(total - 1))
    rc, out, _ = _run(plan_text, fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys)
    assert rc == 0
    assert "CLEAN checked=1" in out
    assert f"capped={ccb._MAX_CITED_IDS} not_examined=15" in out

    ids, seen_total = ccb.extract_cited_ids_with_total(plan_text, self_issue=8000)
    assert len(ids) == ccb._MAX_CITED_IDS
    assert seen_total == total


def test_no_cap_no_disclosure_noise(fake_repo, capsys):
    """Blocker 9 companion: an uncapped run must not grow a `capped=` token."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "persist #9991 body", BEFORE)
    rc, out, _ = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 0
    assert "capped=" not in out and "not_examined=" not in out


def test_unknown_verdict_discloses_cap_truncation(fake_repo, capsys):
    """Round-3 item 3. The UNKNOWN text branch printed the cited/unresolved/
    git_failed counts but not the cap, so a capped plan whose in-cap ids all
    failed to probe reported `cited=40` as though 40 were the whole citation
    set. Same reasoning as blocker 9's CLEAN/STALE disclosure."""
    total = ccb._MAX_CITED_IDS + 7
    plan_text = " ".join(f"#{9000 + i}" for i in range(total))  # none resolvable
    rc, out, _ = _run(plan_text, fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys)
    assert rc == 0
    assert "UNKNOWN reason=no cited id probed successfully" in out
    assert f"capped={ccb._MAX_CITED_IDS} not_examined=7" in out


def test_cap_disclosure_in_json_report(fake_repo, capsys):
    """Blocker 9: the machine-readable leg carries the same disclosure."""
    plan_text = " ".join(f"#{9000 + i}" for i in range(ccb._MAX_CITED_IDS + 3))
    rc, out, _ = _run(
        plan_text, fake_repo, ["--issue", "8000", "--since-unix", str(REF), "--json"], capsys
    )
    assert rc == 0
    payload = json.loads(out)
    assert payload["capped"] is True
    assert payload["not_examined"] == 3
    assert payload["cited_total"] == ccb._MAX_CITED_IDS + 3


def test_usage_error_exits_2_with_no_verdict_line(capsys):
    """Blocker 11 (Minor). Exit 2 is argparse's USAGE error, outside the
    documented CLEAN/UNKNOWN(0) / STALE(3) vocabulary. It must stay
    distinguishable: no ``CITED-BODY-CURRENCY:`` line is printed, so a caller
    keying on the verdict line can never read it as STALE."""
    with pytest.raises(SystemExit) as exc:
        ccb.main([])  # --issue is required
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "CITED-BODY-CURRENCY:" not in captured.out
    assert "CITED-BODY-CURRENCY:" not in captured.err


def test_detail_render_failure_cannot_downgrade_exit_3(fake_repo, monkeypatch, capsys):
    """Blocker 12 (Minor). STALE is printed BEFORE the best-effort detail
    render. Pre-fix a raise in the renderer reached the top-level fail-soft
    handler, which printed a SECOND line (``UNKNOWN``) and returned 0 — so a
    positively-established stale citation was reported as both stale and
    unknown, and the process exited 0."""
    folder = _make_task(fake_repo, 9991)
    _commit(fake_repo, [folder / "body.md"], "correct #9991 body", AFTER)

    def _boom(*_a, **_k):
        raise RuntimeError("diff render exploded")

    monkeypatch.setattr(ccb, "body_diff_since", _boom)
    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3
    assert "STALE ids=9991" in out
    assert "UNKNOWN" not in out
    assert "diff render failed for #9991" in err


def test_git_output_decoding_is_lenient(fake_repo, capsys):
    """Blocker 3 (Critical) helper-side. git echoes commit SUBJECTS verbatim
    and they are not guaranteed UTF-8; under strict decoding
    ``subprocess.run`` raises ``UnicodeDecodeError`` — a ``ValueError``, not
    an ``OSError`` — from inside the probe. A latin-1 subject must not stop
    the gate from reaching its verdict."""
    folder = _make_task(fake_repo, 9991)
    stamp = f"{AFTER} +0000"
    _git(fake_repo, "add", "--", str(folder / "body.md"))
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    msg = fake_repo / "msg.txt"
    msg.write_bytes(b"correct \xe9\xe8 body")  # latin-1, invalid UTF-8
    subprocess.run(
        ["git", "-C", str(fake_repo), "commit", "-q", "-F", str(msg)],
        check=True,
        capture_output=True,
        env=env,
    )
    msg.unlink()

    rc, out, err = _run(
        "Grounds on #9991.", fake_repo, ["--issue", "8000", "--since-unix", str(REF)], capsys
    )
    assert rc == 3, out + err
    assert "STALE ids=9991" in out
