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
    # window and correctly stops: one R100 row carrying both endpoints.
    rows = ccb._window_history(new_rel, REF, repo_root=fake_repo)
    assert [r.status for r in rows] == ["R100"], why
    assert rows[0].path == new_rel and rows[0].prev_path == old_rel, why

    # The harm the parse defect caused, end to end: the advisory label.
    _, label = ccb.body_diff_since(new_rel, REF, repo_root=fake_repo)
    assert label == "rename-only", why


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
