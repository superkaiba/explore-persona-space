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


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    )


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

    monkeypatch.setattr(ccb, "extract_cited_ids", _boom)
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
