"""verify_plan c75 (cited-body currency) — the WARN-only plan-time backstop (#2384).

Five cases per plan #2384 §3.5, all against `_draft_start_reference` +
`check_cited_body_currency` directly (the c23/c62 out-of-band idiom): the
reference is the MIN over the legs that RESOLVE (leg (2) is a FLOOR, never
an ``else`` branch — `test_mixed_breadcrumb_window_still_warns` is the
regression pin for the min()-vs-else defect the plan bounced), the prior
version's timestamp comes from git (never a clobbered st_mtime), an
unresolved reference SKIPs (the check never guesses), and a stale citation
is a WARN — never a FAIL.

Fixture: the fake-repo pattern of tests/test_task_workflow_list_children.py
with deterministic GIT_AUTHOR_DATE/GIT_COMMITTER_DATE commits.
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


def _load_verify_plan():
    spec = importlib.util.spec_from_file_location(
        "verify_plan", REPO_ROOT / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_plan", mod)
    spec.loader.exec_module(mod)
    return sys.modules["verify_plan"]


vp = _load_verify_plan()

# Deterministic timeline (unix seconds).
T_INIT = 1_699_000_000
T_OLD = 1_700_000_000
T_MID = 1_700_050_000
T_NEW = 1_700_100_000
PLAN_MTIME = 1_700_200_000.0


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    )


def _commit(repo: Path, paths: list[Path], msg: str, unix: int) -> None:
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
    """git-init tmp_path and rebind task_workflow's resolvers to it (needed
    by `_c75_cited_body_path`'s worktree-safe resolution)."""
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


def _make_task(repo: Path, issue: int, status: str = "running") -> Path:
    folder = repo / "tasks" / status / str(issue)
    folder.mkdir(parents=True)
    (folder / "body.md").write_text(f"---\nid: {issue}\n---\n\n# Task {issue}\n")
    return folder


def test_reference_is_min_of_breadcrumb_and_prior_version(fake_repo):
    """Both legs resolve => the EARLIER one is the reference, in BOTH orders."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "persist plan v1", T_MID)

    # Order A: breadcrumb (T_OLD) older than v1's commit (T_MID).
    _write_events(folder, [T_OLD])
    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v2.md", repo_root=fake_repo
    )
    assert ref == T_OLD
    assert "breadcrumb" in why

    # Order B: v1's commit (T_MID) older than the only breadcrumb (T_NEW).
    _write_events(folder, [T_NEW])
    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v2.md", repo_root=fake_repo
    )
    assert ref == T_MID
    assert "v1.md" in why


def test_mixed_breadcrumb_window_still_warns(fake_repo):
    """The min()-vs-else regression pin (#2384 §2.1): v{K-1} committed at t0,
    the only breadcrumb at t2 > t0, the cited body corrected at t1 in
    (t0, t2). An ``else``-branch reference (breadcrumb-first) would read t2
    and certify t1 CLEAN; the MIN reads t0 and WARNs."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "persist plan v1", T_OLD)
    _write_events(folder, [T_NEW])
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", T_MID)

    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v2.md", repo_root=fake_repo
    )
    assert ref == T_OLD

    res = vp.check_cited_body_currency(
        "Plan grounds on the #9991 result.",
        self_issue=8000,
        reference_unix=ref,
        reason=why,
        repo_root=fake_repo,
    )
    assert res.is_warn is True
    assert "#9991" in res.detail


def test_prior_version_ts_from_git_not_mtime(fake_repo):
    """A checkout clobbers mtimes: with v1.md's st_mtime pushed past its
    commit time, the reference still reads the git commit time."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "persist plan v1", T_MID)
    os.utime(plans / "v1.md", times=(T_NEW, T_NEW))

    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v2.md", repo_root=fake_repo
    )
    assert ref == T_MID
    assert why.startswith("git commit time")


def test_skip_when_neither_resolves(fake_repo):
    """v1 draft (no prior version), no breadcrumbs => (None, reason) and the
    check SKIPs — it never guesses a reference."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()

    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v1.md", repo_root=fake_repo
    )
    assert ref is None
    assert "neither leg resolved" in why

    res = vp.check_cited_body_currency(
        "Plan grounds on the #9991 result.",
        self_issue=8000,
        reference_unix=ref,
        reason=why,
        repo_root=fake_repo,
    )
    assert res.skipped is True
    assert res.passed is True
    assert "draft-start reference unresolved" in res.detail


def test_warn_not_fail(fake_repo):
    """A positively-stale citation is a WARN (passed=True, is_warn=True) —
    c75 is the advisory backstop; the blocking arm is the pre-persist
    helper, and the WARN guidance names it."""
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", T_MID)

    res = vp.check_cited_body_currency(
        "Plan grounds on the #9991 result.",
        self_issue=8000,
        reference_unix=T_OLD,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert res.is_warn is True
    assert res.passed is True
    assert res.skipped is False
    assert "scripts/check_cited_body_currency.py" in res.detail
