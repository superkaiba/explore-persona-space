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

import ast
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import textwrap
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


def _load_ccb():
    """The pre-persist helper, loaded by PATH so the WORKTREE copy is the one
    compared against c75 (the same idiom `test_c75_and_helper_extract_identical_ids`
    uses inline)."""
    spec = importlib.util.spec_from_file_location(
        "check_cited_body_currency_pin", REPO_ROOT / "scripts" / "check_cited_body_currency.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_cited_body_currency_pin"] = mod
    spec.loader.exec_module(mod)
    return mod


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


# ── ROUND-2 regressions (#2384 code-review v1 blockers 1, 3, 4, 5, 6, 9) ─────


def test_oldest_of_several_breadcrumbs_wins_and_warns(fake_repo):
    """Blocker 6 (Major) — the load-bearing gap. Every prior test used a
    SINGLE breadcrumb, so `_c75_oldest_dispatch_unix`'s ``unix < oldest``
    comparison was never exercised: a min->max flip passed the whole suite.

    Fixture is the exact shape the check exists to catch:
    ``t1 < body_commit < t2 < plan_mtime`` — two planner-dispatch rounds, and
    a cited body corrected BETWEEN them. The reference must be t1 (drafting
    began at the FIRST round), which puts the correction inside the window =>
    WARN. Under ``max`` the reference is t2, the correction reads as
    predating the draft, and the check certifies a stale citation CLEAN —
    exactly the false-CLEAN #2384 exists to prevent."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    _write_events(folder, [T_OLD, T_NEW])  # t1 = T_OLD, t2 = T_NEW
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", T_MID)  # t1 < T_MID < t2

    ref, why = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v1.md", repo_root=fake_repo
    )
    assert ref == T_OLD, "reference must be the OLDEST breadcrumb, not the newest"
    assert ref != T_NEW
    assert "breadcrumb" in why

    res = vp.check_cited_body_currency(
        "Plan grounds on the #9991 result.",
        self_issue=8000,
        reference_unix=ref,
        reason=why,
        repo_root=fake_repo,
    )
    assert res.is_warn is True
    assert "#9991" in res.detail


def test_breadcrumbs_newer_than_plan_mtime_are_still_excluded(fake_repo):
    """Blocker 6 companion: the oldest-wins rule must not swallow the
    mtime ceiling — a breadcrumb from a LATER round (ts > plan mtime) belongs
    to a future round and cannot be this draft's start."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    later = int(PLAN_MTIME) + 10_000
    _write_events(folder, [T_NEW, later])
    ref, _ = vp._draft_start_reference(
        folder, PLAN_MTIME, plan_path=plans / "v1.md", repo_root=fake_repo
    )
    assert ref == T_NEW


def test_prior_version_reference_skips_status_move_commits(fake_repo):
    """Blocker 1 (Critical). A persisted plan version is written ONCE and then
    only MOVED — `task.py set-status` ``git mv``s the whole task folder — so
    ``git log -1 -- <path>`` answers with the STATUS-MOVE commit, minutes to
    days after the real persist. Used as the draft-start reference that
    shifts the window FORWARD, hiding every cited-body correction between the
    true persist and the move.

    Fixture: v1 persisted at T_OLD, folder moved at T_NEW. The reference must
    be T_OLD."""
    folder = _make_task(fake_repo, 8000, status="running")
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "task #8000: plan v1", T_OLD)

    dest = fake_repo / "tasks" / "completed" / "8000"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(fake_repo, "mv", str(folder), str(dest))
    stamp = f"{T_NEW} +0000"
    _git(
        fake_repo,
        "commit",
        "-q",
        "-m",
        "task #8000: running -> completed",
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )
    moved_plans = dest / "plans"

    # The naive probe answers with the status move; the content probe does not.
    assert vp._c75_last_commit_unix(moved_plans / "v1.md", repo_root=fake_repo) == T_NEW
    assert vp._c75_last_content_commit_unix(moved_plans / "v1.md", repo_root=fake_repo) == T_OLD

    ref, why = vp._draft_start_reference(
        dest, PLAN_MTIME, plan_path=moved_plans / "v2.md", repo_root=fake_repo
    )
    assert ref == T_OLD, "reference must be the persist commit, not the status move"
    assert "v1.md" in why


def test_status_move_reference_catches_a_correction_the_move_would_hide(fake_repo):
    """Blocker 1, end-to-end: a cited body corrected BETWEEN the plan persist
    and the status move. With the move as reference the correction reads
    CLEAN; with the persist as reference it WARNs."""
    folder = _make_task(fake_repo, 8000, status="running")
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "task #8000: plan v1", T_OLD)
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", T_MID)

    dest = fake_repo / "tasks" / "completed" / "8000"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(fake_repo, "mv", str(folder), str(dest))
    stamp = f"{T_NEW} +0000"
    _git(
        fake_repo,
        "commit",
        "-q",
        "-m",
        "task #8000: running -> completed",
        env_extra={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp},
    )

    ref, why = vp._draft_start_reference(
        dest, PLAN_MTIME, plan_path=dest / "plans" / "v2.md", repo_root=fake_repo
    )
    res = vp.check_cited_body_currency(
        "Plan grounds on the #9991 result.",
        self_issue=8000,
        reference_unix=ref,
        reason=why,
        repo_root=fake_repo,
    )
    assert res.is_warn is True, f"ref={ref} why={why} detail={res.detail}"
    assert "#9991" in res.detail


def test_mtime_fallback_requires_positive_untracked_evidence(fake_repo, monkeypatch):
    """Blocker 4 (Major). ``st_mtime`` is documented as unusable (a checkout
    clobbers it) and permitted ONLY for a file that exists uncommitted.
    Pre-fix the fallback fired on ANY ``None`` from the git probe, so a FAILED
    probe on a TRACKED file silently substituted the clobbered mtime.

    Here the file is tracked with a real commit, but every git probe is forced
    to fail: the leg must go UNRESOLVED, not mtime."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "persist plan v1", T_MID)
    os.utime(plans / "v1.md", times=(T_NEW, T_NEW))

    monkeypatch.setattr(vp, "_c75_git_out", lambda *_a, **_k: None)
    unix, why = vp._c75_prior_version_unix(folder, plan_path=plans / "v2.md", repo_root=fake_repo)
    assert unix is None, "a failed probe must never substitute st_mtime"
    assert unix != T_NEW
    assert "mtime not substituted" in why


def test_mtime_fallback_fires_for_a_genuinely_untracked_prior_version(fake_repo):
    """Blocker 4 companion: the documented case still works — an UNCOMMITTED
    prior version falls back to st_mtime, on positive `git ls-files`
    evidence."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")  # never committed
    os.utime(plans / "v1.md", times=(T_MID, T_MID))

    assert vp._c75_is_untracked(plans / "v1.md", repo_root=fake_repo) is True
    unix, why = vp._c75_prior_version_unix(folder, plan_path=plans / "v2.md", repo_root=fake_repo)
    assert unix == T_MID
    assert "untracked" in why


def test_tracked_file_is_not_reported_untracked(fake_repo):
    """Blocker 4: the tracked/untracked discriminator itself."""
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("v1 body\n")
    _commit(fake_repo, [plans / "v1.md"], "persist plan v1", T_MID)
    assert vp._c75_is_untracked(plans / "v1.md", repo_root=fake_repo) is False


def test_undecodable_events_jsonl_does_not_raise(fake_repo):
    """Blocker 3 (Critical). ``events.jsonl`` was read with a bare
    ``read_text()``: one non-UTF-8 byte in a marker note raised
    ``UnicodeDecodeError`` out of a WARN-only check and aborted
    verify_plan.py. Well-formed rows must still parse around the bad byte."""
    folder = _make_task(fake_repo, 8000)
    rows = [
        json.dumps(
            {"ts": _iso(T_OLD), "kind": "epm:progress", "note": "planner-dispatch round=1"}
        ).encode()
    ]
    (folder / "events.jsonl").write_bytes(
        rows[0] + b"\n" + b'{"ts": "x", "kind": "epm:progress", "note": "\xff\xfe bad"}\n'
    )
    assert vp._c75_oldest_dispatch_unix(folder, ceiling_unix=PLAN_MTIME) == T_OLD


def test_unreadable_events_jsonl_does_not_raise(fake_repo, monkeypatch):
    """Blocker 3: an OSError on the read (permissions, IO fault) is absorbed
    too — the leg goes unresolved, the verifier keeps running."""
    folder = _make_task(fake_repo, 8000)
    (folder / "events.jsonl").write_text("")

    def _boom(*_a, **_k):
        raise PermissionError("no read for you")

    monkeypatch.setattr(Path, "read_text", _boom)
    assert vp._c75_oldest_dispatch_unix(folder, ceiling_unix=PLAN_MTIME) is None


def test_git_probe_decodes_leniently(fake_repo):
    """Blocker 3: git echoes commit SUBJECTS verbatim and they are not
    guaranteed UTF-8. Under strict decoding ``subprocess.run`` raises
    ``UnicodeDecodeError`` — a ``ValueError``, which the pre-fix
    ``(OSError, TimeoutExpired)`` tuple did not catch — from inside a
    WARN-only check."""
    folder = _make_task(fake_repo, 9991)
    stamp = f"{T_MID} +0000"
    _git(fake_repo, "add", "--", str(folder / "body.md"))
    env = os.environ.copy()
    env.update({"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    msg = fake_repo / "msg.txt"
    msg.write_bytes(b"correct \xe9\xe8 body")
    subprocess.run(
        ["git", "-C", str(fake_repo), "commit", "-q", "-F", str(msg)],
        check=True,
        capture_output=True,
        env=env,
    )
    msg.unlink()
    assert vp._c75_last_commit_unix(folder / "body.md", repo_root=fake_repo) == T_MID


def test_c75_degrades_to_skip_when_it_errors(fake_repo, monkeypatch, tmp_path, capsys):
    """Blocker 3: the fail-soft boundary in main(). c75 is the only appended
    check that shells out to git, reads a sibling task's events.jsonl, stats
    the filesystem, and imports the task resolver — four external surfaces
    whose faults are unrelated to the plan under review. An escape aborts
    verify_plan.py for every caller on the fleet, so it must degrade to SKIP.
    """
    folder = _make_task(fake_repo, 8000)
    plans = folder / "plans"
    plans.mkdir()
    (plans / "v1.md").write_text("# Plan\n\nGrounds on #9991.\n")
    # Empty registry: find_task_path falls back to the on-disk status scan,
    # which resolves tasks/running/8000 without hand-building an entry shape.
    (fake_repo / "tasks" / "REGISTRY.json").write_text(json.dumps({"tasks": {}}))

    def _boom(*_a, **_k):
        raise RuntimeError("c75 exploded")

    monkeypatch.setattr(vp, "_draft_start_reference", _boom)
    monkeypatch.setattr(sys, "argv", ["verify_plan.py", "--issue", "8000"])
    rc = vp.main()
    out = capsys.readouterr().out
    assert "[SKIP] plan cites a task body corrected after drafting began" in out, out
    assert "degraded to SKIP: RuntimeError: c75 exploded" in out, out
    # The verifier still produced a full report and a normal exit code — the
    # whole point: a WARN-only check's fault must not abort the run.
    assert "OVERALL:" in out, out
    assert isinstance(rc, int)


@pytest.mark.parametrize(
    ("label", "plan_text", "expected"),
    [
        ("mixed_delimiters", "#111\n```\n#222\n~~~\n#333\n```\n#444", ["111", "444"]),
        ("longer_opener", "#111\n````\n#222\n```\n#333\n````\n#444", ["111", "444"]),
        ("indented_is_not_a_fence", "#111\n    ```\n#222", ["111", "222"]),
        ("three_space_indent_is_a_fence", "#111\n   ```\n#222\n   ```\n#333", ["111", "333"]),
        ("unclosed_fence", "#111\n```\n#222", ["111"]),
    ],
)
def test_c75_fence_semantics_are_delimiter_aware(label, plan_text, expected):
    """Blocker 5 (Major), c75 side. The shared ``_fence_mask`` toggles on any
    line whose stripped form starts with three backticks or tildes, so a
    mismatched delimiter, a shorter inner run, or a 4-space-indented code line
    inverts in/out for the rest of the document."""
    assert vp._c75_extract_cited_ids(plan_text, self_issue=1)[0] == expected, label


def test_c75_dedups_leading_zero_variants():
    """Blocker 5 fallout, caught by the cross-leg comparison below: c75
    deduped on the raw STRING while every call site does ``int(tid)``, so
    ``#039`` and ``#39`` were two ids pointing at ONE body — a doubled probe,
    a burned cap slot, and a task nameable twice in one WARN."""
    ids, total = vp._c75_extract_cited_ids("Grounds on #39 and #039 and #0039.", self_issue=1)
    assert ids == ["39"]
    assert total == 1


def test_c75_and_helper_extract_identical_ids():
    """#2384 §3.4's core contract: the WARN-only backstop and the blocking
    pre-persist helper must read the SAME citation set, or the two legs
    disagree about what was even checked. Pinned over adversarial fences,
    the lineage-list idiom, URL adjacency, and leading zeros."""
    spec = importlib.util.spec_from_file_location(
        "check_cited_body_currency_pin", REPO_ROOT / "scripts" / "check_cited_body_currency.py"
    )
    ccb = importlib.util.module_from_spec(spec)
    sys.modules["check_cited_body_currency_pin"] = ccb
    spec.loader.exec_module(ccb)

    samples = [
        "Grounds on #884/#1045/#1134 and the #9991 result.",
        "#111\n```\n#222\n~~~\n#333\n```\n#444",
        "#111\n````\n#222\n```\n#333\n````\n#444",
        "#111\n    ```\n#222",
        "#111\n```\n#222",
        "See https://example.com/issues#123 and #456.",
        "Grounds on #39 and #039.",
        "Self #8000 is never a citation; #9991 is.",
        " ".join(f"#{9000 + i}" for i in range(60)),
    ]
    for text in samples:
        a = vp._c75_extract_cited_ids(text, self_issue=8000)[0]
        b = [str(x) for x in ccb.extract_cited_ids(text, self_issue=8000)]
        assert a == b, text[:60]


def test_c75_discloses_cap_truncation(fake_repo):
    """Blocker 9 (Minor, raised by BOTH reviewers). A bare ``checked=N`` PASS
    over a plan citing more than the cap reads as full coverage."""
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "persist #9991 body", T_OLD)
    total = vp._C75_MAX_IDS + 12
    plan_text = "#9991 " + " ".join(f"#{7000 + i}" for i in range(total - 1))
    res = vp.check_cited_body_currency(
        plan_text,
        self_issue=8000,
        reference_unix=T_NEW,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert f"capped={vp._C75_MAX_IDS} not_examined=12" in res.detail


def test_c75_no_cap_no_disclosure_noise(fake_repo):
    """Blocker 9 companion: an uncapped run must not grow a `capped=` token."""
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "persist #9991 body", T_OLD)
    res = vp.check_cited_body_currency(
        "Grounds on #9991.",
        self_issue=8000,
        reference_unix=T_NEW,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert "capped=" not in res.detail


# ── ROUND-3 items (#2384 reconciler: B7 reference plausibility, B9 residue,
#    the fence-extractor drift pin) ────────────────────────────────────────


@pytest.mark.parametrize(
    ("bad", "label"),
    [
        (-62135596800, "0001-01-01T00:00:00Z, the malformed-but-parseable ts"),
        (0, "unix epoch"),
        (vp._C75_MIN_PLAUSIBLE_REF_UNIX - 1, "one second under the floor"),
        (4_102_444_800, "year 2100 — far future"),
    ],
)
def test_c75_implausible_reference_skips(fake_repo, bad, label):
    """Round-3 item 2. c75 had a CEILING on leg (1) only (a breadcrumb newer
    than the plan mtime belongs to a later round) and NO floor on either leg,
    so any non-None resolved reference went straight into the comparisons.

    Both directions silently invert the check — far past marks every cited
    body stale, far future certifies every one CLEAN — so neither may pass as
    a verdict. The helper has carried this band since round 2
    (`_MIN_PLAUSIBLE_REF_UNIX`); this is its c75 twin."""
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "persist #9991 body", T_OLD)
    res = vp.check_cited_body_currency(
        "Grounds on #9991.",
        self_issue=8000,
        reference_unix=bad,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert res.skipped is True, label
    assert res.is_warn is False
    assert "implausible draft-start reference" in res.detail
    assert str(bad) in res.detail


def test_c75_plausible_reference_still_warns(fake_repo):
    """Mutation guard for the band above: a reference INSIDE it must still
    reach the comparison and WARN — a floor that swallowed every reference
    would pass the four rejection cases while disabling the check."""
    cited = _make_task(fake_repo, 9991, status="completed")
    _commit(fake_repo, [cited / "body.md"], "correct #9991 body", T_MID)
    res = vp.check_cited_body_currency(
        "Grounds on #9991.",
        self_issue=8000,
        reference_unix=T_OLD,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert res.is_warn is True
    assert res.skipped is False


def test_c75_zero_probed_skip_discloses_cap(fake_repo):
    """Round-3 item 3. The zero-probed SKIP returned BEFORE the `capped=`
    string was built, so `40 cited id(s) but none probed` over a 49-citation
    plan read as if 40 were the whole citation set."""
    total = vp._C75_MAX_IDS + 9
    plan_text = " ".join(f"#{7000 + i}" for i in range(total))  # none resolvable
    res = vp.check_cited_body_currency(
        plan_text,
        self_issue=8000,
        reference_unix=T_NEW,
        reason="oldest planner-dispatch breadcrumb",
        repo_root=fake_repo,
    )
    assert res.skipped is True
    assert "none probed" in res.detail
    assert f"capped={vp._C75_MAX_IDS} not_examined=9" in res.detail


def _fn_body_ast(fn, *, rename: dict[str, str] | None = None) -> str:
    """``ast.dump`` of ``fn``'s body with the docstring dropped and ``rename``
    applied to Name loads — comment-blind and formatting-blind by
    construction (``ast.dump`` omits line attributes by default)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    body = list(tree.body[0].body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # the two copies document different rationales BY DESIGN
    module = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and rename and node.id in rename:
            node.id = rename[node.id]
    return "\n".join(ast.dump(n) for n in body)


def test_c75_and_helper_fence_extractors_are_code_identical():
    """Round-3 item 6. `test_c75_and_helper_extract_identical_ids` pins the two
    extractors by ID-SET agreement over 9 samples, which any drift those 9
    samples happen not to distinguish walks straight through — and the whole
    #2384 §3.4 contract is that the WARN-only backstop and the blocking helper
    read the SAME citation set.

    This asserts CODE identity instead: the fence regex byte-for-byte, and the
    two function bodies as normalized ASTs (docstrings dropped, the regex
    global renamed, comments and formatting invisible)."""
    ccb = _load_ccb()
    assert vp._C75_FENCE_RE.pattern == ccb._FENCE_RE.pattern
    assert vp._C75_FENCE_RE.flags == ccb._FENCE_RE.flags
    assert vp._C75_ISSUE_REF_RE.pattern == ccb._ISSUE_REF_RE.pattern
    assert vp._C75_ISSUE_REF_RE.flags == ccb._ISSUE_REF_RE.flags
    assert _fn_body_ast(
        vp._c75_strip_code_blocks, rename={"_C75_FENCE_RE": "_FENCE_RE"}
    ) == _fn_body_ast(ccb._strip_code_blocks)
