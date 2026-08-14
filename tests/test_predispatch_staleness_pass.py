"""Tests for the #2134 pre-dispatch task-premise staleness scanner + watcher pass.

Pure-logic tests drive ``scripts/predispatch_staleness.py`` (tokens /
targets / scan via the injected ``git_log_fn`` seam — no git subprocess is
ever spawned); pass-level tests drive
``autonomous_session_watch.predispatch_staleness_pass`` with the
collect/git seams + ``PROJECT_ROOT`` / ``AUTONOMOUS_REGISTRY_DIR`` /
``_telegram_push`` monkeypatched (the registry-drift test shape).

Hard invariants pinned here (plan #2134 acceptance 3/4 + critic notes):

- report-only: the pass NEVER composes a ``set-status`` / ``archived``
  ``task.py`` argv (asserted on the SUBCOMMAND token — the flag-marker note
  deliberately QUOTES ``set-status <N> archived`` as the human adjudication
  affordance, so a substring scan would false-fail) and never touches the
  infra-drain queue / proposed-infra-sweep state files;
- kill switch / interval throttle / per-(issue, fingerprint) dedup + TTL
  re-alert / marker cap with sidecar-only overflow / persisted cursor;
- fail toward silence on unreadable inputs (a failed git log never reads
  as "no staleness");
- ``dry_run=True`` performs zero writes (no state file, no sidecar row, no
  marker subprocess, no wire push).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import predispatch_staleness as pds  # noqa: E402

_NOW = 1_800_000_000.0


# ─── pure logic: informative_tokens ──────────────────────────────────────────


def test_informative_tokens_shapes():
    toks = pds.informative_tokens("Task #2134: Pre-dispatch STALENESS pass for queued_tasks")
    # lowercase, >=4 chars, [a-z0-9_] runs; stopwords (task, pass) excluded.
    assert "staleness" in toks
    assert "queued_tasks" in toks  # underscore runs are one token
    assert "2134" in toks  # digit runs count (issue-number overlap is signal)
    assert "task" not in toks and "pass" not in toks  # stopwords
    assert "for" not in toks  # < 4 chars
    assert pds.informative_tokens("") == set()
    assert pds.informative_tokens(None) == set()


# ─── pure logic: parse_targets ───────────────────────────────────────────────


def test_parse_targets_observed_live_shapes():
    body = (
        "## Provenance\n"
        "- workflow_fix_target: scripts/autonomous_session_watch.py\n"
        "workflow_fix_target: src/explore_persona_space/eval/utils.py (parse_judge_json step 4)\n"
        "- workflow_fix_target: scripts/a.py, scripts/b.py\n"
        "- workflow_fix_target: scripts/workflow_lint.py, .claude/rules/LESSONS.md\n"
        "- workflow_fix_target: `CLAUDE.md`\n"
    )
    assert pds.parse_targets(body) == [
        "scripts/autonomous_session_watch.py",
        "src/explore_persona_space/eval/utils.py",
        "scripts/a.py",
        "scripts/b.py",
        "scripts/workflow_lint.py",
        ".claude/rules/LESSONS.md",
        "CLAUDE.md",
    ]


def test_parse_targets_dedupes_and_fails_toward_silence():
    body = "- workflow_fix_target: scripts/a.py\n- workflow_fix_target: scripts/a.py\n"
    assert pds.parse_targets(body) == ["scripts/a.py"]
    assert pds.parse_targets("no provenance section here") == []
    assert pds.parse_targets("") == []
    assert pds.parse_targets(None) == []
    # A value with no path-ish piece parses to [] rather than raising.
    assert pds.parse_targets("workflow_fix_target: (tbd)\n") == []


def test_parse_targets_rejects_absolute_and_bare_slash_paths():
    """#2134 v2 fold: a ``workflow_fix_target`` must be repo-relative. The
    live #1067 prose value tokenized to the bare ``/`` pathspec — a
    guaranteed git exit-128 + stderr line EVERY firing."""
    # Literal #1067 shape: the "/" token is dropped, nothing survives.
    assert pds.parse_targets("workflow_fix_target: my-goat / Happy pairing (...)\n") == []
    # Absolute paths are dropped too; siblings on the same line survive.
    assert pds.parse_targets("workflow_fix_target: /etc/passwd\n") == []
    assert pds.parse_targets("- workflow_fix_target: /abs/dir/x.py, scripts/rel.py\n") == [
        "scripts/rel.py"
    ]

    # Zero surviving paths => the task never reaches a git-backed signal.
    def _forbidden(paths, since):
        raise AssertionError("task with zero surviving targets must not reach git")

    recs = pds.scan(
        [_task(10, targets="my-goat / Happy pairing (Claude Code on the phone)")],
        git_log_fn=_forbidden,
    )
    assert recs == []


def test_parse_created_at():
    assert pds.parse_created_at("---\ncreated_at: '2026-08-06T07:22:21Z'\n---\n") == (
        "2026-08-06T07:22:21Z"
    )
    assert pds.parse_created_at("---\ncreated_at: 2026-08-06T07:22:21Z\n---\n") == (
        "2026-08-06T07:22:21Z"
    )
    assert pds.parse_created_at("no frontmatter") is None
    assert pds.parse_created_at(None) is None


# ─── pure logic: scan ────────────────────────────────────────────────────────


def _task(
    issue,
    *,
    status="proposed",
    kind="infra",
    created="2026-08-01T00:00:00Z",
    title="watcher daemon liveness escalation",
    targets="scripts/foo.py",
):
    return {
        "id": issue,
        "status": status,
        "kind": kind,
        "created_ts": created,
        "title": title,
        "body": f"premise text about daemon liveness escalation\n"
        f"## Provenance\n- workflow_fix_target: {targets}\n",
    }


def test_scan_fires_on_acceptance_fixture():
    # Acceptance 1: a proposed infra task with workflow_fix_target:
    # scripts/foo.py and a post-creation commit on scripts/foo.py whose
    # subject shares >= 3 informative tokens with the task body -> ONE
    # stale-premise flag record.
    seen: list[tuple[list[str], str]] = []

    def git_log_fn(paths, since):
        seen.append((paths, since))
        return [("a" * 40, "task #99: watcher daemon liveness escalation rework")]

    recs = pds.scan([_task(10)], git_log_fn=git_log_fn)
    stale = [r for r in recs if r.kind == "stale-premise"]
    assert len(stale) == 1
    rec = stale[0]
    assert rec.issue == 10
    assert rec.target == "scripts/foo.py"
    assert rec.evidence["sha"] == "a" * 40
    assert len(rec.evidence["matched_tokens"]) >= 3
    assert "daemon" in rec.evidence["matched_tokens"]
    assert isinstance(rec.fingerprint, str) and len(rec.fingerprint) == 12
    # The injected seam received the task's targets + creation timestamp.
    assert seen == [(["scripts/foo.py"], "2026-08-01T00:00:00Z")]


def test_scan_no_fire_on_unrelated_subject():
    recs = pds.scan(
        [_task(10)],
        git_log_fn=lambda p, s: [("b" * 40, "unrelated figure regeneration cleanup")],
    )
    assert [r for r in recs if r.kind == "stale-premise"] == []


def test_scan_no_fire_below_min_tokens():
    # Exactly 2 shared informative tokens (daemon, liveness) < min_tokens=3.
    recs = pds.scan(
        [_task(10)],
        git_log_fn=lambda p, s: [("c" * 40, "task #99: daemon liveness")],
    )
    assert [r for r in recs if r.kind == "stale-premise"] == []


def test_scan_queue_collision_one_record_per_file():
    # Acceptance 2: two queued tasks naming the same target file -> ONE
    # collision record (per file), listing both ids.
    recs = pds.scan(
        [
            _task(10, title="alpha premise", targets="scripts/foo.py"),
            _task(11, title="beta premise", targets="scripts/foo.py"),
        ],
        git_log_fn=lambda p, s: [],
    )
    coll = [r for r in recs if r.kind == "queue-collision"]
    assert len(coll) == 1
    assert coll[0].target == "scripts/foo.py"
    assert coll[0].evidence["colliding_issues"] == [10, 11]
    assert coll[0].issue == 10


def test_scan_collision_grouping_spans_beyond_window():
    """#2134 v2 fold: ``collision_tasks`` (the FULL collected queue) feeds
    the git-free collision grouping while ``tasks`` (the cap+cursor window)
    keeps the git budget bounded — colliding tasks on opposite sides of a
    window boundary co-detect."""
    full = [
        _task(10, title="alpha premise", targets="scripts/foo.py"),
        _task(11, title="beta premise", targets="scripts/bar.py"),
        _task(12, title="gamma premise", targets="scripts/foo.py"),
    ]
    window = full[:2]
    seen: list[list[str]] = []

    def git_log_fn(paths, since):
        seen.append(list(paths))
        return []

    recs = pds.scan(window, git_log_fn=git_log_fn, collision_tasks=full)
    coll = [r for r in recs if r.kind == "queue-collision"]
    assert len(coll) == 1
    # #12 sits OUTSIDE the window and is still co-detected.
    assert coll[0].evidence["colliding_issues"] == [10, 12]
    # Git-backed signals stay WINDOWED: only the two window tasks hit git.
    assert seen == [["scripts/foo.py"], ["scripts/bar.py"]]


def test_scan_landed_sibling_collision_excludes_own_id():
    recs = pds.scan(
        [_task(10)],
        git_log_fn=lambda p, s: [
            ("d" * 40, "task #10: own follow-up commit"),  # own id — excluded
            ("e" * 40, "task #99: sibling landing on the same file"),
            ("f" * 40, "issue-77 second sibling"),
        ],
    )
    sib = [r for r in recs if r.kind == "landed-sibling-collision"]
    assert len(sib) == 1  # aggregated per task
    sib_ids = {i for row in sib[0].evidence["siblings"] for i in row["sibling_issues"]}
    assert sib_ids == {77, 99}


def test_scan_git_failure_fails_toward_silence(capsys):
    # A raising git_log_fn skips THAT task's commit-backed signals with one
    # stderr line; the other task still scans, and the failed task still
    # participates in (git-free) queue-collision grouping. A failed git log
    # never reads as "no staleness" — no records are fabricated for it.
    def git_log_fn(paths, since):
        if "scripts/foo.py" in paths:
            raise RuntimeError("git wedged")
        return [("9" * 40, "task #99: watcher daemon liveness escalation rework")]

    recs = pds.scan(
        [
            _task(10, targets="scripts/foo.py"),
            _task(11, targets="scripts/bar.py"),
            _task(12, targets="scripts/foo.py", title="gamma premise"),
        ],
        git_log_fn=lambda p, s: git_log_fn(p, s),
    )
    err = capsys.readouterr().err
    assert "git log failed for #10" in err
    stale_issues = {r.issue for r in recs if r.kind == "stale-premise"}
    assert stale_issues == {11}
    coll = [r for r in recs if r.kind == "queue-collision"]
    assert len(coll) == 1 and coll[0].evidence["colliding_issues"] == [10, 12]


def test_scan_scope_filter_and_missing_created(capsys):
    def _forbidden(p, s):
        raise AssertionError("out-of-scope task must not reach git")

    # Out-of-scope status/kind: skipped entirely (no collision participation).
    recs = pds.scan(
        [
            _task(10, status="on_hold"),
            _task(11, kind="experiment"),
            _task(12, status="running"),
        ],
        git_log_fn=_forbidden,
    )
    assert recs == []
    # Missing created_ts: commit scan skipped with a stderr line, but the
    # task still participates in queue-collision grouping.
    recs = pds.scan(
        [_task(20, created=None), _task(21, created=None, title="delta premise")],
        git_log_fn=_forbidden,
    )
    assert "no parseable created_at" in capsys.readouterr().err
    assert {r.kind for r in recs} == {"queue-collision"}


def test_scan_fingerprint_changes_with_new_commit():
    # A NEW invalidating commit must produce a NEW fingerprint (so the
    # pass's (issue, fingerprint) dedup re-fires on new evidence).
    subj = "task #99: watcher daemon liveness escalation rework"
    r1 = pds.scan([_task(10)], git_log_fn=lambda p, s: [("a" * 40, subj)])
    r2 = pds.scan([_task(10)], git_log_fn=lambda p, s: [("b" * 40, subj)])
    fp1 = [r.fingerprint for r in r1 if r.kind == "stale-premise"]
    fp2 = [r.fingerprint for r in r2 if r.kind == "stale-premise"]
    assert fp1 and fp2 and fp1 != fp2


# ─── real-body coverage for the seam-stubbed collectors ─────────────────────
# The pass tests below stub `pds.collect_tasks` / `pds.git_log_for_paths`;
# these two tests execute the REAL bodies (code-style § one production-body
# test per seam-stubbed function) — fakes only at the filesystem/git
# boundary via a tmp registry tree + a real throwaway git repo.


def _write_fixture_task(root: Path, issue: int, status: str, kind: str, body: str) -> None:
    d = root / "tasks" / status / str(issue)
    d.mkdir(parents=True, exist_ok=True)
    (d / "body.md").write_text(body, encoding="utf-8")


def test_collect_tasks_real_body(tmp_path, monkeypatch, capsys):
    import explore_persona_space.task_workflow as tw

    root = tmp_path / "repo"
    body = (
        "---\ncreated_at: '2026-08-01T00:00:00Z'\n---\n"
        "## Provenance\n- workflow_fix_target: scripts/foo.py\n"
    )
    _write_fixture_task(root, 10, "proposed", "infra", body)
    _write_fixture_task(root, 11, "proposed", "experiment", body)  # out-of-scope kind
    _write_fixture_task(root, 12, "blocked", "batch", body)
    reg = {
        "tasks": {
            "10": {
                "status": "proposed",
                "kind": "infra",
                "path": "tasks/proposed/10",
                "title": "alpha",
            },
            "11": {
                "status": "proposed",
                "kind": "experiment",
                "path": "tasks/proposed/11",
                "title": "exp",
            },
            "12": {
                "status": "blocked",
                "kind": "batch",
                "path": "tasks/blocked/12",
                "title": "beta",
            },
            # In-scope row whose body.md is MISSING -> fail-soft skip + stderr.
            "13": {
                "status": "proposed",
                "kind": "infra",
                "path": "tasks/proposed/13",
                "title": "gone",
            },
        }
    }
    reg_path = root / "tasks" / "REGISTRY.json"
    reg_path.write_text(json.dumps(reg))
    monkeypatch.setattr(tw, "registry_path", lambda: reg_path)

    tasks, repo_root = pds.collect_tasks()
    assert repo_root == root
    assert [t["id"] for t in tasks] == [10, 12]
    assert tasks[0]["title"] == "alpha"
    assert tasks[0]["created_ts"] == "2026-08-01T00:00:00Z"
    assert "workflow_fix_target" in tasks[0]["body"]
    assert "body read failed for #13" in capsys.readouterr().err


def test_git_log_for_paths_real_git(tmp_path):
    import subprocess

    repo = tmp_path / "gitrepo"
    repo.mkdir()

    def _git(*args):
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@e.st", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    _git("init", "-q")
    (repo / "foo.py").write_text("x = 1\n")
    _git("add", "foo.py")
    _git("commit", "-q", "-m", "task #99: watcher daemon liveness escalation rework")
    (repo / "bar.py").write_text("y = 2\n")
    _git("add", "bar.py")
    _git("commit", "-q", "-m", "unrelated other file commit")

    out = pds.git_log_for_paths(repo, ["foo.py"], "2020-01-01T00:00:00Z")
    assert len(out) == 1
    sha, subject = out[0]
    assert len(sha) == 40
    assert subject == "task #99: watcher daemon liveness escalation rework"
    # A path never touched since the cutoff yields an empty, non-raising scan.
    assert pds.git_log_for_paths(repo, ["nonexistent.py"], "2020-01-01T00:00:00Z") == []
    # scan() end-to-end over the REAL git seam: the acceptance-1 fixture fires.
    recs = pds.scan(
        [_task(10, targets="foo.py")],
        git_log_fn=lambda paths, since: pds.git_log_for_paths(repo, paths, since),
    )
    assert [r.kind for r in recs if r.issue == 10] == [
        "stale-premise",
        "landed-sibling-collision",
    ]


# ─── watcher pass: seams + isolation helper ──────────────────────────────────


def _pds_isolate(asw, monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point the pass's state (AUTONOMOUS_REGISTRY_DIR) + sidecar
    (PROJECT_ROOT-derived) at tmp_path; return (state_path, sidecar_path).
    Mirrors the registry-drift ``_rdrift_isolate`` shape."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path / "root")
    (tmp_path / "root").mkdir(parents=True, exist_ok=True)
    return (
        tmp_path / "registry" / "predispatch-staleness.json",
        tmp_path / "root" / ".claude" / "cache" / "predispatch-staleness-events.jsonl",
    )


def _stub_collect(monkeypatch, tasks, tmp_path):
    monkeypatch.setattr(pds, "collect_tasks", lambda: (list(tasks), tmp_path))
    monkeypatch.setattr(
        pds,
        "git_log_for_paths",
        lambda root, paths, since: [
            ("a" * 40, "task #99: watcher daemon liveness escalation rework")
        ],
    )


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# ─── watcher pass: behavior ──────────────────────────────────────────────────


def test_pass_fires_sidecar_marker_and_push(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)
    _stub_collect(monkeypatch, [_task(10)], tmp_path)
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg) or True)
    argvs: list[list[str]] = []

    def fake_run(argv, **kw):
        argvs.append(list(argv))
        return _FakeCompleted()

    monkeypatch.setattr(asw.subprocess, "run", fake_run)

    assert asw.predispatch_staleness_pass(dry_run=False) is True
    assert len(pushes) == 1
    assert "predispatch-staleness" in pushes[0]
    assert "Nothing was changed automatically" in pushes[0]
    rows = [json.loads(x) for x in sidecar_path.read_text().splitlines()]
    # stale-premise + landed-sibling for #10 (subject names sibling #99).
    kinds = sorted(r["flag_kind"] for r in rows)
    assert kinds == ["landed-sibling-collision", "stale-premise"]
    assert all(r["issue"] == 10 for r in rows)
    # Marker argvs: one per fired flag (2 < cap 5), all epm:progress posts
    # whose note leads with the anti-liveness sentinel.
    marker_argvs = [a for a in argvs if "task.py" in " ".join(a)]
    assert len(marker_argvs) == 2
    for argv in marker_argvs:
        sub = argv[argv.index("scripts/task.py") + 1]
        assert sub == "post-marker"
        note = argv[argv.index("--note") + 1]
        assert note.startswith(asw._PREDISPATCH_STALENESS_NOTE_SENTINEL)
        assert "clarifier adjudicates at dispatch" in note
    state = json.loads(state_path.read_text())
    assert isinstance(state["last_run_ts"], float)
    assert set(state["flagged"]) == {"10"}
    assert len(state["flagged"]["10"]) == 2  # one entry per fingerprint


def test_pass_kill_switch_skips_everything(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    monkeypatch.setenv("EPM_DISABLE_PREDISPATCH_STALENESS", "1")
    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)

    def _forbidden(*a, **kw):
        raise AssertionError("no collect / IO / push under the kill switch")

    monkeypatch.setattr(pds, "collect_tasks", _forbidden)
    monkeypatch.setattr(asw, "_telegram_push", _forbidden)
    assert asw.predispatch_staleness_pass(dry_run=False) is False
    assert not state_path.exists() and not sidecar_path.exists()


def test_pass_interval_throttle_honored(tmp_path, monkeypatch):
    # Forbidden-raiser collect + pre-seeded state byte-unchanged: a broken
    # throttle gate would raise into the fail-soft path, which WRITES an
    # error sidecar row (asserted absent) — not a vacuous returns-False pin.
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)

    def _forbidden(*a, **kw):
        raise AssertionError("throttled tick must not collect / push")

    monkeypatch.setattr(pds, "collect_tasks", _forbidden)
    monkeypatch.setattr(asw, "_telegram_push", _forbidden)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    seeded = json.dumps({"last_run_ts": time.time() - 60.0})
    state_path.write_text(seeded)
    assert asw.predispatch_staleness_pass(dry_run=False) is False
    assert state_path.read_text() == seeded
    assert not sidecar_path.exists()


def test_pass_dedup_latch_and_ttl_refire(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_INTERVAL_HOURS", "0")
    _stub_collect(monkeypatch, [_task(10)], tmp_path)
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg) or True)
    monkeypatch.setattr(asw.subprocess, "run", lambda argv, **kw: _FakeCompleted())

    # Run 1 fires; run 2 (same fingerprints, inside the 168h TTL) is latched:
    # no new sidecar rows, no push, returns False.
    assert asw.predispatch_staleness_pass(dry_run=False) is True
    n_rows_1 = len(sidecar_path.read_text().splitlines())
    assert asw.predispatch_staleness_pass(dry_run=False) is False
    assert len(pushes) == 1
    assert len(sidecar_path.read_text().splitlines()) == n_rows_1

    # Age the flagged timestamps past the TTL -> re-fires.
    state = json.loads(state_path.read_text())
    aged = {fp: ts - 169 * 3600.0 for fp, ts in state["flagged"]["10"].items()}
    state["flagged"]["10"] = aged
    state_path.write_text(json.dumps(state))
    assert asw.predispatch_staleness_pass(dry_run=False) is True
    assert len(pushes) == 2


def test_pass_marker_cap_overflow_stays_sidecar_only(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)
    # 8 tasks, one stale-premise flag each (git subject shares tokens with
    # every task body; no sibling ids in the subject) > marker cap 5.
    tasks = [_task(100 + i) for i in range(8)]
    monkeypatch.setattr(pds, "collect_tasks", lambda: (tasks, tmp_path))
    monkeypatch.setattr(
        pds,
        "git_log_for_paths",
        lambda root, paths, since: [("a" * 40, "watcher daemon liveness escalation rework")],
    )
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    argvs: list[list[str]] = []

    def fake_run(argv, **kw):
        argvs.append(list(argv))
        return _FakeCompleted()

    monkeypatch.setattr(asw.subprocess, "run", fake_run)

    assert asw.predispatch_staleness_pass(dry_run=False) is True
    rows = [json.loads(x) for x in sidecar_path.read_text().splitlines()]
    # Every fired flag has a sidecar row (8 stale-premise + 1 queue-collision
    # on the shared scripts/foo.py target fanned to 8 issues = 16 targets),
    # but only 5 markers were composed.
    assert len([a for a in argvs if "task.py" in " ".join(a)]) == 5
    assert len(rows) == 16
    assert sum(1 for r in rows if r["marker"]) == 5
    # #2134 v2 fold: state remembers ONLY the MARKERED fingerprints — the
    # 11 over-cap flags stay UNstamped so they re-enter the marker queue at
    # the next daily firing. Sorted (issue, fp) order gives the 5 slots to
    # both flags of #100 and #101 plus one flag of #102.
    state = json.loads(state_path.read_text())
    assert set(state["flagged"]) == {"100", "101", "102"}
    assert sum(len(v) for v in state["flagged"].values()) == 5


def test_pass_overcap_flag_unstamped_gets_marker_next_firing(tmp_path, monkeypatch):
    """#2134 v2 fold: the dedup TTL is stamped ONLY for flags whose marker
    was composed this firing. An over-cap (sidecar-only) flag stays
    unstamped and takes a marker slot at the NEXT firing; a markered flag
    IS stamped and never re-fires inside the TTL."""
    import autonomous_session_watch as asw

    state_path, _sidecar = _pds_isolate(asw, monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_INTERVAL_HOURS", "0")
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_MARKER_CAP", "1")
    # Two tasks, DISTINCT targets (no queue collision), one stale-premise
    # flag each (the commit subject carries no sibling ids).
    tasks = [
        _task(10, targets="scripts/a.py"),
        _task(11, targets="scripts/b.py", title="beta daemon premise"),
    ]
    monkeypatch.setattr(pds, "collect_tasks", lambda: (list(tasks), tmp_path))
    monkeypatch.setattr(
        pds,
        "git_log_for_paths",
        lambda root, paths, since: [("a" * 40, "watcher daemon liveness escalation rework")],
    )
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    argvs: list[list[str]] = []

    def fake_run(argv, **kw):
        argvs.append(list(argv))
        return _FakeCompleted()

    monkeypatch.setattr(asw.subprocess, "run", fake_run)

    def _marker_issues() -> list[str]:
        return [a[a.index("post-marker") + 1] for a in argvs if "scripts/task.py" in a]

    # Firing 1: cap 1 -> only #10 (lowest (issue, fp)) gets the marker;
    # #11 fires sidecar-only and is NOT stamped.
    assert asw.predispatch_staleness_pass(dry_run=False) is True
    assert _marker_issues() == ["10"]
    assert set(json.loads(state_path.read_text())["flagged"]) == {"10"}

    # Firing 2: #10 is TTL-latched; the overflowed #11 re-enters the marker
    # queue and takes the slot (previously it waited out the 168h TTL).
    assert asw.predispatch_staleness_pass(dry_run=False) is True
    assert _marker_issues() == ["10", "11"]
    assert set(json.loads(state_path.read_text())["flagged"]) == {"10", "11"}

    # Firing 3: both stamped inside the TTL -> nothing fires.
    assert asw.predispatch_staleness_pass(dry_run=False) is False
    assert _marker_issues() == ["10", "11"]


def test_pass_collision_codetected_across_window_boundary(tmp_path, monkeypatch):
    """#2134 v2 fold: the pass hands scan() the FULL collected list for
    collision grouping while the git-backed signals stay windowed — two
    colliding tasks on opposite sides of a cap-2 window boundary co-detect
    on the FIRST firing."""
    import autonomous_session_watch as asw

    _state, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_TASK_CAP", "2")
    tasks = [
        _task(100, targets="scripts/shared.py", title="alpha premise"),
        _task(101, targets="scripts/only101.py", title="beta premise"),
        _task(102, targets="scripts/only102.py", title="gamma premise"),
        _task(103, targets="scripts/only103.py", title="delta premise"),
        _task(104, targets="scripts/shared.py", title="epsilon premise"),
    ]
    monkeypatch.setattr(pds, "collect_tasks", lambda: (tasks, tmp_path))
    git_paths: list[list[str]] = []

    def fake_git(root, paths, since):
        git_paths.append(list(paths))
        return []

    monkeypatch.setattr(pds, "git_log_for_paths", fake_git)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    monkeypatch.setattr(asw.subprocess, "run", lambda argv, **kw: _FakeCompleted())

    assert asw.predispatch_staleness_pass(dry_run=False) is True
    rows = [json.loads(x) for x in sidecar_path.read_text().splitlines()]
    # ONE collision record on the shared target, fanned to BOTH issues —
    # #104 sits outside the [100, 101] window and is still co-detected.
    assert {r["flag_kind"] for r in rows} == {"queue-collision"}
    assert {r["issue"] for r in rows} == {100, 104}
    assert all(r["evidence"]["colliding_issues"] == [100, 104] for r in rows)
    # Git-backed signals stay WINDOW-scoped: only the window tasks hit git.
    assert git_paths == [["scripts/shared.py"], ["scripts/only101.py"]]


def test_pass_never_mutates_status_or_sweep_state(tmp_path, monkeypatch):
    """Report-only hard invariant (acceptance 3 + critic note): assert on the
    argv SUBCOMMAND token (position after ``scripts/task.py``) — NOT a
    substring scan, because the flag note deliberately QUOTES
    ``set-status <N> archived`` as the human adjudication affordance — and
    assert the infra-drain queue + proposed-infra-sweep state files come out
    byte-unchanged."""
    import autonomous_session_watch as asw

    _state, _sidecar = _pds_isolate(asw, monkeypatch, tmp_path)
    _stub_collect(monkeypatch, [_task(10), _task(11, title="beta premise")], tmp_path)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    argvs: list[list[str]] = []

    def fake_run(argv, **kw):
        argvs.append(list(argv))
        return _FakeCompleted()

    monkeypatch.setattr(asw.subprocess, "run", fake_run)

    # Seed the sweep/drain files the pass must never touch.
    sweep_state = asw._proposed_infra_sweep_state_path()
    drain_queue = asw._infra_drain_queue_path()
    sweep_state.parent.mkdir(parents=True, exist_ok=True)
    sweep_state.write_text('{"attempts": {"999": "sentinel"}}')
    drain_queue.write_text('{"queue": ["sentinel"]}')

    assert asw.predispatch_staleness_pass(dry_run=False) is True
    task_py_argvs = [a for a in argvs if "scripts/task.py" in a]
    assert task_py_argvs, "expected at least one flag marker post"
    for argv in task_py_argvs:
        sub = argv[argv.index("scripts/task.py") + 1]
        assert sub == "post-marker", f"forbidden task.py subcommand composed: {sub}"
        assert sub not in {"set-status", "archive"}
        # The note BODY may quote `set-status <N> archived` (the human
        # affordance); the subcommand-token assert above is the invariant.
    # No non-task.py mutation subprocess either (git log goes via the
    # stubbed pds seam; nothing else may run).
    for argv in argvs:
        assert "scripts/task.py" in argv
    assert sweep_state.read_text() == '{"attempts": {"999": "sentinel"}}'
    assert drain_queue.read_text() == '{"queue": ["sentinel"]}'


def test_pass_dry_run_writes_nothing(tmp_path, monkeypatch):
    """dry_run=True threads through the whole code path and performs ZERO
    writes: no state file, no sidecar row, no marker subprocess, no wire
    push (the push helper is invoked with dry_run=True only — the
    unfolded-round dry-run contract)."""
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)
    _stub_collect(monkeypatch, [_task(10)], tmp_path)
    push_dry_flags: list[bool] = []
    monkeypatch.setattr(
        asw, "_telegram_push", lambda msg, dry_run: push_dry_flags.append(dry_run) or True
    )

    def _forbidden_run(*a, **kw):
        raise AssertionError("dry_run must not spawn any subprocess")

    monkeypatch.setattr(asw.subprocess, "run", _forbidden_run)

    assert asw.predispatch_staleness_pass(dry_run=True) is True
    assert not state_path.exists()
    assert not sidecar_path.exists()
    assert push_dry_flags and all(f is True for f in push_dry_flags)

    # Dry-run left dedup state untouched: a follow-up real run still fires.
    monkeypatch.setattr(asw.subprocess, "run", lambda argv, **kw: _FakeCompleted())
    real_pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: real_pushes.append(msg) or True)
    assert asw.predispatch_staleness_pass(dry_run=False) is True
    assert len(real_pushes) == 1
    assert state_path.exists() and sidecar_path.exists()


def test_pass_cursor_persists_and_wraps(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    state_path, _sidecar = _pds_isolate(asw, monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_INTERVAL_HOURS", "0")
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_TASK_CAP", "2")
    tasks = [_task(100 + i) for i in range(5)]
    monkeypatch.setattr(pds, "collect_tasks", lambda: (tasks, tmp_path))
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    monkeypatch.setattr(asw.subprocess, "run", lambda argv, **kw: _FakeCompleted())
    windows: list[list[int]] = []
    real_scan = pds.scan

    def recording_scan(window, git_log_fn, **kw):
        windows.append([t["id"] for t in window])
        return real_scan(window, git_log_fn=lambda p, s: [], **kw)

    monkeypatch.setattr(pds, "scan", recording_scan)

    asw.predispatch_staleness_pass(dry_run=False)
    assert json.loads(state_path.read_text())["cursor_idx"] == 2
    asw.predispatch_staleness_pass(dry_run=False)
    assert json.loads(state_path.read_text())["cursor_idx"] == 4
    asw.predispatch_staleness_pass(dry_run=False)
    assert json.loads(state_path.read_text())["cursor_idx"] == 0  # wrapped
    assert windows == [[100, 101], [102, 103], [104]]


def test_pass_fail_soft_on_collect_exception(tmp_path, monkeypatch, capsys):
    import autonomous_session_watch as asw

    state_path, sidecar_path = _pds_isolate(asw, monkeypatch, tmp_path)

    def _boom():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(pds, "collect_tasks", _boom)

    def _no_push(*a, **kw):
        raise AssertionError("failed collect must not push")

    monkeypatch.setattr(asw, "_telegram_push", _no_push)
    assert asw.predispatch_staleness_pass(dry_run=False) is False
    assert "predispatch-staleness: pass failed (fail-soft)" in capsys.readouterr().err
    rows = [json.loads(x) for x in sidecar_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "predispatch-staleness-error"
    assert "registry unreadable" in rows[0]["error"]
    # Attempt stamp (saved BEFORE the collect) bounds a crashing collect to
    # one error row per throttle interval; no flagged/cursor state landed.
    state = json.loads(state_path.read_text())
    assert isinstance(state["last_run_ts"], float)
    assert "flagged" not in state


def test_pass_prunes_flagged_entries_out_of_scope(tmp_path, monkeypatch):
    import autonomous_session_watch as asw

    state_path, _sidecar = _pds_isolate(asw, monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_PREDISPATCH_STALENESS_INTERVAL_HOURS", "0")
    _stub_collect(monkeypatch, [_task(10)], tmp_path)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    monkeypatch.setattr(asw.subprocess, "run", lambda argv, **kw: _FakeCompleted())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"flagged": {"777": {"deadbeefcafe": _NOW}}})  # left the queue long ago
    )
    asw.predispatch_staleness_pass(dry_run=False)
    state = json.loads(state_path.read_text())
    assert "777" not in state["flagged"]
    assert "10" in state["flagged"]


# ─── pure decision helpers ───────────────────────────────────────────────────


def test_decide_flags_dedup_ttl_and_cap():
    import autonomous_session_watch as asw

    rec = object()
    targets = [(10, "fp1", rec), (11, "fp2", rec), (12, "fp3", rec)]
    realert_s = 168 * 3600.0
    # Never alerted -> fire; garbled ts -> fire; fresh ts -> latched.
    flagged = {"11": {"fp2": "garbled"}, "12": {"fp3": _NOW - 60.0}}
    actions = asw.decide_predispatch_staleness_flags(targets, flagged, _NOW, realert_s, 1)
    by_issue = {a["issue"]: a for a in actions}
    assert by_issue[10]["fire"] is True and by_issue[10]["marker"] is True  # cap slot 1
    assert by_issue[11]["fire"] is True and by_issue[11]["marker"] is False  # over cap
    assert by_issue[12]["fire"] is False and by_issue[12]["marker"] is False  # latched
    # TTL expiry re-fires.
    flagged = {"12": {"fp3": _NOW - realert_s - 1.0}}
    actions = asw.decide_predispatch_staleness_flags(
        [(12, "fp3", rec)], flagged, _NOW, realert_s, 5
    )
    assert actions[0]["fire"] is True
    # marker_cap=0 -> fired flags are sidecar-only.
    actions = asw.decide_predispatch_staleness_flags([(10, "fp1", rec)], {}, _NOW, realert_s, 0)
    assert actions[0]["fire"] is True and actions[0]["marker"] is False


def test_marker_targets_fan_out_queue_collision():
    import autonomous_session_watch as asw

    stale = pds.FlagRecord(10, "stale-premise", "scripts/foo.py", {"sha": "a"}, "fp1")
    coll = pds.FlagRecord(
        10, "queue-collision", "scripts/foo.py", {"colliding_issues": [10, 11, 12]}, "fp2"
    )
    targets = asw._predispatch_marker_targets([stale, coll])
    assert (10, "fp1", stale) in targets
    assert {(t[0], t[1]) for t in targets if t[2] is coll} == {
        (10, "fp2"),
        (11, "fp2"),
        (12, "fp2"),
    }


def test_sentinel_registered_for_anti_liveness():
    import autonomous_session_watch as asw

    assert asw._PREDISPATCH_STALENESS_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


# ─── main() wiring ───────────────────────────────────────────────────────────


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point AUTONOMOUS_REGISTRY_DIR at a tmp dir in BOTH spawn_session (the
    canonical home) and autonomous_session_watch (the stalled-detector suite's
    fixture shape) so a full-main() run never touches live fleet state."""
    import autonomous_session_watch as asw
    import spawn_session

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def test_main_wires_predispatch_staleness_before_dispatch_passes(isolated_registry, monkeypatch):
    # The pass must run BEFORE both dispatch passes (flags land on
    # events.jsonl before this tick's dispatches) while preserving the
    # pinned infra-drain -> proposed-infra-sweep adjacency ordering.
    import autonomous_session_watch as asw

    from tests.conftest import _stub_fleet_mutating_passes

    order: list[str] = []
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_session_ids", lambda: set())
    monkeypatch.setattr(asw, "_live_children", lambda: [])
    monkeypatch.setattr(asw, "_live_pids_by_sid_or_none", lambda: None)
    _stub_fleet_mutating_passes(asw, monkeypatch)
    for name in (
        "vm_disk_pass",
        "triage_observer_pass",
        "pod_safety_pass",
        "stalled_session_pass",
        "orphan_sweep_pass",
        "stale_blocked_flag_pass",
        "session_reconcile_pass",
        "zombie_wrapper_pass",
        "idle_unmapped_pass",
    ):
        monkeypatch.setattr(asw, name, lambda *a, **kw: None)
    monkeypatch.setattr(
        asw, "predispatch_staleness_pass", lambda *a, **kw: order.append("predispatch")
    )
    monkeypatch.setattr(asw, "infra_drain_pass", lambda *a, **kw: order.append("infra_drain"))
    monkeypatch.setattr(
        asw, "proposed_infra_sweep_pass", lambda *a, **kw: order.append("proposed_infra_sweep")
    )
    rc = asw.main([])
    assert rc == 0
    assert order.index("predispatch") < order.index("infra_drain")
    assert order.index("infra_drain") < order.index("proposed_infra_sweep")


def test_main_predispatch_staleness_only_flag(isolated_registry, monkeypatch):
    # --predispatch-staleness-only runs JUST this pass and exits; --dry-run
    # threads the kwarg through the dispatch branch (the plan-v2 dry-run pin).
    import autonomous_session_watch as asw

    calls: list[bool] = []
    monkeypatch.setattr(asw, "predispatch_staleness_pass", lambda dry_run: calls.append(dry_run))
    monkeypatch.setattr(
        asw, "vm_disk_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    monkeypatch.setattr(
        asw, "infra_drain_pass", lambda *a, **kw: pytest.fail("ran another pass under --only")
    )
    rc = asw.main(["--predispatch-staleness-only", "--dry-run"])
    assert rc == 0
    assert calls == [True]
    rc = asw.main(["--predispatch-staleness-only"])
    assert rc == 0
    assert calls == [True, False]
