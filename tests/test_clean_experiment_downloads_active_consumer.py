"""Tests for the active-CONSUMER reap gate in
``scripts/clean_experiment_downloads.py`` (task #773).

A ``data/issue_<M>/{hf_dl,g*_dl}`` cache must NEVER be deleted while a
DIFFERENT, currently-ACTIVE task declares ``data/issue_<M>/`` as a planned
input in its ``plans/plan.md`` or ``body.md``. The owning-issue terminal-status
gate (in ``vm_disk_guard.py``) allows the reap once ``M`` itself is terminal,
but says nothing about OTHER consumers — so a panic clean of ``#658``
(``awaiting_promotion``) reaped UltraChat/Betley input artifacts out from under
``#742`` (``running``), whose round-3 launch then died on ``FileNotFoundError``.
The new gate fails toward keep: any active consumer blocks the reap and the
skip is sidecar-logged (``kind: "active-consumer-reap-skipped"``).

The script lives under ``scripts/`` (not an importable package), so it is
loaded via importlib exactly like ``tests/test_clean_experiment_downloads_parity.py``.
A fake ``tasks/`` tree is built under the ``fake_repo`` tmp dir and
``ced.repo_root()`` is pointed at it (``tasks_dir`` / ``list_by_status`` both
resolve under ``repo_root()`` per call, so no extra monkeypatching is needed).
``failure_classifier.classify_failure`` is imported to pin the non-regression
invariant that the new sidecar reason string is never misrouted.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
fc = _load("failure_classifier")


# ─── fixtures / helpers ──────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Point every repo-root resolver at a temp dir so the sidecar, rel-name,
    and ``tasks/`` walk all resolve under one temp filesystem.

    Two resolver surfaces must be rebound: (1) ``ced.repo_root`` /
    ``ced.tasks_dir`` — the names ``clean_experiment_downloads`` calls directly
    (the sidecar path, ``_rel_name``, and the ``tasks/`` walk root); and (2)
    ``task_workflow``'s OWN resolvers — ``list_by_status`` was imported into
    ``ced`` as a bound function and internally calls ``task_workflow.tasks_dir``
    (which calls the cached ``task_workflow.repo_root``), so without rebinding
    those it would walk the REAL ``tasks/`` tree. ``invalidate_cache`` drops the
    process-local repo-root LRU before the rebind."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    # The names ced calls directly (it imported repo_root + tasks_dir from tw).
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "tasks_dir", lambda: tmp_path / "tasks")
    return tmp_path


def _make_cache(data_root: Path, issue_n: int) -> Path:
    """A normal re-downloadable issue cache: hf_dl + g1_dl (each a file), plus a
    SIBLING store/ (kept by the keep/delete contract, untouched here)."""
    issue_dir = data_root / f"issue_{issue_n}"
    for cache in ("hf_dl", "g1_dl"):
        d = issue_dir / cache
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 2048)
    sib_store = issue_dir / "store"
    sib_store.mkdir(parents=True)
    (sib_store / "v0_summaries.pt").write_bytes(b"y" * 4096)
    return issue_dir


def _make_task(
    repo: Path,
    *,
    issue_n: int,
    status: str,
    body: str = "",
    plan: str | None = None,
) -> Path:
    """Create ``tasks/<status>/<issue_n>/{body.md[,plans/plan.md]}`` under the
    fake repo. ``body.md`` carries minimal frontmatter so ``list_by_status``'s
    ``_read_body`` parses it; ``plan`` (when given) writes ``plans/plan.md``."""
    task_dir = repo / "tasks" / status / str(issue_n)
    task_dir.mkdir(parents=True)
    frontmatter = f"---\ntitle: task {issue_n}\nkind: experiment\n---\n\n"
    (task_dir / "body.md").write_text(frontmatter + body)
    if plan is not None:
        plans = task_dir / "plans"
        plans.mkdir()
        (plans / "plan.md").write_text(plan)
    return task_dir


def _read_sidecar(repo: Path) -> list[dict]:
    path = repo / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── case 1: hero (the #742 scenario) ────────────────────────────────────────


def test_hero_active_consumer_blocks_reap_and_escalates(fake_repo):
    """A=658 at awaiting_promotion with hf_dl + g1_dl caches; B=742 at running
    whose plan.md references data/issue_658/store/v0_summaries.pt. Reaping 658
    is SKIPPED (both cache dirs), the reason names #742, and one sidecar row per
    cache dir records the active-consumer skip."""
    data_root = fake_repo / "data"
    _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=742,
        status="running",
        body="Reuses data/issue_658/store/v0_summaries.pt (sha-pinned).",
        plan="Phase 2 loads data/issue_658/store/v0_summaries.pt as input.",
    )

    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)

    assert res.removed == []
    assert sorted(name for name, _ in res.skipped) == [
        "data/issue_658/g1_dl",
        "data/issue_658/hf_dl",
    ]
    # Skip reason names the consumer.
    assert all("#742" in reason for _, reason in res.skipped)
    # The caches survive on disk.
    assert (data_root / "issue_658" / "hf_dl" / "blob.bin").exists()
    assert (data_root / "issue_658" / "g1_dl" / "blob.bin").exists()

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 2  # one per kept cache dir
    for row in rows:
        assert row["kind"] == "active-consumer-reap-skipped"
        assert row["task"] == 658
        assert row["consumers"] == [742]  # the new field, dedup'd + sorted
        assert row["path"] in ("data/issue_658/hf_dl", "data/issue_658/g1_dl")
        assert "ts" in row


# ─── case 2: negative — terminal owning issue, no active consumer ────────────


def test_no_active_consumer_reaps_normally(fake_repo):
    """A=658 terminal, NO active task references it -> both cache dirs reaped,
    nothing skipped, no sidecar row."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    # An INACTIVE task referencing 658 must not protect it (see case 4); here we
    # add none at all.
    _make_task(fake_repo, issue_n=658, status="awaiting_promotion", body="self")

    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_658/g1_dl", "data/issue_658/hf_dl"]
    assert res.skipped == []
    assert res.failed == []
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    assert (issue_dir / "store" / "v0_summaries.pt").exists()  # sibling store kept
    assert _read_sidecar(fake_repo) == []


# ─── case 3: self-reap exempt ────────────────────────────────────────────────


def test_self_reference_does_not_block_own_reap(fake_repo):
    """A=658's OWN plan/body references data/issue_658/, and NO other active task
    does -> the reap proceeds (a task never protects itself)."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=658,
        status="awaiting_promotion",
        body="My own caches live under data/issue_658/hf_dl/.",
        plan="Generated data/issue_658/store/ from data/issue_658/hf_dl/.",
    )

    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_658/g1_dl", "data/issue_658/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()
    assert _read_sidecar(fake_repo) == []


# ─── case 4: inactive consumer does not protect ──────────────────────────────


@pytest.mark.parametrize("inactive_status", ["completed", "archived", "on_hold", "blocked"])
def test_inactive_consumer_does_not_protect(fake_repo, inactive_status):
    """B references data/issue_658/ but B is at an INACTIVE status -> no
    protection -> A's caches reaped."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=742,
        status=inactive_status,
        body="Once read data/issue_658/store/v0_summaries.pt.",
    )

    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_658/g1_dl", "data/issue_658/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()
    assert _read_sidecar(fake_repo) == []


# ─── case 5: incremental wrapper skips the guard ─────────────────────────────


def test_incremental_path_skips_active_consumer_guard(fake_repo, monkeypatch):
    """The within-run incremental path is self-aware: it opts out of the
    active-consumer gate (``_skip_active_consumer_guard=True``) and does NOT even
    walk tasks/. Monkeypatch the protected-set helper to raise; the incremental
    reap must NOT call it AND must reap the caches even with an active consumer
    B present."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=742,
        status="running",
        body="Reads data/issue_658/store/v0_summaries.pt.",
    )

    def _boom(*a, **k):
        raise AssertionError(
            "_active_consumer_protected_issues must NOT be called on the incremental path"
        )

    monkeypatch.setattr(ced, "_active_consumer_protected_issues", _boom)

    res = ced.clean_issue_downloads_incremental(658, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_658/g1_dl", "data/issue_658/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()


# ─── case 6: empty tasks/ tree is a clean no-op ──────────────────────────────


def test_empty_tasks_tree_is_noop(fake_repo):
    """_active_consumer_protected_issues returns {} under a fake_repo with no
    tasks/ tree at all (the parity suite's indirect dependence: an absent tasks/
    walk must never raise)."""
    assert ced._active_consumer_protected_issues(901) == {}


# ─── case 7: boundary regex ──────────────────────────────────────────────────


def test_data_issue_ref_boundary(fake_repo):
    """data/issue_65/ and data/issue65_x/ do NOT match 658; data/issue_658/,
    data/issue658/, data/issue658_slug/ DO match 658."""

    def _refs(text: str) -> set[int]:
        return {int(m.group(1)) for m in ced._DATA_ISSUE_REF.finditer(text)}

    # Positives for 658.
    assert 658 in _refs("path data/issue_658/store/x.pt")
    assert 658 in _refs("path data/issue658/hf_dl/x.bin")
    assert 658 in _refs("path data/issue658_marker_slug/store/x.pt")
    # Negatives — neither 65 nor 658 spuriously captured for 658.
    assert _refs("path data/issue_65/store/x.pt") == {65}
    assert _refs("path data/issue65_x/store/x.pt") == {65}
    assert 658 not in _refs("path data/issue_65/store/x.pt")
    assert 658 not in _refs("path data/issue65_x/store/x.pt")


def test_boundary_regex_via_protected_map(fake_repo):
    """End-to-end: a consumer referencing data/issue65_x/ does NOT protect 658,
    while one referencing data/issue658_slug/ does."""
    # A near-miss consumer (65, not 658).
    _make_task(
        fake_repo,
        issue_n=742,
        status="running",
        body="reads data/issue65_x/store/x.pt",
    )
    protected = ced._active_consumer_protected_issues(self_issue_n=999)
    assert 658 not in protected
    assert protected.get(65) == [742]

    # A no-underscore-with-slug consumer of 658.
    _make_task(
        fake_repo,
        issue_n=743,
        status="running",
        body="reads data/issue658_marker_slug/hf_dl/x.bin",
    )
    protected = ced._active_consumer_protected_issues(self_issue_n=999)
    assert protected.get(658) == [743]


# ─── case 8: dry-run reports but never deletes / persists sidecar ────────────


def test_dry_run_reports_skip_no_delete_no_persist(fake_repo):
    """apply=False with an active consumer -> the cache dirs are reported in
    .skipped (would-skip) but nothing is deleted; apply=False also reports the
    sidecar row only (does not persist it)."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=742,
        status="running",
        body="reads data/issue_658/store/v0_summaries.pt",
    )

    res = ced.clean_issue_downloads(658, apply=False, data_root=data_root)

    assert res.removed == []
    assert sorted(name for name, _ in res.skipped) == [
        "data/issue_658/g1_dl",
        "data/issue_658/hf_dl",
    ]
    assert (issue_dir / "hf_dl").exists()  # nothing deleted in dry-run
    assert (issue_dir / "g1_dl").exists()
    assert _read_sidecar(fake_repo) == []  # apply=False does not persist


# ─── case 9: missing plans/plan.md fail-soft (body-only still protects) ───────


def test_missing_plan_md_fail_soft_body_only_protects(fake_repo):
    """A consumer at planning/proposed with body.md referencing data/issue_658/
    but NO plans/plan.md (planning-status tasks before new_plan_version runs may
    lack the symlink): (a) no crash; (b) the body-only reference still
    protects."""
    data_root = fake_repo / "data"
    issue_dir = _make_cache(data_root, 658)
    _make_task(
        fake_repo,
        issue_n=742,
        status="planning",
        body="Phase 2 will load data/issue_658/store/v0_summaries.pt.",
        plan=None,  # no plans/plan.md
    )

    # (a) no crash; (b) body-only reference protects.
    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)

    assert res.removed == []
    assert sorted(name for name, _ in res.skipped) == [
        "data/issue_658/g1_dl",
        "data/issue_658/hf_dl",
    ]
    assert (issue_dir / "hf_dl").exists()
    rows = _read_sidecar(fake_repo)
    assert all(r["consumers"] == [742] for r in rows)


# ─── case 10: failure_classifier non-regression ──────────────────────────────


def test_active_consumer_reason_not_misrouted_by_failure_classifier(fake_repo):
    """The active-consumer skip reason string must never be misrouted by
    classify_failure (it is never fed there, but pin the non-regression
    invariant: the reason carries no failure_class field, CUDA-OOM, DataLoader
    wrap, or 'No space left on device / ENOSPC / disk full' infra pattern, so it
    falls through to the conservative 'code' default)."""
    reason = (
        "active task(s) #742 declare data/issue_658/ as a planned input — "
        "reaping data/issue_658/hf_dl could strand their run; KEPT"
    )
    assert fc.classify_failure(reason) == "code"


# ─── case 11: partition totality ─────────────────────────────────────────────


def test_consumer_inactive_statuses_partition_totality(fake_repo):
    """_CONSUMER_INACTIVE_STATUSES is a subset of STATUSES, and the active set
    (STATUSES - inactive) covers every remaining status (no silent default).
    Recurs whenever the status enum changes."""
    assert ced._CONSUMER_INACTIVE_STATUSES.issubset(set(ced.STATUSES))
    active = set(ced.STATUSES) - ced._CONSUMER_INACTIVE_STATUSES
    # Union of active + inactive covers all of STATUSES (a clean partition).
    assert active | ced._CONSUMER_INACTIVE_STATUSES == set(ced.STATUSES)
    # And the active set is non-empty (the gate must be able to fire).
    assert active
