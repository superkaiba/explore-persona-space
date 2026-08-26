"""Tests for the #2342 reconcile-family marker-kind CLI guard.

Covers the pure predicate (`task._reconcile_kind_violation`) and the
`task.py post-marker` CLI wiring: any marker kind containing the substring
``reconcil`` other than the ONE canonical marker-mode kind
``epm:review-reconcile`` refuses with exit 2 and appends NOTHING
(byte-checked); the canonical kind and non-reconcile kinds post unchanged;
library-level `post_event` stays deliberately ungated (the #2309
precedent — programmatic posters unaffected, the CLI is the chokepoint).

The deviant parametrization below is the COMPLETE observed fleet census
(2026-08-24 grep over ``tasks/*/*/events.jsonl``: 908 canonical rows vs
88 deviant rows across these 15 improvised variants) — every variant the
fleet has actually produced must trip the guard.

The CLI is exercised at the handler-function layer against a fake repo
(the branch-guarded resolver can't be redirected across a process
boundary — see test_task_workflow_post_marker_echo.py).

Workflow-invariant family: run after any edit to `scripts/task.py` or
`task_workflow.py`'s marker surface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import task as task_cli

from explore_persona_space import task_workflow as tw_mod

# The complete observed fleet census of DEVIANT reconcile-family kinds
# (2026-08-24; see module docstring). epm:plan-critique-reconcile is the
# in-context stdout tag — 17 rows were wrongly posted as events; it gets
# a dedicated error message (tested separately below) but refuses like
# the rest.
OBSERVED_DEVIANT_KINDS = [
    "epm:clean-result-critic-reconciler",
    "epm:clean-result-critique-reconcile",
    "epm:clean-result-critique-reconciler",
    "epm:clean-result-reconcile",
    "epm:code-review-reconcile",
    "epm:code-review-reconciled",
    "epm:code-review-reconciler",
    "epm:interp-critique-reconcile",
    "epm:interp-critique-reconciled",
    "epm:interp-reconcile",
    "epm:plan-critique-reconcile",
    "epm:reconciled",
    "epm:reconciler-decision",
    "epm:reconciler-verdict",
    "epm:review-reconciliation",
]

# The plan's representative set adds the registered documentation alias
# (its own workflow.yaml entry says the reconciler posts the ONE
# canonical kind) — never observed in events.jsonl, refused by design.
DOC_ALIAS_KIND = "epm:followup-value-critique-reconcile"

RECONCILE_BODY = "**Role under adjudication:** code-reviewer\n\nVerdict: PASS\n"


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo with task_workflow's resolvers rebound
    (the test_task_workflow.py convention)."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    tw_mod.invalidate_cache()
    monkeypatch.setattr(tw_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw_mod, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw_mod, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw_mod, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw_mod, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.setattr(tw_mod, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    monkeypatch.setattr(tw_mod, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")

    tid = tw_mod.create_task(tw_mod.NewTaskRequest(kind="infra", title="reconcile guard fixture"))
    return tmp_path, tw_mod, tid


def _events_path(tw, tid: int) -> Path:
    return tw.find_task_path(tid) / "events.jsonl"


def _ns(tid: int, marker: str, note: str | None, **overrides) -> argparse.Namespace:
    base = dict(
        number=tid,
        marker=marker,
        version=None,
        by="test",
        note=note,
        file=None,
        allow_nonconforming_report=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ─── Pure-predicate unit tests ─────────────────────────────────────────────


def test_predicate_canonical_kind_is_none():
    assert task_cli._reconcile_kind_violation("epm:review-reconcile") is None


def test_predicate_non_reconcile_kinds_are_none():
    for kind in ("epm:progress", "epm:results", "epm:code-review", "epm:status-changed"):
        assert task_cli._reconcile_kind_violation(kind) is None, kind


@pytest.mark.parametrize("kind", [*OBSERVED_DEVIANT_KINDS, DOC_ALIAS_KIND])
def test_predicate_trips_on_every_observed_variant(kind):
    """Every fleet-observed deviant variant + the doc alias trips the
    guard (the plan's implementation-time census assertion)."""
    msg = task_cli._reconcile_kind_violation(kind)
    assert msg is not None
    assert "epm:review-reconcile" in msg
    assert "Role under adjudication" in msg
    assert "NOTHING was appended" in msg


def test_predicate_plan_critique_message_names_stdout_tag():
    msg = task_cli._reconcile_kind_violation("epm:plan-critique-reconcile")
    assert msg is not None
    assert "stdout" in msg
    assert "never post it to events.jsonl" in msg


def test_predicate_generic_message_bans_role_derivation():
    msg = task_cli._reconcile_kind_violation("epm:code-review-reconcile")
    assert msg is not None
    assert "Never derive the marker kind" in msg
    assert "workflow.yaml" in msg
    assert "reconciler.md" in msg


# ─── CLI wiring tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", [*OBSERVED_DEVIANT_KINDS, DOC_ALIAS_KIND])
def test_cli_refuses_deviant_kind_exit_2_nothing_appended(fake_repo, capsys, kind):
    """Acceptance criteria 1+2: exit code 2, stderr names the canonical
    recipe, events.jsonl is BYTE-unchanged."""
    _, tw, tid = fake_repo
    before = _events_path(tw, tid).read_bytes()
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_post_event(_ns(tid, kind, RECONCILE_BODY))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "epm:review-reconcile" in err
    assert _events_path(tw, tid).read_bytes() == before


def test_cli_refuses_before_reading_file_channel(fake_repo, tmp_path):
    """The guard fires on the KIND alone — a --file post of a deviant kind
    refuses identically (and before the note file matters)."""
    _, tw, tid = fake_repo
    before = _events_path(tw, tid).read_bytes()
    body = tmp_path / "reconcile.md"
    body.write_text(RECONCILE_BODY)
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_post_event(_ns(tid, "epm:code-review-reconcile", None, file=str(body)))
    assert exc.value.code == 2
    assert _events_path(tw, tid).read_bytes() == before


def test_cli_canonical_kind_posts_unchanged(fake_repo):
    """Acceptance criterion 3: epm:review-reconcile behavior is unchanged
    (exit 0 path, row appended with the right kind + note)."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:review-reconcile", RECONCILE_BODY))
    rows = [json.loads(x) for x in _events_path(tw, tid).read_text().splitlines()]
    assert rows[-1]["kind"] == "epm:review-reconcile"
    assert "Role under adjudication" in rows[-1]["note"]


def test_cli_non_reconcile_kind_unaffected(fake_repo):
    """Acceptance criterion 3: kinds not containing `reconcil` post as
    before."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:progress", "ordinary note"))
    rows = [json.loads(x) for x in _events_path(tw, tid).read_text().splitlines()]
    assert rows[-1]["kind"] == "epm:progress"


def test_library_post_event_stays_ungated(fake_repo):
    """Acceptance criterion 4: `task_workflow.post_event` is deliberately
    NOT gated (the #2309 precedent) — the CLI is the chokepoint; a direct
    library call still appends. reconciler.md declares the library path
    out of contract at the instruction layer."""
    _, tw, tid = fake_repo
    tw.post_event(tid, "epm:code-review-reconcile", note="library path, deliberately ungated")
    rows = [json.loads(x) for x in _events_path(tw, tid).read_text().splitlines()]
    assert rows[-1]["kind"] == "epm:code-review-reconcile"
