"""Tests for explore_persona_space.campaign_state (task #586).

Same fake-repo pattern as tests/test_task_workflow.py: each test runs in a
temporary git repo and rebinds task_workflow's resolver functions, so
campaign_state's path resolution (which goes through find_task_path) lands
in the tmp tree without touching the real tasks/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ─── Fake-repo fixture (mirrors tests/test_task_workflow.py) ───────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """git-init tmp_path and rebind task_workflow's resolvers to it. Returns
    ``(repo_root, task_workflow_module, campaign_state_module)``."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.campaign_state as cs
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    # Hermetic registry-caps source: never read the real ~/.eps-autonomous
    # (a stray campaign-<small-N>.json there would leak caps into tests).
    monkeypatch.setattr(cs, "AUTONOMOUS_REGISTRY_DIR", tmp_path / ".eps-autonomous")
    (tmp_path / "tasks").mkdir()
    return tmp_path, tw, cs


BRIEF_BODY = """# Campaign: does persona distance predict leakage?

## Goal

Settle q:leak-predictor.

## Campaign Brief

**Question anchor:** q:leak-predictor

### Hypotheses

- H1: base-model persona distance predicts marker leakage.

### Experiments

| id | title | hypothesis | depends_on | gpu_hours_est |
|----|-------|------------|------------|---------------|
| exp-01 | Distance sweep | H1 holds across 8 sources | - | 30 |
| exp-02 | Held-out replication | H1 transfers to unseen personas | exp-01 | 40 |
| exp-03 | Negative-panel ablation | gradient needs contrastive negatives | - | 25 |

## Notes

Not part of the brief table.
"""


def _make_campaign(tw, body: str = BRIEF_BODY, **fm_extra):
    """Create a kind=campaign task with ``body`` and return (task_id, fm, body)."""
    task_id = tw.create_task(
        tw.NewTaskRequest(kind="campaign", title="campaign: leak predictor", body=body)
    )
    if fm_extra:
        # Write extra frontmatter keys directly (permissive freeform YAML).
        path = tw.find_task_path(task_id) / "body.md"
        fm, body_only = tw._read_body(path)
        fm.update(fm_extra)
        tw._write_body(path, fm, body_only)
    task = tw.get_task(task_id)
    return task_id, task["frontmatter"], task["body"]


# ─── init_state_from_brief + round-trip ─────────────────────────────────────


def test_init_state_roundtrip_defaults(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)

    # Persisted where the schema says, and load_state returns the same doc.
    path = tw.find_task_path(task_id) / "artifacts" / cs.STATE_FILENAME
    assert path.is_file()
    assert cs.load_state(task_id) == state

    assert state["schema_version"] == 1
    assert state["campaign_task"] == task_id
    assert state["question_anchor"] == "q:leak-predictor"
    assert state["budget"] == {"gpu_hours_total": 250.0, "gpu_hours_committed": 0.0}
    assert state["limits"]["max_experiments"] == 8
    assert state["limits"]["max_concurrent_children"] == 4
    assert state["limits"]["per_child_gpu_hours_cap"] == 100.0
    assert state["stop"]["dry_limit"] == 3
    assert state["stop"]["confidence_target"] == "HIGH"
    assert state["stop"]["current_confidence"] is None
    assert [e["id"] for e in state["experiments"]] == ["exp-01", "exp-02", "exp-03"]
    assert state["experiments"][1]["depends_on"] == ["exp-01"]
    assert state["experiments"][0]["depends_on"] == []
    assert all(e["status"] == "planned" for e in state["experiments"])
    assert state["experiments"][0]["gpu_hours_est"] == 30.0


def test_init_state_honors_frontmatter_overrides(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(
        tw,
        campaign={"gpu_hours_total": 60, "max_concurrent_children": 2, "dry_limit": 1},
    )
    state = cs.init_state_from_brief(task_id, fm, body)
    assert state["budget"]["gpu_hours_total"] == 60.0
    assert state["limits"]["max_concurrent_children"] == 2
    assert state["stop"]["dry_limit"] == 1
    # Untouched keys keep defaults.
    assert state["limits"]["max_experiments"] == 8


def test_init_state_caps_precedence_registry_then_frontmatter(fake_repo):
    """Budget seeding precedence: frontmatter `campaign:` overrides >
    registry caps (campaign-<N>.json from spawn-campaign flags) > defaults.
    Pins the single-pathed enforcement of `spawn-campaign --budget-gpu-hours`."""
    _, tw, cs = fake_repo

    # Registry caps only -> they beat the module defaults.
    task_id, fm, body = _make_campaign(tw)
    reg_dir = cs.AUTONOMOUS_REGISTRY_DIR
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / f"campaign-{task_id}.json").write_text(
        json.dumps({"budget_gpu_hours": 50.0, "max_concurrent": 2, "per_child_gpu_hours_cap": 20.0})
    )
    state = cs.init_state_from_brief(task_id, fm, body)
    assert state["budget"]["gpu_hours_total"] == 50.0
    assert state["limits"]["max_concurrent_children"] == 2
    assert state["limits"]["per_child_gpu_hours_cap"] == 20.0
    # Keys the registry never carries keep the module defaults.
    assert state["limits"]["max_experiments"] == cs.DEFAULT_MAX_EXPERIMENTS

    # Frontmatter override on the SAME key beats the registry cap.
    task_id2, fm2, body2 = _make_campaign(tw, campaign={"gpu_hours_total": 75})
    (reg_dir / f"campaign-{task_id2}.json").write_text(
        json.dumps({"budget_gpu_hours": 50.0, "max_concurrent": 2})
    )
    state2 = cs.init_state_from_brief(task_id2, fm2, body2)
    assert state2["budget"]["gpu_hours_total"] == 75.0  # frontmatter wins
    assert state2["limits"]["max_concurrent_children"] == 2  # registry still applies


def test_init_state_empty_table_row_fails_loud(fake_repo):
    _, tw, cs = fake_repo
    bad = BRIEF_BODY.replace(
        "| exp-03 | Negative-panel ablation | gradient needs contrastive negatives | - | 25 |",
        "|  |  |  |  |  |",
    )
    task_id, fm, body = _make_campaign(tw, body=bad)
    with pytest.raises(ValueError, match="row is empty"):
        cs.init_state_from_brief(task_id, fm, body)


def test_init_state_unknown_override_key_fails_loud(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw, campaign={"max_experiment": 5})  # typo
    with pytest.raises(ValueError, match="unknown campaign override"):
        cs.init_state_from_brief(task_id, fm, body)


def test_init_state_missing_brief_fails_loud(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw, body="# No brief here\n\nJust prose.\n")
    with pytest.raises(ValueError, match="Campaign Brief"):
        cs.init_state_from_brief(task_id, fm, body)


def test_init_state_unknown_dependency_fails_loud(fake_repo):
    _, tw, cs = fake_repo
    bad = BRIEF_BODY.replace("| exp-01 | 40 |", "| exp-99 | 40 |")
    task_id, fm, body = _make_campaign(tw, body=bad)
    with pytest.raises(ValueError, match="unknown id 'exp-99'"):
        cs.init_state_from_brief(task_id, fm, body)


def test_init_state_dependency_cycle_fails_loud(fake_repo):
    _, tw, cs = fake_repo
    cyclic = BRIEF_BODY.replace(
        "| exp-01 | Distance sweep | H1 holds across 8 sources | - | 30 |",
        "| exp-01 | Distance sweep | H1 holds across 8 sources | exp-02 | 30 |",
    )
    task_id, fm, body = _make_campaign(tw, body=cyclic)
    with pytest.raises(ValueError, match="cycle"):
        cs.init_state_from_brief(task_id, fm, body)


# ─── save_state atomicity + validation ──────────────────────────────────────


def test_save_state_atomic_and_validates_before_write(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    path = tw.find_task_path(task_id) / "artifacts" / cs.STATE_FILENAME
    on_disk_before = path.read_text()

    # An invalid mutation must raise BEFORE any write — the on-disk state
    # is untouched and no temp file is left behind.
    broken = json.loads(on_disk_before)
    broken["experiments"][0]["status"] = "not-a-status"
    with pytest.raises(ValueError, match="not-a-status"):
        cs.save_state(task_id, broken)
    assert path.read_text() == on_disk_before
    assert not list(path.parent.glob("*.tmp"))

    # A valid mutation lands cleanly (and leaves no temp file).
    state["budget"]["gpu_hours_committed"] = 30.0
    cs.save_state(task_id, state)
    assert cs.load_state(task_id)["budget"]["gpu_hours_committed"] == 30.0
    assert not list(path.parent.glob("*.tmp"))


def test_save_state_rejects_wrong_task_binding(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    state["campaign_task"] = task_id + 1
    with pytest.raises(ValueError, match="bound to task"):
        cs.save_state(task_id, state)


def test_save_state_rejects_unknown_current_confidence(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    state["stop"]["current_confidence"] = "VERY-HIGH"
    with pytest.raises(ValueError, match="current_confidence"):
        cs.save_state(task_id, state)
    # Known levels are accepted case-insensitively; null stays valid.
    for value in ("moderate", "HIGH", None):
        state["stop"]["current_confidence"] = value
        cs.save_state(task_id, state)


def test_load_state_missing_raises_filenotfound(fake_repo):
    _, tw, cs = fake_repo
    task_id, _, _ = _make_campaign(tw)
    with pytest.raises(FileNotFoundError, match="campaign state missing"):
        cs.load_state(task_id)


# ─── scheduling reads ───────────────────────────────────────────────────────


def test_ready_experiments_honors_depends_on(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)

    # exp-02 depends on exp-01 (still planned) -> only the roots are ready.
    assert [e["id"] for e in cs.ready_experiments(state)] == ["exp-01", "exp-03"]

    # exp-01 in flight does NOT unblock exp-02; ingested does.
    state["experiments"][0]["status"] = "running"
    assert [e["id"] for e in cs.ready_experiments(state)] == ["exp-03"]
    state["experiments"][0]["status"] = "ingested"
    assert [e["id"] for e in cs.ready_experiments(state)] == ["exp-02", "exp-03"]

    # An ABANDONED dependency keeps the dependent un-ready.
    state["experiments"][0]["status"] = "abandoned"
    assert [e["id"] for e in cs.ready_experiments(state)] == ["exp-03"]


def test_open_slots_and_budget_headroom(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    assert cs.open_slots(state) == 4
    state["experiments"][0]["status"] = "running"
    state["experiments"][1]["status"] = "filed"
    state["experiments"][2]["status"] = "landed"
    assert cs.open_slots(state) == 1
    # waiting-user / ingested / abandoned do NOT occupy slots.
    state["experiments"][2]["status"] = "waiting-user"
    assert cs.open_slots(state) == 2

    assert cs.budget_headroom(state) == 250.0
    state["budget"]["gpu_hours_committed"] = 70.0
    assert cs.budget_headroom(state) == 180.0


# ─── check_stop ordering ────────────────────────────────────────────────────


def _all_triggers_state(state: dict) -> dict:
    """Mutate ``state`` so EVERY stop criterion is simultaneously true:
    deadline passed, budget exhausted, max experiments reached, campaign
    working belief at target, dry counter at limit."""
    state["wall_clock_deadline"] = "2020-01-01T00:00:00Z"
    state["budget"]["gpu_hours_committed"] = state["budget"]["gpu_hours_total"]
    state["limits"]["max_experiments"] = 1
    state["experiments"][0]["status"] = "ingested"
    state["stop"]["current_confidence"] = "HIGH"
    state["stop"]["dry_counter"] = state["stop"]["dry_limit"]
    return state


def test_check_stop_fixed_order(fake_repo):
    """Peel triggers off one at a time and assert the reason follows the
    fixed order: user-stop -> wall-clock -> budget -> max-experiments ->
    confidence target -> dry counter."""
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = _all_triggers_state(cs.init_state_from_brief(task_id, fm, body))
    now = datetime(2026, 6, 11, tzinfo=UTC)

    stop, reason = cs.check_stop(state, now, user_stop=True)
    assert stop and "user-stop" in reason

    stop, reason = cs.check_stop(state, now)
    assert stop and "wall-clock" in reason

    state["wall_clock_deadline"] = (now + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop, reason = cs.check_stop(state, now)
    assert stop and "budget" in reason

    state["budget"]["gpu_hours_committed"] = 0.0
    stop, reason = cs.check_stop(state, now)
    assert stop and "max experiments" in reason

    state["limits"]["max_experiments"] = 8
    stop, reason = cs.check_stop(state, now)
    assert stop and "confidence target" in reason

    state["stop"]["current_confidence"] = "LOW"
    stop, reason = cs.check_stop(state, now)
    assert stop and "dry counter" in reason

    state["stop"]["dry_counter"] = 0
    stop, reason = cs.check_stop(state, now)
    assert (stop, reason) == (False, None)


def test_single_high_child_does_not_stop_campaign(fake_repo):
    """BLOCKER fix on #586 (`campaign-stop-confidence-per-claim`): per-child
    clean-result confidence is PER-CLAIM — an ingested HIGH child must NOT
    stop the campaign while the CAMPAIGN-LEVEL working belief
    (`stop.current_confidence`) is null or below target."""
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    now = datetime.now(tz=UTC)

    # First child lands HIGH on its narrow claim; working belief still null.
    state["experiments"][0]["status"] = "ingested"
    state["experiments"][0]["confidence"] = "HIGH"
    assert cs.check_stop(state, now) == (False, None)

    # Working belief set but below the HIGH target: still no stop.
    state["stop"]["current_confidence"] = "MODERATE"
    assert cs.check_stop(state, now) == (False, None)


def test_check_stop_confidence_target_reads_campaign_level_belief(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    now = datetime.now(tz=UTC)

    # Working belief at/above target trips the stop — independent of any
    # per-child tag (no experiment row is even ingested here).
    state["stop"]["current_confidence"] = "DETERMINATE"
    stop, reason = cs.check_stop(state, now)
    assert stop and "confidence target" in reason and "working belief" in reason

    # Unknown string never satisfies the target.
    state["stop"]["current_confidence"] = "VERY-SURE"
    assert cs.check_stop(state, now) == (False, None)


def test_check_stop_already_stopped_short_circuits(fake_repo):
    _, tw, cs = fake_repo
    task_id, fm, body = _make_campaign(tw)
    state = cs.init_state_from_brief(task_id, fm, body)
    state["stop"]["stopped"] = True
    state["stop"]["stop_reason"] = "budget exhausted (test)"
    stop, reason = cs.check_stop(state, datetime.now(tz=UTC))
    assert stop and reason == "budget exhausted (test)"
