"""Unit tests for the #1490 ALERT-ONLY orphan-gcp-handle pod-safety arm.

The watcher-side D2 companion of the poller-side D1 provision-residue reclaim
(``backend_poll._reclaim_failed_runpod_provision``, pinned in
``tests/test_backend_poll.py``): a RUNNING bare ``pod-<N>`` whose run-handle
sidecar still points at ``backend=gcp`` past a grace window is likely
provision residue from a failed GCP->RunPod failover launch the poller-side
reclaim missed (poller crash between provision and reclaim, or a failed /
unattributable terminate — the #1417 shape: pod-1417 idled 45 min on 4xH100
until a human noticed). Covers:

* the end-to-end arm in ``_process_pod`` — exactly ONE alert per pod
  incarnation (``orphan_gcp_noted`` dedup), never a stop/terminate;
* the predicate's negative controls (runpod sidecar / absent sidecar /
  unreadable sidecar / suffixed pod name / within grace / keep-running /
  follow-up shields — passed-flag AND lazy-re-check legs);
* the additive-only contract: the firing arm never blocks the pre-existing
  status-class decision (the DONE-task auto-stop still happens).

Follows ``tests/test_autonomous_session_watch_wedge.py`` conventions: PodInfo
fixtures, the patched state dir, ``task.py`` reads monkeypatched, no network.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

from explore_persona_space.backends.base import RunHandle  # noqa: E402
from explore_persona_space.backends.issue_dispatch import write_handle_sidecar  # noqa: E402

GRACE = asw.ORPHAN_GCP_HANDLE_GRACE_SEC  # 1800s (env-overridable at call time)


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the wedge suite)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _healthy_info(pod_id: str = "p1417", name: str = "pod-1417") -> PodInfo:
    """A RUNNING pod with a public port (so the #692 wedge arm never handles it)."""
    return PodInfo(
        pod_id=pod_id, name=name, desired_status="RUNNING", ssh_host="1.2.3.4", ssh_port=22001
    )


def _gcp_handle(issue: int = 1417) -> RunHandle:
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-1",
        pod_name=f"eps-issue-{issue}",
        scratch_dir=f"/workspace/eps-issue-{issue}",
        log_path=f"/workspace/logs/issue-{issue}.log",
        extra={"issue": issue},
    )


def _seed_state(tmp_path, issue: int, *, pod_id: str, first_seen: float, missed: int = 0) -> None:
    """Write a pod-safety state file with an aged incarnation clock."""
    payload = {
        "pod_id": pod_id,
        "missed": missed,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": first_seen,
    }
    (tmp_path / f"pod-safety-{issue}.json").write_text(json.dumps(payload))


@pytest.fixture
def gcp_sidecar(tmp_path, monkeypatch):
    """A REAL backend=gcp handle sidecar, resolved by the arm's real read leg."""
    sidecar = tmp_path / "issue-1417-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.resolve_handle_sidecar_path",
        lambda issue: (sidecar, False),
    )
    return sidecar


@pytest.fixture
def marker_recorder(monkeypatch):
    posts: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label=None: posts.append((issue, note, label)),
    )
    return posts


@pytest.fixture
def stop_recorder(monkeypatch):
    stops: list[int] = []
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    return stops


@pytest.fixture
def active_task(monkeypatch):
    """A RUNNING-status task with fresh progress (status-class action=keep)."""
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: time.time() - 600)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)


def _orphan_posts(posts):
    return [p for p in posts if asw._ORPHAN_GCP_HANDLE_NOTE_SENTINEL in p[1]]


# ---------------------------------------------------------------------------
# 9. The end-to-end alert (once per pod incarnation) — AC 6
# ---------------------------------------------------------------------------


def test_orphan_gcp_handle_pod_alerts_once(
    isolated_registry, gcp_sidecar, marker_recorder, stop_recorder, active_task
):
    """AC 6: RUNNING bare pod-1417 + backend=gcp sidecar + age past grace and
    no exemptions -> exactly ONE alert on tick 1, none on tick 2
    (orphan_gcp_noted persisted); NO stop/terminate ever."""
    now = time.time()
    _seed_state(isolated_registry, 1417, pod_id="p1417", first_seen=now - GRACE - 600)

    asw._process_pod(1417, "p1417", _healthy_info(), now, dry_run=False, threshold=2)
    orphan = _orphan_posts(marker_recorder)
    assert len(orphan) == 1
    issue, note, label = orphan[0]
    assert issue == 1417
    assert label == "orphan-gcp-handle"
    assert "pod-1417" in note
    assert "backend=gcp" in note
    assert "pod.py terminate --issue 1417" in note
    assert stop_recorder == []
    state = json.loads((isolated_registry / "pod-safety-1417.json").read_text())
    assert state["orphan_gcp_noted"] is True

    # Tick 2: the persisted dedup flag short-circuits — no second alert.
    asw._process_pod(1417, "p1417", _healthy_info(), now + 600, dry_run=False, threshold=2)
    assert len(_orphan_posts(marker_recorder)) == 1
    assert stop_recorder == []


def test_orphan_arm_realerts_on_new_pod_incarnation(
    isolated_registry, gcp_sidecar, marker_recorder, stop_recorder, active_task
):
    """The dedup flag is per pod INCARNATION (the wedge-clock reset
    convention): a stored orphan_gcp_noted under a DIFFERENT pod_id does not
    suppress the fresh incarnation's alert."""
    now = time.time()
    payload = {
        "pod_id": "p_OLD",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - GRACE - 600,
        "orphan_gcp_noted": True,
    }
    (isolated_registry / "pod-safety-1417.json").write_text(json.dumps(payload))

    asw._process_pod(1417, "p_NEW", _healthy_info(pod_id="p_NEW"), now, dry_run=False, threshold=2)
    assert len(_orphan_posts(marker_recorder)) == 1
    assert stop_recorder == []

    # End-to-end once-per-incarnation pin (r2): after the alert tick, the
    # persisted state carries the NEW pod_id + the noted flag, and a second
    # tick on the same incarnation stays quiet. On THIS (healthy-pod) path the
    # wedge arm's _clear_wedge_state laundering save re-keys the state to the
    # new pod_id BEFORE the status-class load, so the flag survives even
    # pre-r2; the direct pre-fix-failing pin of the r2 pod_id mirror is
    # test_orphan_arm_mirror_survives_status_class_save_on_new_incarnation.
    state = json.loads((isolated_registry / "pod-safety-1417.json").read_text())
    assert state["pod_id"] == "p_NEW"
    assert state["orphan_gcp_noted"] is True

    asw._process_pod(
        1417, "p_NEW", _healthy_info(pod_id="p_NEW"), now + 600, dry_run=False, threshold=2
    )
    assert len(_orphan_posts(marker_recorder)) == 1
    assert stop_recorder == []


def test_orphan_arm_mirror_survives_status_class_save_on_new_incarnation(
    isolated_registry, gcp_sidecar, marker_recorder, monkeypatch
):
    """r2 Minor fix (pod_id-change-tick carry clobber): when NO earlier save
    re-keyed the state to the new incarnation (the wedged-DONE-status MF6
    fall-through skips the wedge arm's _clear_wedge_state laundering save),
    the caller's in-memory prev_state still holds the OLD pod_id when the arm
    alerts. The arm mirrors the NEW pod_id alongside the flag, so a later
    status-class-arm save in the SAME tick (orphan_gcp_noted at the
    pod_id-keyed None carry, prev=the in-memory snapshot) carries the
    just-persisted flag forward — pre-fix it recomputed same_pod=False off
    the stale OLD pod_id and clobbered the flag back to False, costing one
    duplicate alert on the next tick."""
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    now = time.time()
    prev_state = {
        "pod_id": "p_OLD",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": now - GRACE - 600,
        "orphan_gcp_noted": True,
    }
    alerted = asw._maybe_flag_orphan_gcp_handle_pod(
        1417, _healthy_info(pod_id="p_NEW"), now, False, False, prev_state, False
    )
    assert alerted is True
    assert prev_state["pod_id"] == "p_NEW"  # the r2 mirror

    # The status-class arm's later same-tick save (keep-shaped: the flag left
    # at its None carry, prev=the in-memory snapshot the arm just mirrored).
    asw._save_pod_safety_state(
        1417, "p_NEW", missed=0, alerted=False, last_progress_ts=None, prev=prev_state
    )
    state = json.loads((isolated_registry / "pod-safety-1417.json").read_text())
    assert state["orphan_gcp_noted"] is True  # pre-fix: clobbered to False
    assert state["pod_id"] == "p_NEW"


# ---------------------------------------------------------------------------
# 10. Negative controls (parametrized where the predicate leg allows)
# ---------------------------------------------------------------------------


def _run_helper(
    isolated_registry,
    *,
    info: PodInfo,
    now: float,
    first_seen: float,
    keep_running: bool = False,
    followup_active: bool = False,
) -> bool:
    prev_state = {
        "pod_id": info.pod_id,
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": first_seen,
    }
    return asw._maybe_flag_orphan_gcp_handle_pod(
        1417, info, now, keep_running, followup_active, prev_state, False
    )


def test_orphan_arm_negative_control_runpod_sidecar(
    isolated_registry, tmp_path, monkeypatch, marker_recorder, active_task
):
    """A sidecar already re-pointed at backend=runpod never alerts (the
    failover completed; the pod is the legitimate fallback run)."""
    sidecar = tmp_path / "issue-1417-handle.json"
    write_handle_sidecar(
        RunHandle(
            backend="runpod",
            cluster=None,
            job_id="pod-fake",
            pod_name="pod-1417",
            scratch_dir="/workspace",
            log_path="/workspace/logs/issue-1417.log",
            extra={"issue": 1417},
        ),
        sidecar,
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.resolve_handle_sidecar_path",
        lambda issue: (sidecar, False),
    )
    now = time.time()
    assert not _run_helper(
        isolated_registry, info=_healthy_info(), now=now, first_seen=now - GRACE - 600
    )
    assert _orphan_posts(marker_recorder) == []


def test_orphan_arm_negative_control_sidecar_absent(
    isolated_registry, tmp_path, monkeypatch, marker_recorder, active_task
):
    """No resolvable sidecar -> quiet (bias quiet; nothing to attribute)."""
    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.resolve_handle_sidecar_path",
        lambda issue: (tmp_path / "no-such-handle.json", False),
    )
    now = time.time()
    assert not _run_helper(
        isolated_registry, info=_healthy_info(), now=now, first_seen=now - GRACE - 600
    )
    assert _orphan_posts(marker_recorder) == []


def test_orphan_arm_negative_control_sidecar_unreadable(
    isolated_registry, tmp_path, monkeypatch, marker_recorder, active_task, capsys
):
    """An unreadable/garbled sidecar -> quiet, with ONE stderr diagnostic line
    (plan v2 fold-in: the quiet miss is diagnosable)."""
    sidecar = tmp_path / "issue-1417-handle.json"
    sidecar.write_text("{not json")
    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.resolve_handle_sidecar_path",
        lambda issue: (sidecar, False),
    )
    now = time.time()
    assert not _run_helper(
        isolated_registry, info=_healthy_info(), now=now, first_seen=now - GRACE - 600
    )
    assert _orphan_posts(marker_recorder) == []
    assert "sidecar unreadable" in capsys.readouterr().err


def test_orphan_arm_negative_control_suffixed_pod_name(
    isolated_registry, gcp_sidecar, marker_recorder, active_task
):
    """A suffixed follow-up pod (pod-1417-b) is NOT the failover's bare
    canonical name -> never alerts."""
    now = time.time()
    assert not _run_helper(
        isolated_registry,
        info=_healthy_info(name="pod-1417-b"),
        now=now,
        first_seen=now - GRACE - 600,
    )
    assert _orphan_posts(marker_recorder) == []


def test_orphan_arm_negative_control_within_grace(
    isolated_registry, gcp_sidecar, marker_recorder, active_task
):
    """A pod younger than the grace window stays quiet (a healthy failover
    bootstrap can hold a gcp sidecar briefly)."""
    now = time.time()
    assert not _run_helper(
        isolated_registry, info=_healthy_info(), now=now, first_seen=now - GRACE + 60
    )
    assert _orphan_posts(marker_recorder) == []


@pytest.mark.parametrize("flag", ["keep_running", "followup_active"])
def test_orphan_arm_negative_control_passed_shield_flags(
    isolated_registry, gcp_sidecar, marker_recorder, active_task, flag
):
    """The caller-passed keep-running / follow-up shields suppress the alert
    (the documented shields for deliberate parallel pods)."""
    now = time.time()
    assert not _run_helper(
        isolated_registry,
        info=_healthy_info(),
        now=now,
        first_seen=now - GRACE - 600,
        **{flag: True},
    )
    assert _orphan_posts(marker_recorder) == []


@pytest.mark.parametrize("helper", ["_task_keep_running", "_task_followup_active"])
def test_orphan_arm_negative_control_lazy_shield_recheck(
    isolated_registry, gcp_sidecar, marker_recorder, active_task, monkeypatch, helper
):
    """The lazy shield re-check: on an ACTIVE-status task the caller passes
    False flags (the lazy exemptions never ran), so the arm re-checks the tag /
    follow-up signal directly and stays quiet when either is live."""
    now = time.time()
    if helper == "_task_keep_running":
        monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    else:
        monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: True)
    assert not _run_helper(
        isolated_registry, info=_healthy_info(), now=now, first_seen=now - GRACE - 600
    )
    assert _orphan_posts(marker_recorder) == []


def test_orphan_arm_dry_run_alerts_but_persists_nothing(
    isolated_registry, gcp_sidecar, marker_recorder, stop_recorder, active_task
):
    """Dry-run: the alert marker goes through _post_progress_marker (which
    no-ops real posts under dry_run in production) but NO state file write."""
    now = time.time()
    _seed_state(isolated_registry, 1417, pod_id="p1417", first_seen=now - GRACE - 600)
    asw._process_pod(1417, "p1417", _healthy_info(), now, dry_run=True, threshold=2)
    assert len(_orphan_posts(marker_recorder)) == 1
    state = json.loads((isolated_registry / "pod-safety-1417.json").read_text())
    assert "orphan_gcp_noted" not in state  # the seeded file was never rewritten
    assert stop_recorder == []


# ---------------------------------------------------------------------------
# 11. Additive-only: the arm never blocks the existing status-class decision
# ---------------------------------------------------------------------------


def test_orphan_arm_never_blocks_existing_decisions(
    isolated_registry, gcp_sidecar, marker_recorder, stop_recorder, monkeypatch
):
    """With the orphan arm FIRING, the pod still reaches the pre-existing
    status-class branch unchanged: a DONE-status escaped pod past the miss
    threshold is STILL auto-stopped on the same tick (additive-only)."""
    now = time.time()
    monkeypatch.setattr(asw, "_task_status", lambda issue: "completed")
    monkeypatch.setattr(asw, "_task_events", lambda issue: [])
    monkeypatch.setattr(asw, "_latest_progress_ts", lambda events: None)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    # missed=1 -> new_missed=2 == threshold -> the auto-stop fires this tick.
    _seed_state(isolated_registry, 1417, pod_id="p1417", first_seen=now - GRACE - 600, missed=1)

    asw._process_pod(1417, "p1417", _healthy_info(), now, dry_run=False, threshold=2)
    assert len(_orphan_posts(marker_recorder)) == 1  # the arm fired...
    assert stop_recorder == [1417]  # ...and the canonical auto-stop still ran
