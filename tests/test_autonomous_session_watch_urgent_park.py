"""Urgent-park router pass tests (task #1681, plan §5 tests 1-13).

Covers the declared-token grammar parser, the pure routing decision, the
two-tier red-on-main verification (incl. the rc-mapping rows that kill the
``rc != 0 -> confirmed`` shortcut), the day cap (#1241 quiet-at-cap parse
shape), dry-run zero-writes, the dedup belts, the AC4 round-trip
(routed-record -> ``sweep_parked_wf_candidates.sweep()`` suppression on the
SAME fixture tree, task-borne AND cache-borne), the #1680 sweep-fp-verbatim
pin + the ``_WATCHER_NOTE_SENTINELS`` membership pin, the wf_fix true/false
body-file routes, the rule-file grammar<->doc sync pin, and the AC1
main()-wiring source-pin (test 13 — the linear-block call site, without which
the pass is dead on production ticks).

Subprocess boundaries are faked with ``unittest.mock.create_autospec``
(signature-conformant by construction — the one-production-body-test rule);
tasks-root / cache-file / state / sidecar paths use tmp overrides so no test
touches live fleet state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# scripts/ holds autonomous_session_watch.py + sweep_parked_wf_candidates.py
# (the sibling test_autonomous_session_watch.py bootstrap shape).
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import sweep_parked_wf_candidates as spw  # noqa: E402

from explore_persona_space.task_workflow import wf_fix_fingerprint  # noqa: E402

# The verbatim #1643 park note (tasks/completed/1643/events.jsonl, ts
# 2026-07-24T07:08:11Z) — the motivating incident's PROSE park shape: names
# the failing node but carries NO formal block and NO urgency fields, so it
# must stay nightly-routed (AC2 / plan §12 detection-predicate trace).
INCIDENT_1643_NOTE = (
    "parked — running under workflow_fix_target recursion guard (see "
    ".claude/rules/workflow-fix-on-bug.md § Recursion guard). Implementer r1 prose "
    "follow-up: tests/test_shared_vm_thread_caps.py::"
    "test_no_new_torch_before_dotenv_vm_entrypoints is pre-existing-red on origin/main, "
    "naming only scripts/issue1586_*.py (heavy import before load_dotenv, #847 class). "
    "NOTE: target files are experiment scripts — OUT of workflow-fix scope; belongs "
    "to the #1586 line / its session, not a wf-fix filing. Logged for the nightly /daily "
    "parked-candidate sweep; no task filed."
)

BUG = "node X is red on origin/main after the refactor"
PROPOSED = "fix the import order in the example script"
NODE = "tests/test_example.py::test_x"


def _urgent_note(
    *,
    urgency: str | None = "main-red",
    failing_test: str | None = NODE,
    wf_fix: str | None = "false",
    target_file: str = "scripts/example.py",
) -> str:
    """A parked note embedding the formal block + the urgent triple; pass
    None for a field to DROP its line (the test-2 each-field-dropped rows)."""
    lines = [
        f"target_file: {target_file}",
        f"bug_observed: {BUG}",
        f"proposed_change: {PROPOSED}",
        "diff_sketch: |",
        "  + fix",
        "confidence: medium",
        "related_task: #1643",
    ]
    if urgency is not None:
        lines.append(f"urgency: {urgency}")
    if failing_test is not None:
        lines.append(f"failing_test: {failing_test}")
    if wf_fix is not None:
        lines.append(f"wf_fix: {wf_fix}")
    return (
        "parked — running under workflow_fix_target recursion guard.\n"
        "<!-- workflow-fix-candidate v1 -->\n" + "\n".join(lines) + "\n"
        "<!-- /workflow-fix-candidate -->"
    )


def _seed_task_park(tasks_root: Path, issue: int, note: str, ts: str) -> Path:
    """One tasks/<status>/<id>/events.jsonl park row; returns the events path."""
    task_dir = tasks_root / "running" / str(issue)
    task_dir.mkdir(parents=True, exist_ok=True)
    events = task_dir / "events.jsonl"
    row = {"ts": ts, "kind": "epm:workflow-fix-candidate", "version": 1, "note": note}
    with open(events, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return events


def _recent_ts(hours_ago: float = 1.0) -> str:
    return (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture()
def tmp_paths(tmp_path, monkeypatch):
    """tmp tasks-root + state/sidecar path overrides; kill-switch env cleared."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    state = tmp_path / "urgent-wf-park-router.json"
    sidecar = tmp_path / "urgent-wf-park-events.jsonl"
    monkeypatch.delenv("EPM_DISABLE_URGENT_WF_PARK_PASS", raising=False)
    monkeypatch.delenv("EPM_URGENT_WF_PARK_ROUTES_PER_DAY", raising=False)
    monkeypatch.setattr(asw, "_urgent_wf_park_state_path", lambda: state)
    monkeypatch.setattr(asw, "_urgent_wf_park_sidecar_path", lambda: sidecar)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: True)
    return tasks_root, state, sidecar


def _sidecar_rows(sidecar: Path) -> list[dict]:
    if not sidecar.is_file():
        return []
    return [json.loads(line) for line in sidecar.read_text().split("\n") if line.strip()]


# ── 1-3: parse_urgent_fields ────────────────────────────────────────────────


def test_parse_urgent_fields_complete_block():
    fields = asw.parse_urgent_fields(_urgent_note(wf_fix="false"))
    assert fields is not None
    assert fields.target_file == "scripts/example.py"
    assert fields.failing_test == NODE
    assert fields.wf_fix is False
    assert fields.bug_observed == BUG
    assert fields.proposed_change == PROPOSED
    assert fields.confidence == "medium"
    assert fields.related_task == "#1643"
    assert fields.fingerprint == wf_fix_fingerprint(PROPOSED, BUG)
    assert fields.block.startswith("<!-- workflow-fix-candidate v1 -->")
    # wf_fix: true parses to True
    true_fields = asw.parse_urgent_fields(_urgent_note(wf_fix="true"))
    assert true_fields is not None and true_fields.wf_fix is True


def test_parse_missing_any_field_not_routable():
    # Each of the three urgent fields dropped in turn -> None (AC2).
    assert asw.parse_urgent_fields(_urgent_note(urgency=None)) is None
    assert asw.parse_urgent_fields(_urgent_note(failing_test=None)) is None
    assert asw.parse_urgent_fields(_urgent_note(wf_fix=None)) is None
    # Unrecognized urgency value / malformed wf_fix -> None.
    assert asw.parse_urgent_fields(_urgent_note(urgency="soonish")) is None
    assert asw.parse_urgent_fields(_urgent_note(wf_fix="maybe")) is None
    # The verbatim #1643 PROSE park (no formal block) -> None: the historical
    # shape stays nightly-routed by construction.
    assert asw.parse_urgent_fields(INCIDENT_1643_NOTE) is None


def test_parse_rejects_malformed_node_id():
    # Whitespace / flag injection / shell metacharacters / not a node id.
    for bad in (
        "tests/x.py::t -x",
        "; rm -rf /",
        "tests/x.py",
        "tests/x.py::a tests/y.py::b",
        "tests/x.py::t; echo pwned",
        "$(true)::test",
    ):
        assert asw.parse_urgent_fields(_urgent_note(failing_test=bad)) is None, bad
    # Parametrize brackets + a class segment are allowed.
    ok = asw.parse_urgent_fields(
        _urgent_note(failing_test="tests/test_x.py::TestC::test_y[case-1]")
    )
    assert ok is not None


# ── 4-5: decide_urgent_route + day cap ──────────────────────────────────────


def _fields() -> asw.UrgentFields:
    fields = asw.parse_urgent_fields(_urgent_note())
    assert fields is not None
    return fields


def test_decide_route_confirmed_with_budget_routes():
    assert asw.decide_urgent_route(_fields(), "confirmed", 0, 2, {}) == "route"
    # Not-yet-verified with budget -> verify; deferred tier-2 -> defer.
    assert asw.decide_urgent_route(_fields(), None, 0, 2, {}) == "verify"
    assert asw.decide_urgent_route(_fields(), "deferred", 0, 2, {}) == "defer"


def test_refuted_latches_never_files():
    assert asw.decide_urgent_route(_fields(), "refuted", 0, 2, {}) == "refuted"
    for latched in ("routed", "deduped", "refuted", "not-routable", "unverifiable"):
        assert (
            asw.decide_urgent_route(_fields(), None, 0, 2, {"verdict": latched}) == "skip-latched"
        )


def test_indeterminate_two_attempts_then_unverifiable():
    assert asw.decide_urgent_route(_fields(), "indeterminate", 0, 2, {}) == "retry"
    assert (
        asw.decide_urgent_route(_fields(), "indeterminate", 0, 2, {"attempts": 1}) == "unverifiable"
    )


def test_day_cap_quiet_and_malformed_env_falls_back(tmp_paths, monkeypatch):
    # Decision level: at-cap -> cap-exhausted (checked before verification).
    assert asw.decide_urgent_route(_fields(), None, 2, 2, {}) == "cap-exhausted"
    # Env parse (#1241 shape): malformed / <1 falls back to the default 2.
    monkeypatch.setenv("EPM_URGENT_WF_PARK_ROUTES_PER_DAY", "abc")
    assert asw._urgent_wf_park_routes_per_day() == asw.URGENT_WF_PARK_ROUTES_PER_DAY
    monkeypatch.setenv("EPM_URGENT_WF_PARK_ROUTES_PER_DAY", "0")
    assert asw._urgent_wf_park_routes_per_day() == asw.URGENT_WF_PARK_ROUTES_PER_DAY
    monkeypatch.setenv("EPM_URGENT_WF_PARK_ROUTES_PER_DAY", "5")
    assert asw._urgent_wf_park_routes_per_day() == 5
    monkeypatch.delenv("EPM_URGENT_WF_PARK_ROUTES_PER_DAY")
    # Pass level (AC5): the (N+1)-th route is not filed — sidecar row only,
    # NO latch (re-eligible tomorrow), no verification attempt.
    tasks_root, state, sidecar = tmp_paths
    _seed_task_park(tasks_root, 9999, _urgent_note(), _recent_ts())
    import time as _time

    now = _time.time()
    day_key = _time.strftime("%Y-%m-%d", _time.gmtime(now))
    state.write_text(json.dumps({"route_day": day_key, "routes_today": 2}))
    filed = create_autospec(asw._urgent_wf_park_file_and_dispatch)
    monkeypatch.setattr(asw, "_urgent_wf_park_file_and_dispatch", filed)
    verify = create_autospec(asw.verify_main_red)
    monkeypatch.setattr(asw, "verify_main_red", verify)
    asw.urgent_wf_park_pass(False, now=now, tasks_root=tasks_root, cache_file=None)
    filed.assert_not_called()
    verify.assert_not_called()
    rows = _sidecar_rows(sidecar)
    assert any(r.get("action") == "cap-exhausted" for r in rows)
    persisted = json.loads(state.read_text())
    assert not (persisted.get("episodes") or {})  # no latch at cap


# ── 6: kill switch + dry-run contract ───────────────────────────────────────


def test_kill_switch_disables_pass(tmp_paths, monkeypatch):
    tasks_root, state, sidecar = tmp_paths
    _seed_task_park(tasks_root, 9999, _urgent_note(), _recent_ts())
    monkeypatch.setenv("EPM_DISABLE_URGENT_WF_PARK_PASS", "1")
    gate = create_autospec(asw._urgent_park_candidate_gate)
    monkeypatch.setattr(asw, "_urgent_park_candidate_gate", gate)
    asw.urgent_wf_park_pass(False, tasks_root=tasks_root, cache_file=None)
    gate.assert_not_called()
    assert not state.exists() and not sidecar.exists()


def test_dry_run_zero_writes_zero_subprocess(tmp_paths, monkeypatch, capsys):
    tasks_root, state, sidecar = tmp_paths
    _seed_task_park(tasks_root, 9999, _urgent_note(), _recent_ts())

    def _boom(*a, **kw):  # signature-agnostic tripwire: any call is a failure
        raise AssertionError("subprocess.run must not run under --dry-run")

    run_spy = create_autospec(subprocess.run, side_effect=_boom)
    monkeypatch.setattr(asw.subprocess, "run", run_spy)
    asw.urgent_wf_park_pass(True, tasks_root=tasks_root, cache_file=None)
    out = capsys.readouterr().out
    assert f"would verify {NODE} then file+dispatch" in out
    run_spy.assert_not_called()
    assert not state.exists() and not sidecar.exists()  # zero writes


# ── 7: two-tier verification ────────────────────────────────────────────────


def _ledger(refreshed_at: str, *, classname: str = "") -> dict:
    return {
        "refreshed_at": refreshed_at,
        "failing_tests": [
            {"file": "tests/test_example.py", "classname": classname, "name": "test_x"}
        ],
    }


def test_verification_ledger_hit_skips_pytest(tmp_path, monkeypatch):
    import step9c_baseline

    fresh = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(step9c_baseline, "try_load_ledger", lambda root: _ledger(fresh))

    def _boom(*a, **kw):
        raise AssertionError("tier-1 ledger hit must not run pytest")

    monkeypatch.setattr(asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_boom))
    park_ts = datetime.now(tz=UTC) - timedelta(hours=1)
    verdict, detail, used_pytest = asw.verify_main_red(
        NODE, False, park_ts=park_ts, project_root=tmp_path
    )
    assert verdict == "confirmed"
    assert used_pytest is False
    assert "step9c ledger" in detail


def test_verification_ledger_stale_falls_to_pytest(tmp_path, monkeypatch):
    import step9c_baseline

    # refreshed_at BEFORE the park: a matching entry cannot tier-1-confirm —
    # the stale-ledger false-confirm window; tier 2 reads live truth.
    stale = (datetime.now(tz=UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(step9c_baseline, "try_load_ledger", lambda root: _ledger(stale))
    seen: dict = {}

    def _fake_run(argv, **kw):
        if argv[:3] == ["uv", "run", "pytest"]:
            seen["env"] = kw.get("env")
            return subprocess.CompletedProcess(argv, 1, stdout="1 failed", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="abc1234", stderr="")

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    park_ts = datetime.now(tz=UTC) - timedelta(hours=1)
    verdict, detail, used_pytest = asw.verify_main_red(
        NODE, False, park_ts=park_ts, project_root=tmp_path
    )
    assert verdict == "confirmed"
    assert used_pytest is True
    assert "rc=1" in detail
    # #1950/#2030: the Tier-2 probe's rc IS a verdict — its child env carries
    # the stale-bytecode guard token.
    assert seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (0, "refuted"),
        (1, "confirmed"),
        (2, "indeterminate"),  # interrupted/usage error — NOT a confirm
        (5, "indeterminate"),  # no tests collected (node deleted/renamed)
    ],
)
def test_verification_pytest_rc_mapping(tmp_path, monkeypatch, rc, expected):
    import step9c_baseline

    monkeypatch.setattr(step9c_baseline, "try_load_ledger", lambda root: None)

    def _fake_run(argv, **kw):
        if argv[:3] == ["uv", "run", "pytest"]:
            return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="abc1234", stderr="")

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    verdict, _detail, used_pytest = asw.verify_main_red(
        NODE, False, park_ts=datetime.now(tz=UTC), project_root=tmp_path
    )
    assert verdict == expected
    assert used_pytest is True


def test_verification_timeout_is_indeterminate(tmp_path, monkeypatch):
    import step9c_baseline

    monkeypatch.setattr(step9c_baseline, "try_load_ledger", lambda root: None)

    def _timeout(argv, **kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=180)

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_timeout)
    )
    verdict, _detail, used_pytest = asw.verify_main_red(
        NODE, False, park_ts=datetime.now(tz=UTC), project_root=tmp_path
    )
    assert verdict == "indeterminate"
    assert used_pytest is True


def test_verification_deferred_when_pytest_budget_spent(tmp_path, monkeypatch):
    import step9c_baseline

    monkeypatch.setattr(step9c_baseline, "try_load_ledger", lambda root: None)

    def _boom(*a, **kw):
        raise AssertionError("allow_pytest=False must not run pytest")

    monkeypatch.setattr(asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_boom))
    verdict, _detail, used_pytest = asw.verify_main_red(
        NODE, False, park_ts=datetime.now(tz=UTC), project_root=tmp_path, allow_pytest=False
    )
    assert verdict == "deferred"
    assert used_pytest is False


# ── 8: dedup belts ──────────────────────────────────────────────────────────


def test_dedup_open_wf_fix_task_posts_deduped_record(tmp_paths, monkeypatch):
    import explore_persona_space.task_workflow as tw

    tasks_root, state, _sidecar = tmp_paths
    monkeypatch.setattr(tw, "is_open_workflow_fix_task", lambda tf, fp=None: 42)
    hit = asw._urgent_wf_park_dedup(_fields(), "deadbeef1234", tasks_root)
    assert hit == (42, "open-wf-fix-task")
    # Pass level, cache-borne park: the REAL routed-record poster appends the
    # deduped record to the cache stream (no subprocess on the cache leg).
    # The #1853 dedup-target escalation's view probe is the ONLY subprocess
    # here — fake it hermetically (a non-proposed target -> untouched).
    cache = tasks_root.parent / "workflow-fix-events.jsonl"
    note = _urgent_note()
    cache.write_text(
        json.dumps(
            {
                "ts": _recent_ts(),
                "marker": "epm:workflow-fix-candidate",
                "note": note,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        asw, "verify_main_red", lambda *a, **kw: ("confirmed", "test detail", False)
    )

    def _fake_view(argv, **kw):
        assert "view" in argv, f"unexpected subprocess on the cache dedup leg: {argv}"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"status": "completed", "frontmatter": {"kind": "infra"}}),
            stderr="",
        )

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_view)
    )
    asw.urgent_wf_park_pass(False, tasks_root=tasks_root, cache_file=cache)
    recorded = [json.loads(line) for line in cache.read_text().split("\n") if line.strip()]
    filed_rows = [r for r in recorded if r.get("kind") == "epm:workflow-fix-task-filed"]
    assert len(filed_rows) == 1
    assert "n/a (deduped against #42)" in filed_rows[0]["note"]
    persisted = json.loads(state.read_text())
    (episode,) = persisted["episodes"].values()
    assert episode["verdict"] == "deduped" and episode["filed_task"] == 42


def test_dedup_failing_node_containment(tmp_paths, monkeypatch):
    import explore_persona_space.task_workflow as tw

    tasks_root, _state, _sidecar = tmp_paths
    monkeypatch.setattr(tw, "is_open_workflow_fix_task", lambda tf, fp=None: None)
    open_dir = tasks_root / "proposed" / "77"
    open_dir.mkdir(parents=True)
    (open_dir / "body.md").write_text(
        f"---\nkind: infra\ntitle: some open fix\n---\n\n## Goal\n\nfix `{NODE}` red\n"
    )
    hit = asw._urgent_wf_park_dedup(_fields(), "deadbeef1234", tasks_root)
    assert hit == (77, "failing-node-containment")
    # A TERMINAL task carrying the node does NOT dedup (closed fixes never
    # block a genuine re-raise — the workflow-fix-on-bug.md dedup rule).
    import shutil

    shutil.move(str(open_dir.parent), str(tasks_root / "completed"))
    assert asw._urgent_wf_park_dedup(_fields(), "deadbeef1234", tasks_root) is None


# ── 8b: dedup-target escalation into the sweep's urgent lane (#1853 leg a) ───


def _escalation_view_payload(status="proposed", kind="infra", tags=None) -> str:
    """A minimal `task.py view --json` payload (status top-level; kind + tags
    in frontmatter — the cmd_view shape)."""
    return json.dumps({"status": status, "frontmatter": {"kind": kind, "tags": tags or []}})


def test_urgent_wf_park_dedup_escalates_proposed_infra_target(tmp_paths, monkeypatch):
    # A dedup hit whose target is a ripe `proposed` infra task gets
    # `urgent-main-red` added (ONE bounded view probe + ONE idempotent
    # add-tag) + the escalated sidecar row, so the SAME-tick sweep's urgent
    # lane dispatches it.
    _tasks_root, _state, sidecar = tmp_paths
    seen: list[list[str]] = []

    def _fake_run(argv, **kw):
        seen.append(list(argv))
        if "view" in argv:
            assert argv[-3:] == ["view", "42", "--json"]
            return subprocess.CompletedProcess(
                argv, 0, stdout=_escalation_view_payload(), stderr=""
            )
        assert argv[-3:] == ["add-tag", "42", "urgent-main-red"]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    asw._urgent_wf_park_escalate_dedup_target(42, "task:9999", False)
    assert len(seen) == 2  # exactly one view probe + one add-tag
    rows = _sidecar_rows(sidecar)
    assert len(rows) == 1
    assert rows[0]["action"] == "deduped-target-escalated"
    assert rows[0]["key"] == "task:9999" and rows[0]["task"] == 42


@pytest.mark.parametrize(
    ("status", "kind"),
    [("completed", "infra"), ("running", "infra"), ("proposed", "experiment")],
)
def test_urgent_wf_park_dedup_leaves_nonproposed_target(tmp_paths, monkeypatch, status, kind):
    # Any other status/kind: untouched (today's behavior) — no add-tag
    # subprocess, no sidecar row.
    _tasks_root, _state, sidecar = tmp_paths
    seen: list[list[str]] = []

    def _fake_run(argv, **kw):
        seen.append(list(argv))
        assert "view" in argv, f"non-view subprocess on an untouched target: {argv}"
        return subprocess.CompletedProcess(
            argv, 0, stdout=_escalation_view_payload(status=status, kind=kind), stderr=""
        )

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    asw._urgent_wf_park_escalate_dedup_target(42, "task:9999", False)
    assert len(seen) == 1  # the view probe only
    assert _sidecar_rows(sidecar) == []


def test_urgent_wf_park_dedup_holds_needs_human_target(tmp_paths, monkeypatch):
    # #706 beats #1853: a `needs-human`-tagged proposed infra target is NOT
    # tagged (the sweep would never dispatch it; an "escalated" row would
    # misleadingly imply a pending dispatch) — the sidecar records the held
    # state instead.
    _tasks_root, _state, sidecar = tmp_paths
    seen: list[list[str]] = []

    def _fake_run(argv, **kw):
        seen.append(list(argv))
        assert "view" in argv, f"add-tag must not run on a needs-human target: {argv}"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_escalation_view_payload(tags=["needs-human", "daily-held"]),
            stderr="",
        )

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    asw._urgent_wf_park_escalate_dedup_target(42, "task:9999", False)
    assert len(seen) == 1  # the view probe only — no tag write
    rows = _sidecar_rows(sidecar)
    assert len(rows) == 1
    assert rows[0]["action"] == "deduped-target-held-needs-human"
    assert rows[0]["key"] == "task:9999" and rows[0]["task"] == 42


def test_urgent_wf_park_dedup_escalation_dry_run_no_subprocess(tmp_paths, monkeypatch, capsys):
    # dry_run runs NO subprocess at all — including the view probe — and
    # writes nothing.
    _tasks_root, _state, sidecar = tmp_paths

    def _boom(*a, **kw):
        raise AssertionError("subprocess.run must not run under dry_run")

    run_spy = create_autospec(subprocess.run, side_effect=_boom)
    monkeypatch.setattr(asw.subprocess, "run", run_spy)
    asw._urgent_wf_park_escalate_dedup_target(42, "task:9999", True)
    run_spy.assert_not_called()
    assert not sidecar.exists()
    assert "would probe dedup target #42" in capsys.readouterr().out


def test_urgent_wf_park_dedup_route_escalates_before_latch(tmp_paths, monkeypatch):
    # Wiring pin (production-body: the REAL route + REAL escalation helper,
    # fakes only at the subprocess boundary): the dedup branch escalates the
    # target BEFORE the episode latch — the escalated sidecar row precedes
    # the deduped one, and the episode still latches `deduped`.
    import explore_persona_space.task_workflow as tw

    tasks_root, state, sidecar = tmp_paths
    monkeypatch.setattr(tw, "is_open_workflow_fix_task", lambda tf, fp=None: 42)
    cache = tasks_root.parent / "workflow-fix-events.jsonl"
    cache.write_text(
        json.dumps(
            {"ts": _recent_ts(), "marker": "epm:workflow-fix-candidate", "note": _urgent_note()}
        )
        + "\n"
    )
    monkeypatch.setattr(
        asw, "verify_main_red", lambda *a, **kw: ("confirmed", "test detail", False)
    )

    def _fake_run(argv, **kw):
        if "view" in argv:
            return subprocess.CompletedProcess(
                argv, 0, stdout=_escalation_view_payload(), stderr=""
            )
        assert argv[-3:] == ["add-tag", "42", "urgent-main-red"]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    asw.urgent_wf_park_pass(False, tasks_root=tasks_root, cache_file=cache)
    actions = [r.get("action") for r in _sidecar_rows(sidecar)]
    assert "deduped-target-escalated" in actions and "deduped" in actions
    assert actions.index("deduped-target-escalated") < actions.index("deduped")
    persisted = json.loads(state.read_text())
    (episode,) = persisted["episodes"].values()
    assert episode["verdict"] == "deduped" and episode["filed_task"] == 42


# ── 9: the AC4 round-trip (routed-record suppresses in the sweep) ───────────


def test_roundtrip_routed_record_suppresses_in_sweep(tmp_paths, monkeypatch):
    tasks_root, state, _sidecar = tmp_paths
    note = _urgent_note()
    events = _seed_task_park(tasks_root, 9999, note, _recent_ts())
    monkeypatch.setattr(
        asw, "verify_main_red", lambda *a, **kw: ("confirmed", "test detail", False)
    )
    monkeypatch.setattr(asw, "_urgent_wf_park_dedup", lambda *a, **kw: None)
    monkeypatch.setattr(
        asw,
        "_urgent_wf_park_file_and_dispatch",
        lambda *a, **kw: (123, True, "filed + dispatched #123"),
    )
    posted: dict = {}

    def _fake_run(argv, **kw):
        # The task-borne routed-record post (task.py post-marker) — the fake
        # appends the row the real subprocess would have committed.
        assert "post-marker" in argv and "epm:workflow-fix-task-filed" in argv
        issue = argv[argv.index("post-marker") + 1]
        record_note = argv[argv.index("--note") + 1]
        posted["issue"], posted["note"] = issue, record_note
        row = {
            "ts": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": "epm:workflow-fix-task-filed",
            "version": 1,
            "note": record_note,
        }
        with open(events, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    asw.urgent_wf_park_pass(False, tasks_root=tasks_root, cache_file=None)
    assert posted["issue"] == "9999"
    # The sweep-reported fp rides the record VERBATIM (#1680 lesson).
    expected_fp = wf_fix_fingerprint(PROPOSED, BUG)
    assert f"fingerprint: {expected_fp}" in posted["note"]
    # Round-trip: the record closes the park for the nightly sweep.
    result = spw.sweep(tasks_root, None, include_routed=True)
    (cand,) = result["candidates"]
    assert cand["suppressed"] is True
    assert cand["suppressed_by"]["kind"] == "same-stream-filed"
    persisted = json.loads(state.read_text())
    (episode,) = persisted["episodes"].values()
    assert episode["verdict"] == "routed" and episode["filed_task"] == 123
    assert persisted["routes_today"] == 1  # the day slot was consumed


def test_roundtrip_cache_borne_leg(tmp_paths, monkeypatch):
    tasks_root, _state, _sidecar = tmp_paths
    cache = tasks_root.parent / "workflow-fix-events.jsonl"
    cache.write_text(
        json.dumps(
            {
                "ts": _recent_ts(),
                "marker": "epm:workflow-fix-candidate",
                "note": _urgent_note(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        asw, "verify_main_red", lambda *a, **kw: ("confirmed", "test detail", False)
    )
    monkeypatch.setattr(asw, "_urgent_wf_park_dedup", lambda *a, **kw: None)
    monkeypatch.setattr(
        asw,
        "_urgent_wf_park_file_and_dispatch",
        lambda *a, **kw: (124, False, "filed #124"),
    )
    asw.urgent_wf_park_pass(False, tasks_root=tasks_root, cache_file=cache)
    result = spw.sweep(tasks_root, cache, include_routed=True)
    (cand,) = result["candidates"]
    assert cand["suppressed"] is True
    assert cand["suppressed_by"]["kind"] == "same-stream-filed"


# ── 10: routed-record fields + sentinel pins ────────────────────────────────


def test_routed_record_carries_sweep_fp_verbatim_and_origin_ts():
    fields = _fields()
    # The fp passed in (the SWEEP-reported value) rides verbatim — even when
    # it differs from a recomputation (the #1680 abridged-text incident).
    note = asw._urgent_wf_park_routed_note(
        fields, "deadbeef1234", "#55", True, "2026-07-25T00:00:00Z", "origin text here"
    )
    assert "fingerprint: deadbeef1234" in note
    assert fields.fingerprint not in note  # never the recomputed value
    assert "origin_candidate_ts: 2026-07-25T00:00:00Z" in note
    assert f"target_file: {fields.target_file}" in note
    assert "source: watcher-urgent-park-router" in note
    assert "filed_task: #55" in note
    assert asw._URGENT_WF_PARK_NOTE_SENTINEL in note
    # A miss here silently resets orphan/stalled staleness clocks.
    assert asw._URGENT_WF_PARK_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS


# ── 11: body-file composition (wf_fix true/false routes) ────────────────────


def test_wf_fix_true_body_provenance_and_tags(tmp_path, monkeypatch):
    fields = asw.parse_urgent_fields(_urgent_note(wf_fix="true"))
    assert fields is not None
    title, body, tags = asw._urgent_wf_park_compose_body(fields, "deadbeef1234", "vline")
    assert title.startswith("workflow-fix: ")
    assert tags == ["wf-fix", "wf-fix-fp:deadbeef1234", "urgent-main-red"]
    assert "- workflow_fix_target: scripts/example.py" in body
    assert "- fingerprint: deadbeef1234" in body
    assert "verified-at-filing: vline" in body
    assert "<!-- workflow-fix-candidate v1 -->" in body  # verbatim block appended
    # Body coverage of the REAL file+dispatch path (autospec'd subprocess
    # boundary): argv carries the wrapper flags; the body file holds the body.
    captured: dict = {}

    def _fake_run(argv, **kw):
        captured["argv"] = argv
        captured["body"] = Path(argv[argv.index("--body-file") + 1]).read_text()
        return subprocess.CompletedProcess(
            argv, 0, stdout="filed + dispatched #123: spawned", stderr=""
        )

    monkeypatch.setattr(
        asw.subprocess, "run", create_autospec(subprocess.run, side_effect=_fake_run)
    )
    filed_id, spawned, _detail = asw._urgent_wf_park_file_and_dispatch(
        title, body, tags, fields.block, False, project_root=tmp_path
    )
    assert (filed_id, spawned) == (123, True)
    argv = captured["argv"]
    assert argv[3] == "scripts/file_infra_task.py"
    assert argv[argv.index("--title") + 1] == title
    assert captured["body"] == body
    tag_values = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--tag"]
    assert tag_values == tags


def test_wf_fix_false_body_fp_line_no_target_line():
    fields = asw.parse_urgent_fields(_urgent_note(wf_fix="false"))
    assert fields is not None
    title, body, tags = asw._urgent_wf_park_compose_body(fields, "deadbeef1234", "vline")
    # The daily-fix: prefix keeps the filing visible to the title-prefix-gated
    # is_open_workflow_fix_task predicate (consistency-round fix; no
    # WF_FIX_TITLE_PREFIXES widening).
    assert title.startswith("daily-fix: fix red main: test_example.py::test_x")
    assert tags == ["urgent-main-red"]
    assert "workflow_fix_target:" not in body  # the child is NOT guard-bound
    assert "- fingerprint: deadbeef1234" in body  # suppression rule 2's belt


# ── 12: grammar <-> doc sync pin ────────────────────────────────────────────


def test_rule_file_documents_urgent_token_grammar():
    rule = (
        Path(__file__).resolve().parent.parent / ".claude" / "rules" / "workflow-fix-on-bug.md"
    ).read_text()
    assert "urgency: main-red" in rule
    assert "failing_test:" in rule
    assert "wf_fix:" in rule
    assert "EPM_DISABLE_URGENT_WF_PARK_PASS" in rule
    assert "urgent_wf_park_pass" in rule


# ── 13: the AC1 main()-wiring source-pin ────────────────────────────────────


def test_main_wiring_linear_block_invokes_pass():
    """Without the LINEAR-block call the pass is dead on production ticks
    while every other test, the docstring-count lint, and the --only smoke
    all stay green (Statistics Must-Fix 1)."""
    import inspect

    src = inspect.getsource(asw.main)
    assert src.count("urgent_wf_park_pass(args.dry_run)") >= 2  # ladder + linear
    assert "if args.urgent_wf_park_only:" in src  # the --only ladder entry
    # The linear-block call sits in the daemon-independent sequence between
    # completed_unmerged_pass and vm_ledger_reap_pass (the plan's wiring).
    tail = src[src.rindex("completed_unmerged_pass(args.dry_run)") :]
    segment = tail[: tail.index("vm_ledger_reap_pass(args.dry_run)")]
    assert "urgent_wf_park_pass(args.dry_run)" in segment
