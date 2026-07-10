"""Tests for the CPU/memory-pressure guard pass (task #849).

The watcher had disk guards but no CPU/memory-pressure detection or
attribution channel (2026-07-02 incident: load 186-226 for hours; earlyoom
SIGTERM sweeps silently killed analysis workers with exit 143 and no
traceback). The CPU guard pass is escalate-only: sidecar rows + deduped
pushes, NEVER a kill / renice / signal.

Covers (plan #849 § Test plan, cases 1-15):
  * pure predicates: parse_psi_avg10, parse_meminfo_avail_pct,
    cpu_tick_hot_reasons, decide_cpu_guard_fire (streak + urgent single-tick
    + dedup/re-alert), parse_earlyoom_kill_line, attribute_issue,
    attribute_kill (pre-kill snapshot matching),
  * cpu_guard_pass end-to-end with monkeypatched readers: pressure fire +
    row shape, recovery -> re-arm second episode (7b), earlyoom rows +
    cursor dedup, mem-avail single-tick fire + snapshot, seeded-snapshot
    kill attribution, dry-run zero-write/zero-subprocess, kill switch,
    garbled state + fail-soft arms + journal lookback bounds,
  * the never-kills grep over the new source block,
  * round-2 review fixes (cases 16-19): wrong-TYPE valid-JSON state fields
    (recent_kill_keys container, alerted bool) degrade + self-repair, a
    snapshot match without an int issue is honestly unattributed on BOTH
    match paths, newest-keys dedup truncation, 15-char kernel-comm matching.

Mirrors tests/test_vm_disk_subfloor_sentinel.py's bootstrap + watcher_roots
fixture (monkeypatch asw.PROJECT_ROOT + asw.AUTONOMOUS_REGISTRY_DIR).
"""

import json
import sys
import time
from pathlib import Path

import pytest

# Bootstrap sys.path the same way the sibling watcher tests do (scripts/ on
# the path so autonomous_session_watch imports by name).
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402

# ─── fixtures ────────────────────────────────────────────────────────────────

# Real captured earlyoom kill line (2026-06-28, `journalctl -o short-iso`).
_KILL_LINE = (
    "2026-06-28T21:44:03-0700 eps-vm earlyoom[2703914]: sending SIGTERM to "
    'process 4087688 uid 1001 "pytest": badness 984, VmRSS 3390 MiB'
)

_PSI_CPU_BODY = "some avg10=76.08 avg60=61.25 avg300=52.44 total=123456789\n"
_PSI_MEM_BODY = (
    "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    "full avg10=12.50 avg60=3.10 avg300=1.00 total=99\n"
)
_MEMINFO_BODY = (
    "MemTotal:       131898036 kB\n"
    "MemFree:         2340000 kB\n"
    "MemAvailable:   26379607 kB\n"
    "Buffers:          123456 kB\n"
)


@pytest.fixture
def watcher_roots(tmp_path, monkeypatch):
    """Pin PROJECT_ROOT (sidecar) and AUTONOMOUS_REGISTRY_DIR (state) at a
    temp dir so the pass is fully offline."""
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    return tmp_path


def _read_sidecar(root: Path) -> list[dict]:
    path = root / ".claude" / "cache" / "cpu-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _pressure_rows(root: Path) -> list[dict]:
    return [r for r in _read_sidecar(root) if r["kind"] == "vm-cpu-pressure"]


def _kill_rows(root: Path) -> list[dict]:
    return [r for r in _read_sidecar(root) if r["kind"] == "earlyoom-kill"]


def _read_state(root: Path) -> dict:
    path = root / "reg" / "vm-cpu-guard.json"
    return json.loads(path.read_text()) if path.is_file() else {}


class _Signals:
    """Mutable holder driving the monkeypatched readers across invocations."""

    def __init__(self):
        self.load: tuple[float, float] | None = (0.5, 0.5)
        self.psi_cpu: float | None = 0.0
        self.psi_mem: float | None = 0.0
        self.mem_avail: float | None = 80.0
        self.kills: list[dict] | None = []
        self.since_calls: list[float] = []
        self.pushes: list[str] = []
        self.top = [
            {
                "pid": 4087688,
                "pcpu": 250.0,
                "rss_mib": 3390.0,
                "argv": "pytest tests/test_x.py",
                "issue": 731,
            }
        ]


@pytest.fixture
def guard_env(watcher_roots, monkeypatch):
    """Monkeypatch every reader / side-effect helper the pass touches so no
    real /proc, journalctl, ps, or push script is involved."""
    sig = _Signals()
    monkeypatch.setattr(asw, "_read_loadavg", lambda: sig.load)
    monkeypatch.setattr(
        asw,
        "_read_psi_avg10",
        lambda path, kind: sig.psi_cpu if "cpu" in path else sig.psi_mem,
    )
    monkeypatch.setattr(asw, "_read_mem_avail_pct", lambda: sig.mem_avail)

    def _kills_since(since_epoch, *, dry_run=False):
        sig.since_calls.append(since_epoch)
        if dry_run:
            return None
        return sig.kills

    monkeypatch.setattr(asw, "_earlyoom_kills_since", _kills_since)
    monkeypatch.setattr(
        asw,
        "_top_processes",
        lambda top_n=asw.CPU_GUARD_TOP_N, *, dry_run=False: [] if dry_run else sig.top,
    )

    def _push(msg, dry_run):
        sig.pushes.append(msg)
        return True

    monkeypatch.setattr(asw, "_telegram_push", _push)
    monkeypatch.setattr(asw.os, "cpu_count", lambda: 32)
    return sig


# ─── 1. parse_psi_avg10 ──────────────────────────────────────────────────────


def test_parse_psi_avg10_real_fixture():
    assert asw.parse_psi_avg10(_PSI_CPU_BODY, "some") == pytest.approx(76.08)
    assert asw.parse_psi_avg10(_PSI_MEM_BODY, "full") == pytest.approx(12.5)
    assert asw.parse_psi_avg10(_PSI_MEM_BODY, "some") == pytest.approx(0.0)
    assert asw.parse_psi_avg10("garbled nonsense\n", "some") is None
    assert asw.parse_psi_avg10("some avg10=bogus avg60=1.0\n", "some") is None
    assert asw.parse_psi_avg10("", "full") is None


# ─── 2. cpu_tick_hot_reasons ─────────────────────────────────────────────────


def test_cpu_tick_hot_reasons_each_signal_and_failsoft():
    f = asw.cpu_tick_hot_reasons
    assert f(200.0, 32, 0.0, 0.0, 80.0) == ["loadavg"]
    assert f(1.0, 32, 60.0, 0.0, 80.0) == ["psi-cpu"]
    assert f(1.0, 32, 0.0, 20.0, 80.0) == ["psi-memory"]
    assert f(1.0, 32, 0.0, 0.0, 10.0) == ["mem-avail"]
    assert f(None, 32, None, None, None) == []
    # A missing PSI signal (None) does not block a loadavg fire (fail-soft).
    assert f(200.0, 32, None, None, None) == ["loadavg"]
    # Healthy full utilization (load ~= nproc) never fires.
    assert f(32.0, 32, 0.0, 0.0, 80.0) == []


# ─── 3. decide_cpu_guard_fire: streak + urgent ───────────────────────────────


def test_decide_cpu_guard_fire_streak_and_urgent():
    d = asw.decide_cpu_guard_fire
    # Streaked reason: no fire until the streak reaches CPU_GUARD_TICKS.
    assert d(["loadavg"], 0, False, None, None, 200.0) == (False, 1)
    assert d(["loadavg"], asw.CPU_GUARD_TICKS - 1, False, None, None, 200.0) == (
        True,
        asw.CPU_GUARD_TICKS,
    )
    # Not hot -> reset.
    assert d([], 5, True, 200.0, ["loadavg"], 1.0) == (False, 0)
    # Urgent reason (mem-avail) fires on tick 1 — no streak requirement.
    assert d(["mem-avail"], 0, False, None, None, 1.0) == (True, 1)


# ─── 4. decide_cpu_guard_fire: dedup + re-alert ──────────────────────────────


def test_decide_cpu_guard_fire_dedup_and_realert():
    d = asw.decide_cpu_guard_fire
    # Alerted + stable load -> suppressed (streak keeps counting).
    assert d(["loadavg"], 2, True, 200.0, ["loadavg"], 205.0) == (False, 3)
    # >25% load5 growth -> re-alert.
    assert d(["loadavg"], 2, True, 200.0, ["loadavg"], 260.0) == (True, 3)
    # Reason-set change (cpu -> cpu+memory) -> re-alert.
    assert d(["loadavg", "psi-memory"], 2, True, 200.0, ["loadavg"], 200.0) == (True, 3)


# ─── 5. parse_earlyoom_kill_line ─────────────────────────────────────────────


def test_parse_earlyoom_kill_line():
    k = asw.parse_earlyoom_kill_line(_KILL_LINE)
    assert k == {
        "journal_ts": "2026-06-28T21:44:03-0700",
        "signal": "SIGTERM",
        "pid": 4087688,
        "uid": 1001,
        "comm": "pytest",
        "badness": 984,
        "vmrss_mib": 3390,
    }
    # Non-kill journal chatter parses to None.
    assert (
        asw.parse_earlyoom_kill_line(
            "2026-07-02T06:31:41-0700 eps-vm earlyoom[123]: mem avail: 10000 of 128000 MiB (7.81%)"
        )
        is None
    )
    assert (
        asw.parse_earlyoom_kill_line(
            "2026-07-02T06:31:41-0700 eps-vm earlyoom[123]: "
            "sending SIGTERM when mem <= 10.00% and swap <= 10.00%"
        )
        is None
    )


# ─── 6. attribute_issue ──────────────────────────────────────────────────────


def test_attribute_issue():
    # Worktree cwd wins (even over a conflicting cmdline hint).
    assert asw.attribute_issue("/home/u/eps/.claude/worktrees/issue-849/src", "--issue 111") == 849
    assert asw.attribute_issue(None, "uv run python scripts/dispatch_issue.py --issue 731") == 731
    assert asw.attribute_issue(None, "python eval.py issue_845.json") == 845
    assert asw.attribute_issue("/home/u", "sshd: something") is None


# ─── 7. pass-level: pressure fire + row shape + in-episode dedup ─────────────


def test_cpu_guard_pass_fires_and_writes_row(guard_env, watcher_roots):
    sig = guard_env
    sig.load = (210.0, 200.0)  # hot: 200 > 1.5 * 32
    # Tick 1: streak building, no row yet.
    assert asw.cpu_guard_pass(dry_run=False) is False
    assert _pressure_rows(watcher_roots) == []
    # Tick 2: fires.
    assert asw.cpu_guard_pass(dry_run=False) is True
    rows = _pressure_rows(watcher_roots)
    assert len(rows) == 1
    row = rows[0]
    assert row["reasons"] == ["loadavg"]
    assert row["load5"] == 200.0
    assert row["nproc"] == 32
    assert row["top_processes"][0]["issue"] == 731
    st = _read_state(watcher_roots)
    assert st["alerted"] is True
    assert st["last_top_snapshot"]["procs"][0]["pid"] == 4087688
    assert len(sig.pushes) == 1
    # Tick 3 at stable load: suppressed within the episode.
    assert asw.cpu_guard_pass(dry_run=False) is False
    assert len(_pressure_rows(watcher_roots)) == 1
    assert len(sig.pushes) == 1


# ─── 7b. pass-level: recovery -> re-arm -> SECOND episode row ────────────────


def test_cpu_guard_pass_recovery_rearms_second_episode(guard_env, watcher_roots):
    sig = guard_env
    sig.load = (210.0, 200.0)
    asw.cpu_guard_pass(dry_run=False)  # hot tick 1
    assert asw.cpu_guard_pass(dry_run=False) is True  # hot tick 2 -> FIRE
    # Cold tick -> episode reset (a dropped reset branch fails here).
    sig.load = (0.5, 0.5)
    assert asw.cpu_guard_pass(dry_run=False) is False
    st = _read_state(watcher_roots)
    assert st["alerted"] is False
    assert st["consecutive_hot"] == 0
    # Re-overload -> a SECOND vm-cpu-pressure row (fresh episode).
    sig.load = (210.0, 200.0)
    asw.cpu_guard_pass(dry_run=False)  # hot tick 1
    assert asw.cpu_guard_pass(dry_run=False) is True  # hot tick 2 -> second FIRE
    assert len(_pressure_rows(watcher_roots)) == 2


# ─── 8. pass-level: earlyoom rows + cursor/key dedup ─────────────────────────


def test_cpu_guard_pass_earlyoom_rows_and_cursor_dedup(guard_env, watcher_roots):
    sig = guard_env  # pressure signals all cold by default
    k1 = asw.parse_earlyoom_kill_line(_KILL_LINE)
    k2 = dict(k1, pid=999, journal_ts="2026-06-28T21:44:05-0700", comm="python3")
    sig.kills = [k1, k2]
    assert asw.cpu_guard_pass(dry_run=False) is True
    rows = _kill_rows(watcher_roots)
    assert len(rows) == 2
    # No pre-kill snapshot in state -> explicit unattributed.
    assert all(r["attribution_status"] == "unattributed" for r in rows)
    assert all(r["issue"] is None for r in rows)
    assert len(sig.pushes) == 1  # kill push, rate-limited to 1/h
    # Second pass over the same window: zero duplicate rows, no new push.
    assert asw.cpu_guard_pass(dry_run=False) is False
    assert len(_kill_rows(watcher_roots)) == 2
    assert len(sig.pushes) == 1
    # Kill rows appeared with pressure thresholds all cold.
    assert _pressure_rows(watcher_roots) == []
    # Journal cursor advanced on the successful scans.
    assert isinstance(_read_state(watcher_roots)["last_journal_epoch"], float)


# ─── 9. dry-run writes nothing, zero subprocess ──────────────────────────────


def test_cpu_guard_pass_dry_run_writes_nothing(guard_env, watcher_roots, monkeypatch):
    sig = guard_env
    sig.load = (210.0, 200.0)
    sig.mem_avail = 5.0  # urgent -> the decision path would fire

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run called during dry-run")

    monkeypatch.setattr(asw.subprocess, "run", _boom)
    asw.cpu_guard_pass(dry_run=True)
    assert _read_sidecar(watcher_roots) == []
    assert not (watcher_roots / "reg" / "vm-cpu-guard.json").exists()


def test_real_helpers_zero_subprocess_in_dry_run(monkeypatch):
    """The REAL (un-monkeypatched) subprocess helpers short-circuit under
    dry_run with zero subprocess.run calls (#681 r3 convention)."""

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run called during dry-run")

    monkeypatch.setattr(asw.subprocess, "run", _boom)
    assert asw._top_processes(dry_run=True) == []
    assert asw._earlyoom_kills_since(0.0, dry_run=True) is None


# ─── 10. kill switch ─────────────────────────────────────────────────────────


def test_cpu_guard_kill_switch(guard_env, watcher_roots, monkeypatch, capsys):
    monkeypatch.setenv("EPM_DISABLE_CPU_GUARD_PASS", "1")
    sig = guard_env
    sig.load = (210.0, 200.0)
    sig.mem_avail = 5.0
    assert asw.cpu_guard_pass(dry_run=False) is False
    assert _read_sidecar(watcher_roots) == []
    assert not (watcher_roots / "reg" / "vm-cpu-guard.json").exists()
    assert "disabled" in capsys.readouterr().out


# ─── 11. warn-only: no process mutation in the new block ─────────────────────


def test_cpu_guard_never_kills():
    """Grep the CPU-guard source block (header sentinel -> the dedicated
    END-OF-CPU-GUARD-BLOCK sentinel after cpu_guard_pass) for process-mutation
    tokens: zero hits.

    The tokens are CODE-shaped (call syntax / quoted argv strings) so the
    block's own "never kills / renices" prose cannot trip the check. #1155:
    the old end anchor (`src.index("def _status_class")`) matched a backticked
    prose mention inside the block's own header comment, 11 lines after the
    header — the scanned span was ~10 comment lines, none of the pass. Both
    anchors are asserted unique and the span is self-checked to cover the
    real implementation, so a re-anchoring regression fails loudly.
    """
    src = (_SCRIPTS / "autonomous_session_watch.py").read_text()
    header = "CPU/memory-pressure guard pass (task #849)"
    end_sentinel = "END-OF-CPU-GUARD-BLOCK (task #849)"
    assert src.count(header) == 1, "header sentinel must be unique in source"
    assert src.count(end_sentinel) == 1, "end sentinel must be unique in source"
    start = src.index(header)
    end = src.index(end_sentinel)
    assert start < end, "block sentinels out of order"
    block = src[start:end]
    # Span self-check: the scanned block must cover the pass implementation —
    # the pass entrypoint plus the two subprocess-running helpers most able
    # to mutate processes.
    for span_probe in (
        "def cpu_guard_pass(",
        "def _earlyoom_kills_since(",
        "def _top_processes(",
    ):
        assert span_probe in block, f"never-kills span no longer covers {span_probe!r}"
    banned = (
        "os.kill(",
        '"renice"',
        "'renice'",
        ".send_signal(",
        ".terminate(",
        "killpg(",
        'os.kill"',
    )
    for token in banned:
        assert token not in block, f"process-mutation token {token!r} in CPU-guard block"


# ─── 12. parse_meminfo_avail_pct ─────────────────────────────────────────────


def test_parse_meminfo_avail_pct():
    pct = asw.parse_meminfo_avail_pct(_MEMINFO_BODY)
    assert pct == pytest.approx(100.0 * 26379607 / 131898036)
    assert asw.parse_meminfo_avail_pct("MemTotal: 1000 kB\n") is None  # no MemAvailable
    assert asw.parse_meminfo_avail_pct("MemTotal: abc kB\nMemAvailable: 10 kB\n") is None
    assert asw.parse_meminfo_avail_pct("MemTotal: 0 kB\nMemAvailable: 0 kB\n") is None
    assert asw.parse_meminfo_avail_pct("") is None


# ─── 13. attribute_kill (pure) + pass-level seeded-snapshot attribution ──────


def test_attribute_kill_from_snapshot():
    snap = {
        "ts": 1.0,
        "procs": [
            {
                "pid": 4087688,
                "pcpu": 250.0,
                "rss_mib": 3390.0,
                "argv": "pytest tests/x.py",
                "issue": 731,
            },
            {
                "pid": 500,
                "pcpu": 10.0,
                "rss_mib": 100.0,
                "argv": "/usr/bin/python3 foo.py",
                "issue": None,
            },
        ],
    }
    kill = {"pid": 4087688, "comm": "pytest"}
    assert asw.attribute_kill(kill, snap) == (731, "attributed")
    # Unique comm match (pid rolled, comm unique among snapshot argv basenames).
    assert asw.attribute_kill({"pid": 1, "comm": "pytest"}, snap) == (731, "attributed")
    # No match / no snapshot -> explicit unattributed.
    assert asw.attribute_kill({"pid": 2, "comm": "sshd"}, snap) == (None, "unattributed")
    assert asw.attribute_kill(kill, None) == (None, "unattributed")


def test_attribute_kill_bool_issue_is_unattributed():
    """Python ``bool`` subclasses ``int``: a corrupt / hand-edited snapshot
    row ``{"issue": true}`` must NOT attribute (r2-reconciler concern
    ``bool-issue-attributed`` — ``isinstance(issue, int)`` returned
    ``(True, "attributed")``)."""
    # (a) Pid-match path with a bool issue -> honest unattributed.
    snap_pid = {
        "ts": 1.0,
        "procs": [
            {"pid": 4087688, "argv": "pytest tests/x.py", "issue": True},
        ],
    }
    kill = {"pid": 4087688, "comm": "pytest"}
    assert asw.attribute_kill(kill, snap_pid) == (None, "unattributed")
    # (b) Unique-comm path with a bool issue -> honest unattributed.
    snap_comm = {
        "ts": 1.0,
        "procs": [
            {"pid": 7, "argv": "pytest tests/y.py", "issue": False},
        ],
    }
    assert asw.attribute_kill({"pid": 1, "comm": "pytest"}, snap_comm) == (None, "unattributed")
    # (c) Sanity: a real int issue still attributes on both paths.
    snap_int = {
        "ts": 1.0,
        "procs": [
            {"pid": 4087688, "argv": "pytest tests/z.py", "issue": 849},
        ],
    }
    assert asw.attribute_kill(kill, snap_int) == (849, "attributed")
    assert asw.attribute_kill({"pid": 1, "comm": "pytest"}, snap_int) == (849, "attributed")


def test_kill_row_attributed_from_seeded_snapshot(guard_env, watcher_roots):
    sig = guard_env
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "vm-cpu-guard.json").write_text(
        json.dumps(
            {
                "last_top_snapshot": {
                    "ts": 1.0,
                    "procs": [
                        {
                            "pid": 4087688,
                            "pcpu": 250.0,
                            "rss_mib": 3390.0,
                            "argv": "pytest tests/x.py",
                            "issue": 731,
                        }
                    ],
                }
            }
        )
    )
    sig.kills = [asw.parse_earlyoom_kill_line(_KILL_LINE)]
    asw.cpu_guard_pass(dry_run=False)
    rows = _kill_rows(watcher_roots)
    assert len(rows) == 1
    assert rows[0]["issue"] == 731
    assert rows[0]["attribution_status"] == "attributed"
    assert rows[0]["attribution_source"] == "pre-kill-snapshot"


# ─── 14. mem-avail single-tick fire + snapshot ───────────────────────────────


def test_mem_avail_single_tick_fire_stores_snapshot(guard_env, watcher_roots):
    sig = guard_env
    sig.mem_avail = 12.0  # below the 20% floor; every other signal cold
    # Fresh state: fires on the FIRST invocation (no 2-tick streak).
    assert asw.cpu_guard_pass(dry_run=False) is True
    rows = _pressure_rows(watcher_roots)
    assert len(rows) == 1
    assert rows[0]["reasons"] == ["mem-avail"]
    st = _read_state(watcher_roots)
    assert st["last_top_snapshot"]["procs"][0]["pid"] == 4087688


# ─── 15. garbled state / fail-soft arms / journal lookback bounds ────────────


def test_garbled_state_and_failsoft_arms(guard_env, watcher_roots, capsys):
    sig = guard_env
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    state_file = reg / "vm-cpu-guard.json"
    state_file.write_text("{not json!!")
    # (a) Garbled state -> pass runs from {} without raising.
    t0 = time.time()
    assert asw.cpu_guard_pass(dry_run=False) is False
    t1 = time.time()
    # (c) First-run journal lookback deliberately bounded: ~30 min + overlap.
    assert len(sig.since_calls) == 1
    lookback = 1800 + asw.CPU_GUARD_JOURNAL_OVERLAP_S
    assert t0 - lookback - 1 <= sig.since_calls[0] <= t1 - lookback + 1
    # (b) Kill arm None -> visible degradation, cursor NOT advanced.
    prev_epoch = _read_state(watcher_roots)["last_journal_epoch"]
    sig.kills = None
    asw.cpu_guard_pass(dry_run=False)
    assert "kill arm unavailable" in capsys.readouterr().err
    assert _read_state(watcher_roots)["last_journal_epoch"] == pytest.approx(prev_epoch)
    # (d) A pressure row fired while the kill arm is degraded carries the
    # explicit kill_arm=unavailable field (no silent protection illusion).
    sig.mem_avail = 5.0  # urgent single-tick fire
    asw.cpu_guard_pass(dry_run=False)
    rows = _pressure_rows(watcher_roots)
    assert rows and rows[-1]["kill_arm"] == "unavailable"
    # (e) Post-outage re-scan capped at 24 h.
    sig.mem_avail = 80.0
    sig.kills = []
    cur = _read_state(watcher_roots)
    cur["last_journal_epoch"] = time.time() - 200000  # ~2.3 days ago
    state_file.write_text(json.dumps(cur))
    sig.since_calls.clear()
    t0 = time.time()
    asw.cpu_guard_pass(dry_run=False)
    t1 = time.time()
    assert sig.since_calls[-1] >= t0 - asw.CPU_GUARD_JOURNAL_MAX_LOOKBACK_S - 1
    assert sig.since_calls[-1] <= t1 - asw.CPU_GUARD_JOURNAL_MAX_LOOKBACK_S + 1


# ─── 16. wrong-TYPE (valid-JSON) state fields — degrade + self-repair ────────


def test_state_wrong_type_recent_kill_keys_never_crashes(guard_env, watcher_roots):
    """A valid-JSON state file with `recent_kill_keys: 5` (truthy non-
    iterable) must NOT raise: the pass is called unwrapped in main()'s
    daemon-independent block, so a TypeError here would abort the ENTIRE
    watcher tick (pod-safety, crash-recovery, GC) every 10 min — AC4.
    Pre-fix this crashed with `TypeError: 'int' object is not iterable`;
    the pass now degrades to "no keys" and self-repairs the field."""
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "vm-cpu-guard.json").write_text(json.dumps({"recent_kill_keys": 5}))
    assert asw.cpu_guard_pass(dry_run=False) is False  # must not raise
    repaired = _read_state(watcher_roots)
    assert isinstance(repaired["recent_kill_keys"], list)


def test_alerted_wrong_type_does_not_suppress_episode(guard_env, watcher_roots):
    """A wrong-type truthy `alerted` (e.g. "yes") must NOT suppress a real
    episode: unguarded, decide_cpu_guard_fire sees alerted=True with no
    valid last_alert_reasons/load5 and never emits the first pressure row
    until a cold recovery tick. The isinstance(bool) guard reads it as
    False, so the (second) hot tick fires."""
    sig = guard_env
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "vm-cpu-guard.json").write_text(json.dumps({"alerted": "yes", "consecutive_hot": 1}))
    sig.load = (210.0, 200.0)  # hot: 200 > 1.5 * 32
    # consecutive_hot is already 1, so THIS tick completes the 2-tick streak.
    assert asw.cpu_guard_pass(dry_run=False) is True
    assert len(_pressure_rows(watcher_roots)) == 1
    assert _read_state(watcher_roots)["alerted"] is True  # self-repaired to bool


# ─── 17. a snapshot match WITHOUT an int issue is honestly unattributed ──────


def test_kill_row_pid_match_without_issue_is_unattributed(guard_env, watcher_roots):
    """Pid match against a snapshot row with `issue: None` -> the kill row is
    explicitly unattributed. `issue: null` + `attribution_status: attributed`
    is banned (the honesty contract the field exists for)."""
    sig = guard_env
    snap = {
        "ts": 1.0,
        "procs": [
            {
                "pid": 4087688,
                "pcpu": 250.0,
                "rss_mib": 3390.0,
                "argv": "pytest tests/x.py",
                "issue": None,
            }
        ],
    }
    kill = asw.parse_earlyoom_kill_line(_KILL_LINE)  # pid 4087688 — pid match
    assert asw.attribute_kill(kill, snap) == (None, "unattributed")
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "vm-cpu-guard.json").write_text(json.dumps({"last_top_snapshot": snap}))
    sig.kills = [kill]
    asw.cpu_guard_pass(dry_run=False)
    rows = _kill_rows(watcher_roots)
    assert len(rows) == 1
    assert rows[0]["attribution_status"] == "unattributed"
    assert rows[0]["issue"] is None
    assert rows[0]["attribution_source"] is None


def test_kill_row_comm_match_without_issue_is_unattributed(guard_env, watcher_roots):
    """Same honesty contract on the unique-comm fallback path: a unique comm
    match to a snapshot row with `issue: None` is unattributed."""
    sig = guard_env
    snap = {
        "ts": 1.0,
        "procs": [
            {
                "pid": 500,  # kill pid 4087688 does NOT match -> comm path
                "pcpu": 10.0,
                "rss_mib": 100.0,
                "argv": "pytest tests/y.py",
                "issue": None,
            }
        ],
    }
    kill = asw.parse_earlyoom_kill_line(_KILL_LINE)  # comm "pytest", unique
    assert asw.attribute_kill(kill, snap) == (None, "unattributed")
    reg = watcher_roots / "reg"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "vm-cpu-guard.json").write_text(json.dumps({"last_top_snapshot": snap}))
    sig.kills = [kill]
    asw.cpu_guard_pass(dry_run=False)
    rows = _kill_rows(watcher_roots)
    assert len(rows) == 1
    assert rows[0]["attribution_status"] == "unattributed"
    assert rows[0]["issue"] is None


# ─── 18. dedup-key truncation keeps the NEWEST keys ──────────────────────────


def test_recent_kill_keys_truncation_keeps_newest(guard_env, watcher_roots):
    """A >50-kill backlog scan keeps the NEWEST keys under the 50-key cap
    (journal order is oldest-first), so a kill inside the 60 s overlap tail
    survives the cap and is not re-emitted on the next tick."""
    sig = guard_env
    base = asw.parse_earlyoom_kill_line(_KILL_LINE)
    sig.kills = [
        dict(base, pid=1000 + i, journal_ts=f"2026-06-28T21:44:{i:02d}-0700") for i in range(55)
    ]
    asw.cpu_guard_pass(dry_run=False)
    assert len(_kill_rows(watcher_roots)) == 55
    keys = _read_state(watcher_roots)["recent_kill_keys"]
    assert len(keys) == 50
    newest = f"{sig.kills[-1]['journal_ts']}:{sig.kills[-1]['pid']}"
    oldest = f"{sig.kills[0]['journal_ts']}:{sig.kills[0]['pid']}"
    assert newest in keys
    assert oldest not in keys
    # Overlap re-scan of the newest tail: zero duplicate rows.
    sig.kills = sig.kills[-5:]
    asw.cpu_guard_pass(dry_run=False)
    assert len(_kill_rows(watcher_roots)) == 55


# ─── 19. kernel 15-char comm truncation still comm-matches ───────────────────


def test_attribute_kill_comm_matches_kernel_15_char_truncation():
    """Kernel `comm` is 15-char truncated; a long-named process must still
    unique-comm-match by 15-char basename prefix."""
    snap = {
        "ts": 1.0,
        "procs": [
            {"pid": 7, "argv": "/usr/bin/my_very_long_worker_name --flag", "issue": 845},
        ],
    }
    kill = {"pid": 1, "comm": "my_very_long_wo"}  # 15-char kernel truncation
    assert asw.attribute_kill(kill, snap) == (845, "attributed")
    # A short comm still requires an exact basename match (no prefix creep).
    assert asw.attribute_kill({"pid": 1, "comm": "my_very"}, snap) == (None, "unattributed")


# ─── env knob fail-soft ──────────────────────────────────────────────────────


def test_env_float_knob(monkeypatch):
    name = "EPM_TEST_CPU_GUARD_FLOAT_KNOB"
    monkeypatch.setenv(name, "2.5")
    assert asw._env_float(name, 1.5, lo=0.1, hi=100.0) == 2.5
    monkeypatch.setenv(name, "garbage")
    assert asw._env_float(name, 1.5, lo=0.1, hi=100.0) == 1.5
    monkeypatch.setenv(name, "500")
    assert asw._env_float(name, 1.5, lo=0.1, hi=100.0) == 1.5
    monkeypatch.delenv(name)
    assert asw._env_float(name, 1.5, lo=0.1, hi=100.0) == 1.5
