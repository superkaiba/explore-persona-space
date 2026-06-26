"""Pin the #672 GCP-validation orchestrator's round-2 measurement-correctness fixes.

`scripts/issue672_validate.py` certifies that GCP works again post-#669/#671. Three
round-2 BLOCKERs (reconciler-binding, code-review round 1) each added a PERMANENT
invariant; this file is the fail-pre-fix / pass-post-fix regression lock for each:

1. **BLOCKER 1 — failover-count scoping.** `_count_failover_relaunches(issue,
   since_ts=...)` must EXCLUDE stale pre-injection failover markers (a Section-A
   failover, a prior #672 attempt) so the live-recovery predicate cannot
   false-PASS on a marker that pre-dates the iptables injection.
2. **BLOCKER 3 — Section-A coverage gate.** `analyze_section_a` must NOT certify
   "flat memory" on a degenerate one-entry log (the exact #671 truncated-logger
   failure class); it requires ≥30 entries + max iter ≥30 + ≥2 post-warmup
   non-None samples per PRIMARY gauge, surfacing the shortfall in
   `section_A.coverage_issue`.
3. **Defense-in-depth #4 — route_verdict conjunction.** The unqualified live
   headline requires `live_injection_pass is True AND failover_count == 1 AND
   fallback_outcome is None`; a hand-built inconsistent dict routes to a residual
   gap, never the live PASS headline.

CPU-only; no GCP / no GPU. Stubs `_events` + a tmp memory_log.json.
"""

from __future__ import annotations

import datetime
import json
import re

import pytest

import scripts.issue672_validate as m

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 1 — _count_failover_relaunches must be scoped to the current injection
# ─────────────────────────────────────────────────────────────────────────────


def _failover_event(ts: str) -> dict:
    """An events.jsonl row carrying a RunPod failover reason in its note."""
    return {
        "kind": "epm:backend-selected",
        "ts": ts,
        "note": json.dumps(
            {"chosen_kind": "runpod", "reason": "gcp_workload_failover_runpod_async"},
            sort_keys=True,
        ),
    }


def test_count_failover_excludes_pre_inject_marker(monkeypatch):
    """A failover marker that PRE-DATES `since_ts` must not count (BLOCKER 1).

    Pre-fix the function ignored `since_ts` and counted ALL historical failover
    markers, so a stale prior failover false-passed the live predicate.
    """
    stale = _failover_event("2026-06-26T10:00:00Z")
    monkeypatch.setattr(m, "_events", lambda _issue: [stale])

    since = datetime.datetime.fromisoformat("2026-06-26T11:00:00+00:00")  # AFTER the stale marker
    count, ts = m._count_failover_relaunches(672, since_ts=since)
    assert count == 0, f"stale pre-inject failover marker must be excluded, got count={count}"
    assert ts is None


def test_count_failover_includes_post_inject_marker(monkeypatch):
    """A failover marker AT/AFTER `since_ts` counts; a stale one alongside does not."""
    stale = _failover_event("2026-06-26T10:00:00Z")
    fresh = _failover_event("2026-06-26T11:05:00Z")
    monkeypatch.setattr(m, "_events", lambda _issue: [stale, fresh])

    since = datetime.datetime.fromisoformat("2026-06-26T11:00:00+00:00")
    count, ts = m._count_failover_relaunches(672, since_ts=since)
    assert count == 1, f"exactly the post-inject marker must count, got count={count}"
    assert ts == "2026-06-26T11:05:00Z"


def test_count_failover_unparseable_ts_excluded_when_scoped(monkeypatch):
    """A failover marker with a missing/garbage ts cannot prove it post-dates the
    injection, so it is EXCLUDED when `since_ts` is set."""
    no_ts = {
        "kind": "epm:cluster-launched",
        "note": json.dumps({"reason": "gcp_workload_failover_runpod"}),
    }
    monkeypatch.setattr(m, "_events", lambda _issue: [no_ts])
    since = datetime.datetime.fromisoformat("2026-06-26T11:00:00+00:00")
    count, _ts = m._count_failover_relaunches(672, since_ts=since)
    assert count == 0


def test_count_failover_unscoped_counts_all(monkeypatch):
    """`since_ts=None` preserves the unfiltered count (fallback-block diagnostic use)."""
    monkeypatch.setattr(
        m,
        "_events",
        lambda _issue: [
            _failover_event("2026-06-26T10:00:00Z"),
            _failover_event("2026-06-26T11:05:00Z"),
        ],
    )
    count, _ts = m._count_failover_relaunches(672)
    assert count == 2


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 3 — analyze_section_a coverage gate
# ─────────────────────────────────────────────────────────────────────────────


def _write_memlog(tmp_path, issue: int, out_subdir: str, entries: list[dict]) -> None:
    d = tmp_path / "eval_results" / f"issue_{issue}" / out_subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory_log.json").write_text(json.dumps(entries))


def test_section_a_one_entry_log_fails_coverage(monkeypatch, tmp_path):
    """A single-entry log must NOT pass (BLOCKER 3) even with npz+hf OK and the
    one sample trivially 'flat'; `coverage_issue` must be populated.

    Pre-fix `_flatness([x])` returned flat=True and `analyze_section_a` had no
    minimum-sample gate, so a truncated logger trivially certified flat memory.
    """
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_backend_selected_sku", lambda _issue: None)
    _write_memlog(
        tmp_path,
        672,
        "secA_smoke",
        [
            {
                "iter": 0,
                "memory_reserved_gib": 22.0,
                "nvidia_smi_used_gib": 22.0,
                "memory_allocated_gib": 18.0,
            }
        ],
    )

    block = m.analyze_section_a(672, "secA_smoke", npz_present=True, hf_ok=True)
    assert block["pass"] is False, "one-entry log must not pass the coverage gate"
    assert block["coverage_issue"], "coverage_issue must name the shortfall"
    assert "require ≥30" in block["coverage_issue"]


def test_section_a_nvidia_none_on_all_post_warmup_fails_coverage(monkeypatch, tmp_path):
    """≥30 reserved samples but nvidia-smi None on every post-warmup sample must
    fail the per-gauge post-warmup depth requirement with a populated
    coverage_issue (the 'nvidia_smi None on every post-warmup sample' case)."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_backend_selected_sku", lambda _issue: None)
    entries = [
        {
            "iter": i,
            "memory_reserved_gib": 22.0,
            "nvidia_smi_used_gib": None,  # absent on every sample
            "memory_allocated_gib": 18.0,
        }
        for i in range(0, 35)
    ]
    _write_memlog(tmp_path, 672, "secA_smoke", entries)

    block = m.analyze_section_a(672, "secA_smoke", npz_present=True, hf_ok=True)
    assert block["pass"] is False
    assert block["coverage_issue"]
    assert "nvidia_smi" in block["coverage_issue"]


def test_section_a_full_coverage_flat_passes(monkeypatch, tmp_path):
    """A well-formed flat log (≥30 entries, max iter ≥30, both PRIMARY gauges with
    ≥2 post-warmup non-None samples) PASSes — the gate does not over-block."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_backend_selected_sku", lambda _issue: None)
    entries = [
        {
            "iter": i,
            "memory_reserved_gib": 22.0 + 0.001 * i,  # < 1 GiB span over 35 samples
            "nvidia_smi_used_gib": 23.0 + 0.001 * i,
            "memory_allocated_gib": 18.0,
        }
        for i in range(0, 35)
    ]
    _write_memlog(tmp_path, 672, "secA_smoke", entries)

    block = m.analyze_section_a(672, "secA_smoke", npz_present=True, hf_ok=True)
    assert block["coverage_issue"] is None, block.get("coverage_issue")
    assert block["pass"] is True, block.get("pass_reason")


# ─────────────────────────────────────────────────────────────────────────────
# Defense-in-depth #4 — route_verdict conjunction
# ─────────────────────────────────────────────────────────────────────────────

_A_PASS = {"pass": True, "pass_reason": "ok"}
_C_PASS = {"pass": True}


def test_route_verdict_inconsistent_dict_does_not_get_live_headline():
    """A hand-built `{live_injection_pass: True, failover_count: 2}` must NOT earn
    the unqualified live headline; it routes to a residual gap (#4)."""
    b_bad = {"live_injection_pass": True, "failover_count": 2}
    verdict = m.route_verdict(_A_PASS, b_bad, _C_PASS)
    assert "self-recovers" not in verdict["verdict"], verdict
    assert verdict["verdict"].startswith("specific residual gap")
    assert verdict["headline_downgraded"] is True


def test_route_verdict_live_pass_full_conjunction():
    """The unqualified live headline fires only on the full conjunction."""
    b_good = {"live_injection_pass": True, "failover_count": 1, "fallback_outcome": None}
    verdict = m.route_verdict(_A_PASS, b_good, _C_PASS)
    assert verdict["verdict"].endswith("self-recovers")
    assert verdict["headline_downgraded"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 Critical #1 — the poller is LOOPED by the validator, not a single
# pre-injection Popen against the one-shot backend_poll.py
# ─────────────────────────────────────────────────────────────────────────────


def test_section_b_dry_run_constructs_looped_poller():
    """The Section-B dry-run must show the poller is LOOPED (round-3 Critical #1).

    Pre-fix (round 2) the poller was a single bg Popen launched BEFORE the 180s
    pre-injection sleep — useless against the one-shot ``backend_poll.py`` (it
    polls a healthy VM once and is dead by watchdog-kill time). The dry-run must
    now surface a ``poller_loop`` block (``looped=True`` + cadence/budget) and
    echo the poller call REPEATEDLY, never once.
    """
    block = m.run_section_b(672, dry_run=True)
    assert block["dry_run"] is True
    loop = block["poller_loop"]
    assert loop["looped"] is True, "poller must be validator-looped, not a single Popen"
    assert "backend_poll.py" in loop["poller_argv"]
    assert loop["interval_s"] == m.POLLER_INTERVAL_S
    assert loop["budget_s"] == m.POLLER_LOOP_BUDGET_S
    assert loop["quiet_period_s"] == m.POLLER_QUIET_PERIOD_S
    assert loop["max_invocations"] >= 2
    # The repeated poller invocation is demonstrated, not a single pre-inject call.
    assert len(block["poller_invocations"]) >= 2, "dry-run must echo the LOOP (≥2 poller calls)"
    assert all("ts" in inv and "exit_code" in inv for inv in block["poller_invocations"])
    assert block["poller_exit_codes"] == [inv["exit_code"] for inv in block["poller_invocations"]]


def test_section_b_dispatch_plan_loop_is_satisfiable():
    """The loop budget / interval admit at least one one-shot invocation, and the
    poller argv is the one-shot backend_poll.py (no --watch / --loop flag)."""
    plan = m.section_b_dispatch_plan(672)
    loop = plan["poller_loop"]
    assert loop["budget_s"] // loop["interval_s"] >= 1
    poller_argv = plan["argvs"]["poller"]
    assert poller_argv[:4] == ["uv", "run", "python", "scripts/backend_poll.py"]
    assert "--watch" not in poller_argv and "--loop" not in poller_argv


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 Critical #2 — Section-A dispatch must use --log-mem-every 1 so the
# coverage gate is SATISFIABLE on the plan §9 <100-forward smoke slice
# ─────────────────────────────────────────────────────────────────────────────

PLAN_SECTION_A_FORWARD_UPPER_BOUND = 100  # plan §9 line 119: "<100 hooked 7B forwards"


def test_section_a_dispatch_uses_log_mem_every_1():
    """The Section-A workload-cmd must carry ``--log-mem-every 1`` (round-3 Crit #2).

    At ``--log-mem-every 10`` the plan §9 <100-forward smoke yields only ~10 log
    rows, below the round-2 ``MIN_LOG_ENTRIES=30`` coverage floor — so a HEALTHY
    Section A could never PASS. Every-iter cadence yields up to ~100 rows.
    """
    argv = m.section_a_dispatch_argv(672, "secA_smoke")
    workload = argv[-1]  # the --workload-cmd value is the last element
    mobj = re.search(r"--log-mem-every\s+(\d+)", workload)
    assert mobj, f"--log-mem-every not found in workload cmd: {workload!r}"
    every = int(mobj.group(1))
    assert every == 1, f"expected --log-mem-every 1, got {every}"
    # Mechanizable invariant: the realized row count on the plan's forward bound
    # must clear the coverage floor.
    rows_emitted = PLAN_SECTION_A_FORWARD_UPPER_BOUND // every
    assert rows_emitted >= m.MIN_LOG_ENTRIES, (
        f"--log-mem-every {every} yields ~{rows_emitted} rows on a "
        f"{PLAN_SECTION_A_FORWARD_UPPER_BOUND}-forward smoke; coverage floor is "
        f"{m.MIN_LOG_ENTRIES}"
    )


def test_section_a_30_entry_every_iter_log_passes_coverage(monkeypatch, tmp_path):
    """A 30-entry every-iter log (max iter ≥30, ≥2 post-warmup per PRIMARY gauge)
    SATISFIES the coverage gate — proving the gate is reachable with the
    ``--log-mem-every 1`` dispatch on a ~30-forward slice (round-3 Crit #2)."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_backend_selected_sku", lambda _issue: None)
    # Every-iter cadence: iter == row index, 0..30 (31 rows, max iter 30).
    entries = [
        {
            "iter": i,
            "memory_reserved_gib": 22.0 + 0.001 * i,  # flat: <1 GiB span
            "nvidia_smi_used_gib": 23.0 + 0.001 * i,
            "memory_allocated_gib": 18.0,
        }
        for i in range(0, 31)
    ]
    _write_memlog(tmp_path, 672, "secA_smoke", entries)
    block = m.analyze_section_a(672, "secA_smoke", npz_present=True, hf_ok=True)
    assert block["coverage_issue"] is None, block.get("coverage_issue")
    assert block["pass"] is True, block.get("pass_reason")


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 Major #3 — quiet-period: a 2nd failover during the quiet window is
# observed so the "failover fires twice" kill criterion trips
# ─────────────────────────────────────────────────────────────────────────────


def _stub_loop_clock_and_poller(monkeypatch, *, count_sequence: list[int]):
    """Drive `_loop_poller_until_failover` deterministically.

    - `time.time()` advances by POLLER_INTERVAL_S per call (so the loop budget
      and quiet-period clocks are exercised in virtual time, no real sleeps).
    - `time.sleep` is a no-op.
    - `_invoke_poller_once` is a no-op stub returning a record.
    - `_count_failover_relaunches` returns the next value in `count_sequence`
      (clamped to the last value once exhausted), simulating markers landing.
    """
    clock = {"t": 1000.0}

    def fake_time():
        clock["t"] += m.POLLER_INTERVAL_S
        return clock["t"]

    monkeypatch.setattr(m.time, "time", fake_time)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        m, "_invoke_poller_once", lambda _issue: {"ts": "T", "exit_code": 0, "stdout_head": "{}"}
    )
    seq = list(count_sequence)
    idx = {"i": 0}

    def fake_count(_issue, *, since_ts=None):
        i = idx["i"]
        val = seq[i] if i < len(seq) else seq[-1]
        idx["i"] = i + 1
        ts = "2026-06-26T11:05:00Z" if val >= 1 else None
        return val, ts

    monkeypatch.setattr(m, "_count_failover_relaunches", fake_count)


def test_loop_poller_detects_second_failover_in_quiet_period(monkeypatch):
    """A 2nd failover landing during the quiet period must be COUNTED (Major #3).

    Pre-fix `_poll_failover_relaunch` broke on the first marker, so a second
    marker landing seconds later was missed and the "failover fires twice" kill
    criterion never tripped. The looped poller keeps polling through a
    quiet-period; the FINAL scoped count must be 2.
    """
    # poll-1 -> count 1 (first marker, starts quiet clock)
    # poll-2 -> count 2 (NEW marker, resets quiet clock)
    # poll-3.. -> stays 2 until quiet-period elapses
    _stub_loop_clock_and_poller(monkeypatch, count_sequence=[1, 2, 2, 2, 2, 2])
    since = datetime.datetime.fromisoformat("2026-06-26T11:00:00+00:00")
    count, ts, invocations = m._loop_poller_until_failover(672, since_ts=since)
    assert count == 2, f"second failover in the quiet period must be counted, got {count}"
    assert ts == "2026-06-26T11:05:00Z"
    assert len(invocations) >= 2


def test_loop_poller_single_failover_after_quiet_period(monkeypatch):
    """Exactly one failover (no second marker through the quiet window) -> count 1."""
    _stub_loop_clock_and_poller(monkeypatch, count_sequence=[0, 1, 1, 1, 1, 1])
    since = datetime.datetime.fromisoformat("2026-06-26T11:00:00+00:00")
    count, _ts, invocations = m._loop_poller_until_failover(672, since_ts=since)
    assert count == 1, f"exactly-one failover must yield count 1, got {count}"
    assert len(invocations) >= 1


def test_run_section_b_double_failover_routes_residual_gap(monkeypatch):
    """End-to-end: a doubled failover -> live_injection_pass False, failover_count
    2, and route_verdict routes to a residual gap (Major #3 + DiD #4)."""
    # Short-circuit the live launch/inject/watchdog/terminate gates to True.
    monkeypatch.setattr(m, "_run", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(m, "_poll_watchdog_fired", lambda _issue: True)
    monkeypatch.setattr(m, "_poll_vm_terminated", lambda _issue: True)
    monkeypatch.setattr(
        m,
        "_loop_poller_until_failover",
        lambda _issue, *, since_ts: (
            2,
            "2026-06-26T11:05:00Z",
            [
                {"ts": "T", "exit_code": 0, "stdout_head": "{}"},
                {"ts": "T", "exit_code": 0, "stdout_head": "{}"},
            ],
        ),
    )
    block = m.run_section_b(672, dry_run=False)
    assert block["failover_count"] == 2
    assert block["live_injection_pass"] is False
    assert block["fallback_outcome"] == "residual_gap"
    assert block["headline_downgraded"] is True
    verdict = m.route_verdict(_A_PASS, block, _C_PASS)
    assert verdict["verdict"].startswith("specific residual gap")
    assert "self-recovers" not in verdict["verdict"]


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 Major #4 — poller_exit_codes present on ALL Section-B return paths
# (success, fallback, exception) — never omitted on an early fallback return
# ─────────────────────────────────────────────────────────────────────────────


def test_run_section_b_success_surfaces_poller_exit_codes(monkeypatch):
    """The success path surfaces ``poller_exit_codes`` as a list of ints (Major #4)."""
    monkeypatch.setattr(m, "_run", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(m, "_poll_watchdog_fired", lambda _issue: True)
    monkeypatch.setattr(m, "_poll_vm_terminated", lambda _issue: True)
    monkeypatch.setattr(
        m,
        "_loop_poller_until_failover",
        lambda _issue, *, since_ts: (
            1,
            "2026-06-26T11:05:00Z",
            [
                {"ts": "T", "exit_code": 0, "stdout_head": "{}"},
                {"ts": "T", "exit_code": 0, "stdout_head": "{}"},
            ],
        ),
    )
    block = m.run_section_b(672, dry_run=False)
    assert block["live_injection_pass"] is True
    assert "poller_exit_codes" in block
    assert block["poller_exit_codes"] == [0, 0]
    assert all(isinstance(c, int) for c in block["poller_exit_codes"])
    assert len(block["poller_invocations"]) == 2


def test_run_section_b_failover_count_zero_fallback_surfaces_exit_codes(monkeypatch):
    """The 'dead VM never re-dispatched' fallback (after the poller loop) still
    surfaces ``poller_exit_codes`` (Major #4: no early return omits it)."""
    monkeypatch.setattr(m, "_run", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr(m, "_poll_watchdog_fired", lambda _issue: True)
    monkeypatch.setattr(m, "_poll_vm_terminated", lambda _issue: True)
    monkeypatch.setattr(
        m,
        "_loop_poller_until_failover",
        lambda _issue, *, since_ts: (
            0,
            None,
            [
                {"ts": "T", "exit_code": 0, "stdout_head": "{}"},
                {"ts": "T", "exit_code": 1, "stdout_head": "{}"},
            ],
        ),
    )
    block = m.run_section_b(672, dry_run=False)
    assert block["live_injection_pass"] is False
    assert block["fallback_outcome"] == "inconclusive_live_validation"
    assert "poller_exit_codes" in block, "fallback must not omit poller_exit_codes"
    assert block["poller_exit_codes"] == [0, 1]


def test_run_section_b_early_fallback_includes_poller_invocations_key(monkeypatch):
    """An EARLY fallback (e.g. iptables injection fails BEFORE the poller loop)
    still carries ``poller_invocations`` (empty list) — the audit-trail key is
    never absent (Major #4)."""
    # launch ok (rc 0) then iptables fails (rc 1).
    calls = {"n": 0}

    def fake_run(argv, *, dry_run, timeout=None):
        calls["n"] += 1
        # 1st _run is the VM launch (ok); 2nd is iptables (fail).
        return (0, "", "") if calls["n"] == 1 else (1, "", "iptables boom")

    monkeypatch.setattr(m, "_run", fake_run)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    block = m.run_section_b(672, dry_run=False)
    assert block["fallback_outcome"] == "inconclusive_live_validation"
    assert "iptables" in block["fallback_reason"]
    assert "poller_invocations" in block
    assert block["poller_invocations"] == []
    # Major #4: poller_exit_codes is present (empty) on the early fallback too.
    assert block.get("poller_exit_codes") == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
