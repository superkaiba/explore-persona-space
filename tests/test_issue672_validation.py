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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
