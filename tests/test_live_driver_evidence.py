"""Tests for ``task_workflow.live_driver_evidence`` (#1326).

The /issue Step 0 single-orchestrator guard's registry-blind second leg: a
sanctioned inline-chat follow-up driver (the CLAUDE.md user-chat carve-out,
explicit-override clause) never appears in the spawn-session registry, so the
guard ALSO probes the task's own marker trail for fresh live-driver evidence
within a freshness window — compute-launch markers (``epm:run-launched`` /
``epm:cluster-launched``) and ``stage-dispatch `` breadcrumbs, the two marker
shapes only an actively-orchestrating driver posts. Anti-liveness parity with
``stage_dispatch_should_skip`` (#810): a ``deliberate-stop `` note,
``by == "spawn_session-stop"`` rows, and bracketed watcher / spawn-session
telemetry never count, so a predecessor's death record cannot hold the guard.

``INCIDENT_952_ROWS`` below are the four REAL #952 events (2026-07-14/15
incident, ``tasks/.../952/events.jsonl``) VERBATIM — Z-suffixed ``ts`` strings
and real note text — pinning the ``Z -> +00:00`` parse path and the real note
shapes the exclusions must discriminate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from explore_persona_space.task_workflow import (
    LIVE_DRIVER_EVIDENCE_KINDS,
    live_driver_evidence,
)

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _ev(kind: str, age_minutes: float, note: str = "", by: str = "unknown") -> dict:
    """Literal event row whose ``ts`` sits ``age_minutes`` before ``NOW``."""
    ts = (NOW - timedelta(minutes=age_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"ts": ts, "kind": kind, "by": by, "note": note}


INCIDENT_952_ROWS: list[dict] = [
    {
        "ts": "2026-07-15T04:43:38Z",
        "kind": "epm:followup-scope",
        "version": 3,
        "by": "unknown",
        "note": (
            "<!-- epm:followup-scope v1 -->\nfollowup_label: china-politics-topup\nsource: user-c"
            'hat\nquestion_relation: same\nest_gpu_hours: 2\norigin_prompt (verbatim): "can we r'
            'erun the china politics stuff" [context: the china-politics divergence-bank categor'
            "y was dropped at the 20-pair eligibility floor with 18 kept pairs of 60 candidates; "
            "user wants the category read]\n\nSCOPE — one changed variable: divergence-bank "
            "CATEGORY COVERAGE (china-politics included), everything else pinned to the committed"
            " recipe.\n1. TOP-UP: generate new china-politics candidate pairs via the EXISTING is"
            "sue952_bank_build.py machinery (CCP-sensitive-prompts source rows not yet used + Son"
            "net same-template entity-swapped controls; both models answer under the committed ge"
            "neration recipe — Qwen vLLM T=1.0 top_p 0.95 seed 42 max_tokens 1024, Claude cl"
            "aude-sonnet-4-5-20250929 T=1.0; graded judge, SAME committed keep gates keep>=47 / m"
            "argin>=-23 — do NOT redesign the rubric this round) until >=20 china pairs pass"
            ". Budget ~15-25 new candidates given the parent's 18/60 yield. Bank content rule bin"
            "ds: item text referenced by file+index only, never printed into markers/bodies/repor"
            "ts.\n2. REUSE the parent's 18 kept pairs; check whether their bank activations were "
            "captured in the parent run (artifact-reuse fitness check) — capture only what i"
            "s missing (teacher-forced, fp16, layer-20 decision cell + bank layers 14/23/26 if ne"
            "ar-free; own + external-plain arms minimum).\n3. RE-RUN the H3 bank read INCLUDING c"
            "hina-politics as a third category: arm-matched read AND the cross transfer cell (own"
            " map x plain targets; eval_results/issue_952/divergence_transfer_cell machinery), Ho"
            "lm across the now-3 surviving categories; report pooled with/without china and per-c"
            "ategory; carry the judge-calibration-inversion caveat forward unchanged.\n4. Fold th"
            "e finding into the #952 clean-result body (Takeaways + Result-divergence section) vi"
            "a the standard analyzer pass; the null-vs-penalty outcome on china rewrites the bank"
            " bullet either way.\nArtifacts under eval_results/issue_952/china-politics-topup/ + "
            "HF issue952_position_divergence/followups/china_politics_topup/."
        ),
    },
    {
        "ts": "2026-07-15T04:46:33Z",
        "kind": "epm:progress",
        "version": 53,
        "by": "spawn_session-stop",
        "note": (
            "deliberate-stop pid=n/a target=happy-session:cmrllhz6nxgitwc0ub6ytqop2 reason=operat"
            "or stop via spawn_session.py stop"
        ),
    },
    {
        "ts": "2026-07-15T04:47:05Z",
        "kind": "epm:progress",
        "version": 54,
        "by": "unknown",
        "note": (
            "china-politics-topup INLINE DISPATCH — user override (verbatim: 'run it inline "
            "so it is faster. run in parallel as much as possible'); spawned session cmrllhz6nxgi"
            "twc0ub6ytqop2 stopped, round runs inline from the user chat session under the armed "
            "epm:followup-scope v3 (satisfier epm:same-issue-followup-run will be posted on compl"
            "etion). Parallel plan: [A] VM API leg (new candidate pairs from unused CCP-sensitive"
            "-prompts subjects via issue952_bank_build.py machinery + Sonnet template-swap contro"
            "ls + Claude answers; parent capture-coverage check) CONCURRENT WITH [B] pod provisio"
            "n (pod-952, intent eval, 1xH100 — RunPod interactive carve-out: inline SSH-orch"
            "estrated follow-up; keep-running tag SET pre-provision, epm:run-launched to be poste"
            "d at pod-up); then [B] Qwen vLLM generation (one batched generate, ~100 rows incl. c"
            "ontrols, committed recipe seed 42) then judge (API, ~50 pairs x 5+3 draws, sync via "
            "api_dispatch, drop-never-coerce, transport-retry) CONCURRENT WITH capture (teacher-f"
            "orced fp16, new pairs + any missing parent-18 coverage, arms own+external-plain, lay"
            "ers 14/20/23/26, batched forwards); then stats re-read (VM CPU, committed batched-GE"
            "MM sign-flip machinery incl. the transfer cell). Compute character: generation ~100 "
            "rows one vLLM batch (<10 min H100); capture <=48 pairs x 2 roles x 2 arms batched te"
            "acher-forced forwards (<15 min); judge ~500 sync calls under the Sonnet 100-cap; sta"
            "ts one batched GEMM battery (<1 min). Projected pod wall <=1h (~1 GPU-h vs est 2). B"
            "ank content rule binds throughout: file+index refs only, no item text in any output."
        ),
    },
    {
        "ts": "2026-07-15T04:56:03Z",
        "kind": "epm:run-launched",
        "version": 1,
        "by": "unknown",
        "note": (
            "pod-952 (eval, 1xH100) up for inline china-politics-topup round — RunPod intera"
            "ctive carve-out (inline SSH-orchestrated follow-up); keep-running tag already set pr"
            "e-provision"
        ),
    },
]


def test_fresh_run_launched_is_evidence():
    # BOTH compute-launch kinds count as evidence; the line names the kind.
    for kind in sorted(LIVE_DRIVER_EVIDENCE_KINDS):
        line = live_driver_evidence([_ev(kind, 10.0, "pod up")], window_minutes=30, now=NOW)
        assert line is not None, kind
        assert kind in line


def test_stale_run_launched_is_not_evidence():
    events = [_ev("epm:run-launched", 45.0, "pod up")]
    assert live_driver_evidence(events, window_minutes=30, now=NOW) is None


def test_fresh_stage_dispatch_breadcrumb_is_evidence():
    note = "stage-dispatch stage=followup-upload round=1 sid=abc123"
    line = live_driver_evidence([_ev("epm:progress", 5.0, note)], window_minutes=30, now=NOW)
    assert line is not None
    assert "epm:progress" in line


def test_anti_liveness_rows_never_count():
    # CANDIDATE-SHAPED anti-liveness rows: each is an evidence-kind row that
    # ONLY the exclusion branch can drop — deleting either exclusion flips
    # the corresponding assertion red (plain-progress fixtures never reach
    # the exclusion branch because the candidate gate drops them first).
    watcher_on_behalf = _ev(
        "epm:run-launched",
        5.0,
        "[autonomous_session_watch:pod-safety] on-behalf run marker for pod-952",
    )
    assert live_driver_evidence([watcher_on_behalf], window_minutes=30, now=NOW) is None

    stop_authored = _ev("epm:run-launched", 5.0, "pod up", by="spawn_session-stop")
    assert live_driver_evidence([stop_authored], window_minutes=30, now=NOW) is None

    # A candidate-shaped breadcrumb whose note is a deliberate-stop record.
    stop_prefixed = _ev("epm:cluster-launched", 5.0, "deliberate-stop pid=n/a target=self ...")
    assert live_driver_evidence([stop_prefixed], window_minutes=30, now=NOW) is None

    # Additional NON-candidate cases (dropped at the candidate gate).
    non_candidates = [
        _ev("epm:progress", 5.0, "deliberate-stop pid=n/a target=self reason=stale-wake-yield"),
        _ev("epm:progress", 5.0, "[spawn-session:issue-952] bookkeeping sentinel"),
    ]
    assert live_driver_evidence(non_candidates, window_minutes=30, now=NOW) is None


def test_plain_progress_note_is_not_evidence():
    # A 0-GPU inline free-analysis DISPATCH note is a generic epm:progress —
    # it never holds the guard (scoping: read-only rounds race nothing).
    note = "inline free-analysis dispatch — re-reading committed eval JSONs for a summary"
    events = [_ev("epm:progress", 2.0, note)]
    assert live_driver_evidence(events, window_minutes=30, now=NOW) is None


def test_malformed_and_future_ts_fail_toward_dispatch():
    future_ts = (NOW + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [
        {"ts": "not-a-timestamp", "kind": "epm:run-launched", "by": "unknown", "note": "pod up"},
        {"ts": None, "kind": "epm:run-launched", "by": "unknown", "note": "pod up"},
        {"ts": future_ts, "kind": "epm:run-launched", "by": "unknown", "note": "pod up"},
    ]
    assert live_driver_evidence(events, window_minutes=30, now=NOW) is None


def test_incident_952_replay_yields():
    # At the duplicate /issue 952 session's Step-0 window (~05:16Z) the
    # inline driver's 04:56:03Z epm:run-launched is ~19.9 min old — the
    # probe FIRES on its own motivating incident (a 15-min window would
    # miss it). The 04:46:33Z deliberate-stop (by=spawn_session-stop) and
    # the 04:47:05Z plain override note never count.
    step0 = datetime(2026, 7, 15, 5, 16, tzinfo=UTC)
    line = live_driver_evidence(INCIDENT_952_ROWS, window_minutes=30, now=step0)
    assert line is not None
    assert "epm:run-launched" in line
    assert "2026-07-15T04:56:03Z" in line
    assert "19.9" in line  # age, loosely — not 20.0

    # Decay (fixture-relative): by 05:47Z the four-row fixture's freshest
    # evidence is >=30 min old — no #845 12h-registration wedge shape. (The
    # full live artifact still fires at 05:47 via its 05:17:46Z breadcrumb;
    # artifact-level decay assertions must use >=05:48Z.)
    later = datetime(2026, 7, 15, 5, 47, tzinfo=UTC)
    assert live_driver_evidence(INCIDENT_952_ROWS, window_minutes=30, now=later) is None


def test_exact_window_boundary_is_not_evidence():
    # Age exactly == window_minutes is NOT evidence (the >= boundary, AC4).
    events = [_ev("epm:run-launched", 30.0, "pod up")]
    assert live_driver_evidence(events, window_minutes=30, now=NOW) is None


def test_window_minutes_parameter_is_honored():
    # A helper hardcoding 30 fails this: 10-min-old evidence, 5-min window.
    events = [_ev("epm:run-launched", 10.0, "pod up")]
    assert live_driver_evidence(events, window_minutes=5, now=NOW) is None
    assert live_driver_evidence(events, window_minutes=30, now=NOW) is not None
