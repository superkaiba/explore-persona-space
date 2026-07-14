"""Tests for the verdict-disagree observer (#1170; origin incident #825).

Two layers are pinned here:

1. **Pure predicate** — ``task_workflow.unreconciled_disagreement_rounds``
   flags, per doubled marker-mode review site, the LATEST round whose
   Claude + Codex verdict markers both exist with parseable OPPOSITE-class
   verdicts, no role-matched ``epm:review-reconcile``, and (Tier-2 only)
   no Codex no-show evidence. The founding #825 shape is replayed twice:
   synthetically (Claude PASS head-sentinel v5 vs Codex FAIL bare
   version 7, 2 min apart) and from the REAL events.jsonl rows committed
   verbatim at ``tests/fixtures/issue825_verdict_disagree_rows.jsonl``.
2. **Watcher pass** — ``autonomous_session_watch.verdict_disagree_pass``
   is observe/alert only: sidecar + one deduped push per finding, NEVER a
   task marker / status mutation / session stop (pinned at the
   subprocess-argv level, mirroring the triage-observer tests' posture).
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# scripts/ holds autonomous_session_watch.py; src/ holds
# explore_persona_space.task_workflow. Inserted ahead of any installed copy
# so THIS checkout's helpers win (#894; same shim as
# tests/test_autonomous_session_watch.py).
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import explore_persona_space.task_workflow as tw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

CR = "epm:code-review"
CRX = "epm:code-review-codex"
REC = "epm:review-reconcile"
FU = "epm:followup-value-critique"
FUX = "epm:followup-value-critique-codex"

CR_PASS = ("pass", "concerns")
CR_FAIL = ("fail",)


def _ev(kind: str, ts: str, *, version: int = 1, note: str = "", by: str = "test") -> dict:
    return {"ts": ts, "kind": kind, "version": version, "note": note, "by": by}


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _tier1_pair(
    claude_verdict: str = "PASS",
    codex_verdict: str = "FAIL",
    *,
    round_n: int = 2,
    t_claude: str = "2026-07-03T08:00:00Z",
    t_codex: str = "2026-07-03T08:03:00Z",
) -> list[dict]:
    """Round-aligned pair: both sides carry the SAME head-sentinel round."""
    return [
        _ev(
            CR,
            t_claude,
            version=round_n,
            note=f"<!-- epm:code-review v{round_n} -->\n**Verdict:** {claude_verdict}",
        ),
        _ev(
            CRX,
            t_codex,
            version=round_n,
            note=f"<!-- epm:code-review-codex v{round_n} -->\n**Verdict:** {codex_verdict}",
        ),
    ]


# now 2h after the later verdict: matured past the 1h grace default.
_TIER1_NOW = _epoch("2026-07-03T08:03:00Z") + 7200.0


def _tier2_pair() -> list[dict]:
    """The #825-drift shape: Codex FAIL bare version 7 (no sentinel) +
    Claude PASS head-sentinel v5, 2 min apart — Tier 1 cannot round-align
    them (round 5 has no Codex marker), so pairing falls to proximity."""
    return [
        _ev(
            CRX,
            "2026-07-03T08:33:30Z",
            version=7,
            note="Codex code-review round 1 (follow-up).\n**Verdict:** FAIL",
        ),
        _ev(
            CR,
            "2026-07-03T08:35:50Z",
            version=5,
            note="<!-- epm:code-review v5 -->\n**Verdict:** PASS",
        ),
    ]


_TIER2_NOW = _epoch("2026-07-03T08:35:50Z") + 7200.0


# ─── Pure predicate: pairing + flagging ──────────────────────────────────────


def test_tier1_disagreement_flagged():
    # (a) Tier-1 disagreement, matured, no reconcile, no evidence -> exactly
    # one finding with the round-scoped key + expected classes.
    out = tw.unreconciled_disagreement_rounds(_tier1_pair(), now_ts=_TIER1_NOW)
    assert len(out) == 1
    f = out[0]
    assert f["key"] == "code-reviewer|r2"
    assert f["tier"] == "round"
    assert f["role"] == "code-reviewer"
    assert (f["claude_class"], f["codex_class"]) == ("pass", "fail")
    assert (f["claude_ts"], f["codex_ts"]) == ("2026-07-03T08:00:00Z", "2026-07-03T08:03:00Z")


def test_round_matched_reconcile_unflags():
    # (b) A round-matched, role-matched reconcile satisfies the pair.
    events = [
        *_tier1_pair(),
        _ev(
            REC,
            "2026-07-03T08:20:00Z",
            version=1,
            note=(
                "<!-- epm:review-reconcile v2 -->\n"
                "**Role under adjudication:** code-reviewer\n"
                "**Round:** 2\n**Verdict:** FAIL"
            ),
        ),
    ]
    assert tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW) == []


def test_ts_scoped_reconcile_round_label_mismatch_unflags():
    # (b2) #825's real shape: the reconcile's own round label (sentinel v1,
    # **Round:** 1) matches NEITHER side's derived round (5 / 7) — the
    # ts-scoped role-matched clause must still satisfy the pair.
    pair = _tier2_pair()
    assert len(tw.unreconciled_disagreement_rounds(pair, now_ts=_TIER2_NOW)) == 1  # control
    events = [
        *pair,
        _ev(
            REC,
            "2026-07-03T08:45:34Z",
            version=1,
            note=(
                "<!-- epm:review-reconcile v1 -->\n\n## Reconciler Verdict — FAIL\n\n"
                "**Role under adjudication:** code-reviewer\n"
                "**Round:** 1 (same-issue follow-up)\n**Verdict:** FAIL"
            ),
        ),
    ]
    assert tw.unreconciled_disagreement_rounds(events, now_ts=_TIER2_NOW) == []


def test_reconcile_for_different_role_still_flags():
    # (c) A same-round reconcile adjudicating a DIFFERENT role never
    # satisfies this site's pair.
    events = [
        *_tier1_pair(),
        _ev(
            REC,
            "2026-07-03T08:20:00Z",
            version=1,
            note=(
                "<!-- epm:review-reconcile v2 -->\n"
                "**Role under adjudication:** interpretation-critic\n"
                "**Round:** 2\n**Verdict:** PASS"
            ),
        ),
    ]
    assert len(tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW)) == 1


@pytest.mark.parametrize(
    ("evidence", "expect_flagged"),
    [
        # (d) Each no-show evidence class suppresses a TIER-2 pairing.
        (
            _ev(
                "epm:codex-task-failed",
                "2026-07-03T08:30:00Z",
                note="codex job failed rc=9; reason codex-quota-exhausted",
                by="codex_task",
            ),
            False,
        ),
        (
            _ev(
                "epm:failure",
                "2026-07-03T08:30:00Z",
                note="<!-- failure_class: codex-output-malformed -->\nwrapper output malformed",
            ),
            False,
        ),
        (
            _ev(
                "epm:failure",
                "2026-07-03T08:30:00Z",
                note="failure_class: infra\nCodex wrapper died: transient registry blip",
            ),
            False,
        ),
        (
            _ev(
                "epm:progress",
                "2026-07-03T08:30:00Z",
                note=(
                    "codex composers skipped — quota sentinel live until "
                    "2026-08-06T13:26:00+00:00 (#1204 pre-spawn check); "
                    "single-Claude per no-show fallback"
                ),
            ),
            False,
        ),
        # Scoping negative 1: a generic pod-infra failure (no "codex" in the
        # note) must NOT suppress.
        (
            _ev(
                "epm:failure",
                "2026-07-03T08:30:00Z",
                note="failure_class: infra\npod provision failed: SUPPLY_CONSTRAINT",
            ),
            True,
        ),
        # Scoping negative 2: evidence BEFORE min(pair_ts) - lookback (2h
        # before 08:33:30 = 06:33:30) never counts.
        (
            _ev(
                "epm:codex-task-failed",
                "2026-07-03T06:00:00Z",
                note="codex job failed rc=9",
                by="codex_task",
            ),
            True,
        ),
    ],
)
def test_no_show_evidence_scopes_tier2(evidence, expect_flagged):
    events = [evidence, *_tier2_pair()]
    out = tw.unreconciled_disagreement_rounds(events, now_ts=_TIER2_NOW)
    assert (len(out) == 1) is expect_flagged


def test_tier1_pair_is_never_evidence_suppressed():
    # (d2) v2 Must-Fix pin: no-show evidence cannot explain away two
    # PRESENT, parseable verdicts at the same round — a Tier-1 pair is
    # never evidence-suppressed (a prior-round codex-task-failed within the
    # lookback would otherwise blind the observer exactly during
    # Codex-unstable periods, #1126).
    events = [
        _ev(
            "epm:codex-task-failed",
            "2026-07-03T07:30:00Z",  # 30 min BEFORE the pair
            note="codex job failed rc=9; reason codex-quota-exhausted",
            by="codex_task",
        ),
        *_tier1_pair(),
    ]
    out = tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW)
    assert len(out) == 1
    assert out[0]["tier"] == "round"


@pytest.mark.parametrize(
    ("claude_verdict", "codex_verdict"),
    [("PASS", "PASS"), ("FAIL", "FAIL"), ("PASS", "CONCERNS")],
)
def test_agreeing_verdicts_not_flagged(claude_verdict, codex_verdict):
    # (e) Same-class pairs never flag — PASS+CONCERNS is same-class for the
    # code-review site per workflow.yaml.
    events = _tier1_pair(claude_verdict, codex_verdict)
    assert tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW) == []


def test_latest_round_only_supersedes_earlier_disagreement():
    # (f) An earlier-round disagreement superseded by a later AGREEING round
    # is moot — only the latest round per (issue, site) is evaluated.
    events = _tier1_pair(
        "PASS", "FAIL", round_n=1, t_claude="2026-07-03T07:00:00Z", t_codex="2026-07-03T07:03:00Z"
    ) + _tier1_pair("PASS", "PASS", round_n=2)
    assert tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW) == []


def test_issue825_replica_tier2_flagged():
    # (f2) The synthetic #825 replica pairs at Tier 2 (proximity).
    out = tw.unreconciled_disagreement_rounds(_tier2_pair(), now_ts=_TIER2_NOW)
    assert len(out) == 1
    f = out[0]
    assert f["tier"] == "proximity"
    assert f["role"] == "code-reviewer"
    assert (f["claude_class"], f["codex_class"]) == ("pass", "fail")
    assert f["key"] == "code-reviewer|t2|2026-07-03T08:35:50Z|2026-07-03T08:33:30Z"


def test_grace_window_defers_then_flags():
    # (h) 30 min after the later verdict: still inside the 1h grace ->
    # deferred; 2h after: flagged.
    events = _tier1_pair()
    early = _epoch("2026-07-03T08:03:00Z") + 1800.0
    assert tw.unreconciled_disagreement_rounds(events, now_ts=early) == []
    assert len(tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW)) == 1


def test_grace_boundary_exact_is_flagged():
    # (h2) now - max(pair_ts) == grace_s exactly -> flagged (strict <).
    events = _tier1_pair()
    at_boundary = _epoch("2026-07-03T08:03:00Z") + 3600.0
    assert len(tw.unreconciled_disagreement_rounds(events, now_ts=at_boundary)) == 1


def test_pair_proximity_boundary_exact_still_pairs():
    # (h2) abs(t_claude - t_codex) == pair_proximity_s exactly -> still
    # paired (strict >). 6h apart to the second, same round sentinels.
    events = _tier1_pair(t_claude="2026-07-03T08:00:00Z", t_codex="2026-07-03T14:00:00Z")
    now = _epoch("2026-07-03T14:00:00Z") + 7200.0
    assert len(tw.unreconciled_disagreement_rounds(events, now_ts=now)) == 1


def test_pair_proximity_exceeded_not_flagged():
    # (i) Verdicts 7h apart exceed the 6h proximity bound (cross-epoch
    # aliasing guard).
    events = _tier1_pair(t_claude="2026-07-03T08:00:00Z", t_codex="2026-07-03T15:00:00Z")
    now = _epoch("2026-07-03T15:00:00Z") + 7200.0
    assert tw.unreconciled_disagreement_rounds(events, now_ts=now) == []


def test_follow_up_critic_vocabulary():
    # (j) not-redundant vs redundant is a pass/fail disagreement on the
    # follow-up-critic site; both not-redundant is agreement.
    def _fu_pair(claude_verdict, codex_verdict):
        return [
            _ev(
                FU,
                "2026-07-03T08:00:00Z",
                note=f"<!-- {FU} v1 -->\n**Verdict:** {claude_verdict}",
            ),
            _ev(
                FUX,
                "2026-07-03T08:03:00Z",
                note=f"<!-- {FUX} v1 -->\n**Verdict:** {codex_verdict}",
            ),
        ]

    out = tw.unreconciled_disagreement_rounds(
        _fu_pair("not-redundant", "redundant"), now_ts=_TIER1_NOW
    )
    assert len(out) == 1
    assert out[0]["role"] == "follow-up-critic"
    assert (out[0]["claude_class"], out[0]["codex_class"]) == ("pass", "fail")
    assert (
        tw.unreconciled_disagreement_rounds(
            _fu_pair("not-redundant", "not-redundant"), now_ts=_TIER1_NOW
        )
        == []
    )


def test_present_but_malformed_verdict_not_flagged():
    # (k) Both markers present at the round but one side's note has no
    # parseable Verdict field: NOT a disagreement (never fabricate a
    # verdict, #810 r4) — and because both sides ARE present at the round,
    # the pairing stays at Tier 1 (Tier 2 is not attempted; the earlier
    # parseable round-1 marker below must NOT be dredged up as a pair).
    events = [
        _ev(CR, "2026-07-03T07:00:00Z", note="<!-- epm:code-review v1 -->\n**Verdict:** PASS"),
        _ev(
            CR,
            "2026-07-03T08:00:00Z",
            version=2,
            note="<!-- epm:code-review v2 -->\nterse summary, no verdict field",
        ),
        _ev(
            CRX,
            "2026-07-03T08:03:00Z",
            version=2,
            note="<!-- epm:code-review-codex v2 -->\n**Verdict:** FAIL",
        ),
    ]
    assert tw.unreconciled_disagreement_rounds(events, now_ts=_TIER1_NOW) == []


def test_cross_round_tier2_pairing_is_deliberate():
    # (r) DELIBERATE-behavior documentation (§11-c/-e trade-off): when the
    # two kinds' latest markers name DIFFERENT rounds (sentinel drift),
    # Tier 2 pairs latest-of-each-kind by design — a Claude round-2 PASS
    # 30 min after a Codex round-1 FAIL flags as a proximity pair (the two
    # timestamps embedded in the key make it human-diagnosable).
    events = [
        _ev(
            CRX, "2026-07-03T08:00:00Z", note="<!-- epm:code-review-codex v1 -->\n**Verdict:** FAIL"
        ),
        _ev(
            CR,
            "2026-07-03T08:30:00Z",
            version=2,
            note="<!-- epm:code-review v2 -->\n**Verdict:** PASS",
        ),
    ]
    now = _epoch("2026-07-03T08:30:00Z") + 7200.0
    out = tw.unreconciled_disagreement_rounds(events, now_ts=now)
    assert len(out) == 1
    assert out[0]["tier"] == "proximity"
    assert out[0]["key"] == "code-reviewer|t2|2026-07-03T08:30:00Z|2026-07-03T08:00:00Z"


# ─── Verdict-token edge shapes (p) ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "pass_values", "fail_values", "expected"),
    [
        ("PASS", CR_PASS, CR_FAIL, "pass"),
        ("pass", CR_PASS, CR_FAIL, "pass"),
        ("CONCERNS", CR_PASS, CR_FAIL, "pass"),
        ("FAIL", CR_PASS, CR_FAIL, "fail"),
        ("REVISE (FAIL-class)", ("pass",), ("revise",), "fail"),
        # '**Verdict: PASS**' residue after parse_followup_note_field:
        ("PASS**", CR_PASS, CR_FAIL, "pass"),
        ("**PASS**", CR_PASS, CR_FAIL, "pass"),
        ("not-redundant", ("not-redundant",), ("redundant",), "pass"),
        ("redundant", ("not-redundant",), ("redundant",), "fail"),
        # Exact token match — 'not-redundant' must never substring-match
        # 'redundant' into a fail-class read:
        ("not-redundant", ("pass",), ("redundant",), None),
        ("MAYBE", CR_PASS, CR_FAIL, None),
        ("", CR_PASS, CR_FAIL, None),
        (None, CR_PASS, CR_FAIL, None),
    ],
)
def test_verdict_class_token_shapes(raw, pass_values, fail_values, expected):
    assert tw._verdict_class(raw, pass_values, fail_values) == expected


# ─── #825 golden fixture (q) ─────────────────────────────────────────────────


def test_issue825_golden_fixture():
    # The REAL disagreement-round rows from #825's events.jsonl, committed
    # verbatim: Codex v7 no-sentinel FAIL (08:33:30Z) + Claude sentinel-v5
    # PASS (08:35:50Z) + the corrective reconcile (sentinel v1, Round "1
    # (...)", role code-reviewer, 08:45:34Z). Strongest regression pin at
    # zero cost: pre-reconcile -> exactly one proximity-tier finding;
    # appending the real reconcile row -> zero findings.
    rows = [
        json.loads(line)
        for line in (FIXTURES / "issue825_verdict_disagree_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["kind"] for r in rows] == [CRX, CR, REC]
    now = _epoch("2026-07-03T10:00:00Z")
    pre = tw.unreconciled_disagreement_rounds(rows[:2], now_ts=now)
    assert len(pre) == 1
    assert pre[0]["tier"] == "proximity"
    assert pre[0]["role"] == "code-reviewer"
    assert (pre[0]["claude_class"], pre[0]["codex_class"]) == ("pass", "fail")
    assert tw.unreconciled_disagreement_rounds(rows, now_ts=now) == []


# ─── Site-table parity with workflow.yaml (l) ────────────────────────────────


def _load_workflow_yaml() -> dict:
    """yaml.safe_load first; omegaconf fallback (plan §5: the parity pin
    must ship in some form even if pyyaml leaves the env)."""
    path = REPO_ROOT / ".claude" / "workflow.yaml"
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except ImportError:  # pragma: no cover - env-dependent fallback
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def test_site_table_matches_workflow_yaml():
    # Drift alarm for the hardcoded ENSEMBLE_MARKER_MODE_SITES constant:
    # roles + lowercased vocabularies from § ensemble_review's marker-mode
    # doubled_steps, kinds from § reviewer_pairs.pairs.*.markers.
    wf = _load_workflow_yaml()
    doubled = wf["ensemble_review"]["doubled_steps"]
    marker_sites = {d["role"]: d for d in doubled if d.get("reconcile_mode") == "marker"}
    table = {s["role"]: s for s in tw.ENSEMBLE_MARKER_MODE_SITES}
    assert set(table) == set(marker_sites)
    for role, site in table.items():
        spec = marker_sites[role]
        assert tuple(v.lower() for v in spec["pass_values"]) == tuple(site["pass_values"]), role
        assert tuple(v.lower() for v in spec["fail_values"]) == tuple(site["fail_values"]), role
    pair_kind_tuples = {tuple(p["markers"]) for p in wf["reviewer_pairs"]["pairs"].values()}
    site_kind_tuples = {(s["claude_kind"], s["codex_kind"]) for s in tw.ENSEMBLE_MARKER_MODE_SITES}
    assert site_kind_tuples == pair_kind_tuples


# ─── Watcher pass (g, m, n, o) ───────────────────────────────────────────────


def _matured_tier1_disagreement() -> list[dict]:
    """A Tier-1 PASS-vs-FAIL pair matured 3h ago relative to wall-clock
    (the pass evaluates against time.time())."""
    now = datetime.now(tz=UTC)
    t_claude = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_codex = (now - timedelta(hours=3) + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _tier1_pair(t_claude=t_claude, t_codex=t_codex)


def _vdo_sandbox(tmp_path, monkeypatch, events, *, status="running", issue=321, patch_push=True):
    """Fully sandbox verdict_disagree_pass: tmp registry + task dir (fresh
    events.jsonl mtime), tmp state/sidecar singletons, recorded pushes.
    Returns (asw, reg_path, state_path, sidecar_path, pushes). Mirrors
    tests/test_autonomous_session_watch.py::_triage_observer_sandbox."""
    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    reg_root = tmp_path / "repo"
    task_rel = f"tasks/{status}/{issue}"
    task_dir = reg_root / task_rel
    task_dir.mkdir(parents=True)
    (task_dir / "events.jsonl").write_text("")  # fresh mtime; list_events is patched
    reg_path = reg_root / "tasks" / "REGISTRY.json"
    reg_path.write_text(
        json.dumps(
            {
                "tasks": {
                    str(issue): {
                        "status": status,
                        "path": task_rel,
                        "kind": "experiment",
                        "title": "synthetic",
                        "has_clean_result": False,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(task_workflow, "registry_path", lambda: reg_path)
    monkeypatch.setattr(task_workflow, "list_events", lambda _issue: list(events))
    state_path = tmp_path / "verdict-disagree-observer.json"
    sidecar_path = tmp_path / "verdict-disagree-observer-events.jsonl"
    monkeypatch.setattr(asw, "_verdict_disagree_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_verdict_disagree_sidecar_path", lambda: sidecar_path)
    pushes: list[tuple[str, bool]] = []
    if patch_push:
        monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry: pushes.append((msg, dry)))
    return asw, reg_path, state_path, sidecar_path, pushes


def test_kill_switch_skips_everything(tmp_path, monkeypatch, capsys):
    # (g) EPM_DISABLE_VERDICT_DISAGREE_OBSERVER=1 -> returns False before
    # any read/write (the registry is never touched; zero writes).
    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    monkeypatch.setenv("EPM_DISABLE_VERDICT_DISAGREE_OBSERVER", "1")
    state_path = tmp_path / "state.json"
    sidecar_path = tmp_path / "sidecar.jsonl"
    monkeypatch.setattr(asw, "_verdict_disagree_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_verdict_disagree_sidecar_path", lambda: sidecar_path)

    def _forbidden():
        raise AssertionError("registry must not be read under the kill switch")

    monkeypatch.setattr(task_workflow, "registry_path", _forbidden)
    assert asw.verdict_disagree_pass(dry_run=False) is False
    assert not state_path.exists() and not sidecar_path.exists()
    out = capsys.readouterr()
    assert "disabled via EPM_DISABLE_VERDICT_DISAGREE_OBSERVER" in out.out
    # The kill switch returns cleanly — the top-level fail-soft guard never
    # fired (its line would prove the registry probe ran).
    assert "pass failed (fail-soft)" not in out.err


def test_integration_fire_once_dedup_and_prune(tmp_path, monkeypatch):
    # (m) Tick 1 on a flagging fixture: one sidecar row + one push + a
    # state entry. Tick 2 on unchanged events: nothing new (fire-once via
    # the REAL state round-trip). Registry flip to completed: state entry
    # self-pruned.
    asw, reg_path, state_path, sidecar_path, pushes = _vdo_sandbox(
        tmp_path, monkeypatch, _matured_tier1_disagreement()
    )
    assert asw.verdict_disagree_pass(dry_run=False) is True
    rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["issue"] == 321
    assert rows[0]["role"] == "code-reviewer"
    assert rows[0]["tier"] == "round"
    assert len(pushes) == 1
    msg = pushes[0][0]
    assert "verdict-disagree-observer: #321" in msg
    assert "Claude PASS vs Codex FAIL" in msg
    # v2 strip-class disambiguator rides every push:
    assert "Blocker tags" in msg
    state = json.loads(state_path.read_text())
    assert state["321"]["flagged"] == [rows[0]["role"] + "|" + rows[0]["round_label"]]

    # Tick 2: unchanged events -> no new row / push (fire-once dedup).
    assert asw.verdict_disagree_pass(dry_run=False) is False
    assert len(sidecar_path.read_text().splitlines()) == 1
    assert len(pushes) == 1

    # Prune: the task leaves the sweep set for good -> entry dropped.
    reg = json.loads(reg_path.read_text())
    reg["tasks"]["321"]["status"] = "completed"
    reg["tasks"]["321"]["path"] = "tasks/completed/321"
    reg_path.write_text(json.dumps(reg))
    assert asw.verdict_disagree_pass(dry_run=False) is False
    assert "321" not in json.loads(state_path.read_text())


def test_dry_run_performs_zero_writes(tmp_path, monkeypatch):
    # (m) Backs the post-merge `--verdict-disagree-only --dry-run` smoke: a
    # dry-run must create no sidecar, write no state, spawn no subprocess.
    asw, _reg, state_path, sidecar_path, pushes = _vdo_sandbox(
        tmp_path, monkeypatch, _matured_tier1_disagreement()
    )
    calls: list = []
    monkeypatch.setattr(asw.subprocess, "run", lambda *a, **kw: calls.append(a))
    asw.verdict_disagree_pass(dry_run=True)
    assert calls == []
    assert not state_path.exists()
    assert not sidecar_path.exists()
    assert all(dry is True for _msg, dry in pushes)


def test_pass_never_invokes_task_mutation(tmp_path, monkeypatch):
    # (n) Non-gating pin at the subprocess-argv level (the triage-observer
    # tests' posture): with the REAL _telegram_push routed at a stub script,
    # the pass's ONLY subprocess is the push — never task.py / set-status /
    # post-marker / spawn_session.
    import subprocess as _subprocess

    asw, _reg, _state, _sidecar, _pushes = _vdo_sandbox(
        tmp_path, monkeypatch, _matured_tier1_disagreement(), patch_push=False
    )
    stub = tmp_path / "push-stub.sh"
    stub.write_text("#!/bin/bash\nexit 0\n")
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(stub))
    argvs: list[list[str]] = []

    def _fake_run(cmd, *a, **kw):
        argvs.append([str(c) for c in cmd])
        return _subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(asw.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        asw.subprocess, "check_output", lambda *a, **kw: pytest.fail("check_output invoked")
    )

    assert asw.verdict_disagree_pass(dry_run=False) is True
    assert argvs, "the flag's push must have fired (proves the pass emitted)"
    for cmd in argvs:
        joined = " ".join(cmd)
        assert "task.py" not in joined
        assert "set-status" not in joined
        assert "post-marker" not in joined
        assert "spawn_session" not in joined


def test_fail_soft_list_events_error_continues(tmp_path, monkeypatch):
    # (o) A per-issue evaluation failure skips THAT issue and continues —
    # issue 321 raises, issue 322 still flags.
    import autonomous_session_watch as asw

    from explore_persona_space import task_workflow

    reg_root = tmp_path / "repo"
    for issue in (321, 322):
        task_dir = reg_root / f"tasks/running/{issue}"
        task_dir.mkdir(parents=True)
        (task_dir / "events.jsonl").write_text("")
    reg_path = reg_root / "tasks" / "REGISTRY.json"
    reg_path.write_text(
        json.dumps(
            {
                "tasks": {
                    str(issue): {
                        "status": "running",
                        "path": f"tasks/running/{issue}",
                        "kind": "experiment",
                        "title": "synthetic",
                        "has_clean_result": False,
                    }
                    for issue in (321, 322)
                }
            }
        )
    )
    events = _matured_tier1_disagreement()

    def _list_events(issue):
        if issue == 321:
            raise RuntimeError("boom")
        return list(events)

    monkeypatch.setattr(task_workflow, "registry_path", lambda: reg_path)
    monkeypatch.setattr(task_workflow, "list_events", _list_events)
    state_path = tmp_path / "state.json"
    sidecar_path = tmp_path / "sidecar.jsonl"
    monkeypatch.setattr(asw, "_verdict_disagree_state_path", lambda: state_path)
    monkeypatch.setattr(asw, "_verdict_disagree_sidecar_path", lambda: sidecar_path)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry: None)

    assert asw.verdict_disagree_pass(dry_run=False) is True
    rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    assert [r["issue"] for r in rows] == [322]
    state = json.loads(state_path.read_text())
    assert "322" in state and "321" not in state


def test_fail_soft_predicate_error_never_propagates(tmp_path, monkeypatch):
    # (o) A raise inside the pure predicate is caught by the per-issue
    # guard — the pass returns False without propagating.
    from explore_persona_space import task_workflow

    asw, _reg, _state, sidecar_path, _pushes = _vdo_sandbox(
        tmp_path, monkeypatch, _matured_tier1_disagreement()
    )

    def _raise(*_a, **_kw):
        raise RuntimeError("predicate boom")

    monkeypatch.setattr(task_workflow, "unreconciled_disagreement_rounds", _raise)
    assert asw.verdict_disagree_pass(dry_run=False) is False
    assert not sidecar_path.exists()
