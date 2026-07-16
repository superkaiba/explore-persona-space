"""Spec-doc regression guard for the /daily three-route classifier (#706).

`/daily` is an LLM-driven SKILL.md, not a runtime surface, so the only
mechanical guard against a future cosmetic edit collapsing the three-route
classifier back to a binary apply-vs-hold split is a pure-text assertion that
the prose contract survives. The watcher invariant test
(`test_autonomous_session_watch.py::test_sweep_candidate_query_skips_needs_human`)
pins the runtime half; this file pins the prose half.

Minimal + durable by design — substring assertions, NOT structure parsing.
Each assertion is a string the round-2 implementation MUST keep in
`.claude/skills/daily/SKILL.md`:

* The THREE route labels exist (trivial mechanical / behavior-or-logic / a
  genuine judgment call) — proving the binary classifier was replaced.
* Route 2 wires to `file_infra_task.py` with `--tag daily-auto-filed`.
* Route 3 wires to `file_infra_task.py --no-dispatch` with both
  `--tag needs-human` and `--tag daily-held`.
* The 5-item judgment-call carve-out list survives verbatim (it is REUSED as
  the route-3 trigger, not re-authored).

At the TDD propose stage every assertion FAILs (the SKILL.md still carries
the binary two-bucket classifier); the round-2 implementation makes them pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_SKILL = REPO_ROOT / ".claude" / "skills" / "daily" / "SKILL.md"
RESEARCH_PM = REPO_ROOT / ".claude" / "agents" / "research-pm.md"


@pytest.fixture(scope="module")
def daily_skill_text() -> str:
    assert DAILY_SKILL.is_file(), f"daily SKILL.md not found at {DAILY_SKILL}"
    return DAILY_SKILL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def research_pm_text() -> str:
    assert RESEARCH_PM.is_file(), f"research-pm.md not found at {RESEARCH_PM}"
    return RESEARCH_PM.read_text(encoding="utf-8")


# ── the three route labels replace the binary classifier ──────────────────────


@pytest.mark.parametrize(
    "label",
    [
        "Trivial mechanical",  # route 1 — self-apply, no behavior change
        "behavior/logic change",  # route 2 — file for independent review
        "judgment call",  # route 3 — tracked needs-human task
    ],
)
def test_three_route_labels_present(daily_skill_text: str, label: str):
    assert label in daily_skill_text, (
        f"route label {label!r} missing from daily/SKILL.md — the three-route "
        "classifier may have regressed to a binary apply-vs-hold split"
    )


# ── route 2 files for review via file_infra_task.py + daily-auto-filed tag ─────


def test_route2_files_for_review(daily_skill_text: str):
    assert "file_infra_task.py" in daily_skill_text, (
        "route 2 must wire to scripts/file_infra_task.py (files + auto-dispatches "
        "behavior/logic changes to /issue --auto for independent review)"
    )
    assert "daily-auto-filed" in daily_skill_text, (
        "route 2 must tag filings with daily-auto-filed (distinguishes /daily "
        "review filings from manual workflow-fix-on-bug filings; feeds the PM digest count)"
    )


# ── route 3 files a tracked needs-human task (no dispatch) ─────────────────────


@pytest.mark.parametrize(
    "token",
    [
        "--no-dispatch",  # files at proposed WITHOUT spawning a session
        "needs-human",  # the PM-surfaced, auto-dispatch-excluded tag
        "daily-held",  # marks the held item as /daily-originated
    ],
)
def test_route3_files_tracked_needs_human_task(daily_skill_text: str, token: str):
    assert token in daily_skill_text, (
        f"route 3 must reference {token!r} — a /daily-held judgment call becomes a "
        "TRACKED proposed task (file_infra_task.py --no-dispatch --tag needs-human "
        "--tag daily-held), no longer a dead-end log note"
    )


# ── the 5-item carve-out list survives (reused verbatim as the route-3 trigger) ─


@pytest.mark.parametrize(
    "carve_out_anchor",
    [
        "Scientific-meaning changes",
        "Destructive / irreversible actions",
        "Spends money or launches compute",
        "External side-effects",
        "Genuinely ambiguous intent",
    ],
)
def test_carve_out_list_survives(daily_skill_text: str, carve_out_anchor: str):
    assert carve_out_anchor in daily_skill_text, (
        f"judgment-call carve-out item {carve_out_anchor!r} dropped — the 5-item "
        "list is REUSED verbatim as the route-3 trigger and must be preserved"
    )


# ── round-2: contract surfaces (frontmatter + Telegram template + PM digest) ───
#
# The round-1 reviewer reconciler FAILed because /daily's own canonical contract
# surfaces still advertised the pre-#706 auto-apply binary. These negative
# substring guards pin the round-2 fixes so a future cosmetic edit cannot
# silently re-introduce the stale "auto-apply everything" wording.


@pytest.mark.parametrize(
    "stale_phrase",
    [
        "AUTO-APPLIED by default",  # frontmatter — pre-#706 binary apply-everything
        "Only genuine judgment calls are held",  # frontmatter — binary hold-only-judgment
        # Telegram template — the old applied-only count (no route-2 filed field,
        # route-3 lumped as "Notes: M other"):
        "auto-applied N fix(es) (<w> workflow, <c> code/infra)",
        # Round-3: the residual self-apply contract for .claude/settings.json hook
        # changes (line ~258). It contradicted route 2, which routes hook repairs
        # through file_infra_task.py → /issue --auto; a behavior-changing hook edit
        # is NOT self-appliable under this skill's own verify gate.
        "hook FIXES are auto-appliable",
        # Round-3: the surfacing-flow section was renamed to
        # "Surfacing flow (applied / filed / held)"; the old cross-reference text
        # must not survive as a section reference.
        'see "Auto-apply + surfacing flow"',
        # Round-4: the daily-file BODY STUB (~line 113) still described the
        # pre-#706 binary output contract (`## Applied workflow improvements` =
        # AUTO-APPLIED only). Plan §2(b) + §4 require route-2 filings ALSO appear
        # there as "filed for review #<N>" entries — so the stub may no longer
        # advertise the section as auto-applied-only.
        "WORKFLOW-FIXABLE problems that were AUTO-APPLIED this run",
    ],
)
def test_stale_pre706_contract_phrases_gone(daily_skill_text: str, stale_phrase: str):
    assert stale_phrase not in daily_skill_text, (
        f"stale pre-#706 phrase {stale_phrase!r} survives in daily/SKILL.md — a "
        "contract surface (frontmatter description / Telegram template) regressed "
        "to the binary auto-apply model the three-route classifier replaced"
    )


def test_body_stub_describes_route2_filed_for_review(daily_skill_text: str):
    # Round-4: the daily-file body stub for `## Applied workflow improvements`
    # must describe the route-2 "filed for review #<N>" output (plan §2(b) + §4),
    # not just route-1 self-applied diffs. The `daily-auto-filed` tag is the
    # robust route-2 marker the PM digest counts (M), so the stub references it.
    assert "filed for review" in daily_skill_text, (
        "the daily-file body stub must describe the route-2 output entry "
        '("filed for review #<N>") in `## Applied workflow improvements` — '
        "route-2 filings are recorded there alongside route-1 self-applied fixes"
    )
    assert "daily-auto-filed" in daily_skill_text, (
        "route-2 filings carry the daily-auto-filed tag (feeds the PM digest "
        "count M); the body stub / triage must keep referencing it"
    )


def test_telegram_template_reports_applied_filed_held(daily_skill_text: str):
    # The Telegram digest must report applied/filed/held counts SEPARATELY,
    # matching the §2(f) PM-digest model — not lump route-2/route-3 as a single
    # "Notes: M other" catch-all.
    assert "filed M route-2" in daily_skill_text, (
        "Telegram template must report the route-2 filed-for-review count "
        "separately (`filed M route-2 review task(s)`), matching the PM digest"
    )
    assert "held J route-3" in daily_skill_text, (
        "Telegram template must report the route-3 held needs-human count "
        "separately (`held J route-3 (needs you)`), not a `Notes: M other` lump"
    )


# ── #1061: durable filing dir + incremental filing driver (routes 2 + 3) ──────
#
# The 2026-07-03 nightly died mid-filing with 10/13 route-2 bodies stranded in
# bare /tmp. #1061 prescribes a durable filings dir + the permanent incremental
# driver; these pins keep a future cosmetic edit from silently dropping that
# prescription while every other suite stays green.


def test_durable_filing_driver_prescribed(daily_skill_text: str):
    assert daily_skill_text.count("daily_drive_filings.py") >= 2, (
        "scripts/daily_drive_filings.py must be named in BOTH the route-2 command "
        "block (the multi-item prescription) and the dedicated durable-filing "
        "subsection — hand-looping file_infra_task.py for >1 item regressed"
    )
    assert "logs/daily/filings-" in daily_skill_text, (
        "the durable filings dir `logs/daily/filings-<date>/` must be prescribed — "
        "filing bodies staged in bare /tmp is the 2026-07-03 stranding incident"
    )


def test_durable_filing_subsection_heading_present(daily_skill_text: str):
    assert "Durable filing dir + incremental filing driver" in daily_skill_text, (
        "the 'Durable filing dir + incremental filing driver' subsection dropped — "
        "it carries the durable-first ordering + filed.jsonl ledger contract (#1061)"
    )


def test_dispatch_pending_record_survives(daily_skill_text: str):
    assert "dispatch pending" in daily_skill_text, (
        "the daily-file record must keep the 'dispatch pending' rendering for "
        "not-yet-terminal / spawn-deferred filings (headless rule 1 + the "
        "durable-filing subsection)"
    )


def test_pm_digest_reads_previous_night_not_newest_file(research_pm_text: str):
    # Fix 3: the PM digest must read the PREVIOUS night's PT-dated file, never
    # the "newest dated file" (which surfaces a stale "/daily last night" line on
    # any night /daily failed to run).
    assert "newest dated file" not in research_pm_text, (
        "research-pm.md still says 'newest dated file' for the /daily digest — it "
        "must read the previous night's PT date specifically and omit if absent, "
        "never fall back to an older daily file as 'last night'"
    )
    assert "date -d 'yesterday'" in research_pm_text, (
        "research-pm.md /daily digest must pin the previous night's PT date "
        "(`date -d 'yesterday' +%F` in America/Los_Angeles)"
    )


# ── #1131: retraction re-check (freshness gate before route-2/3 filings) ──────
#
# Incident #1101/#1074 (2026-07-06): the sweep filed #1101 from #1074's
# `epm:pod-terminated v1` prose follow-up while an explicit retraction
# (`epm:progress v26`) sat 37 seconds later on the same events.jsonl.
# Negative-control replay (reasoning-level, verified during planning): #1074's
# LATER rows (`epm:progress v27+`, watcher stall alerts) do NOT pass the
# binding test for that premise — the binding test, not the recall regex
# alone, is the decision rule.


@pytest.mark.parametrize(
    "token",
    [
        "Retraction re-check",
        "retracted upstream",
        "select(.ts > $ts)",
        "FILE anyway",  # the fail-open-on-ambiguity clause (§9-allowed extra pin)
    ],
)
def test_retraction_recheck_present(daily_skill_text: str, token: str):
    """#1131: the freshness gate before route-2/3 filings (incident #1101/#1074)
    must survive future SKILL.md edits."""
    assert token in daily_skill_text, (
        f"#1131 retraction re-check contract token {token!r} missing from daily "
        "SKILL.md — the pre-filing freshness gate (skip when the source trail "
        "retracted the mined premise) must stay in the route-2/3 filing steps."
    )


# ── #1173: route-2 wf-fix body Provenance mandate (durable recursion guard) ────


def test_route2_wf_fix_provenance_mandate_present(daily_skill_text: str):
    """#1173: every route-2 wf-fix body must carry the durable recursion-guard
    Provenance lines (`workflow_fix_target:` + `fingerprint:`) — the env-var leg
    is absent from `spawn-issue --auto` spawns and lost on watcher respawns
    (incident #1134). This pins the SKILL.md mandate paragraph the driver's
    mechanical injection backstops."""
    assert "wf-fix body Provenance mandate" in daily_skill_text, (
        "the route-2 'wf-fix body Provenance mandate' paragraph dropped from daily "
        "SKILL.md — daily-filed wf-fix bodies would regress to carrying no durable "
        "recursion-guard signal (#1173, incident #1134)"
    )
    assert "- workflow_fix_target: <target_file>" in daily_skill_text, (
        "the literal `- workflow_fix_target: <target_file>` Provenance-line template "
        "dropped from daily SKILL.md — the durable signal "
        "task_workflow.is_workflow_fix_session() reads must stay mandated (#1173)"
    )


# ── #1228: route-2 wf_fix: false variant (non-workflow-surface daily filings) ──


def test_route2_wf_fix_false_variant_documented(daily_skill_text: str):
    """#1228: the route-2 non-workflow-surface variant (drop the wf-fix tags,
    keep daily-auto-filed) must stay documented WITH its driver mechanism —
    the `wf_fix: false` manifest key `daily_drive_filings.py` gates on. Without
    the mechanism sentence the variant is unreachable on the batch driver path
    (the driver would re-tag + re-inject unconditionally, the #1228 bug)."""
    assert "(drop the `wf-fix` / `wf-fix-fp` tags, keep `daily-auto-filed`)" in daily_skill_text, (
        "the route-2 drop-tags contract sentence dropped from daily SKILL.md — "
        "experiment-code (non-workflow-surface) daily filings would regress to "
        "carrying wf-fix dedup/recursion-guard tags (#1228)"
    )
    assert "wf_fix: false" in daily_skill_text, (
        "the `wf_fix: false` manifest mechanism dropped from daily SKILL.md — "
        "the drop-tags route-2 variant would be prose-only again, unreachable "
        "through the batch driver daily_drive_filings.py (#1228)"
    )


# ── Step D: nightly title-sync drift sweep invoker (#1196 → #1235) ────────────


def test_title_sync_sweep_pass_present(daily_skill_text: str):
    """The nightly /daily invoker for the #1196 title-sync drift sweep survives
    (task #1235): the command, its WARN-only/exit-0 contract, and the
    surface-only routing must not be silently dropped by a later edit."""
    assert (
        "scripts/audit_clean_results_body_discipline.py --title-sync-sweep" in daily_skill_text
    ), "the Step D title-sync sweep command is missing from daily/SKILL.md"
    assert "title-sync drift sweep (Step D)" in daily_skill_text, (
        "the Step D pass header is missing — the nightly title-sync invoker "
        "(#1235) may have been dropped"
    )


# ── #1272: verified-at-filing grep-evidence line (rule template + route 2) ─────


def test_verified_at_filing_line_required(daily_skill_text: str):
    """#1272 pin: the verified-at-filing filing-time grep-evidence mandate
    survives in the daily route-2 text, the rule's Body-file template, and the
    workflow.yaml orchestrator_actions prose (#1221/#1229/#1249: three
    stale-claim filings in two days, each burning a spawned session's
    verification rounds). #1307 extends the pin: the grep must BIND to the
    claim (per-target confirmation + relocation grep) on all three surfaces."""
    assert "verified-at-filing:" in daily_skill_text, (
        "the route-2 'verified-at-filing mandate' sentence dropped from daily "
        "SKILL.md — daily-filed wf-fix bodies would regress to carrying no "
        "filing-time grep evidence (#1272)"
    )
    rule_text = (REPO_ROOT / ".claude" / "rules" / "workflow-fix-on-bug.md").read_text(
        encoding="utf-8"
    )
    assert rule_text.count("verified-at-filing:") >= 3, (
        "workflow-fix-on-bug.md must keep >=3 'verified-at-filing:' occurrences "
        "(template line + Before-emitting sentence + anti-pattern row) (#1272)"
    )
    assert "n/a — " in rule_text, (
        "the 'n/a — <reason>' escape for non-grep-able bug claims dropped from "
        "workflow-fix-on-bug.md (#1272)"
    )
    yaml_text = (REPO_ROOT / ".claude" / "workflow.yaml").read_text(encoding="utf-8")
    assert "verified-at-filing:" in yaml_text, (
        "workflow.yaml orchestrator_actions no longer mention the "
        "verified-at-filing: grep-evidence line (#1272)"
    )
    # #1307: the binding tightening — per-target confirmation + relocation grep
    assert "EACH file named in" in rule_text, (
        "the #1307 per-target-confirmation binding rule dropped from "
        "workflow-fix-on-bug.md (a 0-hit named target must be a mis-target)"
    )
    assert "relocation grep" in rule_text, (
        "the #1307 relocation-grep binding rule dropped from "
        "workflow-fix-on-bug.md (nonexistence claims need a repo-wide grep)"
    )
    assert "relocation grep" in daily_skill_text, (
        "the #1307 binding clause dropped from the daily route-2 "
        "verified-at-filing mandate sentence"
    )
    assert "relocation grep" in yaml_text, (
        "the #1307 binding clause dropped from the workflow.yaml orchestrator_actions grep step"
    )
    # v2 (Statistics-critic suggestion): pin rule (a) on the two compact
    # surfaces too — without these, a future editor could drop the
    # per-target clause from daily/yaml without failing the test.
    assert "per-target hits" in daily_skill_text, (
        "the #1307 per-target-confirmation clause dropped from the daily "
        "route-2 verified-at-filing mandate sentence"
    )
    assert "per-target hits" in yaml_text, (
        "the #1307 per-target-confirmation clause dropped from the "
        "workflow.yaml orchestrator_actions grep step"
    )


def test_context_consistency_clause_present(daily_skill_text: str):
    """#1383 pin: the context-consistency binding clause survives — a
    presence hit whose surrounding text already implements the proposed
    change is a landed fix (dedup, don't file; #1330 filed over the
    landed #1309 fix) — in the rule's Body-file template and the daily
    route-2 verified-at-filing mandate sentence."""
    rule_text = (REPO_ROOT / ".claude" / "rules" / "workflow-fix-on-bug.md").read_text(
        encoding="utf-8"
    )
    assert "context consistency" in rule_text, (
        "the #1383 context-consistency clause dropped from "
        "workflow-fix-on-bug.md (a presence hit that IS the landed fix "
        "must route to dedup, not a new filing)"
    )
    assert "binds on CONTEXT" in daily_skill_text, (
        "the #1383 context-binding sentence dropped from the daily "
        "route-2 verified-at-filing mandate paragraph"
    )


def test_semantic_probe_absence_clause_present(daily_skill_text: str):
    """#1420 pin: the semantic-probe clause (a') for ABSENCE claims about
    text-matching guards survives — a verbatim-literal 0-hit grep alone is
    not absence evidence; require running the predicate against the claimed
    text and/or repo-wide fragment/substring greps, plus a landed-fix
    history check (#1386 filed+spawned ~9h after #1360 landed the shorter
    'queue size reached' substring in orchestrate/hub.py) — in the rule's
    consistency-BINDS clause (a'), the daily route-2 mandate sentence, and
    the workflow.yaml orchestrator_actions grep step."""
    rule_text = (REPO_ROOT / ".claude" / "rules" / "workflow-fix-on-bug.md").read_text(
        encoding="utf-8"
    )
    assert "semantic probe" in rule_text, (
        "the #1420 semantic-probe clause (a') dropped from "
        "workflow-fix-on-bug.md — a verbatim-literal 0-hit grep would again "
        "count as absence evidence for a text-matching guard"
    )
    assert "--since='7 days ago'" in rule_text, (
        "the #1420 landed-fix history check (git log --since) dropped from "
        "workflow-fix-on-bug.md clause (a')"
    )
    assert "semantic probe" in daily_skill_text, (
        "the #1420 semantic-probe sentence dropped from the daily route-2 "
        "verified-at-filing mandate paragraph"
    )
    yaml_text = (REPO_ROOT / ".claude" / "workflow.yaml").read_text(encoding="utf-8")
    assert "semantic probe" in yaml_text, (
        "the #1420 semantic-probe clause dropped from the workflow.yaml "
        "orchestrator_actions grep step"
    )
