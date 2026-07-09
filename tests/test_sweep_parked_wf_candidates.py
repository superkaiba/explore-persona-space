"""Tests for scripts/sweep_parked_wf_candidates.py — the /daily Step C enumerator.

Exercises the REAL module body (``sweep()`` / ``main(argv)``) against synthetic
tmp task trees + tmp cache files via the documented ``--tasks-root`` /
``--cache-file`` overrides (no fakes of production functions; the only fixture
is the filesystem input, per the #906 body-coverage discipline). The 15-case
matrix mirrors plan §7 of task #1132.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sweep_parked_wf_candidates as spc  # noqa: E402

from explore_persona_space.task_workflow import wf_fix_fingerprint  # noqa: E402

CAND_KIND = "epm:workflow-fix-candidate"
FILED_KIND = "epm:workflow-fix-task-filed"

T0 = "2026-07-07T07:34:50Z"
T1 = "2026-07-07T08:00:00Z"
T2 = "2026-07-07T09:00:00Z"


def cand_row(ts: str, note: str, kind: str = CAND_KIND) -> dict:
    return {"ts": ts, "kind": kind, "version": 1, "by": "unknown", "note": note}


def filed_row(ts: str, note: str) -> dict:
    return {"ts": ts, "kind": FILED_KIND, "version": 1, "by": "unknown", "note": note}


def block_note(target_file: str, bug: str, change: str) -> str:
    """A recursion-guard park note embedding a formal candidate block (#1101 shape)."""
    return (
        "parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, "
        "see workflow-fix-on-bug.md § Recursion guard.\n\n"
        "<!-- workflow-fix-candidate v1 -->\n"
        f"target_file: {target_file}\n"
        f"bug_observed: {bug}\n"
        "why_workflow_gap: the workflow surface lacks the guardrail\n"
        f"proposed_change: {change}\n"
        "diff_sketch: |\n  + add the guardrail\n"
        "confidence: medium\n"
        "related_task: #999\n"
        "<!-- /workflow-fix-candidate -->\n"
    )


PROSE_NOTE = (
    "parked — running under workflow_fix_target recursion guard (see "
    ".claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidate surfaced by the "
    "reconciler: the wrapper discards success-stderr. Proposed change: forward it. "
    "target_file: scripts/codex_task.py. confidence: medium. related_task: #1100. "
    "NOT auto-filed (recursion guard); next human/orchestrator pass routes it."
)


def make_task(
    root: Path,
    tid: int,
    status: str,
    *,
    kind: str = "infra",
    title: str = "some task",
    body_extra: str = "",
    events: list[dict] | tuple = (),
    raw_event_lines: list[str] | tuple = (),
) -> Path:
    """Write a minimal tasks/<status>/<tid>/ fixture (body.md + events.jsonl)."""
    d = root / status / str(tid)
    d.mkdir(parents=True)
    fm = yaml.safe_dump({"kind": kind, "title": title}, sort_keys=False)
    (d / "body.md").write_text(f"---\n{fm}---\n\n## Overview\n\n{body_extra}\n", encoding="utf-8")
    lines = [json.dumps(r) for r in events]
    lines.extend(raw_event_lines)
    (d / "events.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return d


def run_sweep(root: Path, cache: Path | None = None, **kw) -> dict:
    return spc.sweep(root, cache, **kw)


def only(result: dict) -> dict:
    assert len(result["candidates"]) == 1, result["candidates"]
    return result["candidates"][0]


# ── 1. formal-block park → enumerated with computed fp ─────────────────────


def test_formal_block_park_enumerated_with_fp(tmp_path: Path) -> None:
    bug = "The sweep mined the claim but not its correction."
    change = "Scan subsequent events for a retraction before filing."
    make_task(
        tmp_path,
        1101,
        "archived",
        events=[cand_row(T0, block_note(".claude/skills/daily/SKILL.md", bug, change))],
    )
    c = only(run_sweep(tmp_path))
    assert c["source"] == "task:1101"
    assert c["formal_block"] is True
    assert c["target_file"] == ".claude/skills/daily/SKILL.md"
    assert c["fingerprint"] == wf_fix_fingerprint(change, bug)
    assert c["park_form"] == "recursion-guard"
    assert c["suppressed"] is False


# ── 2. prose park (#1100 shape) → fp null, target_file parsed (+ advisory) ─


def test_prose_park_enumerated_fp_null_target_parsed(tmp_path: Path) -> None:
    make_task(tmp_path, 1100, "completed", events=[cand_row(T0, PROSE_NOTE)])
    make_task(
        tmp_path,
        50,
        "running",
        title="workflow-fix: forward success-stderr",
        body_extra="## Provenance\n\n- workflow_fix_target: scripts/codex_task.py\n",
        events=[{"ts": T1, "kind": "epm:created", "note": "x"}],
    )
    c = only(run_sweep(tmp_path))
    assert c["fingerprint"] is None
    assert c["formal_block"] is False
    # trailing punctuation stripped from the prose regex capture
    assert c["target_file"] == "scripts/codex_task.py"
    # the advisory sees the open workflow-fix: task
    assert c["open_wf_fix_on_file"] == 50


def test_advisory_sees_daily_fix_titled_open_filing(tmp_path: Path) -> None:
    """#1180: the advisory mirror is no longer blind to daily-fix: titles."""
    make_task(tmp_path, 1100, "completed", events=[cand_row(T0, PROSE_NOTE)])
    make_task(
        tmp_path,
        51,
        "running",
        title="daily-fix: forward success-stderr",
        body_extra="## Provenance\n\n- workflow_fix_target: scripts/codex_task.py\n",
        events=[{"ts": T1, "kind": "epm:created", "note": "x"}],
    )
    c = only(run_sweep(tmp_path))
    assert c["open_wf_fix_on_file"] == 51


# ── 3. later same-stream filed record with MATCHING fp → suppressed ────────


def test_same_stream_filed_with_matching_fp_suppresses(tmp_path: Path) -> None:
    bug, change = "bug one.", "change one."
    fp = wf_fix_fingerprint(change, bug)
    make_task(
        tmp_path,
        7,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(T1, f"filed_task: #1131 / target_file: a/b.md / fingerprint: {fp}"),
        ],
    )
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#1131"}


# ── 4. fp-less candidate + legacy record (no origin ts) → target_file match ─


def test_fp_less_candidate_suppressed_by_target_file_only_record(tmp_path: Path) -> None:
    make_task(
        tmp_path,
        8,
        "completed",
        events=[
            cand_row(T0, PROSE_NOTE),
            filed_row(T1, "filed_task: #1130 / target_file: scripts/codex_task.py"),
        ],
    )
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"]["kind"] == "same-stream-filed"


# ── 5. open infra task with wf-fix-fp tag in body → suppressed ─────────────


def test_open_infra_task_with_fp_tag_suppresses(tmp_path: Path) -> None:
    bug, change = "bug two.", "change two."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 9, "archived", events=[cand_row(T0, block_note("a/b.md", bug, change))])
    make_task(tmp_path, 60, "running", body_extra=f"tags carry wf-fix-fp:{fp} here")
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-open", "ref": "#60"}


# ── 6. terminal fp-tag task: created AFTER → suppressed; BEFORE → not ───────


def test_terminal_fp_task_created_after_candidate_suppresses(tmp_path: Path) -> None:
    bug, change = "bug three.", "change three."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 10, "archived", events=[cand_row(T0, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        61,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[{"ts": T2, "kind": "epm:created", "note": "created after the park"}],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#61"}


def test_terminal_fp_task_created_before_candidate_is_re_raise(tmp_path: Path) -> None:
    bug, change = "bug four.", "change four."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 11, "archived", events=[cand_row(T2, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        62,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[{"ts": T0, "kind": "epm:created", "note": "created before the park"}],
    )
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


# ── 7. window filter: excluded at --window-days 2, included at 0 ────────────


def test_window_filter_excludes_old_candidate(tmp_path: Path) -> None:
    old_ts = "2026-01-01T00:00:00Z"
    make_task(tmp_path, 12, "completed", events=[cand_row(old_ts, PROSE_NOTE)])
    assert run_sweep(tmp_path, window_days=2)["candidates"] == []
    assert len(run_sweep(tmp_path, window_days=0)["candidates"]) == 1


# ── 8. non-park routed candidate + mid-note "parked" → NOT enumerated ──────


def test_non_park_and_mid_note_parked_not_enumerated(tmp_path: Path) -> None:
    make_task(
        tmp_path,
        13,
        "running",
        events=[
            cand_row(T0, "routed: filed #123 — target_file: a/b.md"),
            cand_row(T1, "routing note: stopped-on-parked-task cleanup ran; nothing parked here"),
        ],
    )
    assert run_sweep(tmp_path, include_routed=True)["candidates"] == []


# ── 9. identical duplicate rows dedupe to one (#1100 duplication) ───────────


def test_duplicate_rows_dedupe_to_one(tmp_path: Path) -> None:
    row = cand_row(T0, PROSE_NOTE)
    make_task(tmp_path, 1100, "completed", events=[row, row])
    assert len(run_sweep(tmp_path)["candidates"]) == 1


# ── 10. cache-file park enumerated; cache-file filed row suppresses ─────────


def test_cache_file_park_and_filed_suppression(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    cache = tmp_path / "workflow-fix-events.jsonl"
    bug, change = "cache bug.", "cache change."
    fp = wf_fix_fingerprint(change, bug)
    cache.write_text(json.dumps(cand_row(T0, block_note("c/d.md", bug, change))) + "\n")
    c = only(run_sweep(root, cache))
    assert c["source"] == "cache"
    assert c["fingerprint"] == fp
    with cache.open("a") as fh:
        fh.write(json.dumps(filed_row(T1, f"filed_task: #77 / fingerprint: {fp}")) + "\n")
    assert run_sweep(root, cache)["candidates"] == []


# ── 11. heterogeneous cache rows: kind-less, marker-key parked / filed ──────


def test_heterogeneous_cache_rows(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    cache = tmp_path / "workflow-fix-events.jsonl"
    rows = [
        # kind-less row (no 'kind'/'marker' key) — must not crash the scan
        {"ts": T0, "note": "misc row about a parked pod, no kind key"},
        # marker-key STRUCTURED candidate, routed contains 'parked' → enumerated
        {
            "ts": T1,
            "marker": "epm:workflow-fix-candidate v1",
            "target_file": "e/f.md",
            "proposed_change": "structured change.",
            "bug_observed": "structured bug.",
            "routed": "parked: EPM_WORKFLOW_FIX_SESSION",
            "source": "candidate-block",
        },
        # marker-key structured candidate already routed → NOT enumerated
        {
            "ts": T2,
            "marker": "epm:workflow-fix-candidate v1",
            "target_file": "g/h.md",
            "proposed_change": "other change.",
            "bug_observed": "other bug.",
            "routed": "filed #456",
            "source": "candidate-block",
        },
    ]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = run_sweep(root, cache, include_routed=True)
    c = only(result)
    assert c["target_file"] == "e/f.md"
    assert c["fingerprint"] == wf_fix_fingerprint("structured change.", "structured bug.")
    assert c["formal_block"] is False
    assert result["skipped_rows"] == 0


# ── 12. fp mismatch: file-matching record with a DIFFERENT fp ≠ suppression ─


def test_fp_mismatch_record_does_not_suppress(tmp_path: Path) -> None:
    bug, change = "bug five.", "change five."
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        14,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(T1, f"filed_task: #90 / target_file: a/b.md / fingerprint: {other_fp}"),
        ],
    )
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


# ── 13. two distinct fp-less parks, same file: ts-keyed record hits ONLY A ──


def test_ts_keyed_record_suppresses_only_matching_fp_less_park(tmp_path: Path) -> None:
    park_a = cand_row(T0, PROSE_NOTE)
    park_b = cand_row(T1, PROSE_NOTE.replace("forward it", "a DISTINCT second bug"))
    record = filed_row(
        T2,
        "filed_task: #91 / target_file: scripts/codex_task.py / "
        f"fingerprint: n/a (prose park) / origin_candidate_ts: {T0}",
    )
    make_task(tmp_path, 15, "completed", events=[park_a, park_b, record])
    result = run_sweep(tmp_path, include_routed=True)
    by_ts = {c["ts"]: c for c in result["candidates"]}
    assert by_ts[T0]["suppressed"] is True
    assert by_ts[T1]["suppressed"] is False


# ── 14. mixed ts formats compare under aware-UTC, not string order ──────────


def test_mixed_ts_formats_aware_utc_ordering(tmp_path: Path) -> None:
    bug, change = "bug six.", "change six."
    fp = wf_fix_fingerprint(change, bug)
    # candidate at 09:00 -07:00 == 16:00Z; the filed record at 10:00Z is
    # EARLIER in UTC (string comparison would misorder it as later) → no
    # suppression; a second record at 17:00Z IS later → suppression.
    cand = cand_row("2026-07-07T09:00:00-07:00", block_note("a/b.md", bug, change))
    early = filed_row("2026-07-07T10:00:00Z", f"filed_task: #92 / fingerprint: {fp}")
    make_task(tmp_path, 16, "completed", events=[cand, early])
    assert only(run_sweep(tmp_path))["suppressed"] is False

    late = filed_row("2026-07-07T17:00:00Z", f"filed_task: #93 / fingerprint: {fp}")
    make_task(tmp_path / "b", 17, "completed", events=[cand, early, late])
    c = only(run_sweep(tmp_path / "b", include_routed=True))
    assert c["suppressed"] is True


# ── 15. malformed rows: skipped + counted, CLI exit 0 ───────────────────────


def test_malformed_rows_skipped_counted_exit_0(tmp_path: Path, capsys) -> None:
    root = tmp_path / "tasks"
    make_task(
        root,
        18,
        "running",
        events=[cand_row(T0, PROSE_NOTE), {"kind": CAND_KIND, "note": "parked — but no ts"}],
        raw_event_lines=["{this is not json"],
    )
    cache = tmp_path / "cache.jsonl"
    cache.write_text("also not json\n")
    rc = spc.main(["--tasks-root", str(root), "--cache-file", str(cache)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # 1 corrupt task line + 1 ts-less candidate row + 1 corrupt cache line
    assert out["skipped_rows"] == 3
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["ts"] == T0


# ── 16. raw U+2028 inside a note must not shred the JSONL row (#950 gotcha) ─


def test_u2028_note_parses_as_one_row_enumerates_and_suppresses(tmp_path: Path) -> None:
    """A VALID JSONL row whose note carries a literal U+2028 (as marker notes
    written with ensure_ascii=False do — two live rows in tasks/completed/1032)
    must (a) parse as ONE row, (b) enumerate the parked candidate, and
    (c) leave skipped_rows at 0. splitlines() shredded it pre-fix."""
    sep = "\u2028"  # LINE SEPARATOR, escape form so no invisible char lives in source
    bug, change = "bug with separator prose.", "change with separator prose."
    fp = wf_fix_fingerprint(change, bug)
    note = block_note("a/b.md", bug, change).replace(
        "see workflow-fix-on-bug.md", f"see{sep}workflow-fix-on-bug.md"
    )
    raw = json.dumps(cand_row(T0, note), ensure_ascii=False)
    assert sep in raw and "\n" not in raw  # one physical line carrying a literal U+2028
    make_task(tmp_path, 19, "completed", raw_event_lines=[raw])
    result = run_sweep(tmp_path)
    assert result["skipped_rows"] == 0
    c = only(result)
    assert c["suppressed"] is False
    assert c["fingerprint"] == fp

    # Twin: a U+2028-bearing FILED record still suppresses (rule 1) — a shredded
    # filed record would otherwise let the candidate re-enumerate (double-file).
    filed_note = f"routing{sep}record / filed_task: #94 / fingerprint: {fp}"
    filed_raw = json.dumps(filed_row(T1, filed_note), ensure_ascii=False)
    assert sep in filed_raw
    make_task(tmp_path / "b", 20, "completed", raw_event_lines=[raw, filed_raw])
    result_b = run_sweep(tmp_path / "b", include_routed=True)
    assert result_b["skipped_rows"] == 0
    c_b = only(result_b)
    assert c_b["suppressed"] is True
