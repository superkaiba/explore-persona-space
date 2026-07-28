"""Tests for scripts/sweep_parked_wf_candidates.py — the /daily Step C enumerator.

Exercises the REAL module body (``sweep()`` / ``main(argv)``) against synthetic
tmp task trees + tmp cache files via the documented ``--tasks-root`` /
``--cache-file`` overrides (no fakes of production functions; the only fixture
is the filesystem input, per the #906 body-coverage discipline). The 15-case
matrix mirrors plan §7 of task #1132; section 17 adds the #1248 regressions
(n/a-fp disposition records closing fp-computable candidates), built from
byte-verbatim #815/#880/#917 marker rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
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
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#61", "basis": "creation"}


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
    # #1680: every skip is described by a structured record (no truncation here)
    assert len(out["skipped"]) == out["skipped_rows"]
    for entry in out["skipped"]:
        assert set(entry) == {"source", "path", "line_no", "reason", "kind_hint", "relevant_kind"}
    assert {e["source"] for e in out["skipped"]} == {"task:18", "cache"}


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


# ── 17. #1248: n/a-fp disposition records close fp-computable candidates ────
#
# Byte-verbatim fixture rows: each _RAW_1248_* constant is the EXACT JSONL line
# from the live #815/#880/#917 events.jsonl (the 2026-07-09 daily posted
# `fingerprint: n/a (prose park)` records against formal-block candidates;
# pre-#1248 they never suppressed, so the three candidates re-enumerated every
# night). Consumed via make_task(raw_event_lines=...), so the sweep parses the
# same bytes the live tree carries.

_RAW_1248_CAND_815 = r"""{"ts": "2026-07-01T22:36:29Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "PARKED — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). NOT auto-routed by this session; surfaced for the next human/orchestrator pass (/daily backstop or PM triage).\n\n<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/guard_repo_root_branch.sh\nbug_observed: The PreToolUse hook's detector (line ~102: `grep -qE '\\bgit\\b.*\\b(checkout|switch)\\b'`) matches only `checkout`/`switch` — it does NOT intercept a repo-root `git reset --hard` at runtime, which is exactly how the #778 analyzer's improvised reset executed and clobbered concurrent siblings #812/#813 (2026-07-01).\nwhy_workflow_gap: Task #815 shipped the docs + spec-drift layer (lint check + analyzer.md hard rule + CLAUDE.md line-39 extension), which covers spec drift and instructed behavior — but the #778 failure vector was RUNTIME IMPROVISATION with no spec instruction, which only the PreToolUse hook can block. All 6 Phase-2 critics + the code reviewer on #815 independently named this the higher-leverage remaining fix (\"the durable runtime belt to #815's spec-drift suspenders\").\nproposed_change: Extend the guard_repo_root_branch.sh detector so a repo-root-cwd `git reset --hard` (and consider the bulk-destructive siblings `git clean -f` / `git checkout .` / `git restore .`) is blocked like a branch switch, while NOT flagging worktree-qualified `git -C <path> reset --hard` and NOT flagging surgical path-scoped `git checkout <ref> -- <path>` (live legitimate pattern in issue/SKILL.md Step 5a/10d — see #815 fact-checker adjacent-finding 1).\ndiff_sketch: |\n  # scripts/guard_repo_root_branch.sh ~line 102\n  - echo \"$cmd\" | grep -qE '\\bgit\\b.*\\b(checkout|switch)\\b' || exit 0\n  + echo \"$cmd\" | grep -qE '\\bgit\\b.*\\b(checkout|switch|reset[[:space:]].*--hard)\\b' || exit 0\n  + # then, in the existing repo-root-cwd branch: allow `git -C <non-root path> ...`\n  + # and path-scoped `git checkout <ref> -- <path>` forms; block bare repo-root\n  + # `git reset --hard` with a loud error naming the per-worktree pattern.\nconfidence: high\nrelated_task: #815\n<!-- /workflow-fix-candidate -->\n"}"""  # noqa: E501
_RAW_1248_REC_815 = r"""{"ts": "2026-07-09T07:02:42Z", "kind": "epm:workflow-fix-task-filed", "version": 1, "by": "daily-2026-07-08", "note": "filed_task: n/a (already-fixed on main: guard_repo_root_branch.sh line 523 now matches 'checkout|switch|restore|clean|reset|merge' with dedicated reset/clean/restore block arms (lines 12-13,) / target_file: scripts/guard_repo_root_branch.sh / fingerprint: n/a (prose park) / session_spawned: False / source: daily-parked-candidate-sweep / origin_candidate_ts: 2026-07-01T22:36:29Z / origin_candidate: guard_repo_root_branch.sh line 523 now matches 'checkout|switch|restore|clean|reset|merge' with dedicated reset/clean/restore block arms (lines 12-13, 75, 630-639); commits since 2026-06-30 (bbc007bf9"}"""  # noqa: E501
_RAW_1248_CAND_880 = r"""{"ts": "2026-07-03T00:09:28Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "parked — running under workflow_fix_target recursion guard, see .claude/rules/workflow-fix-on-bug.md § Recursion guard. LOGGED + surfaced, NOT auto-routed by this session.\n\n<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue667_extract.py (and _device()-style descendants — NOTE: experiment scripts, OUT of the workflow-fix surface; this is an experiment-code follow-up, not a workflow-fix task)\nbug_observed: a hand-launch of a launcher-pinned per-cell worker with only --gpu-id silently targets the busy default GPU and crashes at vLLM init_device (#813, 2026-07-02) — the failure is only prose-guarded after #880.\nwhy_workflow_gap: (borderline/out-of-scope by target) the worker class has no runtime guard; a fail-loud assert would bind on ALL launch paths, prose read or not.\nproposed_change: _device() (scripts/issue667_extract.py::_device) asserts CUDA_VISIBLE_DEVICES is set in the environment when gpu_id > 0, raising a self-explaining error naming the env-pin recipe instead of silently returning cuda:0.\ndiff_sketch: |\n  def _device(gpu_id, cpu_only):\n      ...\n  +   if not cpu_only and gpu_id > 0 and \"CUDA_VISIBLE_DEVICES\" not in os.environ:\n  +       raise RuntimeError(\n  +           f\"--gpu-id {gpu_id} is informational in this worker class; launch with \"\n  +           f\"env CUDA_VISIBLE_DEVICES={gpu_id} ... --gpu-id {gpu_id} (see gotchas.md manual-launch CVD entry)\")\n      return torch.device(\"cuda:0\")\nconfidence: medium\nrelated_task: #880 (raised by both alternatives reviewers + the reconciler as the stronger structural fix, out of the #880 workflow-fix surface)\n<!-- /workflow-fix-candidate -->\n"}"""  # noqa: E501
_RAW_1248_REC_880 = r"""{"ts": "2026-07-09T07:02:37Z", "kind": "epm:workflow-fix-task-filed", "version": 1, "by": "daily-2026-07-08", "note": "filed_task: n/a (already-fixed on main: The proposed fail-loud guard landed anyway: scripts/issue667_extract.py L449-458 raises when --gpu-id is set without CUDA_VISIBLE_DEVICES, naming the ) / target_file: scripts/issue667_extract.py (and _device()-style descendants — NOTE: experiment scripts, OUT of the workflow-fix surface; this is an experiment-code follow-up, not a workflow-fix task) / fingerprint: n/a (prose park) / session_spawned: False / source: daily-parked-candidate-sweep / origin_candidate_ts: 2026-07-03T00:09:28Z / origin_candidate: The proposed fail-loud guard landed anyway: scripts/issue667_extract.py L449-458 raises when --gpu-id is set without CUDA_VISIBLE_DEVICES, naming the env-pin recipe. (Target was also self-declared out"}"""  # noqa: E501
_RAW_1248_CAND_917 = r"""{"ts": "2026-07-03T09:54:34Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this session is a workflow-fix session; candidate LOGGED for the next orchestrator pass, NOT auto-filed)\nsource: prose-followup (planner § Follow-ups + Claude code-reviewer bug-class sweep, both confirmed the line)\n\n<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/poll_pipeline.py\nbug_observed: The synthesized-envelope fallback (~lines 846-856) pins \"kind\": \"epm:results\", \"version\": 1 in code; on a follow-up round's re-run or a re-drained sentinel this reproduces the same version-collision class (#389/#825) in the poller path.\nwhy_workflow_gap: The prose surfaces now defer to max+1 (#917), but this code path still hardcodes version 1 — the last checked-in literal for a round-versioned kind, outside #917's declared scope (poll_pipeline.py was must-not-touch).\nproposed_change: The synthesized envelope should omit the version (let task_workflow.post_event derive max+1) or compute max(existing)+1 for the kind; needs its own analysis of multipart/pointer-marker + sentinel-schema interactions.\ndiff_sketch: |\n  - envelope = {\"kind\": \"epm:results\", \"version\": 1, ...}\n  + envelope = {\"kind\": \"epm:results\", ...}  # version omitted -> post_event derives max+1\nconfidence: medium\nrelated_task: #917\n<!-- /workflow-fix-candidate -->\n"}"""  # noqa: E501
_RAW_1248_REC_917 = r"""{"ts": "2026-07-09T07:02:37Z", "kind": "epm:workflow-fix-task-filed", "version": 1, "by": "daily-2026-07-08", "note": "filed_task: n/a (already-fixed on main: Commit fa4f3194f3 (issue-975) 'synthesized results envelope omits version -> post_event derives max+1' is the exact proposed change; #1095 (0c4020eea2) / target_file: scripts/poll_pipeline.py / fingerprint: n/a (prose park) / session_spawned: False / source: daily-parked-candidate-sweep / origin_candidate_ts: 2026-07-03T09:54:34Z / origin_candidate: Commit fa4f3194f3 (issue-975) 'synthesized results envelope omits version -> post_event derives max+1' is the exact proposed change; #1095 (0c4020eea2) extended max+1 derivation to the drain side."}"""  # noqa: E501


@pytest.mark.parametrize(
    ("cand_raw", "rec_raw", "cand_ts", "rec_ts", "fp"),
    [
        pytest.param(
            _RAW_1248_CAND_815,
            _RAW_1248_REC_815,
            "2026-07-01T22:36:29Z",
            "2026-07-09T07:02:42Z",
            "123060ae62e0",
            id="815-exact-equal-target-file",
        ),
        pytest.param(
            _RAW_1248_CAND_880,
            _RAW_1248_REC_880,
            "2026-07-03T00:09:28Z",
            "2026-07-09T07:02:37Z",
            "fe046b20d35c",
            id="880-multiword-target-file-prefix",
        ),
        pytest.param(
            _RAW_1248_CAND_917,
            _RAW_1248_REC_917,
            "2026-07-03T09:54:34Z",
            "2026-07-09T07:02:37Z",
            "ab749bae51d7",
            id="917-routed-parked-note-form",
        ),
    ],
)
def test_issue1248_na_fp_record_suppresses_formal_block_candidate(
    tmp_path: Path, cand_raw: str, rec_raw: str, cand_ts: str, rec_ts: str, fp: str
) -> None:
    """The #1248 fix: a later same-stream n/a-fp record whose origin_candidate_ts
    equals the fp-computable candidate's row ts suppresses it (with the record's
    note-form target_file prefix-compatible with the candidate's)."""
    # round-trip guard: the embedded fixture lines are single valid JSON rows
    assert json.loads(cand_raw)["ts"] == cand_ts
    assert json.loads(rec_raw)["ts"] == rec_ts
    make_task(tmp_path, 815, "completed", raw_event_lines=[cand_raw, rec_raw])
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["formal_block"] is True
    # fp fidelity: matches the fp the 2026-07-10 v2 records later carried
    assert c["fingerprint"] == fp
    assert c["suppressed"] is True
    assert c["suppressed_by"]["kind"] == "same-stream-filed"
    # `filed_task: n/a (...)` never matches _FILED_TASK_RE -> ref is the record ts
    assert c["suppressed_by"]["ref"] == rec_ts
    # and the DEFAULT (unsuppressed) listing drops it
    assert run_sweep(tmp_path)["candidates"] == []


def test_issue1680_differing_fp_record_matching_origin_ts_suppresses(
    tmp_path: Path,
) -> None:
    """#1680 deliberately supersedes the #1248 differing-fp veto (this test's
    predecessor, test_issue1248_real_differing_fp_record_never_suppresses_even_
    with_matching_ts, pinned the inverse): a record naming the candidate's
    exact row ts claims to have routed THAT row — the differing real 12-hex fp
    is a driver recomputation artifact (#1630), not a different bug. Same
    fixture, flipped assertion."""
    bug, change = "bug 1248-a.", "change 1248-a."
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        21,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(
                T1,
                f"filed_task: #95 / target_file: a/b.md / fingerprint: {other_fp} / "
                f"origin_candidate_ts: {T0}",
            ),
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"]["kind"] == "same-stream-filed"


def test_issue1248_na_fp_record_mismatched_origin_ts_does_not_suppress(tmp_path: Path) -> None:
    make_task(
        tmp_path,
        22,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1248-b.", "change 1248-b.")),
            filed_row(
                T2,
                "filed_task: n/a (x) / target_file: a/b.md / fingerprint: n/a (prose park) / "
                f"origin_candidate_ts: {T1}",
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1248_na_fp_record_without_origin_ts_does_not_suppress_fp_candidate(
    tmp_path: Path,
) -> None:
    """No target_file-ONLY fallback for fp-computable candidates (#622 hazard)."""
    make_task(
        tmp_path,
        23,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1248-c.", "change 1248-c.")),
            filed_row(
                T1, "filed_task: n/a (x) / target_file: a/b.md / fingerprint: n/a (prose park)"
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1248_na_fp_record_incompatible_target_file_does_not_suppress(
    tmp_path: Path,
) -> None:
    """The belt-and-suspenders veto: a same-second record for a DIFFERENT file
    must not close the candidate."""
    make_task(
        tmp_path,
        24,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1248-d.", "change 1248-d.")),
            filed_row(
                T1,
                "filed_task: n/a (x) / target_file: some/other.py / "
                f"fingerprint: n/a (prose park) / origin_candidate_ts: {T0}",
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1680_structured_differing_fp_matching_structured_ts_suppresses(
    tmp_path: Path,
) -> None:
    """#1680 supersedes the #1248 structured-fp veto (predecessor:
    test_issue1248_structured_real_fp_key_vetoes_ts_fallback): a cache-stream
    filed row with a STRUCTURED differing 12-hex `fingerprint` key (no note)
    and a matching structured origin_candidate_ts now suppresses — the exact
    row-ts claim overrides the recomputed-fp mismatch. Same fixture, flipped
    assertion."""
    root = tmp_path / "tasks"
    root.mkdir()
    cache = tmp_path / "workflow-fix-events.jsonl"
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    rows = [
        {
            "ts": T0,
            "marker": "epm:workflow-fix-candidate v1",
            "target_file": "e/f.md",
            "proposed_change": "structured change 1248.",
            "bug_observed": "structured bug 1248.",
            "routed": "parked: EPM_WORKFLOW_FIX_SESSION",
        },
        {
            "ts": T1,
            "marker": "epm:workflow-fix-task-filed v1",
            "target_file": "e/f.md",
            "fingerprint": other_fp,
            "origin_candidate_ts": T0,
        },
    ]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert only(run_sweep(root, cache, include_routed=True))["suppressed"] is True


def test_issue1248_comma_separated_record_target_file_is_prefix_compatible(
    tmp_path: Path,
) -> None:
    """A record naming a comma-separated target_file list: _TARGET_FILE_RE truncates
    the capture at ',' -> the first entry is prefix-compatible -> suppresses."""
    make_task(
        tmp_path,
        25,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1248-e.", "change 1248-e.")),
            filed_row(
                T1,
                "filed_task: n/a (x) / target_file: a/b.md,c/d.md / "
                f"fingerprint: n/a (prose park) / origin_candidate_ts: {T0}",
            ),
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"]["kind"] == "same-stream-filed"


# ── 18. #1274 regressions: per-task stream grouping across duplicate folders ─
#
# Byte-verbatim fixture rows from the live #1196 events.jsonl (the #644/#1253
# stale-duplicate-status-folder class): the 2026-07-09 fp-less prose park sits
# byte-identically in BOTH tasks/completed/1196 (canonical) and the stale
# tasks/reviewing/1196 fork, while the 2026-07-10 routed-record
# (filed_task: #1235) lives ONLY in the canonical folder. Pre-#1274 each
# status folder was its OWN stream, so the stale folder's copy escaped
# suppression rule 1 and re-enumerated nightly (incident 2026-07-10,
# hand-deduped as `deduped:#1235`).

_RAW_1196_CAND = r"""{"ts": "2026-07-09T20:54:25Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "parked — running under workflow_fix_target Provenance (recursion guard, see workflow-fix-on-bug.md § Recursion guard). target_file: .claude/skills/daily/SKILL.md. proposed_change: invoke 'uv run python scripts/audit_clean_results_body_discipline.py --title-sync-sweep' from the nightly /daily skill (one bullet + command) so the H1/title drift report surfaces without a manual run. bug_observed: the #1196 sweep is manually-invoked only; no scheduled invoker. confidence: medium. related_task: #1196. routed: parked: EPM_WORKFLOW_FIX_SESSION"}"""  # noqa: E501
_RAW_1196_REC = r"""{"ts": "2026-07-10T06:57:53Z", "kind": "epm:workflow-fix-task-filed", "version": 1, "by": "unknown", "note": "filed_task: #1235 / target_file: .claude/skills/daily/SKILL.md / fingerprint: n/a (prose park) / session_spawned: false / source: daily-parked-candidate-sweep / origin_candidate_ts: 2026-07-09T20:54:25Z / origin_candidate: parked — running under workflow_fix_target Provenance (recursion guard, see workflow-fix-on-bug.md § Recursion guard). target_file: .claude/skills/daily/SKILL.md. proposed_change: "}"""  # noqa: E501


def test_issue1196_stale_duplicate_status_folder_filed_record_suppresses_both_copies(
    tmp_path: Path,
) -> None:
    """#1274 red-green: one task id in two status folders (#644/#1253 class).

    tasks/reviewing/1196 is a stale fork carrying a byte-identical copy of the
    fp-less prose park but NOT the later routed-record; tasks/completed/1196
    (canonical) carries both. The record must close BOTH copies and the
    identical fork copies must collapse to ONE output row.
    """
    # round-trip guard: the embedded fixture lines are single valid JSON rows
    assert json.loads(_RAW_1196_CAND)["ts"] == "2026-07-09T20:54:25Z"
    assert json.loads(_RAW_1196_REC)["ts"] == "2026-07-10T06:57:53Z"
    make_task(tmp_path, 1196, "completed", raw_event_lines=[_RAW_1196_CAND, _RAW_1196_REC])
    make_task(tmp_path, 1196, "reviewing", raw_event_lines=[_RAW_1196_CAND])  # stale fork copy
    assert run_sweep(tmp_path)["candidates"] == []  # RED pre-fix: the stale copy escapes
    c = only(run_sweep(tmp_path, include_routed=True))  # RED pre-fix: 2 rows, no collapse
    assert c["source"] == "task:1196"
    assert c["fingerprint"] is None  # prose park; fp-less origin-ts key decided
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#1235"}


def test_issue1196_record_in_stale_folder_also_closes_canonical_copy(tmp_path: Path) -> None:
    """Reverse direction: the merged per-task pool is order-symmetric — a
    routed-record living ONLY in the stale fork still closes the canonical
    folder's copy (pins the docstring's 'ANY events.jsonl of the task' claim)."""
    make_task(tmp_path, 1196, "completed", raw_event_lines=[_RAW_1196_CAND])
    make_task(tmp_path, 1196, "reviewing", raw_event_lines=[_RAW_1196_CAND, _RAW_1196_REC])
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#1235"}


def test_filed_record_on_different_task_never_suppresses_same_second_park(
    tmp_path: Path,
) -> None:
    """Grain pin (plan §4.2 alt 1 hazard): the filed-record pool is PER TASK,
    never global. An fp-less park on task 4001 must not be closed by ANOTHER
    task's routed-record carrying the same bare-ts origin_candidate_ts (the
    fp-less PRIMARY key has no target_file check, so a global pool could
    false-suppress a same-second park on a different task)."""
    make_task(tmp_path, 4001, "completed", events=[cand_row(T0, PROSE_NOTE)])
    make_task(
        tmp_path,
        4002,
        "completed",
        events=[
            filed_row(
                T1,
                "filed_task: #96 / target_file: scripts/codex_task.py / "
                f"fingerprint: n/a (prose park) / origin_candidate_ts: {T0}",
            )
        ],
    )
    c = only(run_sweep(tmp_path))
    assert c["source"] == "task:4001"
    assert c["suppressed"] is False


def test_distinct_candidate_rows_in_duplicate_folders_both_enumerate(tmp_path: Path) -> None:
    """Grain pin: row dedup collapses only identical (source, ts_raw,
    content-hash) rows — two DISTINCT parks split across a task's duplicate
    status folders both stay enumerated."""
    make_task(tmp_path, 4003, "completed", events=[cand_row(T0, PROSE_NOTE)])
    make_task(tmp_path, 4003, "reviewing", events=[cand_row(T1, PROSE_NOTE)])
    result = run_sweep(tmp_path)
    assert len(result["candidates"]) == 2, result["candidates"]
    assert [c["ts"] for c in result["candidates"]] == [T0, T1]
    assert all(c["source"] == "task:4003" for c in result["candidates"])


def test_cache_park_not_closed_by_task_stream_record_at_matching_origin_ts(
    tmp_path: Path,
) -> None:
    """Grain pin: the cache file stays its OWN group — a task-stream
    routed-record with a matching bare-ts origin key must not close an
    fp-less structured cache park."""
    root = tmp_path / "tasks"
    make_task(
        root,
        4004,
        "completed",
        events=[
            filed_row(
                T1,
                "filed_task: #97 / target_file: g/h.md / "
                f"fingerprint: n/a (prose park) / origin_candidate_ts: {T0}",
            )
        ],
    )
    cache = tmp_path / "workflow-fix-events.jsonl"
    cache_park = {
        "ts": T0,
        "marker": "epm:workflow-fix-candidate v1",
        "target_file": "g/h.md",
        "routed": "parked: EPM_WORKFLOW_FIX_SESSION",
    }
    cache.write_text(json.dumps(cache_park) + "\n")
    c = only(run_sweep(root, cache))
    assert c["source"] == "cache"
    assert c["suppressed"] is False


# ── 19. #1281: mid-note park DECLARATIONS are enumerated ────────────────────
#
# Byte-verbatim fixture row: _RAW_1271_CAND is the EXACT JSONL line 21 of the
# live tasks/completed/1271/events.jsonl (marker `epm:workflow-fix-candidate`,
# ts 2026-07-11T18:39:16Z) — a root-sync recovery record whose note opens with
# unrelated prose, embeds the formal block mid-note, and declares
# 'Routing: parked — … recursion guard …' only at the END. Pre-#1281 it matched
# none of the three accept paths (_PARKED_LEAD_RE.match fails on the prose
# prefix; no 'routed: parked'; no structured routed field), so the 2026-07-11
# /daily Step C run silently skipped it (recovered only via the transcript
# miner, routed as #1280).

_RAW_1271_CAND = r"""{"ts": "2026-07-11T18:39:16Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "Root-sync recovery record (post-merge): the local root was 29 ahead / 4 behind with a genuine content conflict; sync_repo_root aborted cleanly. Resolved per its prescription: sparse scratch worktree detached at origin/main, `merge main` there, conflicts on tasks/1272 (concurrently moved on both sides; origin/main had transiently LOST 1272 entirely) + a 1274 plan-symlink rename mispair resolved to the local registry-canonical state (completed/1272, reviewing/1274, completed/1271), merge f56a2f93f9 pushed HEAD:main, local root fast-forwarded. Post-merge stale-task-folder guard re-run: CLEAN (exactly one folder per task on origin/main).\n\n<!-- workflow-fix-candidate v1 -->\ntarget_file: .claude/skills/issue/SKILL.md\nbug_observed: Step 10d Guard 1 computes FOREIGN from the two-endpoint diff `git diff MAIN_SHA HEAD -- tasks/`, which on ANY behind branch under fleet marker churn lists main-side advancement (33 false-positive paths on #1271, whose replayed commit set touched ZERO tasks/ paths); following the prescribed checkout+strip-commit there would stage main-advancement content into a new branch commit whose server-side replay CONFLICTS (the #1128 shape) — the strip would create the very conflict it exists to prevent.\nwhy_workflow_gap: the guard's diff form conflates \"branch carries foreign tasks/ changes in its replayed commits\" (the real hazard) with \"main advanced since the merge-base\" (benign; rebase keeps main's version of untouched files).\nproposed_change: scope Guard 1's FOREIGN set to the branch's OWN commits — three-dot `git diff --name-only origin/main...HEAD -- tasks/` (or per-commit `git show` over `origin/main..HEAD`) — so the strip fires only when a replayed commit actually touches a foreign tasks/ path; keep the reset-to-MAIN_SHA recipe for that genuine case.\ndiff_sketch: |\n  - if ! git -C \"$WT\" diff --name-only \"$MAIN_SHA\" HEAD -- 'tasks/' \\\n  + if ! git -C \"$WT\" diff --name-only \"$MAIN_SHA\"...HEAD -- 'tasks/' \\\n      > /tmp/issue-<N>-guard1-tasks-diff.txt; then\n  (three-dot: merge-base..HEAD = the replayed set; two-endpoint form retired)\nconfidence: high\nrelated_task: #1271\n<!-- /workflow-fix-candidate -->\n\nRouting: parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard); the nightly /daily parked-candidate sweep routes it.\n"}"""  # noqa: E501
# The embedded block's bug_observed / proposed_change values, verbatim:
_1271_BUG = "Step 10d Guard 1 computes FOREIGN from the two-endpoint diff `git diff MAIN_SHA HEAD -- tasks/`, which on ANY behind branch under fleet marker churn lists main-side advancement (33 false-positive paths on #1271, whose replayed commit set touched ZERO tasks/ paths); following the prescribed checkout+strip-commit there would stage main-advancement content into a new branch commit whose server-side replay CONFLICTS (the #1128 shape) — the strip would create the very conflict it exists to prevent."  # noqa: E501
_1271_CHANGE = "scope Guard 1's FOREIGN set to the branch's OWN commits — three-dot `git diff --name-only origin/main...HEAD -- tasks/` (or per-commit `git show` over `origin/main..HEAD`) — so the strip fires only when a replayed commit actually touches a foreign tasks/ path; keep the reset-to-MAIN_SHA recipe for that genuine case."  # noqa: E501


def test_issue1271_midnote_routing_parked_after_prose_enumerated(tmp_path: Path) -> None:
    """#1281 red-green + durability pin: the byte-verbatim #1271 note (prose
    prefix + embedded formal block + trailing 'Routing: parked — … recursion
    guard') is enumerated with block-computed fields."""
    # round-trip guard: the embedded fixture line is a single valid JSON row
    assert json.loads(_RAW_1271_CAND)["ts"] == "2026-07-11T18:39:16Z"
    make_task(tmp_path, 1271, "completed", raw_event_lines=[_RAW_1271_CAND])
    c = only(run_sweep(tmp_path))
    assert c["source"] == "task:1271"
    assert c["formal_block"] is True
    assert c["target_file"] == ".claude/skills/issue/SKILL.md"
    assert c["fingerprint"] == wf_fix_fingerprint(_1271_CHANGE, _1271_BUG)
    assert c["park_form"] == "recursion-guard"
    assert c["suppressed"] is False


def test_midnote_park_after_formal_block_enumerated(tmp_path: Path) -> None:
    """Arm 3 (the #941/#988/#1233 family): the park declared AFTER the formal
    block — the note STARTS with the block, so _PARKED_LEAD_RE fails."""
    bug, change = "bug 1281-a.", "change 1281-a."
    note = (
        "<!-- workflow-fix-candidate v1 -->\n"
        "target_file: .claude/agents/critic.md\n"
        f"bug_observed: {bug}\n"
        "why_workflow_gap: the workflow surface lacks the guardrail\n"
        f"proposed_change: {change}\n"
        "confidence: low\n"
        "related_task: #999\n"
        "<!-- /workflow-fix-candidate -->\n\n"
        "parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, "
        "see workflow-fix-on-bug.md § Recursion guard.\n"
    )
    make_task(tmp_path, 26, "running", events=[cand_row(T0, note)])
    c = only(run_sweep(tmp_path))
    assert c["formal_block"] is True
    assert c["fingerprint"] == wf_fix_fingerprint(change, bug)
    assert c["park_form"] == "recursion-guard"


def test_architectural_midnote_park_enumerated_and_classified(tmp_path: Path) -> None:
    """Arm 2 (`parked: architectural`) + _park_form coherence: a mid-note
    architectural park classifies correctly the moment the gate admits it."""
    note = (
        "Prose prefix: routing decision recorded after triage.\n\n"
        "parked: architectural — needs user greenlight (plan-approval gate). "
        "target_file: .claude/workflow.yaml"
    )
    make_task(tmp_path, 27, "running", events=[cand_row(T0, note)])
    c = only(run_sweep(tmp_path))
    assert c["park_form"] == "architectural"
    assert c["target_file"] == ".claude/workflow.yaml"


def test_midnote_parked_efs_token_enumerated(tmp_path: Path) -> None:
    """Arm 2, second token (`parked: EPM_WORKFLOW_FIX_SESSION`) — isolated from
    arm 3 (no 'recursion guard' co-mention) and arm 1 (no 'Routing: parked')."""
    note = (
        "Prose prefix: routing decision recorded after triage.\n\n"
        "parked: EPM_WORKFLOW_FIX_SESSION (this session is a workflow-fix session). "
        "target_file: scripts/codex_task.py"
    )
    make_task(tmp_path, 28, "running", events=[cand_row(T0, note)])
    c = only(run_sweep(tmp_path))
    assert c["target_file"] == "scripts/codex_task.py"
    assert c["fingerprint"] is None
    assert c["formal_block"] is False
    assert c["park_form"] == "recursion-guard"
    assert c["suppressed"] is False


def test_casual_parked_negation_near_recursion_guard_not_enumerated(tmp_path: Path) -> None:
    """The sketch-discriminating negative (#1281 plan §4.2): 'was not parked
    under the recursion guard' carries no declaration punctuation after
    'parked' — the task-body sketch regex would false-positive here."""
    make_task(
        tmp_path,
        15,
        "running",
        events=[
            cand_row(
                T0,
                "routed: filed #1290 — this candidate was not parked under "
                "the recursion guard; filed directly",
            ),
        ],
    )
    assert run_sweep(tmp_path, include_routed=True)["candidates"] == []


# ── 20. #1599: merge/close-time subsumption for terminal fp-tag hits ────────
#
# Byte-verbatim fixture rows: each _RAW_1599_* constant is the EXACT JSONL line
# from the live tasks/completed/1579/events.jsonl (lines 1 / 25 / 27; ts values
# verified 2026-07-22). The incident temporal shape: #1577's fp-computable-class
# park at 2026-07-21T10:59:07Z sat BETWEEN #1579's terminal status flip
# (10:46:50Z) and its Step 10d merge (11:00:26Z) — 79 s before the merge — so
# the pre-#1599 creation-only key read it as a genuine re-raise. The close rule
# is max over {epm:merged, epm:done, epm:promoted, terminal epm:status-changed}
# because marker order varies (#1577 posted epm:done AFTER epm:merged).

_RAW_1599_CREATED = r"""{"ts": "2026-07-21T06:38:56Z", "kind": "epm:created", "version": 1, "by": "task.py", "kind_": "infra"}"""  # noqa: E501
_RAW_1599_FLIP = r"""{"ts": "2026-07-21T10:46:50Z", "kind": "epm:status-changed", "version": 1, "by": "task.py", "from": "reviewing", "to": "completed", "note": "Step 10 auto-complete: kind=infra, code-review PASS r1, test-verdict PASS (4236 passed, compare clean), completion audit PASS. No children."}"""  # noqa: E501
_RAW_1599_MERGED = r"""{"ts": "2026-07-21T11:00:26Z", "kind": "epm:merged", "version": 1, "by": "unknown", "note": "Step 10d auto-merge: PR #1356 squash-merged to main (merge_form: squash; merge_attempts: 1; merge sha f0770307ce5c08bea3ed44f90d01d0ec762f425e; branch tip 3351767f5a certified by the pre-push lint gate: verdict pass, BASE_RC=0 GATED_RC=0 TG legs 0/0, pre-gate re-sync no-drift, choom=ok). Worktree kept."}"""  # noqa: E501

# The real #1577 park ts (tasks/completed/1577/events.jsonl, verified 2026-07-22).
_1599_PARK_TS = "2026-07-21T10:59:07Z"


def _make_1599_fix_task(root: Path, fp: str) -> None:
    """The byte-verbatim #1579 fixture: a completed infra fix task carrying fp."""
    # round-trip guard: the embedded fixture lines are single valid JSON rows
    assert json.loads(_RAW_1599_CREATED)["ts"] == "2026-07-21T06:38:56Z"
    flip = json.loads(_RAW_1599_FLIP)
    assert (flip["ts"], flip["to"]) == ("2026-07-21T10:46:50Z", "completed")
    assert json.loads(_RAW_1599_MERGED)["ts"] == "2026-07-21T11:00:26Z"
    make_task(
        root,
        1579,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        raw_event_lines=[_RAW_1599_CREATED, _RAW_1599_FLIP, _RAW_1599_MERGED],
    )


def test_issue1599_terminal_fp_task_merged_after_park_suppresses(tmp_path: Path) -> None:
    """#1599 red-green + durability pin (acceptance criterion 1): a candidate
    parked BETWEEN the fix task's terminal status flip and its Step 10d merge
    is subsumed — the epm:merged arm, not the status flip, decides (the park ts
    postdates the flip, so a status-changed-only key would NOT suppress)."""
    bug, change = "bug 1599-a.", "change 1599-a."
    fp = wf_fix_fingerprint(change, bug)
    make_task(
        tmp_path,
        1577,
        "completed",
        events=[cand_row(_1599_PARK_TS, block_note("a/b.md", bug, change))],
    )
    _make_1599_fix_task(tmp_path, fp)
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#1579", "basis": "close"}
    # and the DEFAULT (unsuppressed) listing drops it
    assert run_sweep(tmp_path)["candidates"] == []


def test_issue1599_candidate_parked_after_merge_is_re_raise(tmp_path: Path) -> None:
    """Criterion 2: a candidate parked AFTER the fix task's merge/close is a
    genuine re-raise and stays enumerated."""
    bug, change = "bug 1599-b.", "change 1599-b."
    fp = wf_fix_fingerprint(change, bug)
    make_task(
        tmp_path,
        30,
        "completed",
        events=[cand_row("2026-07-21T12:00:00Z", block_note("a/b.md", bug, change))],
    )
    _make_1599_fix_task(tmp_path, fp)
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


def test_issue1599_unparseable_events_fail_open_to_enumeration(tmp_path: Path) -> None:
    """Criterion 3: unreadable / garbage / ts-less events.jsonl → neither the
    creation nor the close ts parses → enumerated (fail-open preserved)."""
    bug, change = "bug 1599-c.", "change 1599-c."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 31, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        65,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        raw_event_lines=[
            "{this is not json",
            json.dumps({"kind": "epm:status-changed", "to": "completed", "note": "no ts"}),
        ],
    )
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


def test_issue1599_creation_after_park_takes_creation_basis(tmp_path: Path) -> None:
    """Ordering pin: creation-after-park decides FIRST (the pre-#1599 rule),
    even when a later epm:merged row would also decide."""
    bug, change = "bug 1599-d.", "change 1599-d."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 32, "archived", events=[cand_row(T0, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        66,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T2, "kind": "epm:created", "note": "created after the park"},
            {"ts": "2026-07-07T10:00:00Z", "kind": "epm:merged", "note": "merged later still"},
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#66", "basis": "creation"}


def test_issue1599_non_terminal_to_status_change_is_not_a_close_signal(tmp_path: Path) -> None:
    """Criterion 5: an epm:status-changed row with a NON-terminal ``to`` is
    never a close signal (the structured-``to`` discrimination)."""
    bug, change = "bug 1599-e.", "change 1599-e."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 33, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        67,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T0, "kind": "epm:created", "note": "created before the park"},
            {"ts": T2, "kind": "epm:status-changed", "from": "approved", "to": "reviewing"},
        ],
    )
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


def test_issue1599_archived_close_via_terminal_status_change(tmp_path: Path) -> None:
    """Criterion 6 (the archived shape, grounded on tasks/archived/1101): an
    archived task closes via its ``to: archived`` status-changed row alone —
    archived tasks carry no epm:merged / epm:done rows."""
    bug, change = "bug 1599-f.", "change 1599-f."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 34, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        68,
        "archived",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T0, "kind": "epm:created", "note": "created before the park"},
            {"ts": T2, "kind": "epm:status-changed", "from": "proposed", "to": "archived"},
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#68", "basis": "close"}


def test_issue1599_close_equal_to_park_ts_is_not_subsumption(tmp_path: Path) -> None:
    """Tie boundary (advisory): closed == cand_ts does NOT suppress — the rule
    is strict ``closed > cand_ts``, mirroring the creation check."""
    bug, change = "bug 1599-g.", "change 1599-g."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 35, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        69,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T0, "kind": "epm:created", "note": "created before the park"},
            {"ts": T1, "kind": "epm:merged", "note": "merged at exactly the park ts"},
        ],
    )
    c = only(run_sweep(tmp_path))
    assert c["suppressed"] is False


def test_issue1599_done_only_close_arm_suppresses(tmp_path: Path) -> None:
    """epm:done-only close arm (advisory): a completed task with neither an
    epm:merged row nor a terminal status-changed row still closes via epm:done."""
    bug, change = "bug 1599-h.", "change 1599-h."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 36, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    make_task(
        tmp_path,
        70,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T0, "kind": "epm:created", "note": "created before the park"},
            {"ts": T2, "kind": "epm:done", "note": "outcome recorded"},
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#70", "basis": "close"}


def test_issue1599_scan_continues_past_non_suppressing_terminal_hit(tmp_path: Path) -> None:
    """Multi-hit continue-scanning (advisory): a terminal fp hit that fails
    BOTH temporal checks does not end the scan — a later fp-bearing body may
    still decide."""
    bug, change = "bug 1599-i.", "change 1599-i."
    fp = wf_fix_fingerprint(change, bug)
    make_task(tmp_path, 37, "archived", events=[cand_row(T1, block_note("a/b.md", bug, change))])
    # sorts first (completed/71 < completed/72): created before the park, no close rows
    make_task(
        tmp_path,
        71,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[{"ts": T0, "kind": "epm:created", "note": "created before the park"}],
    )
    # sorts second: closes after the park → decides
    make_task(
        tmp_path,
        72,
        "completed",
        body_extra=f"- fingerprint: {fp}\n",
        events=[
            {"ts": T0, "kind": "epm:created", "note": "created before the park"},
            {"ts": T2, "kind": "epm:merged", "note": "merged after the park"},
        ],
    )
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed_by"] == {"kind": "fp-tag-closed", "ref": "#72", "basis": "close"}


# ── 21. #1680: ts-claim precedence over differing fps + structured skips ────


def test_issue1680_1630_shape_driver_recomputed_fp_record_suppresses(tmp_path: Path) -> None:
    """The #1630 regression shape near-verbatim: the /daily driver recomputed
    the fp from ABRIDGED origin text (differing from the sweep-canonical fp)
    but named the candidate's exact row ts — the record must close it. A
    DECOY full-ISO ts (a DIFFERENT candidate's row ts) planted inside the
    abridged `origin_candidate:` tail must NOT close that other candidate
    (pins _record_origin_ts's field-segment bounding at the ' / ' separator)."""
    bug_a, change_a = "bug 1630-a.", "change 1630-a."
    bug_b, change_b = "bug 1630-b.", "change 1630-b."
    driver_fp = wf_fix_fingerprint("abridged change.", "abridged bug.")
    make_task(
        tmp_path,
        1630,
        "completed",
        events=[
            cand_row(T0, block_note("scripts/workflow_lint.py", bug_a, change_a)),
            cand_row(T1, block_note("scripts/workflow_lint.py", bug_b, change_b)),
            filed_row(
                T2,
                "filed_task: #1648 / target_file: scripts/workflow_lint.py / "
                f"fingerprint: {driver_fp} / session_spawned: best-effort (driver) / "
                f"source: daily-parked-candidate-sweep / origin_candidate_ts: {T0} / "
                f"origin_candidate: abridged tail quoting a decoy row ts {T1} verbatim",
            ),
        ],
    )
    by_ts = {c["ts"]: c for c in run_sweep(tmp_path, include_routed=True)["candidates"]}
    assert by_ts[T0]["suppressed"] is True
    assert by_ts[T0]["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#1648"}
    # the decoy ts sits OUTSIDE the origin_candidate_ts field's value segment
    assert by_ts[T1]["suppressed"] is False


def test_issue1680_differing_fp_record_nonmatching_ts_not_suppressed(tmp_path: Path) -> None:
    """Dedup doctrine preserved: a differing-fp record whose origin_candidate_ts
    names a DIFFERENT row is a different bug — never suppresses."""
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        26,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-b.", "change 1680-b.")),
            filed_row(
                T2,
                f"filed_task: #96 / target_file: a/b.md / fingerprint: {other_fp} / "
                f"origin_candidate_ts: {T1}",
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1680_differing_fp_record_no_origin_ts_not_suppressed(tmp_path: Path) -> None:
    """Pure fp-mismatch with NO row-ts claim keeps the #1248 outcome: a
    differing real fp is a different bug (workflow-fix-on-bug.md § Dedup)."""
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        27,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-c.", "change 1680-c.")),
            filed_row(T1, f"filed_task: #97 / target_file: a/b.md / fingerprint: {other_fp}"),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1680_differing_fp_matching_ts_incompatible_target_file_not_suppressed(
    tmp_path: Path,
) -> None:
    """The target_file prefix-compatibility veto survives the #1680 widening."""
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        28,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-d.", "change 1680-d.")),
            filed_row(
                T1,
                f"filed_task: #98 / target_file: some/other.py / fingerprint: {other_fp} / "
                f"origin_candidate_ts: {T0}",
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False


def test_issue1680_differing_fp_matching_ts_missing_target_file_abstains_suppresses(
    tmp_path: Path,
) -> None:
    """Behavior-table row-3 abstain sub-case: with a matching row-ts claim and
    NO target_file on the record side, the veto abstains (the inherited #1248
    semantics) and the exact row ts decides — suppressed."""
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        29,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-e.", "change 1680-e.")),
            filed_row(T1, f"filed_task: #99 / fingerprint: {other_fp} / origin_candidate_ts: {T0}"),
        ],
    )
    assert only(run_sweep(tmp_path, include_routed=True))["suppressed"] is True


def test_issue1680_multi_ts_record_closes_each_listed_candidate(tmp_path: Path) -> None:
    """A record whose origin_candidate_ts lists several full-ISO timestamps
    (`TS1 + TS2 + TS3`, the live #1630 v4 corrective shape) closes EACH listed
    park; unlisted timestamps stay enumerated."""
    make_task(
        tmp_path,
        30,
        "completed",
        events=[
            cand_row(T0, PROSE_NOTE),
            cand_row(T1, PROSE_NOTE),
            cand_row(T2, PROSE_NOTE),
            filed_row(
                "2026-07-07T10:00:00Z",
                "filed_task: #1650 / fingerprint: n/a (prose park) / "
                f"origin_candidate_ts: {T0} + {T2} / origin_candidate: abridged",
            ),
        ],
    )
    by_ts = {c["ts"]: c for c in run_sweep(tmp_path, include_routed=True)["candidates"]}
    assert by_ts[T0]["suppressed"] is True
    assert by_ts[T2]["suppressed"] is True
    assert by_ts[T1]["suppressed"] is False


def test_issue1680_same_second_sibling_residual_both_suppressed(tmp_path: Path) -> None:
    """Pins the ACCEPTED residual (documented in the module docstring; #1680
    kill criterion 1): two DISTINCT formal blocks at the SAME second on the
    same task with compatible target_files are indistinguishable to the
    ts+target_file key — one differing-fp record naming that second closes
    BOTH. Live candidate rows are minutes apart, and a false closure
    additionally requires a wrong-fp record on top; if this residual ever
    fires on live evidence, the fix direction reverts to fp-primacy.
    (Enumerator note: one row yields at most ONE Candidate — _extract_fields
    parses the FIRST formal block per note — so the two same-second
    candidates here are necessarily two distinct rows.)"""
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path,
        31,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-f.", "change 1680-f.")),
            cand_row(T0, block_note("a/b.md", "bug 1680-g.", "change 1680-g.")),
            filed_row(
                T1,
                f"filed_task: #100 / target_file: a/b.md / fingerprint: {other_fp} / "
                f"origin_candidate_ts: {T0}",
            ),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    assert len(result["candidates"]) == 2
    assert all(c["suppressed"] for c in result["candidates"])


def test_issue1680_present_but_unparseable_origin_ts_no_target_file_fallback(
    tmp_path: Path,
) -> None:
    """The #1248 absent-vs-unparseable distinction survives _record_origin_ts:
    a PRESENT origin_candidate_ts with no parseable full-ISO token is a
    non-match and never falls through to the legacy target_file key — on the
    fp-less primary path AND the fp-bearing ts-claim branch."""
    # fp-less park: the record's target_file matches EXACTLY, but the present
    # (garbage) ts field blocks the legacy target_file fallback
    make_task(
        tmp_path,
        32,
        "completed",
        events=[
            cand_row(T0, PROSE_NOTE),
            filed_row(
                T1,
                "filed_task: #101 / target_file: scripts/codex_task.py / "
                "fingerprint: n/a (prose park) / origin_candidate_ts: not-a-timestamp",
            ),
        ],
    )
    assert only(run_sweep(tmp_path))["suppressed"] is False

    # fp-bearing candidate: differing fp + garbage ts field -> non-match
    other_fp = wf_fix_fingerprint("unrelated change.", "unrelated bug.")
    make_task(
        tmp_path / "b",
        33,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", "bug 1680-h.", "change 1680-h.")),
            filed_row(
                T1,
                f"filed_task: #102 / target_file: a/b.md / fingerprint: {other_fp} / "
                "origin_candidate_ts: not-a-timestamp",
            ),
        ],
    )
    assert only(run_sweep(tmp_path / "b"))["suppressed"] is False


def test_issue1680_note_and_structured_ts_fields_union(tmp_path: Path) -> None:
    """_record_origin_ts UNIONS the note-form field and the structured key: a
    record carrying T0 in its note and T2 in its structured
    origin_candidate_ts key closes fp-less parks at BOTH timestamps."""
    root = tmp_path / "tasks"
    root.mkdir()
    cache = tmp_path / "workflow-fix-events.jsonl"
    rows = [
        {"ts": T0, "marker": "epm:workflow-fix-candidate v1", "note": PROSE_NOTE},
        {"ts": T2, "marker": "epm:workflow-fix-candidate v1", "note": PROSE_NOTE},
        {
            "ts": "2026-07-07T10:00:00Z",
            "marker": "epm:workflow-fix-task-filed v1",
            "note": (
                f"filed_task: #103 / fingerprint: n/a (prose park) / origin_candidate_ts: {T0}"
            ),
            "origin_candidate_ts": T2,
        },
    ]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = run_sweep(root, cache, include_routed=True)
    assert [c["suppressed"] for c in result["candidates"]] == [True, True]


def test_issue1680_skipped_record_json_decode_error_irrelevant_kind(tmp_path: Path) -> None:
    """The #1333:152 shape: a truncated raw line of an IRRELEVANT marker kind
    is a benign skip — relevant_kind False, fully attributed."""
    truncated = (
        '{"ts": "2026-07-11T00:00:00Z", "kind": "epm:experiment-implementation", "note": "trunc'
    )
    make_task(
        tmp_path,
        1333,
        "completed",
        events=[cand_row(T0, PROSE_NOTE)],
        raw_event_lines=[truncated],
    )
    out = run_sweep(tmp_path)
    assert out["skipped_rows"] == 1
    (entry,) = out["skipped"]
    assert entry["reason"] == "json-decode-error"
    assert entry["kind_hint"] == "epm:experiment-implementation"
    assert entry["relevant_kind"] is False
    assert entry["source"] == "task:1333"
    assert entry["path"].endswith("completed/1333/events.jsonl")
    assert entry["line_no"] == 2


def test_issue1680_skipped_record_relevant_kind_malformed_line(tmp_path: Path) -> None:
    """A malformed line of CANDIDATE kind is a possible lost park ->
    relevant_kind True (investigate)."""
    truncated = (
        '{"ts": "2026-07-11T00:00:00Z", "kind": "epm:workflow-fix-candidate", "note": "parked trunc'
    )
    make_task(tmp_path, 34, "completed", raw_event_lines=[truncated])
    out = run_sweep(tmp_path)
    (entry,) = out["skipped"]
    assert entry["reason"] == "json-decode-error"
    assert entry["kind_hint"] == "epm:workflow-fix-candidate"
    assert entry["relevant_kind"] is True


def test_issue1680_skipped_record_missing_ts_and_non_dict(tmp_path: Path) -> None:
    """A ts-less candidate row is relevant by construction (only kind-matched
    rows reach the ts check); a valid-JSON non-dict line is unknown
    (kind_hint None -> relevant_kind None, investigate)."""
    make_task(
        tmp_path,
        35,
        "completed",
        events=[{"kind": CAND_KIND, "note": "parked — but no ts"}],
        raw_event_lines=["[1, 2, 3]"],
    )
    out = run_sweep(tmp_path)
    assert out["skipped_rows"] == 2
    by_reason = {e["reason"]: e for e in out["skipped"]}
    ts_less = by_reason["missing-or-unparseable-ts"]
    assert ts_less["kind_hint"] == CAND_KIND
    assert ts_less["relevant_kind"] is True
    non_dict = by_reason["non-dict-row"]
    assert non_dict["kind_hint"] is None
    assert non_dict["relevant_kind"] is None


def test_issue1680_skipped_list_capped_total_preserved(tmp_path: Path) -> None:
    """The emitted list caps at _SKIPPED_EMIT_CAP while skipped_rows keeps the
    TRUE total — skipped_rows > len(skipped) is the truncation signal."""
    n = spc._SKIPPED_EMIT_CAP + 1
    make_task(tmp_path, 36, "completed", raw_event_lines=["{bad json"] * n)
    out = run_sweep(tmp_path)
    assert out["skipped_rows"] == n
    assert len(out["skipped"]) == spc._SKIPPED_EMIT_CAP


# ── 18. unmatched_record_fps advisory (#1703) ──────────────────────────


def test_unmatched_record_fp_emitted_when_fp_differs_from_candidate(tmp_path: Path) -> None:
    """Filed record carries a real fp differing from the candidate's
    enumerated fp AND names the candidate's ts (the #1630 drift shape):
    the record still SUPPRESSES via the #1680 ts fallback, AND the
    unmatched fp is recorded in the advisory for investigation."""
    bug, change = "drift bug.", "drift change."
    cand_fp = wf_fix_fingerprint(change, bug)
    drift_fp = "abcdef012345"  # 12 hex, ≠ cand_fp
    assert drift_fp != cand_fp
    make_task(
        tmp_path,
        100,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(
                T1,
                f"filed_task: #200 / target_file: a/b.md / "
                f"fingerprint: {drift_fp} / origin_candidate_ts: {T0}",
            ),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    # Suppression semantics UNCHANGED — the ts fallback closes the park.
    c = only(result)
    assert c["suppressed"] is True
    assert c["suppressed_by"]["kind"] == "same-stream-filed"
    # NEW: the drift is surfaced in the advisory field.
    assert result["unmatched_record_fps"] == [{"source": "task:100", "ref": "#200", "fp": drift_fp}]


def test_unmatched_advisory_empty_when_all_fps_match(tmp_path: Path) -> None:
    """Normal suppression case (test 3): filed fp EQUALS candidate fp.
    Advisory is EMPTY — the field exists but no drift surfaced."""
    bug, change = "bug one.", "change one."
    fp = wf_fix_fingerprint(change, bug)
    make_task(
        tmp_path,
        101,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(T1, f"filed_task: #201 / fingerprint: {fp}"),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    assert result["unmatched_record_fps"] == []
    assert only(result)["suppressed"] is True  # semantics preserved


def test_unmatched_advisory_ignores_prose_and_na_fp_records(tmp_path: Path) -> None:
    """Fail-soft: a filed record with prose-only fp values (n/a, empty
    string, missing entirely) contributes NOTHING to the advisory —
    never raises."""
    make_task(
        tmp_path,
        102,
        "completed",
        events=[
            cand_row(T0, PROSE_NOTE),
            filed_row(T1, "filed_task: #202 / fingerprint: n/a (prose park)"),
            filed_row(T2, "filed_task: #203 / target_file: scripts/codex_task.py"),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    assert result["unmatched_record_fps"] == []


def test_unmatched_advisory_dedupes_within_stream(tmp_path: Path) -> None:
    """A drift fp appearing in TWO filed records on the same stream
    emits ONCE (advisory dedup — one investigation entry per drift)."""
    bug, change = "d bug.", "d change."
    drift_fp = "cafebabe0011"
    make_task(
        tmp_path,
        103,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug, change)),
            filed_row(
                T1,
                f"filed_task: #301 / fingerprint: {drift_fp} / origin_candidate_ts: {T0}",
            ),
            filed_row(
                T2,
                f"filed_task: #302 / fingerprint: {drift_fp} / origin_candidate_ts: {T0}",
            ),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    assert len(result["unmatched_record_fps"]) == 1
    entry = result["unmatched_record_fps"][0]
    assert entry["fp"] == drift_fp
    assert entry["source"] == "task:103"


def test_unmatched_advisory_isolates_across_streams(tmp_path: Path) -> None:
    """A drift fp on task:X does not surface on task:Y's stream — the
    advisory is per-source."""
    bug_x, change_x = "x bug.", "x change."
    bug_y, change_y = "y bug.", "y change."
    fp_y = wf_fix_fingerprint(change_y, bug_y)  # matches y's candidate
    drift_fp = "deadbeef0022"  # matches neither
    make_task(
        tmp_path,
        104,
        "completed",
        events=[
            cand_row(T0, block_note("a/b.md", bug_x, change_x)),
            filed_row(
                T1,
                f"filed_task: #401 / fingerprint: {drift_fp} / origin_candidate_ts: {T0}",
            ),
        ],
    )
    make_task(
        tmp_path,
        105,
        "completed",
        events=[
            cand_row(T0, block_note("c/d.md", bug_y, change_y)),
            filed_row(T1, f"filed_task: #402 / fingerprint: {fp_y}"),
        ],
    )
    result = run_sweep(tmp_path, include_routed=True)
    # Only task:104 contributes to the advisory.
    assert result["unmatched_record_fps"] == [{"source": "task:104", "ref": "#401", "fp": drift_fp}]


def test_unmatched_advisory_extracts_from_structured_fingerprint_key(tmp_path: Path) -> None:
    """A filed record carrying its fp under a STRUCTURED ``fingerprint``
    key (not the note) still contributes to the advisory."""
    bug, change = "s bug.", "s change."
    drift_fp = "0123456789ab"
    row = {
        "ts": T1,
        "kind": FILED_KIND,
        "version": 1,
        "by": "unknown",
        "note": "filed_task: #501 (structured fp key)",
        "fingerprint": drift_fp,
    }
    make_task(
        tmp_path,
        106,
        "completed",
        events=[cand_row(T0, block_note("e/f.md", bug, change)), row],
    )
    result = run_sweep(tmp_path, include_routed=True)
    assert result["unmatched_record_fps"] == [{"source": "task:106", "ref": "#501", "fp": drift_fp}]


# ── #1741: URGENT fast-path park accept arms (#1681 grammar; incident #1718) ─

# Byte-verbatim fixture row: _RAW_1718_CAND is the EXACT JSONL line from the
# live #1718 events.jsonl (the 2026-07-27 urgent fast-path park — leads
# `URGENT-PARK`, formal block carries urgency: main-red + failing_test +
# wf_fix, ZERO occurrences of the word "parked"). Pre-#1741 the predicate
# never enumerated it, so BOTH consumers (the watcher's urgent-park router
# and the nightly /daily Step C sweep) were blind for ~16 h while main
# stayed red. Consumed via make_task(raw_event_lines=...), so the sweep
# parses the same bytes the live tree carries.

_RAW_1718_CAND = r"""{"ts": "2026-07-27T14:38:28Z", "kind": "epm:workflow-fix-candidate", "version": 1, "by": "unknown", "note": "URGENT-PARK workflow-fix candidate raised from #1718 Step 10d merge attempt (autonomous session).\n\n## Context\n\n- Session: task #1718 (workflow-fix session — carries `workflow_fix_target: scripts/workflow_lint.py` Provenance; recursion guard ACTIVE).\n- Blocked action: `/issue 1718` Step 10d local merge (`git merge origin/main` after `gh pr merge --squash` reported CONFLICTING).\n- Blocker: pristine `origin/main` fails the agent-spec-size ratchet. Verified two ways:\n  (a) `git show origin/main:.claude/agents/planner.md | wc -c` = 40900 (over the 40000 FAIL threshold).\n  (b) In a scratch worktree detached at `origin/main` (SHA `0a056258de4de8e209ddd1535269ebf38aa477f2` at diagnosis time; `f36fdb2f84` post-`task #1725: set-body` commit), `uv run python scripts/workflow_lint.py --check-agent-spec-size` prints `workflow_lint: .claude/agents/planner.md: 40900 bytes exceeds the 40000-byte agent-spec FAIL threshold — relocate per-scenario content to .claude/rules/ (see #829). workflow_lint: FAIL (1 error(s))`.\n\n- My branch alone is GREEN — `tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes` PASSES on `issue-1718` @ `a6d4a4b9a4` (planner.md there = 39371 bytes). The failure entered my worktree only via the merge that pulled origin/main's planner.md at 40900 bytes.\n\n## Why urgent-park (per `.claude/rules/workflow-fix-on-bug.md` § Recursion guard \"Urgent fast path\", #1681)\n\n- My session is a workflow-fix session ⇒ recursion guard PARKS the candidate; the orchestrator does NOT auto-file/spawn.\n- Every intervening Step 10d merge attempt across the fleet must re-classify this red vs the baseline ledger; the #1643-class fleet-wide cost applies.\n- Direct evidence exists: `test_live_tree_passes` FAILs on pristine `origin/main`, and my in-worktree merge attempt fired the same failure through the pre-commit hook (`workflow-lint-agent-spec-size`).\n\n## Post-park action\n\n- Parking #1718 at status `blocked` with `epm:failure v1 failure_class: infra reason: main-red-planner-md-ratchet-blocks-step10d-merge`.\n- The autonomous-session watcher's `urgent_wf_park_pass` (every 10 min) will detect the token, VERIFY the claim (one bounded pytest run of the named node, expecting rc=1), and file+dispatch via `scripts/file_infra_task.py` — `epm:workflow-fix-task-filed` will land on #1718 with the routed record.\n- Once the routed fix lands (planner.md trimmed to <40 KB and merged to main), a fresh `/issue 1718` respawn (via the completed-unmerged watcher pass or manual) can retry Step 10d cleanly.\n\n## Bug details\n\n- **File:** `.claude/agents/planner.md`\n- **Current size (origin/main):** 40900 bytes\n- **FAIL threshold:** 40000 bytes (`AGENT_SPEC_FAIL_BYTES` in `scripts/workflow_lint.py`)\n- **Overage:** 900 bytes\n- **Policy stance (per #829 / #838):** planner.md was DELIBERATELY not grandfathered — it was structurally trimmed to ≤20 KB and per-scenario content moved to `.claude/rules/`. Concurrent growth pushed it past 40 KB without an offsetting relocation.\n- **Correct fix (per the workflow_lint hook message itself):** \"relocate per-scenario content to .claude/rules/ (see #829)\".\n\n<!-- workflow-fix-candidate v1 -->\ntarget_file: .claude/agents/planner.md\nbug_observed: origin/main planner.md is 40900 bytes, exceeds the 40000-byte AGENT_SPEC_FAIL threshold; pristine origin/main fails workflow_lint --check-agent-spec-size and tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes\nwhy_workflow_gap: planner.md was deliberately NOT grandfathered per #829/#838 (structurally trimmed to <=20 KB, per-scenario content moved to .claude/rules/); concurrent growth pushed it past 40 KB without a relocation, so every Step 10d merge that pulls origin/main hits the pre-commit ratchet hook and cannot land — fleet-wide per-hour cost until fixed\nproposed_change: Relocate per-scenario content in .claude/agents/planner.md to .claude/rules/ subfiles per #829 so planner.md falls below 40000 bytes\ndiff_sketch: |\n  # Scenario-specific content trimming; the spawned session's planner+implementer\n  # will identify the largest relocatable per-scenario sections per the #829\n  # protocol. Verify with:\n  #   uv run pytest tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes\n  #   uv run python scripts/workflow_lint.py --check-agent-spec-size\nconfidence: high\nrelated_task: #1718\nurgency: main-red\nfailing_test: tests/test_workflow_lint_agent_spec_size.py::test_live_tree_passes\nwf_fix: true\n<!-- /workflow-fix-candidate -->\n\nFingerprint (sha256(normalize(proposed_change) + \"||\" + normalize(bug_observed))[:12]): 06bc0203d759\n"}"""  # noqa: E501

_1718_TS = "2026-07-27T14:38:28Z"
_1718_FP = "06bc0203d759"


def test_issue1741_urgent_park_lead_note_enumerated(tmp_path: Path) -> None:
    """Arm (a): the verbatim #1718 URGENT-PARK note enumerates with its fp."""
    # round-trip guard: the embedded fixture line is a single valid JSON row
    row = json.loads(_RAW_1718_CAND)
    assert row["ts"] == _1718_TS
    # the incident's defining property: no "parked" token anywhere
    assert "parked" not in row["note"].lower()
    make_task(tmp_path, 1718, "blocked", raw_event_lines=[_RAW_1718_CAND])
    c = only(run_sweep(tmp_path))
    assert c["source"] == "task:1718"
    assert c["formal_block"] is True
    assert c["fingerprint"] == _1718_FP
    assert c["target_file"] == ".claude/agents/planner.md"
    assert c["suppressed"] is False


def test_issue1741_urgent_block_token_enumerated_without_lead(tmp_path: Path) -> None:
    """Arm (b) independently: `urgency: main-red` INSIDE the formal block
    enumerates with neither an URGENT-PARK lead nor any "parked" token."""
    bug = "check c99 fails on origin/main after the ratchet landed."
    change = "raise the size cap for the offending file."
    note = (
        "Recursion-guard candidate surfaced for the nightly sweep (no routing "
        "performed by this session).\n\n"
        "<!-- workflow-fix-candidate v1 -->\n"
        "target_file: .claude/agents/planner.md\n"
        f"bug_observed: {bug}\n"
        "why_workflow_gap: the ratchet lacks headroom\n"
        f"proposed_change: {change}\n"
        "urgency: main-red\n"
        "failing_test: tests/test_x.py::test_y\n"
        "wf_fix: true\n"
        "confidence: high\n"
        "related_task: #1718\n"
        "<!-- /workflow-fix-candidate -->\n"
    )
    assert "parked" not in note.lower() and not note.lower().startswith("urgent-park")
    make_task(tmp_path, 42, "completed", events=[cand_row(T0, note)])
    c = only(run_sweep(tmp_path))
    assert c["formal_block"] is True
    assert c["fingerprint"] == wf_fix_fingerprint(change, bug)
    assert c["target_file"] == ".claude/agents/planner.md"


def test_issue1741_urgent_park_suppressed_by_matching_filed_record(tmp_path: Path) -> None:
    """The #1718→#1740 no-double-route pin: a LATER same-stream filed record
    carrying the matching fp suppresses the urgent park (suppression rule 1)."""
    rec = filed_row(
        "2026-07-28T06:41:05Z",
        "filed_task: #1740 / target_file: .claude/agents/planner.md / "
        f"fingerprint: {_1718_FP} / session_spawned: true / "
        f"source: daily-parked-candidate-sweep / origin_candidate_ts: {_1718_TS}",
    )
    make_task(tmp_path, 1718, "blocked", events=[rec], raw_event_lines=[_RAW_1718_CAND])
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#1740"}


def test_issue1741_urgency_token_outside_block_not_enumerated(tmp_path: Path) -> None:
    """Prose QUOTING the grammar (`urgency: main-red` with NO formal block, no
    park token) stays out — the casual-mention exclusion is not widened."""
    note = (
        "Discussion of the #1681 urgent fast path: a parking session adds\n"
        "urgency: main-red\n"
        "failing_test: tests/test_x.py::test_y\n"
        "inside its formal block; the router verifies before filing."
    )
    make_task(tmp_path, 43, "completed", events=[cand_row(T0, note)])
    assert run_sweep(tmp_path)["candidates"] == []
    assert run_sweep(tmp_path, include_routed=True)["candidates"] == []


def test_issue1741_routed_urgent_note_suppressed(tmp_path: Path) -> None:
    """A mis-tagged ROUTED urgent-block candidate (`routed: filed #999` note
    lead) is enumerated by arm (b) but closed by its later same-stream filed
    record — the routed corner the arm-(b) rationale leans on (rule 1)."""
    bug = "some urgent bug."
    change = "some urgent change."
    fp = wf_fix_fingerprint(change, bug)
    note = (
        "routed: filed #999\n\n"
        "<!-- workflow-fix-candidate v1 -->\n"
        "target_file: a/b.md\n"
        f"bug_observed: {bug}\n"
        "why_workflow_gap: gap\n"
        f"proposed_change: {change}\n"
        "urgency: main-red\n"
        "failing_test: tests/test_x.py::test_y\n"
        "wf_fix: true\n"
        "confidence: high\n"
        "related_task: #999\n"
        "<!-- /workflow-fix-candidate -->\n"
    )
    make_task(
        tmp_path,
        44,
        "completed",
        events=[
            cand_row(T0, note),
            filed_row(T1, f"filed_task: #999 / target_file: a/b.md / fingerprint: {fp}"),
        ],
    )
    assert run_sweep(tmp_path)["candidates"] == []
    c = only(run_sweep(tmp_path, include_routed=True))
    assert c["suppressed"] is True
    assert c["suppressed_by"] == {"kind": "same-stream-filed", "ref": "#999"}
