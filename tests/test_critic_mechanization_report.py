"""Tests for ``scripts/critic_mechanization_report.py``.

Fixture-driven: builds a tmp ``tasks/<status>/<id>/events.jsonl`` tree and
asserts the per-month tagging counts, the graceful handling of untagged
(pre-2026-06-12) markers, the reconcile-marker exclusion, and the
best-effort verifier-landing ratchet count.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "critic_mechanization_report.py"

_spec = importlib.util.spec_from_file_location("critic_mechanization_report", _SCRIPT)
cmr = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(cmr)  # type: ignore[union-attr]


def _write_events(tasks_dir: Path, status: str, task_id: str, events: list[dict | str]) -> None:
    folder = tasks_dir / status / task_id
    folder.mkdir(parents=True, exist_ok=True)
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    (folder / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "tasks"
    _write_events(
        tasks_dir,
        "running",
        "1",
        [
            # FAIL-class code review with one yes-tag and one no-tag blocker.
            {
                "ts": "2026-06-12T10:00:00.000Z",
                "kind": "epm:code-review",
                "note": (
                    "<!-- epm:code-review v1 -->\n"
                    "## Code Review: foo\n\n**Verdict:** FAIL\n\n"
                    "### Critical\n- `foo.py:12`: missing upload call\n"
                    "  - Mechanizable: yes — grep dispatcher for upload_raw_completions\n"
                    "### Major\n- `bar.py:9`: confusing naming\n"
                    "  - Mechanizable: no\n"
                ),
            },
            # PASS marker with no tags — never counts as fail_untagged.
            {
                "ts": "2026-06-12T11:00:00.000Z",
                "kind": "epm:interp-critique",
                "note": "## Interpretation Critique — Round 2\n\n**Verdict: PASS**\n\nClean.",
            },
            # Pre-tag-era REVISE marker: FAIL-class, zero tags → fail_untagged.
            {
                "ts": "2026-05-03T09:00:00.000Z",
                "kind": "epm:interp-critique",
                "note": "## Interpretation Critique — Round 1\n\n**Verdict: REVISE**\n\nOverclaim.",
            },
            # Reconcile marker — excluded even though prefix-adjacent.
            {
                "ts": "2026-06-12T12:00:00.000Z",
                "kind": "epm:plan-critique-reconcile",
                "note": "**Verdict: REVISE** mechanizable: yes",
            },
            # Derived decision marker quoting a blocker — excluded (would
            # double-count the quoted tag under a prefix match).
            {
                "ts": "2026-06-12T12:30:00.000Z",
                "kind": "epm:code-review-decision",
                "note": "Bounced on: `foo.py:12` missing upload call — Mechanizable: yes",
            },
            # Codex twin marker — INCLUDED (exact allowlist covers -codex),
            # with an unclassifiable verdict head → verdict_unknown.
            {
                "ts": "2026-06-12T14:00:00.000Z",
                "kind": "epm:interp-critique-codex",
                "note": "## Codex Interpretation Critique — Round 1\n\n(truncated output)",
            },
            # Non-critique marker — ignored.
            {
                "ts": "2026-06-12T13:00:00.000Z",
                "kind": "epm:progress",
                "note": "mechanizable: yes (should not be counted)",
            },
            # Malformed line — skipped without crashing.
            "{not json",
        ],
    )
    _write_events(
        tasks_dir,
        "completed",
        "2",
        [
            # clean-result-critique with the `Round K: FAIL` verdict shape + yes tag.
            {
                "ts": "2026-06-13T08:00:00.000Z",
                "kind": "epm:clean-result-critique",
                "note": (
                    "Round 1: FAIL — title leads with the correction story.\n"
                    "- Lens 8: FAIL — mechanizable: yes — regex the H1 for "
                    "'once .* corrected' in verify_task_body.py\n"
                ),
            },
            # Verifier-targeting workflow fix landed → ratchet count.
            {
                "ts": "2026-06-14T08:00:00.000Z",
                "kind": "epm:workflow-fix-applied",
                "note": "merged workflow-fix: add Lens-8 title regex to verify_task_body.py",
            },
            # Workflow fix NOT targeting a verifier → not counted.
            {
                "ts": "2026-06-14T09:00:00.000Z",
                "kind": "epm:workflow-fix-applied",
                "note": "merged workflow-fix: clarify experimenter pod-resume hostname check",
            },
        ],
    )
    return tasks_dir


def test_counts_tags_fail_class_and_untagged(tmp_path: Path) -> None:
    report = cmr.build_report(_fixture(tmp_path))
    june = report["2026-06"]
    # code-review FAIL + interp PASS + codex-twin unknown + clean-result FAIL = 4.
    assert june["critique_markers"] == 4
    assert june["fail_class_markers"] == 2
    assert june["verdict_unknown"] == 1  # the truncated codex-twin marker
    assert june["mechanizable_yes"] == 2  # one in code-review, one in clean-result-critique
    assert june["mechanizable_no"] == 1
    assert june["fail_untagged"] == 0


def test_pre_tag_era_markers_count_as_untagged(tmp_path: Path) -> None:
    report = cmr.build_report(_fixture(tmp_path))
    may = report["2026-05"]
    assert may["critique_markers"] == 1
    assert may["fail_class_markers"] == 1
    assert may["mechanizable_yes"] == 0
    assert may["mechanizable_no"] == 0
    assert may["fail_untagged"] == 1


def test_derived_and_non_critique_markers_excluded(tmp_path: Path) -> None:
    report = cmr.build_report(_fixture(tmp_path))
    june = report["2026-06"]
    # The reconcile marker's yes-tag, the -decision marker's quoted tag, and
    # the epm:progress tag must not leak in (exact allowlist, not prefix).
    assert june["mechanizable_yes"] == 2
    assert june["critique_markers"] == 4


def test_verifier_fix_applied_ratchet_is_best_effort(tmp_path: Path) -> None:
    report = cmr.build_report(_fixture(tmp_path))
    june = report["2026-06"]
    # Only the verify_task_body.py-naming fix counts; the experimenter one doesn't.
    assert june["verifier_fixes_applied"] == 1


def test_classify_verdict_shapes() -> None:
    assert cmr.classify_verdict("**Verdict:** FAIL\n...") == "fail"
    assert cmr.classify_verdict("**Verdict: PASS**") == "pass"
    assert cmr.classify_verdict("## Code-Reviewer Verdict — PASS") == "pass"
    assert cmr.classify_verdict("Round 3: FAIL — issues remain") == "fail"
    assert cmr.classify_verdict("**Rating: REVISE**") == "fail"
    assert cmr.classify_verdict("**Verdict: needs_targeted_fix**") == "fail"
    assert cmr.classify_verdict("no verdict line here") == "unknown"
    # A FAIL word buried deep in prose (past the head window) does not flip the class.
    assert cmr.classify_verdict("x" * 900 + " Verdict: FAIL") == "unknown"


def test_classify_verdict_real_corpus_shapes() -> None:
    # Shapes observed on the production tasks/ tree that a bare-separator
    # regex missed (code-review round 1: 14% unknown before the widening).
    assert cmr.classify_verdict("## Code-Reviewer Verdict — round 3, PASS") == "pass"
    assert cmr.classify_verdict("## Code-Reviewer Verdict (round 2) — PASS") == "pass"
    assert cmr.classify_verdict("## Code Review: PASS\n\nDiff is 38 insertions") == "pass"
    assert cmr.classify_verdict("## Code-Review v3 — FAIL") == "fail"
    # Bare verdict token on its own line right under the sentinel.
    assert cmr.classify_verdict("<!-- epm:code-review v1 -->\nPASS\n\nDetails follow") == "pass"
    assert cmr.classify_verdict("<!-- epm:interp-critique v1 -->\n**REVISE**\n...") == "fail"
    # A bare token deeper than the first few lines does NOT classify.
    assert cmr.classify_verdict("a\nb\nc\nd\ne\nf\nPASS") == "unknown"
    # Round-opener variants from clean-result-critique notes.
    assert cmr.classify_verdict("Round 1: needs_targeted_fix — body is structurally off") == "fail"
    assert cmr.classify_verdict("Round 1: REVISE (needs_targeted_fix) — content gaps") == "fail"
    assert cmr.classify_verdict("Round 2 (Claude-only): PASS — all six items addressed") == "pass"
    assert cmr.classify_verdict("Round 3 (final) — orchestrator-verified PASS.") == "pass"


def test_main_text_and_json_output(tmp_path: Path, capsys) -> None:
    tasks_dir = _fixture(tmp_path)
    assert cmr.main(["--tasks-dir", str(tasks_dir)]) == 0
    out = capsys.readouterr().out
    assert "2026-06" in out and "TOTAL" in out

    assert cmr.main(["--tasks-dir", str(tasks_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["2026-06"]["mechanizable_yes"] == 2


def test_missing_tasks_dir_errors(tmp_path: Path, capsys) -> None:
    assert cmr.main(["--tasks-dir", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().err
