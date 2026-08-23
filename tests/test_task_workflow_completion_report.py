"""Tests for the #2309 four-H3 completion-report contract check.

Covers the pure predicate (`task_workflow.completion_report_violation` +
`completion_report_advisory`) and the `task.py post-marker` CLI wiring:
pre-append refusal (nothing appended on refusal — byte-checked), the
`--allow-nonconforming-report` waiver (recorded as `report_shape_waiver`
on the event row), the multi-part / sentinel / non-contract-kind skips,
and the soft no-signature advisory. The CLI is exercised at the
handler-function layer against a fake repo (the branch-guarded resolver
can't be redirected across a process boundary — see
test_task_workflow_post_marker_echo.py).

Workflow-invariant family: run after any edit to `scripts/task.py` or
`task_workflow.py`'s marker surface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import task as task_cli

from explore_persona_space import task_workflow as tw_mod

# The REAL #2302 epm:results v2 marker (the founding incident's CONFORMING
# re-post — acceptance criterion 2: no false positive on it).
FIXTURE_2302_V2 = (Path(__file__).parent / "fixtures" / "issue2302_epm_results_v2.md").read_text()

# The REAL experiment-implementer.md template header form + all four
# lettered H3s (acceptance criterion 2 for the second contract kind).
CONFORMING_IMPL_REPORT = """<!-- epm:experiment-implementation v2 -->
## Implementation Report — round 2

### (a) What was done
- `scripts/foo.py`: wired the thing.

### (b) Considered but not done
Nothing material.

### (c) How to verify
Run the tests.

### (d) Needs human eyeball
None — confidence high across the diff.
"""

# Signature present, (d) absent — the #2302 round-1 defect class.
MISSING_D_REPORT = """<!-- epm:results v1 -->
## Completion Report

### (a) What was done
- stuff

### (b) Considered but not done
- nothing

### (c) How to verify
- run tests
"""

IMPL_MISSING_D = MISSING_D_REPORT.replace(
    "## Completion Report", "## Implementation Report — round 2"
)

# Workload-results shaped (pod-sentinel drain class): no report header.
SENTINEL_SHAPED_NOTE = json.dumps(
    {"phase": "eval", "cells": 12, "status": "done", "out_root": "/workspace/out"}
)


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo with task_workflow's resolvers rebound
    (the test_task_workflow.py convention)."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    tw_mod.invalidate_cache()
    monkeypatch.setattr(tw_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw_mod, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw_mod, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw_mod, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw_mod, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.setattr(tw_mod, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    monkeypatch.setattr(tw_mod, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")

    tid = tw_mod.create_task(tw_mod.NewTaskRequest(kind="infra", title="report check fixture"))
    return tmp_path, tw_mod, tid


def _events_path(tw, tid: int) -> Path:
    return tw.find_task_path(tid) / "events.jsonl"


def _ns(tid: int, marker: str, note: str | None, **overrides) -> argparse.Namespace:
    base = dict(
        number=tid,
        marker=marker,
        version=None,
        by="test",
        note=note,
        file=None,
        allow_nonconforming_report=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ─── Pure-predicate unit tests ─────────────────────────────────────────────


def test_predicate_non_contract_kind_is_none():
    assert tw_mod.completion_report_violation("epm:progress", MISSING_D_REPORT) is None


def test_predicate_empty_and_none_notes_are_none():
    assert tw_mod.completion_report_violation("epm:results", None) is None
    assert tw_mod.completion_report_violation("epm:results", "") is None


def test_predicate_no_signature_is_none():
    assert tw_mod.completion_report_violation("epm:results", SENTINEL_SHAPED_NOTE) is None


def test_predicate_conforming_returns_empty():
    assert tw_mod.completion_report_violation("epm:results", FIXTURE_2302_V2) == []
    assert (
        tw_mod.completion_report_violation("epm:experiment-implementation", CONFORMING_IMPL_REPORT)
        == []
    )


def test_predicate_missing_d():
    assert tw_mod.completion_report_violation("epm:results", MISSING_D_REPORT) == ["d"]
    assert tw_mod.completion_report_violation("epm:experiment-implementation", IMPL_MISSING_D) == [
        "d"
    ]


def test_predicate_both_header_forms_for_both_kinds():
    # The v2-plan single-form constant would have no-opped one whole kind
    # (39 epm:experiment-implementation rows carry the Completion form).
    for kind in tw_mod.COMPLETION_REPORT_KINDS:
        assert tw_mod.completion_report_violation(kind, MISSING_D_REPORT) == ["d"]
        assert tw_mod.completion_report_violation(kind, IMPL_MISSING_D) == ["d"]


def test_predicate_case_insensitive_headers():
    # Fleet-observed variants: "## Completion report — task #2263 ...",
    # "## Implementation report v2".
    for header in (
        "## Completion report — task #2263 caps",
        "## Implementation report v2",
        "## COMPLETION REPORT",
    ):
        note = MISSING_D_REPORT.replace("## Completion Report", header)
        assert tw_mod.completion_report_violation("epm:results", note) == ["d"]


def test_predicate_signature_mid_note_matches():
    # MULTILINE: the header need not be the first line (#2302 v1's header
    # sits after the marker comment line).
    note = "preamble line\n\n" + MISSING_D_REPORT
    assert tw_mod.completion_report_violation("epm:results", note) == ["d"]


def test_predicate_lenient_section_labels():
    # Label text after the letter is free; only the lettered H3s bind.
    note = (
        "## Completion Report\n"
        "### (a) Everything that changed\n"
        "### (b) Rejected alternatives\n"
        "### (c) Verification recipe\n"
        "### (d) Eyeball items\n"
    )
    assert tw_mod.completion_report_violation("epm:results", note) == []


def test_predicate_part_token_skips():
    note = "part=1/3\n" + MISSING_D_REPORT
    assert tw_mod.completion_report_violation("epm:results", note) is None
    # Spaced '=' variants parse too.
    note2 = "part = 2 / 3\n" + MISSING_D_REPORT
    assert tw_mod.completion_report_violation("epm:results", note2) is None


def test_predicate_prose_part_phrase_does_not_bypass():
    # The '=' form ONLY: prose "(part 1/6)" (the #76 v1 commit-series shape)
    # must NOT open a refusal bypass.
    note = MISSING_D_REPORT + "\nCommits landed as described (part 1/6) of the series.\n"
    assert tw_mod.completion_report_violation("epm:results", note) == ["d"]


def test_predicate_missing_multiple_sections():
    note = "## Completion Report\n### (c) How to verify\nstuff\n"
    assert tw_mod.completion_report_violation("epm:results", note) == ["a", "b", "d"]


def test_advisory_helper():
    assert tw_mod.completion_report_advisory("epm:results", SENTINEL_SHAPED_NOTE) is True
    assert tw_mod.completion_report_advisory("epm:results", None) is True
    assert tw_mod.completion_report_advisory("epm:results", FIXTURE_2302_V2) is False
    assert tw_mod.completion_report_advisory("epm:results", "part=2/3 chunk body") is False
    assert tw_mod.completion_report_advisory("epm:progress", SENTINEL_SHAPED_NOTE) is False


# ─── CLI wiring (handler layer, fake repo, REAL post_event) ────────────────


def test_refusal_exits_nonzero_and_appends_nothing(fake_repo, capsys):
    """Fail-loud pin (#2309 success criteria): the missing-(d) post raises
    SystemExit, names (d), and leaves events.jsonl BYTE-unchanged."""
    _, tw, tid = fake_repo
    before = _events_path(tw, tid).read_bytes()
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_post_event(_ns(tid, "epm:results", MISSING_D_REPORT))
    assert exc.value.code not in (0, None)
    err = capsys.readouterr().err
    assert "(d)" in err
    assert "### (d) Needs human eyeball" in err
    assert "None — confidence high across the diff." in err
    assert "--allow-nonconforming-report" in err
    assert _events_path(tw, tid).read_bytes() == before


def test_refusal_covers_file_channel(fake_repo, tmp_path):
    """The check keys on the RESOLVED note, so --file bodies are covered."""
    _, tw, tid = fake_repo
    body = tmp_path / "report.md"
    body.write_text(MISSING_D_REPORT)
    before = _events_path(tw, tid).read_bytes()
    with pytest.raises(SystemExit):
        task_cli.cmd_post_event(_ns(tid, "epm:results", None, file=str(body)))
    assert _events_path(tw, tid).read_bytes() == before


def test_conforming_2302_v2_posts_unchanged(fake_repo):
    """Acceptance criterion 2: the real #2302 epm:results v2 marker (all
    four sections present) posts with the note stored verbatim."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:results", FIXTURE_2302_V2))
    rows = tw.list_events(tid)
    assert rows[-1]["kind"] == "epm:results"
    assert rows[-1]["note"] == FIXTURE_2302_V2
    assert "report_shape_waiver" not in rows[-1]


def test_conforming_experiment_implementation_posts(fake_repo):
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:experiment-implementation", CONFORMING_IMPL_REPORT))
    rows = tw.list_events(tid)
    assert rows[-1]["kind"] == "epm:experiment-implementation"
    assert rows[-1]["note"] == CONFORMING_IMPL_REPORT


def test_experiment_implementation_missing_d_refused(fake_repo, capsys):
    """The kind the v2-plan single-form signature would have silently
    no-opped on: the REAL template header + a missing (d) → refused."""
    _, tw, tid = fake_repo
    before = _events_path(tw, tid).read_bytes()
    with pytest.raises(SystemExit):
        task_cli.cmd_post_event(_ns(tid, "epm:experiment-implementation", IMPL_MISSING_D))
    assert "(d)" in capsys.readouterr().err
    assert _events_path(tw, tid).read_bytes() == before


def test_non_contract_kind_unaffected(fake_repo):
    """Acceptance criterion 3: the same defective text on epm:progress
    posts unchanged."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:progress", MISSING_D_REPORT))
    assert tw.list_events(tid)[-1]["note"] == MISSING_D_REPORT


def test_sentinel_shaped_results_note_posts(fake_repo, capsys):
    """No report header => the check does not apply (the poller-drain
    shape); the soft advisory NOTE fires instead of any refusal."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:results", SENTINEL_SHAPED_NOTE))
    assert tw.list_events(tid)[-1]["note"] == SENTINEL_SHAPED_NOTE
    err = capsys.readouterr().err
    assert "NOTE:" in err
    assert "## Completion Report" in err  # the per-kind expected header


def test_advisory_names_per_kind_header(fake_repo, capsys):
    _, _tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:experiment-implementation", "run digest: 12 cells ok"))
    err = capsys.readouterr().err
    assert "## Implementation Report — round <n>" in err


def test_no_advisory_on_non_contract_kind(fake_repo, capsys):
    _, _tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:progress", "plain progress note"))
    assert "NOTE:" not in capsys.readouterr().err


def test_part_token_chunk_posts(fake_repo):
    """A part=K/N multi-part chunk with a header but no (d) is excluded
    from refusal by design."""
    _, tw, tid = fake_repo
    note = "part=1/3\n" + MISSING_D_REPORT
    task_cli.cmd_post_event(_ns(tid, "epm:results", note))
    assert tw.list_events(tid)[-1]["note"] == note


def test_prose_part_phrase_still_refused(fake_repo):
    """Prose "(part 1/6)" (space form, no '=') inside a signature-bearing
    note missing (d) is still refused — the tightened token does not match
    prose."""
    _, _tw, tid = fake_repo
    note = MISSING_D_REPORT + "\nLanded as described (part 1/6) of the commit series.\n"
    with pytest.raises(SystemExit):
        task_cli.cmd_post_event(_ns(tid, "epm:results", note))


def test_waiver_posts_and_records_reason(fake_repo):
    _, tw, tid = fake_repo
    reason = "manual repro of 2309 refusal"
    task_cli.cmd_post_event(
        _ns(tid, "epm:results", MISSING_D_REPORT, allow_nonconforming_report=reason)
    )
    row = tw.list_events(tid)[-1]
    assert row["kind"] == "epm:results"
    assert row["report_shape_waiver"] == reason
    assert row["note"] == MISSING_D_REPORT  # posted verbatim, no auto-fill


def test_waiver_short_reason_refused(fake_repo, capsys):
    _, tw, tid = fake_repo
    before = _events_path(tw, tid).read_bytes()
    with pytest.raises(SystemExit) as exc:
        task_cli.cmd_post_event(
            _ns(tid, "epm:results", MISSING_D_REPORT, allow_nonconforming_report="short")
        )
    assert exc.value.code not in (0, None)
    assert ">= 10 chars" in capsys.readouterr().err
    assert _events_path(tw, tid).read_bytes() == before


def test_no_silent_normalization(fake_repo):
    """Fail-loud pin: a posted note is never rewritten or auto-filled —
    conforming and waived-nonconforming notes alike land byte-verbatim,
    and no placeholder (d) is ever injected."""
    _, tw, tid = fake_repo
    task_cli.cmd_post_event(_ns(tid, "epm:results", FIXTURE_2302_V2))
    task_cli.cmd_post_event(
        _ns(
            tid,
            "epm:experiment-implementation",
            IMPL_MISSING_D,
            allow_nonconforming_report="deliberate nonconforming re-post",
        )
    )
    rows = tw.list_events(tid)
    assert rows[-2]["note"] == FIXTURE_2302_V2
    assert rows[-1]["note"] == IMPL_MISSING_D
    assert "### (d)" not in rows[-1]["note"]  # nothing auto-filled


def test_waiver_flag_wired_in_argparse():
    """The escape flag exists on both subcommand spellings (argparse help
    exits before the branch-guarded resolver ever runs, so a subprocess
    probe is safe here)."""
    script = Path(task_cli.__file__).resolve()
    for name in ("post-marker", "post-event"):
        out = subprocess.run(
            [sys.executable, str(script), name, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--allow-nonconforming-report" in out.stdout
