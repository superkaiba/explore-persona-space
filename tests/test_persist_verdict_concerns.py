"""Tests for ``scripts/persist_verdict_concerns.py`` (#2326).

The blind forwarder that copies a Codex verdict's machine-readable
``CONCERN:: `` rows into the per-task concerns ledger (incident #2321: a
Codex verdict carried 8 "Concerns to persist" items, zero were persisted,
and the round-2 prior-concerns gate walked an empty ledger).

Fixture shapes are synthetic but mirror the real #2321 artifact's structure
(a marker envelope carrying a concerns heading; the legacy shape is that
heading with ZERO machine rows). The real verdict prose is deliberately NOT
copied into fixtures.

Covers (plan #2326 §6 item 1):

1.  K valid rows -> K ledger events; re-run -> 0 new events (idempotency).
2.  Resume-recovery family: a current-round codex marker event whose
    ``note`` holds K rows + an EMPTY ledger; the note->``$MB`` extraction +
    forwarder run TWICE yields exactly K events (not 0 — recovery
    persisted; not 2K — the replay no-oped) and stdout carries only counts
    + kebab ids. Plus the legacy shape (heading + 0 rows) at a contract
    site -> exit 3.
3.  ``CONCERN:: none`` sole row -> exit 0, nothing persisted; ``none``
    alongside a real row -> exit 1.
4.  Bad severity / bad id / too-few-fields / duplicate id -> exit 1,
    NOTHING persisted (all-or-nothing), stdout only reason codes.
5.  Heading + zero rows + ``--require-block`` -> exit 3 (the #2321 shape);
    same without the flag -> exit 0 + WARN line.
6.  >200-char summary -> persisted with a word-boundary summary lead and
    the full text in ``evidence``.
7.  ``--validate-only`` writes nothing on both the passing and failing
    shape.

Round-2 additions (#2326 open concerns):

8.  ``_MAX_ROWS``+1 valid rows -> exit 1 (``too-many-rows``), ZERO ledger
    mutations, on the persist AND ``--validate-only`` paths
    (``unbounded-concern-row-fanout``).
9.  Mid-loop operational failure -> exit 4 with a content-free reason
    line (exception CLASS + counts, never summary text), a PARTIAL ledger
    (rows before the failure durably raised), and CONVERGENCE: the re-run
    with a healthy library persists exactly the remaining rows
    (``exit-taxonomy-operational-collision`` +
    ``partial-persist-residual-undocumented``).
10. Non-UTF-8 ``--file`` -> argparse usage exit 2 (the invocation is the
    bug; the marker was never examined).

Round-3 addition (#2326 ``partial-persist-residual-undocumented``,
verified-open r2):

11. Intra-call mirror split: fail the SECOND ``_append_jsonl_line`` inside
    ``raise_concern`` (the events.jsonl mirror) -> exit 4 whose count is
    COMPLETED calls while the failing call's OWN row IS in concerns.jsonl
    (ledger-first append), the row sits appended-but-uncommitted, and the
    healthy replay converges the LEDGER but never restores the missing
    mirror (the idempotency early-return keys on concerns.jsonl).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import persist_verdict_concerns as pvc  # noqa: E402

# ─── Fake-repo fixture (mirrors tests/test_task_workflow.py) ───────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo with task_workflow's resolvers rebound to it."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    monkeypatch.setattr(tw, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")
    return tmp_path, tw


@pytest.fixture
def concerns_task(fake_repo):
    """A clean task; yields (repo, tw, task_id)."""
    repo, tw = fake_repo
    new_id = tw.create_task(tw.NewTaskRequest(kind="infra", title="Forwarder under test"))
    return repo, tw, new_id


def _ledger_rows(tw, tid: int) -> list[dict]:
    path = tw.find_task_path(tid) / "concerns.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _concern_mirror_ids(tw, tid: int) -> list[str]:
    """concern_ids of the ``epm:concern-*`` mirror events on events.jsonl,
    in append order (the audit breadcrumbs ``raise_concern`` mirrors)."""
    path = tw.find_task_path(tid) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e["concern_id"] for e in events if str(e.get("kind", "")).startswith("epm:concern-")]


_THREE_ROWS = (
    "<!-- epm:code-review-codex v2 -->\n"
    "**Verdict:** PASS\n"
    "## Concerns to persist\n"
    "\n"
    "CONCERN:: BLOCKER hf-delete-scope-unbounded first synthetic one-liner\n"
    "CONCERN:: CONCERN pack-dir-sweep-residual second synthetic one-liner\n"
    "CONCERN:: NIT stray-log-line third synthetic one-liner\n"
    "\n"
    "Prose mentioning CONCERN:: mid-line must NOT parse as a row.\n"
    "<!-- /epm:code-review-codex -->\n"
)

_HEADING_NO_ROWS = (
    "<!-- epm:code-review-codex v1 -->\n"
    "**Verdict:** FAIL\n"
    "## Concerns to persist\n"
    "\n"
    "- prose bullet one (the legacy pre-#2326 shape: no machine rows)\n"
    "- prose bullet two\n"
    "<!-- /epm:code-review-codex -->\n"
)


def _run(args: list[str], capsys) -> tuple[int, str]:
    rc = pvc.main(args)
    out = capsys.readouterr().out
    return rc, out


# ─── 1. K rows persist + idempotent replay ─────────────────────────────────


def test_k_valid_rows_persist_k_events_and_replay_noops(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(_THREE_ROWS)
    argv = [str(tid), "--file", str(mb), "--by", "codex-code-reviewer", "--round", "2"]
    rc, out = _run(argv, capsys)
    assert rc == 0
    assert "persisted 3/3 concern(s):" in out
    rows = _ledger_rows(tw, tid)
    assert len(rows) == 3
    assert {r["concern_id"] for r in rows} == {
        "hf-delete-scope-unbounded",
        "pack-dir-sweep-residual",
        "stray-log-line",
    }
    assert {r["severity"] for r in rows} == {"BLOCKER", "CONCERN", "NIT"}
    assert all(r["raised_by"] == "codex-code-reviewer" for r in rows)
    assert all(r["raised_at_round"] == 2 for r in rows)
    # Idempotent replay: same (id, round, severity) -> library no-op.
    rc2, _ = _run(argv, capsys)
    assert rc2 == 0
    assert len(_ledger_rows(tw, tid)) == 3


# ─── 2. Resume-recovery family (note -> $MB -> forwarder, run TWICE) ───────


def _extract_note(events_path: Path, kind: str, version: int, dest: Path) -> None:
    """Python equivalent of the SKILL.md resume-recovery jq pipe:

    ``task.py view <N> --json | jq -er --arg k <kind> --argjson v <n>
    '[.events[] | select(.kind==$k and .version==$v)] | last | .note'``
    """
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    matching = [e for e in events if e.get("kind") == kind and e.get("version") == version]
    assert matching, "fixture must seed the codex marker event"
    dest.write_text(matching[-1]["note"])


def test_resume_recovery_from_marker_note_twice_k_not_2k(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    events_path = tw.find_task_path(tid) / "events.jsonl"
    event = {
        "ts": "2026-08-16T12:21:11Z",
        "kind": "epm:code-review-codex",
        "version": 2,
        "by": "codex-code-reviewer",
        "note": _THREE_ROWS,
    }
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    assert _ledger_rows(tw, tid) == []  # the crash seam: marker posted, nothing persisted

    mb = tmp_path / "recovered-mb.md"
    argv = [
        str(tid),
        "--file",
        str(mb),
        "--by",
        "codex-code-reviewer",
        "--round",
        "2",
        "--require-block",
    ]
    outputs: list[str] = []
    for _ in range(2):
        _extract_note(events_path, "epm:code-review-codex", 2, mb)
        rc, out = _run(argv, capsys)
        assert rc == 0
        outputs.append(out)
    rows = _ledger_rows(tw, tid)
    assert len(rows) == 3, "recovery must persist K rows exactly once (not 0, not 2K)"
    # Stdout discipline: counts + kebab ids only — never summary text.
    for out in outputs:
        assert "synthetic one-liner" not in out
        assert "persisted 3/3 concern(s):" in out
        for cid in ("hf-delete-scope-unbounded", "pack-dir-sweep-residual", "stray-log-line"):
            assert cid in out


def test_resume_legacy_marker_heading_without_rows_exit3(concerns_task, tmp_path, capsys):
    """The pre-fix legacy shape at a contract site: exit 3 (the orchestrator's
    resume disposition WARNs and proceeds — prose, pinned by the lint region)."""
    _, _tw, tid = concerns_task
    mb = tmp_path / "legacy-mb.md"
    mb.write_text(_HEADING_NO_ROWS)
    rc, out = _run(
        [
            str(tid),
            "--file",
            str(mb),
            "--by",
            "codex-code-reviewer",
            "--round",
            "1",
            "--require-block",
        ],
        capsys,
    )
    assert rc == 3
    assert "heading-without-rows" in out


# ─── 3. `CONCERN:: none` handling ──────────────────────────────────────────


def test_none_sole_row_ok_nothing_persisted(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text("## Concerns to persist\n\nCONCERN:: none\n")
    rc, out = _run(
        [
            str(tid),
            "--file",
            str(mb),
            "--by",
            "codex-code-reviewer",
            "--round",
            "1",
            "--require-block",
        ],
        capsys,
    )
    assert rc == 0
    assert "persisted 0/0 concern(s)" in out
    assert _ledger_rows(tw, tid) == []


def test_none_alongside_real_row_exit1(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text("## Concerns to persist\nCONCERN:: none\nCONCERN:: CONCERN real-id a summary\n")
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 1
    assert "none-with-rows" in out
    assert _ledger_rows(tw, tid) == []


# ─── 4. Malformed rows: all-or-nothing + content-free reason codes ─────────


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ("CONCERN:: SEVERE some-id a fine summary", "bad-severity"),
        ("CONCERN:: BLOCKER Bad_ID a fine summary", "bad-id"),
        ("CONCERN:: BLOCKER lonely-id", "too-few-fields"),
    ],
)
def test_malformed_row_exit1_nothing_persisted(concerns_task, tmp_path, capsys, row, code):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    # A VALID sibling row rides along: all-or-nothing means it must not persist.
    mb.write_text(f"## Concerns to persist\nCONCERN:: CONCERN valid-id ok summary\n{row}\n")
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 1
    assert code in out
    assert _ledger_rows(tw, tid) == [], "all-or-nothing: a bad row blocks every row"
    assert "ok summary" not in out and "fine summary" not in out


def test_duplicate_id_exit1(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(
        "CONCERN:: CONCERN dup-id first summary\nCONCERN:: BLOCKER dup-id second summary\n"
    )
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 1
    assert "duplicate-id" in out
    assert _ledger_rows(tw, tid) == []


# ─── 5. Heading / block-presence handling ──────────────────────────────────


def test_heading_zero_rows_require_block_exit3(concerns_task, tmp_path, capsys):
    _, _tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(_HEADING_NO_ROWS)
    rc, out = _run(
        [str(tid), "--file", str(mb), "--by", "x", "--round", "1", "--require-block"], capsys
    )
    assert rc == 3
    assert "heading-without-rows" in out


def test_no_block_at_all_require_block_exit3(concerns_task, tmp_path, capsys):
    _, _tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text("**Verdict:** PASS\nno concerns section anywhere\n")
    rc, out = _run(
        [str(tid), "--file", str(mb), "--by", "x", "--round", "1", "--require-block"], capsys
    )
    assert rc == 3
    assert "missing-concerns-block" in out


def test_heading_zero_rows_without_flag_warns_exit0(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(_HEADING_NO_ROWS)
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 0
    assert "WARN: concerns-heading-without-rows" in out
    assert _ledger_rows(tw, tid) == []


def test_zero_rows_no_heading_without_flag_noop_exit0(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text("**Verdict:** PASS\nnothing here\n")
    rc, _ = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 0
    assert _ledger_rows(tw, tid) == []


# ─── 6. Over-cap summary: word-boundary lead + full text in evidence ───────


def test_long_summary_word_boundary_lead_full_text_in_evidence(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    long_summary = ("wordy " * 60).strip()  # 359 chars, > the 200-char cap
    mb = tmp_path / "mb.md"
    mb.write_text(f"CONCERN:: CONCERN long-summary-id {long_summary}\n")
    rc, _ = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 0
    (row,) = _ledger_rows(tw, tid)
    assert row["concern_id"] == "long-summary-id"
    assert len(row["summary"]) <= 200
    assert row["summary"].endswith("...")
    assert row["evidence"] == long_summary


# ─── 8. MAX_ROWS availability cap (#2326 unbounded-concern-row-fanout) ─────


def test_too_many_rows_exit1_zero_ledger_mutations(concerns_task, tmp_path, capsys):
    """_MAX_ROWS+1 VALID rows are rejected BEFORE any write, with a
    content-free reason code — regression test for the round-2 cap: the
    pre-cap forwarder would have run 51 durable flock+append+commit cycles."""
    _, tw, tid = concerns_task
    rows = "\n".join(
        f"CONCERN:: CONCERN cap-row-{i:03d} synthetic row number {i}"
        for i in range(pvc._MAX_ROWS + 1)
    )
    mb = tmp_path / "mb.md"
    mb.write_text(f"## Concerns to persist\n{rows}\n")
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 1
    assert "too-many-rows" in out
    assert f"{pvc._MAX_ROWS + 1} > {pvc._MAX_ROWS}" in out
    assert _ledger_rows(tw, tid) == [], "over-cap block must mutate NOTHING"
    assert "synthetic row" not in out
    # --validate-only takes the same rejection.
    rc2, out2 = _run(
        [str(tid), "--file", str(mb), "--by", "x", "--round", "1", "--validate-only"], capsys
    )
    assert rc2 == 1
    assert "too-many-rows" in out2
    assert _ledger_rows(tw, tid) == []


def test_max_rows_boundary_passes(concerns_task, tmp_path, capsys):
    """Exactly _MAX_ROWS rows stay legal (the cap is strictly-greater)."""
    _, tw, tid = concerns_task
    rows = "\n".join(f"CONCERN:: NIT boundary-row-{i:03d} synthetic" for i in range(pvc._MAX_ROWS))
    mb = tmp_path / "mb.md"
    mb.write_text(f"## Concerns to persist\n{rows}\n")
    rc, out = _run([str(tid), "--file", str(mb), "--by", "x", "--round", "1"], capsys)
    assert rc == 0
    assert f"persisted {pvc._MAX_ROWS}/{pvc._MAX_ROWS} concern(s):" in out
    assert len(_ledger_rows(tw, tid)) == pvc._MAX_ROWS


# ─── 9. Exit 4: operational failure -> partial ledger -> convergence ───────


def test_operational_failure_exit4_partial_ledger_then_converges(concerns_task, tmp_path, capsys):
    """Regression test for the #2326 exit-taxonomy fix: an operational
    persistence failure mid-loop exits 4 (never the exit-1 MALFORMED class),
    prints only the exception CLASS + counts + kebab ids, leaves the rows
    COMPLETED before the failing call durably persisted (the documented
    partial-ledger floor — this fake fails the WHOLE library call, so the
    failing row lands nothing here; the mirror-split sibling test below
    covers the intra-call case where it DOES land), and the idempotent
    re-run converges the ledger to the complete row set.

    Fails pre-fix: the pre-exit-4 forwarder let the OSError propagate as an
    uncaught traceback (process exit 1, colliding with MALFORMED)."""
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(_THREE_ROWS)
    argv = [str(tid), "--file", str(mb), "--by", "codex-code-reviewer", "--round", "2"]

    real_raise_concern = tw.raise_concern
    calls = {"n": 0}

    def flaky_raise_concern(*args, **kwargs):
        """Delegates to the real library (signature-conformant by
        delegation); raises OSError on the SECOND row only."""
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated flock/disk failure")
        return real_raise_concern(*args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(tw, "raise_concern", flaky_raise_concern):
        rc, out = _run(argv, capsys)
    assert rc == 4
    assert "OPERATIONAL: persist-failed row 2" in out
    assert "OSError" in out
    assert "1/3 persisted" in out
    assert "synthetic one-liner" not in out, "stdout must stay content-free"
    assert "simulated flock/disk failure" not in out, "exception MESSAGE must not print"
    rows = _ledger_rows(tw, tid)
    assert len(rows) == 1, "rows before the failure stay durably persisted (partial ledger)"
    assert rows[0]["concern_id"] == "hf-delete-scope-unbounded"

    # Convergence: the healthy re-run persists the full set exactly once.
    rc2, out2 = _run(argv, capsys)
    assert rc2 == 0
    assert "persisted 3/3 concern(s):" in out2
    assert len(_ledger_rows(tw, tid)) == 3, "idempotent re-run converges (not 4 rows)"


# ─── 11. Intra-call mirror split (#2326 round 3) ───────────────────────────


def test_mirror_append_failure_row_lands_uncounted_replay_never_repairs_mirror(
    concerns_task, tmp_path, capsys
):
    """Fault-injection for the intra-call mirror split (#2326 round 3,
    ``partial-persist-residual-undocumented``): fail the SECOND
    ``_append_jsonl_line`` inside ``raise_concern`` — the events.jsonl
    mirror (``task_workflow.py:8480``) — while the FIRST (the
    concerns.jsonl append, ``:8458``) succeeds. Asserts the documented
    split:

    * exit 4 reports ``0/3 persisted`` (COMPLETED calls) while the failing
      call's own row IS durably in concerns.jsonl — the count is a floor;
    * the covering commit never ran: the ledger row sits
      appended-but-uncommitted (untracked in git) until a later concern
      append commits the file by path;
    * the healthy replay converges the CONCERNS LEDGER (3 rows, no
      duplicate of row 1) but NEVER restores the missing events.jsonl
      mirror — the idempotency early-return (``:8542-8556``) keys on
      concerns.jsonl and returns before ``_append_concern_event``.

    The round-2 fake (the test above) patches ``raise_concern`` wholesale
    and structurally cannot expose this — the split lives INSIDE the
    library call.

    Scope: primary/non-routed mode (the fixture's repo root is a
    primary-style checkout on main); in ROUTED mode a fresh-process
    replay converges row AND mirror after the resolver re-sync discards
    the uncommitted row (see module docstring)."""
    repo, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_text(_THREE_ROWS)
    argv = [str(tid), "--file", str(mb), "--by", "codex-code-reviewer", "--round", "2"]

    real_append = tw._append_jsonl_line

    def mirror_failing_append(path, payload):
        """Signature-conformant by delegation; raises ONLY on the
        events.jsonl mirror append (the second append inside
        ``_append_concern_event``)."""
        if path.name == "events.jsonl":
            raise OSError("simulated mirror-append failure")
        return real_append(path, payload)

    import unittest.mock as mock

    with mock.patch.object(tw, "_append_jsonl_line", mirror_failing_append):
        rc, out = _run(argv, capsys)
    assert rc == 4
    assert "OPERATIONAL: persist-failed row 1 (hf-delete-scope-unbounded)" in out
    assert "0/3 persisted" in out, "the exit-4 count is COMPLETED calls only"
    assert "simulated mirror-append failure" not in out, "exception MESSAGE must not print"
    rows = _ledger_rows(tw, tid)
    assert len(rows) == 1, "the FAILING call's own row landed (ledger-first append)"
    assert rows[0]["concern_id"] == "hf-delete-scope-unbounded"
    assert _concern_mirror_ids(tw, tid) == [], "the events.jsonl mirror must be absent"
    # The covering commit never ran: row 1 is appended-but-uncommitted.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert "concerns.jsonl" in status, "ledger row must be uncommitted after the mirror failure"

    # Healthy replay: the LEDGER converges (3 rows, row 1 not duplicated)...
    rc2, out2 = _run(argv, capsys)
    assert rc2 == 0
    assert "persisted 3/3 concern(s):" in out2
    rows2 = _ledger_rows(tw, tid)
    assert [r["concern_id"] for r in rows2] == [
        "hf-delete-scope-unbounded",
        "pack-dir-sweep-residual",
        "stray-log-line",
    ], "replay converges the ledger: no duplicate of the split row"
    # ...but the missing mirror is NEVER restored: only rows 2 and 3 mirror.
    assert _concern_mirror_ids(tw, tid) == [
        "pack-dir-sweep-residual",
        "stray-log-line",
    ], "replay early-returns off concerns.jsonl and cannot repair the mirror"


# ─── 10. Non-UTF-8 --file -> usage exit 2 ──────────────────────────────────


def test_non_utf8_file_usage_exit2(concerns_task, tmp_path, capsys):
    """An invalid-UTF-8 --file is an INVOCATION bug (the marker was never
    examined): argparse usage exit 2, never the exit-1 MALFORMED class."""
    _, tw, tid = concerns_task
    mb = tmp_path / "mb.md"
    mb.write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
    with pytest.raises(SystemExit) as excinfo:
        pvc.main([str(tid), "--file", str(mb), "--by", "x", "--round", "1"])
    assert excinfo.value.code == 2
    assert _ledger_rows(tw, tid) == []


# ─── 7. --validate-only writes nothing ─────────────────────────────────────


def test_validate_only_writes_nothing_on_pass_and_fail(concerns_task, tmp_path, capsys):
    _, tw, tid = concerns_task
    ok = tmp_path / "ok.md"
    ok.write_text(_THREE_ROWS)
    rc, out = _run(
        [str(tid), "--file", str(ok), "--by", "x", "--round", "2", "--validate-only"], capsys
    )
    assert rc == 0
    assert "concerns-block OK: 3 row(s)" in out
    assert _ledger_rows(tw, tid) == []

    bad = tmp_path / "bad.md"
    bad.write_text(_HEADING_NO_ROWS)
    rc, _ = _run(
        [
            str(tid),
            "--file",
            str(bad),
            "--by",
            "x",
            "--round",
            "1",
            "--require-block",
            "--validate-only",
        ],
        capsys,
    )
    assert rc == 3
    assert _ledger_rows(tw, tid) == []
