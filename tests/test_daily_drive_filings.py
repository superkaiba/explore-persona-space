"""Tests for scripts/daily_drive_filings.py — the /daily durable+incremental filing driver.

Executes the REAL driver body (``main(argv)``) end to end; the only fakes sit at the
external boundaries (#906 discipline): a stub filer EXECUTABLE (run through the driver's
real ``subprocess.run`` path via ``--filer``; it records its argv and snapshots the
ledger at invocation time) and a synthetic ``--tasks-root`` tree whose task frontmatter
is built through the REAL frontmatter writer (``yaml.safe_dump``, matching
``task_workflow._join_frontmatter`` — daily titles contain ``": "`` so the title line is
QUOTED, ``title: 'daily-fix: ...'``; a hand-crafted unquoted fixture would false-PASS a
recovery that never fires in production).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import daily_drive_filings as ddf  # noqa: E402
import file_infra_task as fit  # noqa: E402

from explore_persona_space.task_workflow import wf_fix_fingerprint  # noqa: E402

DATE = "2026-07-05"

STUB_TEMPLATE = """\
import json
import shutil
import sys
from pathlib import Path

DIR = Path({dirpath!r})
ledger = DIR / "filed.jsonl"
snap = DIR / "ledger_snapshot_at_invocation.jsonl"
if ledger.exists():
    shutil.copy(ledger, snap)
else:
    snap.write_text("")
with open(DIR / "filer_calls.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
for line in {stderr_lines!r}:
    print(line, file=sys.stderr)
fail_marker = {fail_marker!r}
if fail_marker and any(fail_marker in a for a in sys.argv[1:]):
    print("stub filer exploded", file=sys.stderr)
    sys.exit(1)
print({output_line!r})
sys.exit({exit_code})
"""


def make_stub(
    tmp_path: Path,
    dirpath: Path,
    *,
    output_line: str = "filed #1234; infra dispatch cap (5) full, NOT dispatching",
    exit_code: int = 0,
    fail_marker: str | None = None,
    name: str = "stub_filer.py",
    stderr_lines: list[str] | None = None,
) -> str:
    """Write a stub filer executable; return the --filer prefix string.

    ``stderr_lines`` (#1529) are printed to the stub's stderr BEFORE the stdout
    line, mirroring the real filer's step-0 advisory ordering; the default (no
    lines) keeps every pre-existing call site byte-identical.
    """
    stub = tmp_path / name
    stub.write_text(
        STUB_TEMPLATE.format(
            dirpath=str(dirpath),
            output_line=output_line,
            exit_code=exit_code,
            fail_marker=fail_marker,
            stderr_lines=list(stderr_lines or []),
        ),
        encoding="utf-8",
    )
    return f"{sys.executable} {stub}"


def make_item(slug: str = "fix-a", route: int = 2, **overrides) -> dict:
    item = {
        "slug": slug,
        "route": route,
        "title": f"daily-fix: {slug}",
        "target": ".claude/skills/daily/SKILL.md",
        "bug": f"bug text for {slug}",
        "change": f"change text for {slug}",
    }
    item.update(overrides)
    return item


def make_filings_dir(tmp_path: Path, items: list[dict], date: str = DATE) -> Path:
    d = tmp_path / f"filings-{date}"
    d.mkdir()
    for it in items:
        body_text = f"## Goal\n\n{it.get('bug', '')}\n"
        (d / f"{it['slug']}.md").write_text(body_text, encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(items), encoding="utf-8")
    return d


def make_task(
    tasks_root: Path,
    status: str,
    tid: int,
    *,
    title: str,
    tags: list[str] | None = None,
    origin_prompt: str | None = None,
) -> Path:
    """Build one synthetic task dir through the REAL frontmatter writer (yaml.safe_dump)."""
    fm = {"title": title, "kind": "infra", "tags": tags or []}
    if origin_prompt is not None:
        fm["origin_prompt"] = origin_prompt
    task_dir = tasks_root / status / str(tid)
    task_dir.mkdir(parents=True)
    fm_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
    body = task_dir / "body.md"
    body.write_text(f"---\n{fm_block}\n---\n## Goal\n\nsynthetic\n", encoding="utf-8")
    return body


def ledger_rows(d: Path) -> list[dict]:
    """Item ledger rows only. The #1735 terminal daily-drive-summary row is
    filtered out by default so every pre-existing per-item assertion stays
    byte-identical; tests that inspect the summary row use
    :func:`ledger_rows_all` below.
    """
    path = d / "filed.jsonl"
    if not path.exists():
        return []
    return [
        row
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
        for row in (json.loads(ln),)
        if row.get("outcome") != "daily-drive-summary"
    ]


def ledger_rows_all(d: Path) -> list[dict]:
    """Every ledger row, INCLUDING the #1735 terminal daily-drive-summary row."""
    path = d / "filed.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def filer_calls(d: Path) -> list[list[str]]:
    path = d / "filer_calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_driver(d: Path, tasks_root: Path, filer: str, *extra: str) -> int:
    argv = ["--dir", str(d), "--tasks-root", str(tasks_root), "--filer", filer, *extra]
    return ddf.main(argv)


def tag_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--tag"]


def title_value(argv: list[str]) -> str:
    """The --title argv value the driver composed (mirrors tag_values)."""
    return argv[argv.index("--title") + 1]


@pytest.fixture()
def tasks_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    root.mkdir()
    return root


# The un-neutralized original — the ``_isolate_closed_sibling_probe`` autouse
# fixture below saves + replaces this; unit tests that call the real function
# through a monkeypatched Hub helper (the ``test_find_closed_sibling_suspects_*``
# family) reach for this alias instead.
_REAL_FIND_CLOSED_SIBLING_SUSPECTS = ddf.find_closed_sibling_suspects


@pytest.fixture(autouse=True)
def _isolate_closed_sibling_probe(monkeypatch):
    """Neutralize the #1711 closed-sibling probe by default across every test.

    The probe reads the LIVE ``task_workflow.registry_path()`` / ``repo_root()``
    globals (they are not seeded by the driver's ``--tasks-root`` CLI seam), so
    without this fixture every test in this file would false-fire against real
    closed-sibling tasks in the working tree — breaking every existing test that
    predates #1711. Tests that specifically exercise the closed-sibling probe
    override ``ddf.find_closed_sibling_suspects`` per-test (see the
    ``test_closed_sibling_probe_*`` section at file bottom). A unit test that
    needs to call the REAL production body (with a fake at the
    ``task_workflow.recent_closed_workflow_fix_tasks`` seam) uses the module-
    level alias ``_REAL_FIND_CLOSED_SIBLING_SUSPECTS`` above.
    """
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda item: ([], []))


# ── case 1: route-2 filing — tags, fp, parsed id ───────────────────────────────


def test_route2_files_with_tags_and_parsed_id(tmp_path, tasks_root):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["attempting", "filed"]
    assert rows[1]["id"] == 1234 and rows[1]["rc"] == 0
    assert all("ts" in r for r in rows)
    (call,) = filer_calls(d)
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    tags = tag_values(call)
    assert tags == ["wf-fix", f"wf-fix-fp:{fp}", "daily-auto-filed"]
    assert "--no-dispatch" not in call
    assert str(d / "fix-a.md") in call


# ── case 2: route-3 filing — held tags, no dispatch, no wf-fix ─────────────────


def test_route3_files_held_tags_no_dispatch(tmp_path, tasks_root):
    d = make_filings_dir(tmp_path, [make_item("hold-a", route=3)])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    (call,) = filer_calls(d)
    assert tag_values(call) == ["daily-held", "needs-human"]
    assert "--no-dispatch" in call
    assert not any(v.startswith("wf-fix") for v in tag_values(call))


# ── case 3: resume — terminal slug skipped, filer not invoked ──────────────────


def test_resume_skips_terminal_slug(tmp_path, tasks_root, capsys):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    seed = {"slug": "fix-a", "outcome": "filed", "id": 1044, "rc": 0, "route": 2}
    (d / "filed.jsonl").write_text(json.dumps(seed) + "\n", encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert "SKIP fix-a" in capsys.readouterr().out
    assert filer_calls(d) == []
    assert ledger_rows(d) == [seed]


# ── case 4: route-2 fp dedup — open tasks dedupe, terminal statuses do not ─────


def test_dedup_route2_open_task_only(tmp_path, tasks_root):
    item = make_item("fix-a")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    make_task(tasks_root, "proposed", 900, title="daily-fix: other", tags=[f"wf-fix-fp:{fp}"])
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["deduped"]
    assert "tasks/proposed/900" in rows[0]["against"].replace(str(tasks_root), "tasks")
    assert filer_calls(d) == []


def test_dedup_ignores_completed_and_archived(tmp_path, tasks_root):
    item = make_item("fix-a")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    make_task(tasks_root, "completed", 900, title="daily-fix: other", tags=[f"wf-fix-fp:{fp}"])
    make_task(tasks_root, "archived", 901, title="daily-fix: other2", tags=[f"wf-fix-fp:{fp}"])
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


# ── case 5: recovery of a trailing `attempting` row ────────────────────────────


def _seed_attempting(d: Path, slug: str, item: dict, id_floor: int) -> None:
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    row = {
        "slug": slug,
        "outcome": "attempting",
        "fp": fp,
        "route": item["route"],
        "id_floor": id_floor,
    }
    with open(d / "filed.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def test_recovery_single_title_match_above_floor(tmp_path, tasks_root, capsys):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title=item["title"], tags=["wf-fix", "daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert rows[-1]["outcome"] == "recovered"
    assert rows[-1]["id"] == 150
    assert rows[-1]["dispatch_unconfirmed"] is True
    assert "RECOVERED fix-a -> #150" in capsys.readouterr().out
    assert filer_calls(d) == []


def test_recovery_no_title_match_files_normally(tmp_path, tasks_root):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title="daily-fix: unrelated", tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert ledger_rows(d)[-1]["outcome"] == "filed"
    assert len(filer_calls(d)) == 1


def test_recovery_decoy_below_id_floor_not_recovered(tmp_path, tasks_root):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    # A same-titled, same-tagged task from a PRIOR night, at or below the floor.
    make_task(tasks_root, "proposed", 50, title=item["title"], tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert not any(r["outcome"] == "recovered" for r in rows)
    assert rows[-1]["outcome"] == "filed"
    assert len(filer_calls(d)) == 1


def test_recovery_two_matches_is_ambiguous_error(tmp_path, tasks_root):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title=item["title"], tags=["daily-auto-filed"])
    make_task(tasks_root, "proposed", 151, title=item["title"], tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 1
    last = ledger_rows(d)[-1]
    assert last["outcome"] == "ERROR" and last["flag"] == "ambiguous-recovery"
    assert filer_calls(d) == []


def test_recovery_requires_route_tag(tmp_path, tasks_root):
    # Same title above the floor but WITHOUT the run's filing tag: not recovered.
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title=item["title"], tags=["needs-human"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert ledger_rows(d)[-1]["outcome"] == "filed"


# ── case 6: filer failure — ERROR row, later items processed, retry semantics ──


def test_filer_failure_records_error_and_continues(tmp_path, tasks_root):
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, fail_marker="fix-a"))
    assert rc == 1
    by_slug = {(r["slug"], r["outcome"]) for r in ledger_rows(d)}
    assert ("fix-a", "ERROR") in by_slug
    assert ("fix-b", "filed") in by_slug  # later items still processed
    err = next(r for r in ledger_rows(d) if r["slug"] == "fix-a" and r["outcome"] == "ERROR")
    assert err["flag"] == "filer-failed" and err["rc"] == 1

    # Re-invocation WITHOUT --retry-errors skips the ERROR slug.
    n_calls = len(filer_calls(d))
    assert run_driver(d, tasks_root, make_stub(tmp_path, d, name="stub2.py")) == 0
    assert len(filer_calls(d)) == n_calls

    # WITH --retry-errors it retries (recovery scan finds nothing -> re-files).
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, name="stub3.py"), "--retry-errors")
    assert rc == 0
    assert len(filer_calls(d)) == n_calls + 1
    assert ("fix-a", "filed") in {(r["slug"], r["outcome"]) for r in ledger_rows(d)}


# ── case 7: dry-run — no subprocess, ledger untouched ──────────────────────────


def test_dry_run_no_subprocess_no_ledger_writes(tmp_path, tasks_root, capsys):
    d = make_filings_dir(tmp_path, [make_item("fix-a")])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    assert "FILE fix-a" in capsys.readouterr().out
    assert filer_calls(d) == []
    assert not (d / "filed.jsonl").exists()


# ── case 8: fail-loud validation — aborts at ZERO filings ──────────────────────


def test_missing_manifest_raises(tmp_path, tasks_root):
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))


def test_missing_body_file_raises_before_any_filing(tmp_path, tasks_root):
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    (d / "fix-b.md").unlink()
    with pytest.raises(FileNotFoundError, match="fix-b"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert filer_calls(d) == []  # up-front validation: item 1 was NOT filed
    assert not (d / "filed.jsonl").exists()


def test_malformed_manifest_item_aborts_at_zero_filings(tmp_path, tasks_root):
    good = make_item("fix-a")
    bad = make_item("fix-b")
    del bad["bug"]
    d = make_filings_dir(tmp_path, [good, bad])
    with pytest.raises(ValueError, match="missing keys"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert filer_calls(d) == []
    assert not (d / "filed.jsonl").exists()


def test_bad_route_rejected(tmp_path, tasks_root):
    d = make_filings_dir(tmp_path, [make_item("fix-a", route=5)])
    with pytest.raises(ValueError, match="route"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))


def test_unresolvable_date_raises(tmp_path, tasks_root):
    d = tmp_path / "not-a-dated-dir"
    d.mkdir()
    (d / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="date"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))


# ── case 9: cwd-independence — relative --dir resolves under the repo root ─────


def expected_repo_root() -> Path:
    out = subprocess.run(
        ["git", "-C", str(SCRIPTS), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    p = Path(out)
    if not p.is_absolute():
        p = (SCRIPTS / p).resolve()
    return p.parent


def test_relative_dir_resolves_under_repo_root_not_cwd(tmp_path, tasks_root, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Plant a decoy at the CWD-relative path: the driver must NOT pick it up.
    decoy = tmp_path / "logs" / "daily" / "filings-2099-01-01"
    decoy.mkdir(parents=True)
    (decoy / "manifest.json").write_text("[]", encoding="utf-8")
    root = expected_repo_root()
    assert ddf.repo_root() == root
    assert not (root / "logs/daily/filings-2099-01-01").exists(), "test precondition"
    with pytest.raises(FileNotFoundError) as exc:
        ddf.main(["--dir", "logs/daily/filings-2099-01-01", "--tasks-root", str(tasks_root)])
    assert str(root) in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


# ── case 10: ordering pin — `attempting` row lands BEFORE the filer subprocess ─


def test_attempting_row_precedes_filer_subprocess(tmp_path, tasks_root):
    d = make_filings_dir(tmp_path, [make_item("fix-a")])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    snap_lines = (
        (d / "ledger_snapshot_at_invocation.jsonl").read_text(encoding="utf-8").splitlines()
    )
    snap = [json.loads(ln) for ln in snap_lines if ln.strip()]
    assert [r["outcome"] for r in snap] == ["attempting"], (
        "at filer-invocation time the ledger must hold the slug's attempting row "
        "and NO terminal row (two-phase ordering)"
    )
    assert "id_floor" in snap[0]
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


# ── case 11: rc=0 with no parseable id -> ERROR no-id-parsed; retry recovers ───


def test_rc0_no_id_is_error_then_retry_recovers_without_refiling(tmp_path, tasks_root):
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, output_line="done, no id line"))
    assert rc == 1
    rows = ledger_rows(d)
    assert not any(r["outcome"] == "filed" for r in rows), "never a filed row with a null id"
    err = rows[-1]
    assert err["outcome"] == "ERROR" and err["flag"] == "no-id-parsed" and err["rc"] == 0

    # The filing may in fact have committed: --retry-errors runs recovery BEFORE re-filing.
    make_task(tasks_root, "proposed", 150, title=item["title"], tags=["daily-auto-filed"])
    n_calls = len(filer_calls(d))
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, name="stub2.py"), "--retry-errors")
    assert rc == 0
    assert len(filer_calls(d)) == n_calls, "recovered WITHOUT invoking the filer"
    last = ledger_rows(d)[-1]
    assert last["outcome"] == "recovered" and last["id"] == 150


# ── case 12: ledger corruption — trailing tolerated, non-trailing fails loud ───


def test_corrupt_trailing_ledger_line_quarantined(tmp_path, tasks_root, capsys):
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    valid = {"slug": "fix-a", "outcome": "filed", "id": 1044, "rc": 0, "route": 2}
    (d / "filed.jsonl").write_text(
        json.dumps(valid) + "\n" + '{"slug": "fix-b", "outc', encoding="utf-8"
    )
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert "quarantined" in capsys.readouterr().err
    assert (d / "filed.jsonl.quarantined").exists()
    rows = ledger_rows(d)
    outcomes = {(r["slug"], r["outcome"]) for r in rows}
    assert ("fix-b", "filed") in outcomes  # the in-flight item resumed normally
    assert all(r["slug"] != "fix-a" or r["outcome"] == "filed" for r in rows)
    assert len(filer_calls(d)) == 1  # fix-a skipped, fix-b filed


def test_corrupt_non_trailing_ledger_line_fails_loud(tmp_path, tasks_root):
    d = make_filings_dir(tmp_path, [make_item("fix-a")])
    valid = {"slug": "fix-a", "outcome": "filed", "id": 1044, "rc": 0, "route": 2}
    (d / "filed.jsonl").write_text('{"broken json\n' + json.dumps(valid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt non-trailing"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))


# ── case 13: #1173 route-2 wf-fix Provenance injection (durable recursion guard) ─


def test_route2_injects_wf_fix_provenance_when_absent(tmp_path, tasks_root, capsys):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])  # default body: `## Goal\n\n{bug}\n`, no line
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    body = (d / "fix-a.md").read_text(encoding="utf-8")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    # The exact consumer needles: is_open_workflow_fix_task's `workflow_fix_target:
    # {target_file}` (single space) + the `fingerprint: {fp}` body fallback.
    assert f"workflow_fix_target: {item['target']}" in body
    assert f"fingerprint: {fp}" in body
    assert body.count("## Provenance") == 1
    assert "INJECTED fix-a" in capsys.readouterr().out
    # Injection happens BEFORE filing; the filer still receives the same body path.
    (call,) = filer_calls(d)
    assert str(d / "fix-a.md") in call


def test_route2_injection_idempotent_when_lines_present(tmp_path, tasks_root, capsys):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    pre = (
        "## Goal\n\nbug text for fix-a\n\n## Provenance\n\n"
        f"- workflow_fix_target: {item['target']}\n- fingerprint: {fp}\n"
    )
    (d / "fix-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert (d / "fix-a.md").read_text(encoding="utf-8") == pre  # byte-unchanged
    assert "INJECTED" not in capsys.readouterr().out


def test_route2_inserts_under_existing_provenance_heading(tmp_path, tasks_root):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    # The #1134 shape: a `## Provenance` heading with only an `- Evidence:` line.
    pre = "## Goal\n\nbug\n\n## Provenance\n\n- Evidence: ccc66ab4 (#825) 09:45Z.\n"
    (d / "fix-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    body = (d / "fix-a.md").read_text(encoding="utf-8")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    assert body.count("## Provenance") == 1  # no duplicate heading
    assert f"workflow_fix_target: {item['target']}" in body
    assert f"fingerprint: {fp}" in body
    assert "- Evidence: ccc66ab4 (#825) 09:45Z." in body  # pre-existing line preserved


def test_route3_body_never_injected(tmp_path, tasks_root):
    item = make_item("hold-a", route=3)
    d = make_filings_dir(tmp_path, [item])
    pre = (d / "hold-a.md").read_text(encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert (d / "hold-a.md").read_text(encoding="utf-8") == pre  # byte-unchanged


def test_dry_run_injection_reports_but_does_not_write(tmp_path, tasks_root, capsys):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    pre = (d / "fix-a.md").read_text(encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    assert (d / "fix-a.md").read_text(encoding="utf-8") == pre  # dry-run stays write-free
    assert not (d / "filed.jsonl").exists()
    assert filer_calls(d) == []
    assert "[will inject workflow_fix_target provenance]" in capsys.readouterr().out


# ── case 13b: #1580 fp reconcile (tag-authoritative) + anchored-line detection ──


def test_route2_reconciles_mismatched_body_fingerprint_to_tag(tmp_path, tasks_root, capsys):
    """A body-carried fp that disagrees with the manifest-computed tag fp is rewritten
    in place to the tag value, preserving the old value as a labeled substring (#1580;
    the #1554-#1571/#1579 mismatch shape)."""
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    old = "44d3a4598f5c"  # the #1570 body-carried fp from the parked candidate block
    assert old != fp
    pre = (
        "## Goal\n\nbug text for fix-a\n\n## Provenance\n\n"
        f"- workflow_fix_target: {item['target']}\n- fingerprint: {old}\n"
    )
    (d / "fix-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    body = (d / "fix-a.md").read_text(encoding="utf-8")
    # The tag value is the ONLY anchored `- fingerprint:` value post-reconcile ...
    assert re.search(rf"(?m)^-\s*fingerprint:\s*{fp}(?![0-9a-f])", body)
    assert f"(tag-authoritative; supersedes body-carried fingerprint: {old})" in body
    # ... while the old value survives ONLY as the substring the OR-predicates match.
    assert f"fingerprint: {old}" in body
    assert re.search(rf"(?m)^\s*-\s*fingerprint:\s*{old}", body) is None
    assert "RECONCILED fix-a" in capsys.readouterr().out
    # Filing proceeded under the manifest-computed tag fp.
    (call,) = filer_calls(d)
    assert f"wf-fix-fp:{fp}" in tag_values(call)
    # Neither 12-hex value trips the #1467 sha scan (SHA_EXCLUDE_LINE_RE skips the line).
    assert "sha-verify (filing-time, #1467)" not in body


def test_route2_injects_fingerprint_when_only_prose_mention_present(tmp_path, tasks_root, capsys):
    """A mid-line prose mention of 'fingerprint:' no longer suppresses injection of the
    anchored dedup line (#1580's own body was the incident: prose mention, no line)."""
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    prose = "- **Bug observed:** filed tag disagrees with its Provenance fingerprint: 44d3a4598f5c"
    pre = (
        "## Goal\n\nbug text for fix-a\n\n## Workflow gap\n\n"
        f"{prose}\n\n## Provenance\n\n- workflow_fix_target: {item['target']}\n"
    )
    (d / "fix-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    body = (d / "fix-a.md").read_text(encoding="utf-8")
    assert re.search(rf"(?m)^-\s*fingerprint:\s*{fp}(?![0-9a-f])", body)  # injected
    assert prose in body  # prose line untouched
    assert "INJECTED fix-a" in capsys.readouterr().out


def test_route2_reconcile_idempotent():
    """One reconcile pass makes every anchored value == tag fp; a second pass is a
    byte no-op returning [] (the reconciled line's mid-line old fp is not bullet-initial)."""
    target = ".claude/skills/daily/SKILL.md"
    fp = "f55e38afc131"
    pre = (
        "## Goal\n\nx\n\n## Provenance\n\n"
        f"- workflow_fix_target: {target}\n- fingerprint: 44d3a4598f5c\n"
    )
    once, actions = ddf.ensure_wf_fix_provenance(pre, target, fp)
    assert actions == ["fp-reconcile"]
    assert f"- fingerprint: {fp} (tag-authoritative" in once
    twice, actions2 = ddf.ensure_wf_fix_provenance(once, target, fp)
    assert actions2 == []
    assert twice == once  # byte-identical


def test_dry_run_reconcile_reports_but_does_not_write(tmp_path, tasks_root, capsys):
    """Dry-run parity for the reconcile branch: reported, never written (#1580)."""
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    pre = (
        "## Goal\n\nbug text for fix-a\n\n## Provenance\n\n"
        f"- workflow_fix_target: {item['target']}\n- fingerprint: 44d3a4598f5c\n"
    )
    (d / "fix-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    assert (d / "fix-a.md").read_text(encoding="utf-8") == pre  # dry-run stays write-free
    assert not (d / "filed.jsonl").exists()
    assert filer_calls(d) == []
    assert "[will reconcile body fingerprint -> tag value]" in capsys.readouterr().out


def test_find_open_fp_duplicate_matches_body_fingerprint_line_not_prose(tasks_root):
    """The route-2 dedup scan matches the tag needle OR an ANCHORED `- fingerprint:`
    Provenance line — never a mid-line prose quote (#1580: bare substring would let
    a body quoting another task's fp false-suppress a genuine re-raise)."""
    fp_line, fp_prose, fp_term = "aabbccddee11", "aabbccddee22", "aabbccddee33"
    t_line = tasks_root / "proposed" / "101"
    t_line.mkdir(parents=True)
    (t_line / "body.md").write_text(
        f"---\ntitle: a\n---\n## Provenance\n\n- fingerprint: {fp_line}\n", encoding="utf-8"
    )
    t_prose = tasks_root / "proposed" / "102"
    t_prose.mkdir(parents=True)
    (t_prose / "body.md").write_text(
        f"---\ntitle: b\n---\n## Goal\n\nquotes a sibling's fingerprint: {fp_prose} in prose\n",
        encoding="utf-8",
    )
    t_term = tasks_root / "completed" / "103"
    t_term.mkdir(parents=True)
    (t_term / "body.md").write_text(
        f"---\ntitle: c\n---\n## Provenance\n\n- fingerprint: {fp_term}\n", encoding="utf-8"
    )
    # Anchored Provenance line, NO tag, open status -> found (the #1580 OR-widening).
    assert ddf.find_open_fp_duplicate(tasks_root, fp_line) == t_line / "body.md"
    # Mid-line prose quote only -> NOT found (no false suppression).
    assert ddf.find_open_fp_duplicate(tasks_root, fp_prose) is None
    # Terminal statuses stay skipped even with an anchored line.
    assert ddf.find_open_fp_duplicate(tasks_root, fp_term) is None


# ── case 14: #1228 wf_fix flag — route-2 non-workflow-surface variant ──────────


def test_route2_wf_fix_false_drops_wf_fix_tags_keeps_daily_auto_filed(tmp_path, tasks_root):
    item = make_item("expcode-a", route=2, wf_fix=False)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["attempting", "filed"]
    # fp stays recorded on every ledger row (audit value, not a tag claim — #1228).
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    assert all(r["fp"] == fp for r in rows)
    (call,) = filer_calls(d)
    assert tag_values(call) == ["daily-auto-filed"]
    assert "--no-dispatch" not in call  # still auto-dispatches (route 2, not route 3)


def test_route2_wf_fix_false_skips_provenance_injection(tmp_path, tasks_root, capsys):
    item = make_item("expcode-a", route=2, wf_fix=False)
    d = make_filings_dir(tmp_path, [item])
    pre = (d / "expcode-a.md").read_text(encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    body = (d / "expcode-a.md").read_text(encoding="utf-8")
    assert body == pre  # byte-unchanged: no normalization for wf_fix: false
    # Recursion-guard-off pin: the filed body must NOT carry the durable signal
    # task_workflow.is_workflow_fix_session() reads.
    assert ddf.WF_FIX_TARGET_KEY not in body
    assert "INJECTED" not in capsys.readouterr().out
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


def test_route2_wf_fix_false_not_deduped_against_open_fp_task(tmp_path, tasks_root):
    item = make_item("expcode-a", route=2, wf_fix=False)
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    # An OPEN task carrying this item's own fp tag would dedup a wf-fix filing;
    # a wf_fix: false item exits the wf-fix key space, so it FILES.
    make_task(tasks_root, "proposed", 900, title="daily-fix: other", tags=[f"wf-fix-fp:{fp}"])
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route2_wf_fix_true_explicit_is_identical_to_default(tmp_path):
    item_default = make_item("fix-a", route=2)
    item_explicit = make_item("fix-a", route=2, wf_fix=True)
    fp = wf_fix_fingerprint(item_default["change"], item_default["bug"])
    cmd_default = ddf._filer_cmd([], item_default, Path("-"), DATE, fp)
    cmd_explicit = ddf._filer_cmd([], item_explicit, Path("-"), DATE, fp)
    assert cmd_default == cmd_explicit
    assert tag_values(cmd_default) == ["wf-fix", f"wf-fix-fp:{fp}", "daily-auto-filed"]


def test_manifest_wf_fix_non_bool_rejected_at_zero_filings(tmp_path, tasks_root):
    # Good item FIRST, bad item second: the abort must happen at VALIDATE time
    # (zero filings), not mid-run after item 1 already filed.
    good = make_item("fix-a", route=2)
    bad = make_item("fix-b", route=2, wf_fix="false")  # JSON string "false" is truthy
    d = make_filings_dir(tmp_path, [good, bad])
    with pytest.raises(ValueError, match="wf_fix"):
        run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert filer_calls(d) == []
    assert ledger_rows(d) == []


def test_dry_run_wf_fix_false_reports_reduced_tags_no_inject(tmp_path, tasks_root, capsys):
    item = make_item("expcode-a", route=2, wf_fix=False)
    d = make_filings_dir(tmp_path, [item])
    pre = (d / "expcode-a.md").read_text(encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("FILE expcode-a"))
    assert "daily-auto-filed" in line
    assert "wf-fix" not in line
    assert "will inject" not in line
    assert (d / "expcode-a.md").read_text(encoding="utf-8") == pre  # write-free
    assert not (d / "filed.jsonl").exists()
    assert filer_calls(d) == []


def test_route3_stray_wf_fix_key_ignored(tmp_path, tasks_root):
    # A stray wf_fix key on a route-3 item is type-checked but semantically ignored.
    d = make_filings_dir(tmp_path, [make_item("hold-a", route=3, wf_fix=True)])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    (call,) = filer_calls(d)
    assert tag_values(call) == ["daily-held", "needs-human"]
    assert "--no-dispatch" in call


def test_route2_wf_fix_false_recovery_still_works(tmp_path, tasks_root, capsys):
    # Kill-window recovery keys on title + the daily-auto-filed route tag + id_floor —
    # all of which a wf_fix: false filing keeps.
    item = make_item("expcode-a", route=2, wf_fix=False)
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "expcode-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title=item["title"], tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert rows[-1]["outcome"] == "recovered" and rows[-1]["id"] == 150
    assert "RECOVERED expcode-a -> #150" in capsys.readouterr().out
    assert filer_calls(d) == []


def test_route2_wf_fix_false_body_with_provenance_line_warns(tmp_path, tasks_root, capsys):
    # WARN-only guard (#1228): a hand-added workflow_fix_target: line on a
    # wf_fix: false body would arm the recursion guard for a non-workflow-fix
    # session — the driver warns on stderr but never blocks or rewrites.
    item = make_item("expcode-a", route=2, wf_fix=False)
    d = make_filings_dir(tmp_path, [item])
    pre = "## Goal\n\nbug\n\n## Provenance\n\n- workflow_fix_target: scripts/foo.py\n"
    (d / "expcode-a.md").write_text(pre, encoding="utf-8")
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert "WARNING expcode-a" in capsys.readouterr().err
    assert (d / "expcode-a.md").read_text(encoding="utf-8") == pre  # never rewritten
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]  # never blocked


# ── #1273: route-2 titles missing a WF_FIX_TITLE_PREFIXES prefix gain daily-fix: ─


def test_route2_bare_title_gains_daily_fix_prefix_before_truncation(tmp_path, tasks_root):
    # Durability pin (#1273 plan §10): prepend happens BEFORE the [:60] cut, so a
    # 55-char bare title files as ("daily-fix: " + "x" * 55)[:60] — len 60, 49 x's.
    item = make_item("fix-a", title="x" * 55)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    title = title_value(filer_calls(d)[0])
    assert title == ("daily-fix: " + "x" * 55)[:60]
    assert title.startswith("daily-fix: ")
    assert len(title) == 60
    assert title.endswith("x" * 49)


def test_route2_prefixed_title_unchanged(tmp_path, tasks_root):
    item = make_item("fix-a")  # default title "daily-fix: fix-a"
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    title = title_value(filer_calls(d)[0])
    assert title == item["title"][:60]
    assert title.count("daily-fix:") == 1


def test_route2_workflow_fix_prefix_not_double_prefixed(tmp_path, tasks_root):
    # The OTHER channel prefix also satisfies the guard — filed verbatim.
    item = make_item("fix-a", title="workflow-fix: xyz")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    title = title_value(filer_calls(d)[0])
    assert title == "workflow-fix: xyz"
    assert "daily-fix:" not in title


def test_route3_title_never_prefixed(tmp_path, tasks_root):
    item = make_item("held-a", route=3, title="held judgment call")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert title_value(filer_calls(d)[0]) == "held judgment call"


def test_route2_wf_fix_false_title_also_prefixed(tmp_path, tasks_root):
    # Scope pin (#1273 plan §4): the prefix is the CHANNEL marker, keyed on
    # route == 2 alone — a wf_fix: false (experiment-code) item is prefixed too.
    item = make_item("expcode-a", wf_fix=False, title="bare experiment-code fix")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert title_value(filer_calls(d)[0]) == "daily-fix: bare experiment-code fix"


def test_recovery_matches_effective_title_for_bare_manifest_title(tmp_path, tasks_root):
    # File-site/recovery consistency: the recovery scan finds the task the
    # POST-fix driver would have filed (effective title), so no double-file.
    item = make_item("fix-a", title="bare fix title")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(
        tasks_root, "proposed", 150, title="daily-fix: bare fix title", tags=["daily-auto-filed"]
    )
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert rows[-1]["outcome"] == "recovered" and rows[-1]["id"] == 150
    assert filer_calls(d) == []


def test_recovery_also_matches_bare_title_from_prefix_migration(tmp_path, tasks_root):
    # Migration window (#1273 constraint 4): a crashed PRE-fix driver filed the
    # BARE title; the post-fix resume must still recover it, not refile.
    item = make_item("fix-a", title="bare fix title")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(tasks_root, "proposed", 150, title="bare fix title", tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert rows[-1]["outcome"] == "recovered" and rows[-1]["id"] == 150
    assert filer_calls(d) == []


def test_recovery_bare_and_prefixed_both_match_is_ambiguous(tmp_path, tasks_root):
    # BOTH title forms above the floor IS a real double-file — the union feeds
    # the existing ambiguity rule: ERROR for manual disposition, no filer call.
    item = make_item("fix-a", title="bare fix title")
    d = make_filings_dir(tmp_path, [item])
    _seed_attempting(d, "fix-a", item, id_floor=100)
    make_task(
        tasks_root, "proposed", 150, title="daily-fix: bare fix title", tags=["daily-auto-filed"]
    )
    make_task(tasks_root, "proposed", 151, title="bare fix title", tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 1
    last = ledger_rows(d)[-1]
    assert last["outcome"] == "ERROR" and last["flag"] == "ambiguous-recovery"
    assert "150" in last["tail"] and "151" in last["tail"]
    assert filer_calls(d) == []


# ── #1467: sha-verify backstop (scan + annotation + ledger key) ─────────────────


def _init_hermetic_repo(tmp_path: Path) -> tuple[Path, str]:
    """A tmp git repo with exactly ONE commit — hermetic resolution oracle (#1467).

    Never the live repo's object set: resolution/non-resolution of every fixture
    token is decided by THIS repo alone. Returns (repo_path, full HEAD sha).
    """
    repo = tmp_path / "hermetic_repo"
    repo.mkdir()

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )

    _git("init", "-q")
    (repo / "f.txt").write_text("x", encoding="utf-8")
    _git("add", "f.txt")
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "c1",
    )
    return repo, _git("rev-parse", "HEAD").stdout.strip()


def test_scan_flags_nonresolving_commit_context_token(tmp_path):
    # BACKTICKED token — the real filed-body shape; pins HEX_TOKEN_RE's \b
    # backtick-boundary behavior against future regex drift.
    repo, _head = _init_hermetic_repo(tmp_path)
    body = "## Workflow gap\n\nthe fix commit `deadbee7f00d` landed yesterday.\n"
    tier1, tier2 = ddf.scan_unresolvable_shas(body, repo)
    assert tier1 == ["deadbee7f00d"]
    assert tier2 == []


def test_scan_resolving_sha_not_flagged(tmp_path):
    repo, head = _init_hermetic_repo(tmp_path)
    body = f"## Workflow gap\n\nthe fix commit `{head}` landed.\n"
    tier1, tier2 = ddf.scan_unresolvable_shas(body, repo)
    assert tier1 == []
    assert tier2 == []


def test_scan_skips_fingerprint_lines_and_all_digit_tokens(tmp_path):
    repo, _head = _init_hermetic_repo(tmp_path)
    body = (
        "## Provenance\n\n"
        "- fingerprint: 9608dfe5771a\n"
        "- wf-fix-fp:aabbccddee11 tag row\n"
        "run 20260716 completed\n"
        "session 4c54094dbeef was idle\n"
    )
    tier1, tier2 = ddf.scan_unresolvable_shas(body, repo)
    assert tier1 == []  # excluded lines + all-digit tokens never reach tier 1
    assert tier2 == ["4c54094dbeef"]  # bare session id, no commit context -> tier 2


def test_check_body_shas_annotates_idempotently(tmp_path):
    repo, _head = _init_hermetic_repo(tmp_path)
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    item = make_item("sha-idem")
    body_path = d / "sha-idem.md"
    body_path.write_text(
        "## Workflow gap\n\nthe fix commit `deadbee7f00d` landed.\n\n"
        "## Provenance\n\n- workflow_fix_target: x.md\n- fingerprint: aabbccddee11\n",
        encoding="utf-8",
    )
    first = ddf._check_body_shas(item, d, repo)
    second = ddf._check_body_shas(item, d, repo)
    assert first == ["deadbee7f00d"]
    assert second == ["deadbee7f00d"]  # still reported; only the injection dedups
    text = body_path.read_text(encoding="utf-8")
    assert text.count("sha-verify (filing-time, #1467)") == 1
    # the ensure_wf_fix_provenance needles survive annotation (no re-injection).
    _new, actions = ddf.ensure_wf_fix_provenance(text, "x.md", "aabbccddee11")
    assert actions == []


def test_check_body_shas_appends_provenance_when_heading_absent(tmp_path):
    repo, _head = _init_hermetic_repo(tmp_path)
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    item = make_item("sha-nohead", route=3)  # route-3-shaped: no Provenance section
    body_path = d / "sha-nohead.md"
    body_path.write_text("## Goal\n\nthe fix landed in `deadbee7f00d`.\n", encoding="utf-8")
    tier1 = ddf._check_body_shas(item, d, repo)
    assert tier1 == ["deadbee7f00d"]
    text = body_path.read_text(encoding="utf-8")
    assert "## Provenance" in text
    assert "sha-verify (filing-time, #1467)" in text
    # bare advisory bullet only — NEVER the recursion-guard line (#1228 semantics).
    assert ddf.WF_FIX_TARGET_KEY not in text


# ── #1808: token-level fp exemption in the #1467 sha walk ───────────────────────

# The 2026-07-28 incident fp — 12-hex WITH letters, so only the exempt set (never
# HAS_HEX_LETTER_RE) can spare it from the walk; unresolvable in the hermetic repo.
_OWN_FP = "06bc0203d759"


def test_scan_exempts_own_fp_midprose_commit_context(tmp_path):
    # The incident shape: the own fp quoted bare (no `fingerprint:` colon substring,
    # so SHA_EXCLUDE_LINE_RE does NOT skip the line) on a commit-context line.
    repo, _head = _init_hermetic_repo(tmp_path)
    body = (
        "## Workflow gap\n\n"
        f"the incident fingerprint `{_OWN_FP}` differs from the landed fix commit.\n"
    )
    # Sanity: without the exempt set the token is tier 1 — the exemption is load-bearing.
    assert ddf.scan_unresolvable_shas(body, repo) == ([_OWN_FP], [])
    assert ddf.scan_unresolvable_shas(body, repo, exempt=frozenset({_OWN_FP})) == ([], [])


def test_scan_exempts_fingerprint_labeled_token_recurring_bare(tmp_path):
    # A token declared via a `fingerprint:` label anywhere in the body is exempt when
    # it recurs bare on another (commit-context) line. Uses the #1580 reconcile-line
    # shape verbatim — TWO captures on ONE line (new fp + superseded body-carried fp).
    repo, _head = _init_hermetic_repo(tmp_path)
    new_fp, old_fp = "aabbccddee99", _OWN_FP
    body = (
        "## Provenance\n\n"
        f"- fingerprint: {new_fp} (tag-authoritative; supersedes body-carried "
        f"fingerprint: {old_fp})\n\n"
        f"## Workflow gap\n\nthe fix landed after {old_fp} was computed.\n"
    )
    exempt = ddf._fp_exempt_tokens(body, None)
    assert exempt == frozenset({new_fp, old_fp})  # both captures on the reconcile line
    assert ddf.scan_unresolvable_shas(body, repo) == ([old_fp], [])  # pre-#1808 shape
    assert ddf.scan_unresolvable_shas(body, repo, exempt=exempt) == ([], [])


def test_scan_still_flags_unrelated_nonresolving_token_with_exempt_set(tmp_path):
    # #1467 regression coverage: the exempt set spares ONLY its own tokens — an
    # unrelated non-resolving 12-hex token in commit context stays tier 1.
    repo, _head = _init_hermetic_repo(tmp_path)
    body = (
        "## Workflow gap\n\n"
        f"fingerprint `{_OWN_FP}` was superseded; the fix commit `deadbee7f00d` landed.\n"
    )
    tier1, tier2 = ddf.scan_unresolvable_shas(body, repo, exempt=frozenset({_OWN_FP}))
    assert tier1 == ["deadbee7f00d"]
    assert tier2 == []


def test_check_body_shas_own_fp_no_advisory(tmp_path):
    # Real path with fp= threaded: a body quoting its OWN fp mid-prose in commit
    # context gains NO advisory and returns [] (task #1808's own body was annotated
    # at filing time for exactly this shape).
    repo, _head = _init_hermetic_repo(tmp_path)
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    item = make_item("fp-own")
    body_path = d / "fp-own.md"
    before = (
        "## Workflow gap\n\n"
        f"the prior filing's fingerprint `{_OWN_FP}` differs — that fix commit landed.\n"
    )
    body_path.write_text(before, encoding="utf-8")
    assert ddf._check_body_shas(item, d, repo, fp=_OWN_FP) == []
    assert body_path.read_text(encoding="utf-8") == before  # body byte-unchanged


def test_check_body_shas_label_scan_arm_works_with_fp_none(tmp_path):
    # The label-scan arm is independent of fp=: a `fingerprint:`-labeled token that
    # recurs bare in commit context is exempt even under the default fp=None.
    repo, _head = _init_hermetic_repo(tmp_path)
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    item = make_item("fp-label")
    body_path = d / "fp-label.md"
    before = (
        "## Provenance\n\n"
        f"- fingerprint: {_OWN_FP}\n\n"
        f"## Workflow gap\n\nthe fix landed after {_OWN_FP} was recorded.\n"
    )
    body_path.write_text(before, encoding="utf-8")
    assert ddf._check_body_shas(item, d, repo) == []
    assert body_path.read_text(encoding="utf-8") == before


def test_dry_run_sha_note_exempts_own_fp(tmp_path):
    # Dry-run mirror parity (#1808): with fp= the note is clean; without it the same
    # body counts one commit-context token — the fp PARAM (no anchored fp line is
    # injected at dry-run time) is what carries the parity with the real path.
    repo, _head = _init_hermetic_repo(tmp_path)
    d = tmp_path / f"filings-{DATE}"
    d.mkdir()
    item = make_item("fp-dry")
    (d / "fp-dry.md").write_text(
        f"## Workflow gap\n\nfingerprint `{_OWN_FP}` predates the landed fix commit.\n",
        encoding="utf-8",
    )
    assert ddf._dry_run_sha_note(item, d, repo, fp=_OWN_FP) == ""
    assert (
        ddf._dry_run_sha_note(item, d, repo)
        == " [sha-scan: 1 commit-context, 0 other non-resolving]"
    )


def test_process_item_records_sha_warnings_in_ledger(tmp_path, tasks_root, monkeypatch):
    # main(argv)-level harness; git resolution stubbed hermetically (the real
    # _sha_resolves body is exercised by the tmp-repo scan tests above).
    item = make_item("sha-ledger", bug="the fix commit `deadbee7f00d` regressed X")
    d = make_filings_dir(tmp_path, [item])

    def fake_sha_resolves(token: str, root: Path) -> bool:
        return token != "deadbee7f00d"

    monkeypatch.setattr(ddf, "_sha_resolves", fake_sha_resolves)
    # Stub filer that snapshots the FILED body at invocation time — pins the
    # annotation-before-filing ordering, not just the post-run body state.
    stub = tmp_path / "stub_body_snap.py"
    stub.write_text(
        "import shutil, sys\n"
        "from pathlib import Path\n"
        "argv = sys.argv[1:]\n"
        "body = Path(argv[argv.index('--body-file') + 1])\n"
        f"shutil.copy(body, Path({str(d)!r}) / 'body_at_invocation.md')\n"
        "print('filed #1234')\n",
        encoding="utf-8",
    )
    rc = run_driver(d, tasks_root, f"{sys.executable} {stub}")
    assert rc == 0  # the backstop never changes the driver exit code
    filed = [r for r in ledger_rows(d) if r["outcome"] == "filed"]
    assert len(filed) == 1
    assert filed[0]["sha_warnings"] == ["deadbee7f00d"]
    snap = (d / "body_at_invocation.md").read_text(encoding="utf-8")
    assert "sha-verify (filing-time, #1467)" in snap
    assert "`deadbee7f00d`" in snap


def test_non_utf8_body_fails_open_scan_skipped(tmp_path, tasks_root, capsys):
    # Round-2 regression (sha-scan-decode-crash): read_text(encoding="utf-8") raises
    # UnicodeDecodeError (a ValueError subclass, NOT an OSError) on a non-UTF-8 body;
    # the fail-open except must catch it — one WARN, no annotation, filing proceeds,
    # exit 0. Pre-fix the exception escaped process_item and aborted the ENTIRE
    # nightly run (plan #1467 §6 never-refuse kill criterion). Hermetic: the decode
    # crash fires before any rev-parse, so no git object set is consulted.
    item = make_item("sha-decode", route=3)
    d = make_filings_dir(tmp_path, [item])
    body_path = d / "sha-decode.md"
    raw = b"## Goal\n\nlog line quoting a stray \x80 byte verbatim.\n"
    body_path.write_bytes(raw)

    # Dry-run first (write-free): exercises the _dry_run_sha_note except leg.
    rc_dry = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc_dry == 0
    assert "[sha-scan skipped: UnicodeDecodeError]" in capsys.readouterr().out
    assert ledger_rows(d) == []  # dry-run stays ledger-write-free

    # Real path: exercises the _check_body_shas except leg end to end.
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0  # fail-open: the scan skip never changes the driver exit code
    err = capsys.readouterr().err
    assert err.count("sha-verify scan skipped") == 1  # ONE loud WARNING
    assert "UnicodeDecodeError" in err
    filed = [r for r in ledger_rows(d) if r["outcome"] == "filed"]
    assert len(filed) == 1  # the filing itself proceeded
    assert "sha_warnings" not in filed[0]  # skipped scan -> no ledger key
    assert body_path.read_bytes() == raw  # no annotation; body byte-untouched


# ── #1483: route-3 open daily-held overlap dedup ───────────────────────────────
#
# Real incident texts (tasks/archived/1140/body.md + tasks/blocked/1472/body.md,
# read 2026-07-17) — the quantitative acceptance replay for test (f). Under the
# §4 predicate the pair shares 7 informative tokens
# (2026-08-06, codex, doubled, every, quota, review, site); title-only shares 1.

TITLE_1140 = "daily-held: Codex quota out to Aug 6 - pay or ride it out"
ORIGIN_1140 = (
    "/daily 2026-07-07 problem sweep (route 3): Every codex_task.py dispatch since "
    "~17:00Z 2026-07-07 fails with a hard usage-limit error (reset 2026-08-06 6:26 AM). "
    "All Codex twins (critic x3 lenses, code-reviewer, interpretation-critic, "
    "clean-result-critic, follow-up-critic) no-show; every doubled review site runs "
    "Claude-only fallback."
)
TITLE_1472 = "daily-held: review diversity during Codex outage to Aug 6"
BUG_1472 = (
    "the Codex org quota is exhausted until ~2026-08-06 (CODEX_QUOTA_LIVE sentinel) — "
    "every doubled review site ran single-Claude across the entire fleet on 2026-07-16 "
    "and will for ~3 more weeks"
)

# ── #1687: the 2026-07-25 wrong-suppression incident, real texts ────────────────
# Item texts from logs/daily/filings-2026-07-24/manifest.json (slug
# issue823-root-draft; suppressed as `already-tracked #1537 (shared: gate,step,warn)`,
# filed.jsonl line 39); task texts from tasks/proposed/1537/body.md frontmatter.

TITLE_823_ITEM = "daily-held: disposition root draft issue823_single_split"
BUG_823_ITEM = (
    "The untracked file scripts/issue823_single_split_protocol.py has sat on the shared "
    "repo root for 11+ hours, degrading every Step 9c gate to scratch-oracle mode - four "
    "gates today each emitted SCRATCH-ORACLE WARN root dirty on this path"
)
TITLE_1537 = "daily-held: enforce body presence on wf-fix filings"
ORIGIN_1537 = (
    "/daily 2026-07-18 problem sweep (route 3): A wf-fix filing without --body/--body-file "
    "lands frontmatter-only silently (#1517, commit 14f2952cab verified); the spawned "
    "session hits the Step 0b empty-body gate. The #1173 WARN half is already landed in "
    "file_infra_task.py; the residual refusal touches the task.py new CLI contract."
)


def _held_item(slug: str, title: str, bug: str) -> dict:
    return make_item(slug, route=3, title=title, bug=bug)


def test_route3_open_daily_held_overlap_dedups_no_filing(tmp_path, tasks_root, capsys):
    # Candidate tokens: {widget, quota, decision, alpha} (title) | {widget, quota, exceeds,
    # gadget, budget, threshold} (bug). Task tokens: {widget, budget, review} (title) |
    # {decide, gadget, policy} (origin). Shared = {widget, budget, gadget} — EXACTLY 3,
    # pinning the >= boundary at ROUTE3_MIN_SHARED_TOKENS (a <-vs-<= off-by-one flips it).
    # #1687: item title carries `widget` so the title anchor = {widget} is satisfied
    # (`widget` was already in the item bug, so the shared set is unchanged).
    item = _held_item(
        "held-a",
        "daily-held: widget quota decision alpha",
        "widget quota exceeds gadget budget threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        77,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert filer_calls(d) == []  # zero filer subprocesses
    (row,) = ledger_rows(d)
    assert row["outcome"] == "already-tracked"
    assert row["against"] == 77
    assert row["against_title"] == "daily-held: widget budget review"
    assert sorted(row["shared"]) == ["budget", "gadget", "widget"]
    assert len(row["shared"]) == ddf.ROUTE3_MIN_SHARED_TOKENS  # exact-boundary pin
    assert row["anchor"] == ["widget"]  # #1687 title-anchor audit field
    assert row["route"] == 3 and "ts" in row
    assert "ALREADY-TRACKED held-a -> #77" in capsys.readouterr().out


def test_route3_two_shared_tokens_files_as_today(tmp_path, tasks_root):
    # Threshold boundary: shared = {widget, gadget} — exactly 2 < 3 — files normally.
    # #1687: the title anchor {widget} IS present, so this test still discriminates
    # on the THRESHOLD, not vacuously on the anchor.
    item = _held_item(
        "held-b",
        "daily-held: widget quota decision alpha",
        "widget quota exceeds gadget budget threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        78,
        title="daily-held: widget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_all_stopword_tokens_files_as_today(tmp_path, tasks_root):
    # An all-stopword/short-token candidate yields an EMPTY token set -> the
    # `if not cand: return None` branch: no dedup possible, files as today.
    item = _held_item(
        "held-empty",
        "daily-held: the for and",
        "with that from into when only over the",
    )
    make_task(
        tasks_root,
        "proposed",
        79,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_dedup_fail_open_files(tmp_path, tasks_root, capsys, monkeypatch):
    # Seam-stub of the added scan (fail-open contract); production-body coverage of
    # the real scan comes from the overlap/boundary/population/replay tests around it.
    def raiser(*_a, **_k):
        raise RuntimeError("synthetic scan explosion")

    monkeypatch.setattr(ddf, "find_open_daily_held_duplicate", raiser)
    item = _held_item("held-c", "daily-held: widget budget review", "widget budget gadget")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0  # fail-open: the scan error never changes the exit code
    err = capsys.readouterr().err
    assert "fail-open, filing proceeds (#1483)" in err
    assert "RuntimeError" in err
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_dedup_ignores_closed_and_untagged(tmp_path, tasks_root):
    # Population gate: a high-overlap CLOSED task and a high-overlap open task
    # WITHOUT the daily-held tag both stay out of the scan population -> files.
    # #1687: the title anchor {widget} would be present, so the test still
    # discriminates on the POPULATION gate.
    item = _held_item(
        "held-d",
        "daily-held: widget quota decision alpha",
        "widget quota exceeds gadget budget threshold",
    )
    make_task(
        tasks_root,
        "completed",
        80,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    make_task(
        tasks_root,
        "proposed",
        81,
        title="daily-held: widget budget review",
        tags=["needs-human"],  # daily-held tag ABSENT
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_dry_run_prints_already_tracked_no_writes(tmp_path, tasks_root, capsys):
    # #1687: item title carries `widget` so the pair is a real match (anchor {widget})
    # and the test keeps exercising dry-run printing.
    item = _held_item(
        "held-e",
        "daily-held: widget quota decision alpha",
        "widget quota exceeds gadget budget threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        82,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    assert "ALREADY-TRACKED held-e -> #82" in capsys.readouterr().out
    assert not (d / "filed.jsonl").exists()  # dry-run stays ledger-write-free
    assert filer_calls(d) == []


def test_route3_replay_1140_1472_pair_dedups(tmp_path, tasks_root, capsys):
    # Quantitative acceptance (plan #1483 §6.1, re-measured for #1687): the REAL
    # #1140/#1472 pair replays as a dedup hit with margin — post-exclusion shared = 5
    # (codex, doubled, quota, review, site; `every` is generic, `2026-08-06` a date)
    # vs threshold 3, and the title anchor is {codex}.
    make_task(
        tasks_root,
        "proposed",
        1140,
        title=TITLE_1140,
        tags=["daily-held"],
        origin_prompt=ORIGIN_1140,
    )
    item = _held_item("codex-outage-refile", TITLE_1472, BUG_1472)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert filer_calls(d) == []
    (row,) = ledger_rows(d)
    assert row["outcome"] == "already-tracked"
    assert row["against"] == 1140
    assert len(row["shared"]) >= ddf.ROUTE3_MIN_SHARED_TOKENS  # measured 5 post-exclusion
    assert {"codex", "quota"} <= set(row["shared"])
    assert row["anchor"] == ["codex"]  # #1687 title-anchor on the one true-dup pair
    assert "ALREADY-TRACKED codex-outage-refile -> #1140" in capsys.readouterr().out


def test_route3_already_tracked_is_terminal_on_resume(tmp_path, tasks_root, capsys):
    # #1687: item title carries `widget` so the pair is a real match (anchor {widget})
    # and the test keeps exercising resume terminality.
    item = _held_item(
        "held-g",
        "daily-held: widget quota decision alpha",
        "widget quota exceeds gadget budget threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        83,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert ledger_rows(d)[-1]["outcome"] == "already-tracked"
    capsys.readouterr()
    # Re-invocation: already-tracked is TERMINAL (TERMINAL_OUTCOMES membership).
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert "SKIP held-g" in capsys.readouterr().out
    assert len(ledger_rows(d)) == 1  # no new rows on resume
    assert filer_calls(d) == []


def test_route2_not_scanned_for_daily_held_overlap(tmp_path, tasks_root):
    # Scope: route-3-only. A route-2 item overlapping an open daily-held task by
    # >= 3 tokens still files normally (route 2 keeps exact fp-dedup only).
    item = make_item(
        "fix-overlap",
        route=2,
        title="daily-fix: widget budget gadget review",
        bug="widget budget gadget threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        84,
        title="daily-held: widget budget review",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): decide gadget policy now",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["attempting", "filed"]
    assert not any(r["outcome"] == "already-tracked" for r in rows)
    assert len(filer_calls(d)) == 1


def test_route3_replay_1687_incident_pair_files(tmp_path, tasks_root, capsys):
    # DURABILITY PIN (#1687): the REAL 2026-07-25 incident replays as a FILING.
    # Raw shared = {gate, step, warn} — all generic workflow vocabulary, excluded
    # from the shared count (post-exclusion 0) AND the titles share no subject
    # token (anchor = {}) — both new conditions independently defeat the match.
    make_task(
        tasks_root,
        "proposed",
        1537,
        title=TITLE_1537,
        tags=["daily-held"],
        origin_prompt=ORIGIN_1537,
    )
    item = _held_item("issue823-root-draft", TITLE_823_ITEM, BUG_823_ITEM)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1
    assert "ALREADY-TRACKED" not in capsys.readouterr().out


def test_route3_generic_vocab_never_counts_toward_overlap(tmp_path, tasks_root):
    # Raw overlap = 8 tokens drawn ONLY from ROUTE3_GENERIC_TOKENS; the subject
    # words are disjoint (quorum/flange vs melon/parade) -> post-exclusion
    # shared = 0 -> files. Pins the exclusion set working alone.
    item = _held_item(
        "held-generic",
        "daily-held: quorum flange analysis",
        "gate step session daily backlog every still across quorum flange",
    )
    make_task(
        tasks_root,
        "proposed",
        85,
        title="daily-held: melon parade cleanup",
        tags=["daily-held"],
        origin_prompt=(
            "/daily 2026-07-01 problem sweep (route 3): "
            "gate step session daily backlog every still across melon parade"
        ),
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_prose_only_overlap_without_title_anchor_files(tmp_path, tasks_root):
    # The #1636 x #1686 class: >= 3 shared SUBJECT tokens confined to bug/origin
    # prose (shared = {widget, gadget, budget, prose} = 4 >= 3), titles sharing
    # no informative token (anchor = {}) -> files. Pins the anchor mechanism
    # specifically (subject tokens survive the exclusion; only the anchor blocks).
    item = _held_item(
        "held-prose",
        "daily-held: alpha analysis",
        "widget gadget budget prose",
    )
    make_task(
        tasks_root,
        "proposed",
        86,
        title="daily-held: melon parade",
        tags=["daily-held"],
        origin_prompt=(
            "/daily 2026-07-01 problem sweep (route 3): widget gadget budget prose cleanup"
        ),
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_date_tokens_never_count_toward_overlap(tmp_path, tasks_root):
    # Titles share `widget` (anchor present), bodies share {widget, gadget,
    # 2026-07-24} — raw 3 would suppress; the date token is excluded so
    # post-exclusion shared = {widget, gadget} = 2 < 3 -> files. Discriminates
    # the DATE exclusion inside the shared count (anchor satisfied).
    item = _held_item(
        "held-date",
        "daily-held: widget alpha analysis",
        "widget gadget 2026-07-24 threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        87,
        title="daily-held: widget melon parade",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): widget gadget 2026-07-24 cleanup",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_generic_tokens_excluded_from_shared_count_with_anchor(tmp_path, tasks_root):
    # Mirror of the date test with GENERIC tokens in place of the date: titles
    # share `widget` (anchor present), bodies share {widget, gate, step} — raw 3
    # would suppress; generic exclusion leaves shared = {widget} = 1 < 3 -> files.
    # Isolates the generic-token exclusion within the SHARED COUNT when an anchor
    # is present (dropping ROUTE3_GENERIC_TOKENS from the shared-count while
    # keeping it in the anchor would fail exactly this test).
    item = _held_item(
        "held-generic-anchored",
        "daily-held: widget alpha analysis",
        "widget gate step threshold",
    )
    make_task(
        tasks_root,
        "proposed",
        88,
        title="daily-held: widget melon parade",
        tags=["daily-held"],
        origin_prompt="/daily 2026-07-01 problem sweep (route 3): widget gate step cleanup",
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    assert len(filer_calls(d)) == 1


def test_route3_generic_set_disjoint_from_true_dup_subjects():
    # Calibration guard (plan #1687 §11 D1): the generic set must never eat the
    # measured true-duplicate subject signal (#1140/#1472 post-exclusion shared =
    # {codex, doubled, quota, review, site}, anchor {codex}). A future
    # over-extension of ROUTE3_GENERIC_TOKENS into these subjects fails here.
    assert ddf.ROUTE3_GENERIC_TOKENS & {"codex", "quota", "doubled", "site", "review"} == set()


# ── #1529: filer sibling-advisory forwarding + ledger persistence ───────────────


def _emit_real_advisories(
    monkeypatch,
    *,
    closed_hits: list[dict] | None = None,
    open_hits: list[dict] | None = None,
    closed_raises: bool = False,
    open_raises: bool = False,
) -> list[str]:
    """Run the REAL file_infra_task advisory emitters; return their stderr lines.

    Emitter-drift coupling (#1529): only the sibling ENUMERATORS are monkeypatched
    (fakes mirroring the real signatures, at the task-scan boundary), so the
    headers / rows / overflow / fail-soft lines are produced by the LIVE emitter
    code — a format drift in file_infra_task.py breaks these tests instead of
    silently regressing extraction to [].
    """

    def _closed(target, title, days=7.0):
        if closed_raises:
            raise RuntimeError("synthetic scan failure")
        return list(closed_hits or [])

    def _open(target, title):
        if open_raises:
            raise RuntimeError("synthetic scan failure")
        return list(open_hits or [])

    monkeypatch.setattr(fit, "recent_closed_workflow_fix_tasks", _closed)
    monkeypatch.setattr(fit, "open_workflow_fix_siblings", _open)
    args = argparse.Namespace(body=None, body_file=None, tag=["wf-fix"], title="workflow-fix: x")
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fit._advise_recent_closed_wf_fix_siblings(args)
        fit._advise_open_wf_fix_siblings(args)
    return [ln for ln in buf.getvalue().splitlines() if ln.strip()]


def _closed_hit(tid: int, title: str = "workflow-fix: prior sibling") -> dict:
    return {
        "id": tid,
        "status": "completed",
        "closed_at": "2026-07-18T01:02:03+00:00",
        "matched": ["target:daily_drive_filings"],
        "title": title,
    }


def _open_hit(tid: int, title: str = "workflow-fix: open sibling") -> dict:
    return {"id": tid, "status": "running", "matched": ["infra-title: advisory"], "title": title}


def test_filed_row_and_stderr_carry_filer_advisories(tmp_path, tasks_root, monkeypatch, capsys):
    # Fixture lines come from the REAL emitters: one closed-arm block (2 rows) +
    # one open-arm block (1 row) — 2 headers + 3 rows.
    fixture = _emit_real_advisories(
        monkeypatch,
        closed_hits=[_closed_hit(1500), _closed_hit(1501)],
        open_hits=[_open_hit(1502)],
    )
    assert len(fixture) == 5
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, stderr_lines=fixture))
    assert rc == 0
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["attempting", "filed"]
    assert rows[1]["id"] == 1234
    assert rows[1]["advisories"] == fixture
    out, err = capsys.readouterr()
    assert "FILED fix-a -> #1234 (rc=0)" in out
    assert "ADVISORY fix-a -> #1234:" in err  # attributing lead line, after FILED
    for ln in fixture:
        assert ln in err  # verbatim re-print on the driver's own stderr


def test_no_advisories_keeps_row_and_output_unchanged(tmp_path, tasks_root, capsys):
    item = make_item("fix-a", route=2)
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    rows = ledger_rows(d)
    assert rows[1]["outcome"] == "filed"
    assert "advisories" not in rows[1]
    # Exactly today's key set — the conditional field keeps no-advisory rows unchanged.
    assert set(rows[1]) == {"slug", "outcome", "id", "rc", "fp", "route", "tail", "ts"}
    out, err = capsys.readouterr()
    assert "ADVISORY" not in out
    assert "ADVISORY" not in err


def test_extract_filer_advisories_predicate():
    stderr = "\n".join(
        [
            "  #7 stray indented row with NO preceding header",
            "WARNING fix-a: sha-verify scan skipped (OSError: boom)",
            "file_infra_task: ADVISORY — 2 closed sibling task(s) overlap this filing:",
            "  #1500  completed  closed 2026-07-18  [target:x]  workflow-fix: t",
            "  ... and 3 more within the window",
            "  [task.py stderr] forwarded child-stderr line (excluded; ends the block)",
            "  #999  a row AFTER the forwarded line sits outside any block",
            "file_infra_task: ADVISORY — 1 OPEN sibling task(s) overlap this filing:",
            "  #1502  running  [infra-title: advisory]  workflow-fix: open sibling",
            "plain filer chatter",
            "file_infra_task: open-sibling advisory leg failed (RuntimeError('x'));"
            " filing proceeds (#1502 fail-soft)",
        ]
    )
    assert ddf.extract_filer_advisories(stderr) == [
        "file_infra_task: ADVISORY — 2 closed sibling task(s) overlap this filing:",
        "  #1500  completed  closed 2026-07-18  [target:x]  workflow-fix: t",
        "  ... and 3 more within the window",
        "file_infra_task: ADVISORY — 1 OPEN sibling task(s) overlap this filing:",
        "  #1502  running  [infra-title: advisory]  workflow-fix: open sibling",
        "file_infra_task: open-sibling advisory leg failed (RuntimeError('x'));"
        " filing proceeds (#1502 fail-soft)",
    ]
    assert ddf.extract_filer_advisories("") == []


def test_extract_filer_advisories_cap_appends_marker():
    header = "file_infra_task: ADVISORY — synthetic oversize block:"
    # Line-count cap: 45 short lines -> first 40 kept + explicit marker line.
    got = ddf.extract_filer_advisories("\n".join([header, *(f"  #{i}  x" for i in range(44))]))
    assert len(got) == ddf._ADVISORY_MAX_FWD_LINES + 1
    assert got[0] == header
    assert got[-1] == "  ... advisory forward capped (45 lines total, #1529)"
    # Char-budget cap: few but huge lines -> truncated under the byte budget + marker.
    long_rows = [f"  #{i}  " + "y" * 990 for i in range(5)]
    got2 = ddf.extract_filer_advisories("\n".join([header, *long_rows]))
    assert got2[-1].startswith("  ... advisory forward capped (6 lines total")
    assert sum(len(ln) for ln in got2[:-1]) <= ddf._ADVISORY_MAX_FWD_CHARS


def test_extractor_captures_every_real_emitter_line(monkeypatch):
    # 12 closed hits (> _ADVISORY_MAX_ROWS = 10) exercise the real overflow line.
    lines = _emit_real_advisories(
        monkeypatch,
        closed_hits=[_closed_hit(1400 + i) for i in range(12)],
        open_hits=[_open_hit(1502)],
    )
    assert any(ln.startswith("  ... and 2 more") for ln in lines)  # overflow emitted
    assert ddf.extract_filer_advisories("\n".join(lines)) == lines

    # Fail-soft one-liners (the scan did NOT run) are captured too.
    failsoft = _emit_real_advisories(monkeypatch, closed_raises=True, open_raises=True)
    assert len(failsoft) == 2
    assert all("advisory leg failed" in ln for ln in failsoft)
    assert ddf.extract_filer_advisories("\n".join(failsoft)) == failsoft


def test_ledger_with_advisories_resumes_and_recovers(tmp_path, tasks_root, capsys):
    # (a) An advisories-bearing filed row is terminal on resume (field never consulted).
    item_a = make_item("fix-a")
    item_b = make_item("fix-b")
    d = make_filings_dir(tmp_path, [item_a, item_b])
    seed = {
        "slug": "fix-a",
        "outcome": "filed",
        "id": 1044,
        "rc": 0,
        "route": 2,
        "advisories": ["file_infra_task: ADVISORY — 1 OPEN sibling task(s) overlap this filing:"],
    }
    (d / "filed.jsonl").write_text(json.dumps(seed) + "\n", encoding="utf-8")
    # (b) A trailing `attempting` row + matching synthetic task: _try_recovery still
    # recovers with the new field present in the ledger (rows stay free-form dicts).
    _seed_attempting(d, "fix-b", item_b, id_floor=100)
    make_task(tasks_root, "proposed", 150, title=item_b["title"], tags=["daily-auto-filed"])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert "SKIP fix-a" in capsys.readouterr().out
    last = ledger_rows(d)[-1]
    assert last["slug"] == "fix-b" and last["outcome"] == "recovered" and last["id"] == 150
    assert filer_calls(d) == []  # neither item reached the filer


def test_error_path_does_not_gain_advisories(tmp_path, tasks_root, monkeypatch, capsys):
    # Success-path-only scope pin: a failing filer that DID emit advisory stderr
    # produces today's ERROR row byte-shape (no `advisories` key, no lead line).
    fixture = _emit_real_advisories(monkeypatch, open_hits=[_open_hit(1502)])
    assert fixture  # the stub really emits advisory stderr on this run
    item = make_item("fix-a")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d, exit_code=1, stderr_lines=fixture))
    assert rc == 1
    err_row = ledger_rows(d)[-1]
    assert err_row["outcome"] == "ERROR" and err_row["flag"] == "filer-failed"
    assert err_row["rc"] == 1
    assert "advisories" not in err_row
    out, err = capsys.readouterr()
    assert "ADVISORY fix-a" not in err  # no #1529 lead line on the error path
    assert "ADVISORY fix-a" not in out


def test_workflow_fix_rule_documents_advisory_forwarding():
    # Doc pin (#1529 Durability pin): limitation (b) documents the /daily leg as
    # CLOSED and no longer claims the advisory is discarded by the ~300-char tail.
    rule = (REPO_ROOT / ".claude" / "rules" / "workflow-fix-on-bug.md").read_text(encoding="utf-8")
    start = rule.index("Open-sibling arm")
    end = rule.index("## Recursion guard")
    section = rule[start:end]
    assert "CLOSED (#1529)" in section
    assert "`advisories`" in section
    assert "persists only a ~300-char output tail" not in rule


# ── #1678: same-target dispatch hold ───────────────────────────────────────────

HOLD_FIELDS = ("held_dispatch", "held_with", "shared_target")


def _filed_row(d: Path, slug: str) -> dict:
    """The terminal ``filed`` ledger row for one slug (exactly one expected)."""
    (row,) = [r for r in ledger_rows(d) if r["slug"] == slug and r["outcome"] == "filed"]
    return row


def test_same_target_second_route2_item_files_without_dispatch(tmp_path, tasks_root, capsys):
    # Durability pin (#1678): the LATER same-target route-2 sibling files with
    # --no-dispatch + the hold ledger fields; the group HEAD stays dispatched
    # and its row carries NONE of the hold fields (an implementation stamping
    # every group member must fail here).
    items = [make_item("fix-a"), make_item("fix-b")]  # shared default target
    d = make_filings_dir(tmp_path, items)
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    call_a, call_b = filer_calls(d)
    assert "--no-dispatch" not in call_a
    assert "--no-dispatch" in call_b
    row_a = _filed_row(d, "fix-a")
    assert not any(f in row_a for f in HOLD_FIELDS)  # head row unchanged
    row_b = _filed_row(d, "fix-b")
    assert row_b["held_dispatch"] is True
    assert row_b["held_with"] == "fix-a"
    assert row_b["shared_target"] == [".claude/skills/daily/SKILL.md"]
    assert "HELD-DISPATCH fix-b" in capsys.readouterr().out


def test_third_same_target_sibling_attributes_to_earliest(tmp_path, tasks_root):
    items = [make_item("fix-a"), make_item("fix-b"), make_item("fix-c")]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    call_a, call_b, call_c = filer_calls(d)
    assert "--no-dispatch" not in call_a
    assert "--no-dispatch" in call_b and "--no-dispatch" in call_c
    assert _filed_row(d, "fix-b")["held_with"] == "fix-a"
    assert _filed_row(d, "fix-c")["held_with"] == "fix-a"  # earliest, not fix-b


def test_distinct_targets_no_hold_no_new_ledger_fields(tmp_path, tasks_root):
    # Byte-parity pin: no overlapping targets -> no --no-dispatch, no hold fields.
    items = [
        make_item("fix-a", target="scripts/task.py"),
        make_item("fix-b", target="scripts/pod.py"),
    ]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    for call in filer_calls(d):
        assert "--no-dispatch" not in call
    for slug in ("fix-a", "fix-b"):
        assert not any(f in _filed_row(d, slug) for f in HOLD_FIELDS)


def test_comma_separated_target_overlap_holds_on_shared_path(tmp_path, tasks_root):
    items = [
        make_item("fix-a", target="scripts/a.py, scripts/b.py"),
        make_item("fix-b", target="./scripts/b.py"),
    ]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    _, call_b = filer_calls(d)
    assert "--no-dispatch" in call_b
    assert _filed_row(d, "fix-b")["shared_target"] == ["scripts/b.py"]


def test_glob_target_literal_token_only(tmp_path, tasks_root):
    # Globs are LITERAL tokens: identical globs overlap; a concrete path under a
    # glob does NOT (the documented false negative — Step 10d stays the backstop).
    items = [
        make_item("fix-a", target=".claude/agents/*.md"),
        make_item("fix-b", target=".claude/agents/*.md"),
        make_item("fix-c", target=".claude/agents/planner.md"),
    ]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    call_a, call_b, call_c = filer_calls(d)
    assert "--no-dispatch" not in call_a
    assert "--no-dispatch" in call_b
    assert "--no-dispatch" not in call_c
    assert not any(f in _filed_row(d, "fix-c") for f in HOLD_FIELDS)


def test_route3_same_target_never_holds_route2(tmp_path, tasks_root):
    # Route-3 items already file --no-dispatch (not a contention source): they
    # neither hold nor are held.
    items = [make_item("hold-a", route=3), make_item("fix-b")]  # same default target
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    call_3, call_2 = filer_calls(d)
    assert "--no-dispatch" in call_3  # route-3's own flag, unchanged
    assert "--no-dispatch" not in call_2  # route-2 NOT held by the route-3 item
    assert not any(f in _filed_row(d, "fix-b") for f in HOLD_FIELDS)


def test_hold_computed_over_full_manifest_not_slice(tmp_path, tasks_root):
    # The hold map is a pure function of the FULL manifest: slicing to item 1
    # only (--start 1) still holds fix-b against the unsliced fix-a.
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d), "--start", "1") == 0
    (call_b,) = filer_calls(d)  # only the sliced item ran
    assert "--no-dispatch" in call_b
    assert _filed_row(d, "fix-b")["held_with"] == "fix-a"


def test_dry_run_surfaces_hold_no_writes(tmp_path, tasks_root, capsys):
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run") == 0
    out = capsys.readouterr().out
    line_a = next(ln for ln in out.splitlines() if ln.startswith("FILE fix-a"))
    line_b = next(ln for ln in out.splitlines() if ln.startswith("FILE fix-b"))
    assert "--no-dispatch" not in line_a
    assert "[held dispatch:" not in line_a
    assert "'--no-dispatch'" in line_b  # rides the printed tags slice, as route 3 does
    assert "[held dispatch: shares .claude/skills/daily/SKILL.md with fix-a]" in line_b
    assert filer_calls(d) == []
    assert not (d / "filed.jsonl").exists()


def test_held_item_terminal_files_then_rerun_skips(tmp_path, tasks_root, capsys):
    # Resume safety (#1678 plan §4.7): a held item that terminal-filed is SKIPped
    # on re-run exactly like any other terminal slug — no re-file, no double-file.
    items = [make_item("fix-a"), make_item("fix-b")]
    d = make_filings_dir(tmp_path, items)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert _filed_row(d, "fix-b")["held_dispatch"] is True
    n_calls = len(filer_calls(d))
    capsys.readouterr()  # drain run-1 output
    assert run_driver(d, tasks_root, make_stub(tmp_path, d, name="stub2.py")) == 0
    assert "SKIP fix-b" in capsys.readouterr().out
    assert len(filer_calls(d)) == n_calls  # no new filer invocations


def test_parse_filed_id_matches_no_dispatch_stdout_shape():
    # Pins the FILED_ID_RE compatibility the hold design leans on: the filer's
    # --no-dispatch success line still yields the task id.
    assert ddf.parse_filed_id("filed #77 (dispatch skipped: --no-dispatch)", "") == 77


# ── #1674: mechanical landed-fix probe ──────────────────────────────────────────
#
# Test seam: driver-level tests monkeypatch ddf.repo_root to a hermetic git repo
# (never the live repo's history — the #1467 isolation principle), so the probe's
# `git -C root log` reads ONLY commits these tests created. Unit tests call
# find_landed_fix_suspects(item, hermetic_repo) directly. No production function
# is stubbed in the driver-level tests — the probe body executes fully through
# real git (#906 discipline).
#
# Synthetic token calibration (live tokenizer, verified 2026-07-25):
#   item tokens (title+bug+change of _probe_item) =
#     {zebra, quokka, lantern, regression, regressed, path}
#   _SUSPECT_SUBJECT tokens ⊃ {zebra, quokka, lantern, path}      -> shared 4 >= 3
#   _TWO_TOKEN_SUBJECT tokens ∩ item tokens = {zebra, quokka}     -> shared 2 <  3

_SUSPECT_SUBJECT = "workflow-fix #999: zebra quokka lantern path hardening"
_TWO_TOKEN_SUBJECT = "workflow-fix #998: zebra quokka unrelated cleanup"


def _hermetic_commit(
    repo: Path, *, subject: str, path: str = "f.txt", commit_date: str | None = None
) -> str:
    """One commit on the hermetic repo touching ``path`` with ``subject`` (#1674).

    ``commit_date`` sets GIT_COMMITTER_DATE + GIT_AUTHOR_DATE so the window test
    can backdate a commit out of the probe's --since window (--since reads the
    COMMITTER date). Returns the abbreviated sha — the %h shape the probe records.
    """
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    prev = f.read_text(encoding="utf-8") if f.exists() else ""
    f.write_text(prev + "x\n", encoding="utf-8")
    env = dict(os.environ)
    if commit_date:
        env["GIT_COMMITTER_DATE"] = commit_date
        env["GIT_AUTHOR_DATE"] = commit_date
    subprocess.run(
        ["git", "-C", str(repo), "add", path], check=True, capture_output=True, text=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            subject,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _probe_item(slug: str = "probe-hit", route: int = 2, target: str = "scripts/foo.py") -> dict:
    """An item whose text shares 4 informative tokens with _SUSPECT_SUBJECT."""
    return make_item(
        slug,
        route=route,
        title="daily-fix: zebra quokka lantern regression fix",
        target=target,
        bug="the zebra quokka lantern path regressed",
        change="fix the zebra quokka lantern path",
    )


def _strip_ts(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "ts"}


def test_landed_fix_probe_suppresses_filing(tmp_path, tasks_root, monkeypatch):
    # Acceptance 1: >= 3 shared subject tokens on an in-window commit touching the
    # item's own target -> exactly one terminal landed-fix-suspect row (full row
    # shape), NO filer subprocess, driver exit 0 (suppression is not an error).
    repo, _head = _init_hermetic_repo(tmp_path)
    sha = _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item()
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "landed-fix-suspect"
    assert row["slug"] == "probe-hit"
    assert row["suspects"] == [
        {"sha": sha, "subject": _SUSPECT_SUBJECT, "shared": ["lantern", "path", "quokka", "zebra"]}
    ]
    assert row["threshold"] == ddf.LANDED_FIX_MIN_SHARED_TOKENS
    assert row["window"] == ddf.LANDED_FIX_WINDOW
    assert row["paths"] == ["scripts/foo.py"]
    assert row["fp"] == wf_fix_fingerprint(item["change"], item["bug"])
    assert row["route"] == 2
    assert "ts" in row


def test_find_landed_fix_suspects_unit_shapes(tmp_path):
    # Unit-level: the probe body against the hermetic repo directly.
    repo, _head = _init_hermetic_repo(tmp_path)
    sha = _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    item = _probe_item("unit")
    assert ddf.find_landed_fix_suspects(item, repo) == [
        {"sha": sha, "subject": _SUSPECT_SUBJECT, "shared": ["lantern", "path", "quokka", "zebra"]}
    ]
    # Degenerate targets (empty token set) never spawn git — empty result.
    assert ddf.find_landed_fix_suspects({**item, "target": " , "}, repo) == []


def test_landed_fix_probe_below_threshold_files(tmp_path, tasks_root, monkeypatch):
    # Acceptance 2: exactly 2 shared tokens -> files normally; the filed row is
    # EQUAL modulo the ledger ts to a no-probe control run (a hermetic repo with
    # no commit on the target path).
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    item = _probe_item("below-thresh")

    repo_a, _ = _init_hermetic_repo(tmp_path / "a")
    _hermetic_commit(repo_a, subject=_TWO_TOKEN_SUBJECT, path="scripts/foo.py")
    d_a = make_filings_dir(tmp_path / "a", [item])
    monkeypatch.setattr(ddf, "repo_root", lambda: repo_a)
    assert run_driver(d_a, tasks_root, make_stub(tmp_path / "a", d_a)) == 0
    assert len(filer_calls(d_a)) == 1  # filer called exactly once

    repo_b, _ = _init_hermetic_repo(tmp_path / "b")
    d_b = make_filings_dir(tmp_path / "b", [item])
    monkeypatch.setattr(ddf, "repo_root", lambda: repo_b)
    assert run_driver(d_b, tasks_root, make_stub(tmp_path / "b", d_b)) == 0

    rows_a = [r for r in ledger_rows(d_a) if r["outcome"] == "filed"]
    rows_b = [r for r in ledger_rows(d_b) if r["outcome"] == "filed"]
    assert len(rows_a) == 1 and len(rows_b) == 1
    assert _strip_ts(rows_a[0]) == _strip_ts(rows_b[0])


def test_landed_fix_probe_scoped_to_target_paths(tmp_path, tasks_root, monkeypatch):
    # Acceptance 2 (path scoping): a >= 3-token commit touching a DIFFERENT file
    # never suppresses — the probe is `git log -- <item targets>`, not repo-wide.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/other.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("path-scoped")  # target scripts/foo.py
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


def test_landed_fix_probe_window_excludes_old_commits(tmp_path, tasks_root, monkeypatch):
    # Acceptance 2 (window): a >= 3-token commit backdated far outside 7 days files
    # normally — pins that the probe's --since reads the committer date.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(
        repo,
        subject=_SUSPECT_SUBJECT,
        path="scripts/foo.py",
        commit_date="2020-01-01T00:00:00 +0000",
    )
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("old-commit")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


def test_landed_fix_probe_git_error_fails_open(tmp_path, tasks_root, capsys, monkeypatch):
    # Acceptance 3: a real `git log` failure (rc=128 in a non-repo root) prints ONE
    # loud stderr WARNING and files normally — fail-open, exit code unchanged.
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()
    monkeypatch.setattr(ddf, "repo_root", lambda: non_repo)
    item = _probe_item("git-err")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    err = capsys.readouterr().err
    assert err.count("landed-fix probe skipped") == 1
    assert "CalledProcessError" in err
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


def test_landed_fix_probe_non_enumerated_exception_propagates(monkeypatch):
    # D6's fail-LOUD half (verify_plan c15): the enumerated fail-open tuple never
    # swallows a non-subprocess driver bug — a TypeError raised inside the probe
    # PROPAGATES out of _landed_fix_or_none (no WARN-and-file).
    def boom(item: dict, root: Path) -> list[dict]:
        raise TypeError("driver bug")

    monkeypatch.setattr(ddf, "find_landed_fix_suspects", boom)
    with pytest.raises(TypeError, match="driver bug"):
        ddf._landed_fix_or_none(make_item("boom"), Path("/nonexistent"))


def test_landed_fix_probe_dry_run_read_only(tmp_path, tasks_root, capsys, monkeypatch):
    # Acceptance 4: --dry-run runs the probe read-only — the LANDED-FIX-SUSPECT
    # line prints, NO ledger row is written, NO filer is spawned.
    repo, _head = _init_hermetic_repo(tmp_path)
    sha = _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("dry-probe")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    out = capsys.readouterr().out
    assert f"LANDED-FIX-SUSPECT dry-probe -> {sha}" in out
    assert "--retry-suspects" in out
    assert ledger_rows(d) == []
    assert filer_calls(d) == []


def test_landed_fix_suspect_terminal_on_resume_and_retry_flag_refiles(
    tmp_path, tasks_root, capsys, monkeypatch
):
    # Acceptance 5: after a suspect run, a plain re-invocation SKIPs (terminal);
    # --retry-suspects re-drives the slug with the probe SKIPPED (the commit is
    # still in-window and would re-fire otherwise) — the filer runs, filed lands.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("retry-flow")
    d = make_filings_dir(tmp_path, [item])
    stub = make_stub(tmp_path, d)
    assert run_driver(d, tasks_root, stub) == 0
    assert [r["outcome"] for r in ledger_rows(d)] == ["landed-fix-suspect"]
    capsys.readouterr()  # drain run-1 output
    assert run_driver(d, tasks_root, stub) == 0
    assert "SKIP retry-flow" in capsys.readouterr().out
    assert filer_calls(d) == []
    assert run_driver(d, tasks_root, stub, "--retry-suspects") == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == [
        "landed-fix-suspect",
        "attempting",
        "filed",
    ]


def test_landed_fix_error_plus_suspect_needs_both_flags(tmp_path, tasks_root, monkeypatch):
    # The _slug_state ERROR-first interplay (#1674 docstring): a slug carrying BOTH
    # an ERROR row and a suspect row re-runs the probe under --retry-errors alone
    # (benign accumulation — another suspect row, no filing); BOTH
    # --retry-errors --retry-suspects re-drive it to a filed row.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("both-rows")
    d = make_filings_dir(tmp_path, [item])
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    ddf.append_row(
        d,
        {
            "slug": "both-rows",
            "outcome": "ERROR",
            "flag": "filer-failed",
            "id": None,
            "rc": 1,
            "fp": fp,
            "route": 2,
            "tail": "",
        },
    )
    ddf.append_row(
        d,
        {
            "slug": "both-rows",
            "outcome": "landed-fix-suspect",
            "suspects": [],
            "threshold": ddf.LANDED_FIX_MIN_SHARED_TOKENS,
            "window": ddf.LANDED_FIX_WINDOW,
            "paths": ["scripts/foo.py"],
            "fp": fp,
            "route": 2,
        },
    )
    stub = make_stub(tmp_path, d)
    assert run_driver(d, tasks_root, stub, "--retry-errors") == 0
    assert filer_calls(d) == []
    assert [r["outcome"] for r in ledger_rows(d)][-1] == "landed-fix-suspect"
    assert run_driver(d, tasks_root, stub, "--retry-errors", "--retry-suspects") == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)][-1] == "filed"


def test_landed_fix_replay_1652_incident_pair_suppresses(tmp_path, tasks_root, monkeypatch):
    # Acceptance 6 / the #1287 predicate-fires-on-its-own-incident pin. REAL
    # artifacts, hard-coded at authoring time (2026-07-25): the item text is task
    # #1652's real title + origin-prompt bug text (tasks/archived/1652
    # frontmatter); the commit subject is ce11dff560's real subject (the #1600
    # fix merge). Measured shared = {pods, runpod, scope}, n=3 -> FIRES at
    # threshold 3 (plan #1674 §11 D1 / §12 A9). A failure here means the
    # tokenizer/exclusion stack drifted: recalibrate per the plan's must-ask
    # list — NEVER tune the threshold to re-green.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(
        repo,
        # git log -1 --format=%s ce11dff560 (read 2026-07-25):
        subject="workflow-fix #1600: scope RunPod pre-flight $/hr guard to managed pods (#1375)",
        path="scripts/pod_lifecycle.py",
    )
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = make_item(
        "replay-1652",
        title="daily-fix: scope RunPod hourly cap to EPS-managed pods",
        target="scripts/pod_lifecycle.py",
        bug=(
            "_assert_under_account_hourly_cap counts the whole shared team account "
            "including ~2855 USD/hr of unmanaged fellows pods so EPS provisions are "
            "falsely blocked"
        ),
        change="scope the RunPod hourly cap to EPS-managed pods only",
    )
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    assert set(rows[0]["suspects"][0]["shared"]) >= {"pods", "runpod", "scope"}


def test_landed_fix_replay_1674_vs_1678_files(tmp_path, tasks_root, monkeypatch):
    # Acceptance 6, negative control: THIS task's own item text vs the #1678
    # driver commit's real subject (0b9d2d330f, read 2026-07-25) on the shared
    # target — measured shared = {driver, filing}, n=2 < 3 -> FILES normally
    # (plan #1674 §11 D1 / §12 A9). A failure here means a legitimate distinct
    # fix on the hot file would be suppressed: recalibrate per the plan's
    # must-ask list — NEVER tune the threshold to re-green.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(
        repo,
        # git log -1 --format=%s 0b9d2d330f (read 2026-07-25):
        subject="task #1678: same-target dispatch hold in the /daily filing driver (#1440)",
        path="scripts/daily_drive_filings.py",
    )
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = make_item(
        "replay-1674",
        title="daily-fix: driver runs mechanical landed-fix probe at filing",
        target="scripts/daily_drive_filings.py",
        bug=(
            "The /daily route-2 channel filed 1652 for a fix that had already landed "
            "1.5 days earlier (1600, merge ce11dff560) - the compose-time landed-fix "
            "git-log duty plus the 1446 closed-sibling advisory both failed to prevent "
            "a duplicate filing and a spawned session, the third recurrence of the "
            "1330/1386 class"
        ),
        change=(
            "Stop the recurring filed-over-landed-fix class mechanically at the driver, "
            "instead of relying on the compose-time git-log duty alone."
        ),
    )
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]


def test_landed_fix_probe_guards_route3(tmp_path, tasks_root, monkeypatch):
    # D5: the probe is route-blind — a route-3 item with a suspect-triggering
    # commit records a landed-fix-suspect row carrying route: 3 and never files.
    repo, _head = _init_hermetic_repo(tmp_path)
    _hermetic_commit(repo, subject=_SUSPECT_SUBJECT, path="scripts/foo.py")
    monkeypatch.setattr(ddf, "repo_root", lambda: repo)
    item = _probe_item("r3-probe", route=3)
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    assert rows[0]["route"] == 3


# ── #1711: mechanical closed-sibling probe ─────────────────────────────────────
#
# Test seams (deviation from brief's SEED-REAL-REGISTRY preference, documented
# per-plan-brief in the implementer report):
#
# For tests 1-4, 7, 10 the brief prefers seeding real closed infra tasks + a
# rebound task_workflow (the `fake_repo` pattern in tests/test_workflow_fix_dedup.py:
# monkeypatch tw.repo_root / tw.tasks_dir / tw.registry_path / lock paths, then
# `create_task` + `set_status`). That harness is a full parallel test-workflow
# rewire and does not compose with THIS file's `--tasks-root` CLI seam (the
# driver-level fixtures) without duplicating fake_repo across every fixture.
# Instead these tests monkeypatch `ddf.find_closed_sibling_suspects` to return
# synthetic hit dicts matching the helper's documented six-field schema (id,
# title, status, target, closed_at, matched — verified from
# task_workflow.recent_closed_workflow_fix_tasks docstring + implementation
# line ~1388). The seams are all AT the boundary this task added, and the
# hit-dict schema is pinned by the helper's own test suite
# (tests/test_workflow_fix_dedup.py::test_recent_closed_*).
#
# Test 6 (fault-injection) uses monkeypatch by contract (matches
# test_landed_fix_probe_non_enumerated_exception_propagates precedent). Test 8
# uses monkeypatched helpers for BOTH probes to pin probe order (matches the
# brief's suggested split for the fixture-complexity concern).


def _closed_sibling_hit(
    tid: int,
    *,
    title: str = "workflow-fix: fix scripts/foo.py",
    status: str = "completed",
    target: str = "scripts/foo.py",
    matched: list[str] | None = None,
    closed_at: str = "2026-07-22T12:00:00Z",
) -> dict:
    """Synthetic hit dict matching recent_closed_workflow_fix_tasks's schema.

    Fields verified from task_workflow.recent_closed_workflow_fix_tasks docstring
    + the return-append at ~line 1388. Default `matched=["target"]` — under
    the #1735 composite arm rule this is a BARE-TARGET advisory hit (bare-
    target alone no longer blocks; a blocking fixture must add a non-stopword
    informative title arm, e.g. `["target", "title:planner"]`). Override to
    `["title:foo,bar"]` for a title-only advisory.
    """
    if matched is None:
        matched = ["target"]
    return {
        "id": tid,
        "title": title,
        "status": status,
        "target": target,
        "closed_at": closed_at,
        "matched": matched,
    }


def test_closed_sibling_probe_blocks_on_target_arm(tmp_path, tasks_root, monkeypatch):
    # Test 1 (plan §15, updated for #1735 composite arm): a prefixed closed
    # wf-fix sibling whose matched arms include BOTH `target` AND an
    # informative title arm (`title:planner` — `planner` is not in
    # CLOSED_SIBLING_TITLE_STOPWORDS) blocks a candidate whose
    # target=scripts/foo.py. Expect ledger row outcome=landed-fix-suspect with
    # suspects[0].kind=closed-sibling and BOTH arms in the matched list,
    # NO filer call, exit 0.
    item = make_item(
        "cs-target-block",
        route=2,
        target="scripts/foo.py",
        bug="foo.py bug",
        change="fix foo.py",
    )
    hit = _closed_sibling_hit(1600, matched=["target", "title:planner"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    row = rows[0]
    assert row["threshold"] is None  # closed-sibling arms are boolean, not token-count
    assert row["window"] == "7.0 days"
    assert row["paths"] == ["scripts/foo.py"]
    assert row["fp"] == wf_fix_fingerprint(item["change"], item["bug"])
    assert row["route"] == 2
    (suspect,) = row["suspects"]
    assert suspect["kind"] == "closed-sibling"
    assert suspect["id"] == 1600
    assert suspect["matched"] == ["target", "title:planner"]
    # #1674's row shape uses `sha` not `id` — the presence of `id` (not `sha`)
    # is the structural discriminator per plan §4.3 (Option A source-compat).
    assert "sha" not in suspect


def test_closed_sibling_probe_blocks_on_infra_target_arm(tmp_path, tasks_root, monkeypatch):
    # Test 2 (plan §15, updated for #1735 composite arm): a widened
    # (non-prefixed) closed kind:infra sibling whose body contains the FULL
    # candidate path (infra-target arm) AND shares an informative title token
    # (via `infra-title:hub,upload` — `hub`/`upload` are not in the driver
    # stoplist) blocks the filing — the vocabulary-divergent landed-fix class
    # #1386/#1360 the whole plan exists for, exercised at the composite arm.
    item = make_item(
        "cs-infra-target",
        route=2,
        target="src/explore_persona_space/orchestrate/hub.py",
        bug="hub upload wedge",
        change="hub xet retry",
    )
    hit = _closed_sibling_hit(
        1360,
        title="hub upload retry cleanup",  # non-prefixed
        matched=["infra-target", "infra-title:hub,upload"],
        target="src/explore_persona_space/orchestrate/hub.py",
    )
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    (suspect,) = rows[0]["suspects"]
    assert suspect["matched"] == ["infra-target", "infra-title:hub,upload"]
    assert suspect["kind"] == "closed-sibling"


def test_closed_sibling_probe_title_only_advises_but_files(
    tmp_path, tasks_root, monkeypatch, capsys
):
    # Test 3 (plan §15): a prefixed sibling with only shared title tokens (NO
    # path overlap) fires an ADVISORY line on stderr but does NOT suppress —
    # the item files normally. Rationale: helper's own docstring flags the
    # unmeasured title-arm FP surface; plan §4.2 defends the PATH-only block.
    item = make_item("cs-title-advise", route=2, target="scripts/foo.py")
    hit = _closed_sibling_hit(
        1500,
        title="workflow-fix: unrelated but tokens shared",
        matched=["title:foo,bar"],  # title-only, NO path arm
    )
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], [hit]))
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    # Filer WAS called; item filed normally.
    assert len(filer_calls(d)) == 1
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["attempting", "filed"]
    err = capsys.readouterr().err
    assert "CLOSED-SIBLING-ADVISORY cs-title-advise -> #1500" in err
    assert "NOT blocking" in err


def test_closed_sibling_probe_infra_title_only_advises_but_files(
    tmp_path, tasks_root, monkeypatch, capsys
):
    # Test 4 (plan §15): the widened title arm (infra-title:*) equivalent —
    # a non-prefixed infra task with >=2 shared title tokens but no path
    # overlap fires the same stderr advisory + files normally.
    item = make_item("cs-infra-title", route=2, target="scripts/foo.py")
    hit = _closed_sibling_hit(
        1400,
        title="watcher retry backstop for pod polling",  # non-prefixed
        matched=["infra-title:retry,backstop"],
    )
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], [hit]))
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    err = capsys.readouterr().err
    assert "CLOSED-SIBLING-ADVISORY cs-infra-title -> #1400" in err
    assert "infra-title:retry,backstop" in err


def test_closed_sibling_probe_no_hits_files_normally(tmp_path, tasks_root, monkeypatch, capsys):
    # Test 5 (plan §15): no closed sibling in the window — normal filing path,
    # NO CLOSED-SIBLING-* lines anywhere. The default autouse-fixture return
    # is ([], []) so this test would pass without the explicit setattr — it is
    # kept explicit to pin the "empty result -> silent proceed" contract.
    item = make_item("cs-nohit")
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], []))
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    output = capsys.readouterr()
    assert "CLOSED-SIBLING" not in output.out
    assert "CLOSED-SIBLING" not in output.err


def test_closed_sibling_probe_helper_error_fails_open(tmp_path, tasks_root, monkeypatch, capsys):
    # Test 6 (plan §15): a bug INSIDE the helper (TypeError) triggers the
    # deliberate broad `except Exception` fail-open — exactly ONE stderr
    # WARNING naming the exception class + fail-open message, the item files
    # normally, driver exit 0. Pins the broad-catch shape (a regression to a
    # narrower enumerated tuple would propagate TypeError and fail this test).
    def boom(item: dict) -> tuple[list[dict], list[dict]]:
        raise TypeError("helper bug")

    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", boom)
    item = make_item("cs-helper-boom")
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d))
    assert rc == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    err = capsys.readouterr().err
    assert err.count("closed-sibling probe skipped") == 1
    assert "TypeError" in err
    assert "helper bug" in err


def test_retry_suspects_re_drives_both_probes(tmp_path, tasks_root, monkeypatch, capsys):
    # Test 7 (plan §15): a pre-seeded landed-fix-suspect row (from a prior
    # closed-sibling hit) is TERMINAL on a plain re-invocation. Passing
    # --retry-suspects makes _slug_state return 'retry-suspect' AND
    # `suspect_eyeballed` short-circuits BOTH probes — filer runs, filed
    # row lands.
    item = make_item("cs-retry-flow")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    d = make_filings_dir(tmp_path, [item])
    # Seed a prior closed-sibling landed-fix-suspect row directly.
    seed = {
        "slug": "cs-retry-flow",
        "outcome": "landed-fix-suspect",
        "suspects": [_closed_sibling_hit(1200) | {"kind": "closed-sibling"}],
        "threshold": None,
        "window": "7.0 days",
        "paths": [".claude/skills/daily/SKILL.md"],
        "fp": fp,
        "route": 2,
    }
    with open(d / "filed.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seed) + "\n")

    # Without --retry-suspects: TERMINAL, skip. (The closed-sibling probe
    # must NOT be called — otherwise the autouse-fixture no-op would still
    # allow re-filing; assert filer never fires.)
    stub = make_stub(tmp_path, d)
    rc = run_driver(d, tasks_root, stub)
    assert rc == 0
    assert "SKIP cs-retry-flow" in capsys.readouterr().out
    assert filer_calls(d) == []

    # With --retry-suspects: `suspect_eyeballed` short-circuits BOTH probes
    # (the closed-sibling probe MUST be skipped even under a monkeypatch that
    # would still return a hit — pins the short-circuit correctness).
    monkeypatch.setattr(
        ddf,
        "find_closed_sibling_suspects",
        lambda it: ([_closed_sibling_hit(1200)], []),
    )
    rc = run_driver(d, tasks_root, stub, "--retry-suspects")
    assert rc == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == [
        "landed-fix-suspect",
        "attempting",
        "filed",
    ]


def test_probe_order_1674_wins_when_both_would_fire(tmp_path, tasks_root, monkeypatch):
    # Test 8 (plan §15, using the brief's suggested split option (a) — a pure
    # unit-style pin via BOTH-monkeypatched helpers): assert the #1674 probe
    # is called FIRST and its non-None return short-circuits before
    # _closed_sibling_outcome is reached. Signalled via a call-order recorder.
    calls: list[str] = []

    def fake_landed(item, root, *, dirpath, fp, dry_run):
        calls.append("landed")
        # Simulate #1674 hit: append the row and return the terminal outcome.
        ddf.append_row(
            dirpath,
            {
                "slug": item["slug"],
                "outcome": "landed-fix-suspect",
                "suspects": [{"sha": "abc1234", "subject": "fake", "shared": ["x", "y", "z"]}],
                "threshold": ddf.LANDED_FIX_MIN_SHARED_TOKENS,
                "window": ddf.LANDED_FIX_WINDOW,
                "paths": ["scripts/foo.py"],
                "fp": fp,
                "route": item["route"],
            },
        )
        return "landed-fix-suspect"

    def fake_closed(item):
        calls.append("closed")
        return ([_closed_sibling_hit(1600)], [])

    monkeypatch.setattr(ddf, "_landed_fix_suspect_outcome", fake_landed)
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", fake_closed)
    item = make_item("cs-order", target="scripts/foo.py")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    # #1674 ran first and short-circuited; #1711 never ran.
    assert calls == ["landed"]
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    # The written row is #1674's shape (sha, not id).
    (suspect,) = rows[0]["suspects"]
    assert "sha" in suspect and "kind" not in suspect


def test_closed_sibling_probe_dry_run_read_only(tmp_path, tasks_root, monkeypatch, capsys):
    # Test 9 (plan §15, updated for #1735 composite arm): --dry-run with a
    # would-block closed sibling (target + non-stopword informative title
    # arm) prints CLOSED-SIBLING-SUSPECT (stdout, operator-facing), NO ledger
    # row written, NO filer call.
    item = make_item("cs-dry", target="scripts/foo.py")
    monkeypatch.setattr(
        ddf,
        "find_closed_sibling_suspects",
        lambda it: (
            [
                _closed_sibling_hit(
                    1600, title="workflow-fix: dry test", matched=["target", "title:planner"]
                )
            ],
            [],
        ),
    )
    d = make_filings_dir(tmp_path, [item])
    rc = run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run")
    assert rc == 0
    out = capsys.readouterr().out
    assert "CLOSED-SIBLING-SUSPECT cs-dry -> #1600" in out
    assert "--retry-suspects" in out
    # A dry-run must not write per-item ledger rows. The #1735 terminal
    # SUMMARY line prints on stderr but does NOT append the daily-drive-summary
    # ledger row under --dry-run (read-only by construction, plan §4.4).
    assert ledger_rows(d) == []
    assert filer_calls(d) == []


def test_closed_sibling_probe_route3_also_guarded(tmp_path, tasks_root, monkeypatch):
    # Test 10 (plan §15, updated for #1735 composite arm): the probe is
    # route-blind — a route-3 item with a COMPOSITE-blocking closed sibling
    # (target + non-stopword informative title arm) records a
    # landed-fix-suspect row carrying route: 3 and never files (parallel to
    # #1674's test_landed_fix_probe_guards_route3 at ~line 2246).
    item = make_item("cs-r3", route=3, target="scripts/foo.py")
    hit = _closed_sibling_hit(1600, matched=["target", "title:planner"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    assert rows[0]["route"] == 3
    (suspect,) = rows[0]["suspects"]
    assert suspect["kind"] == "closed-sibling"


def test_closed_sibling_error_plus_suspect_needs_both_flags(tmp_path, tasks_root, monkeypatch):
    # Concern-fold (Statistics critic): mirror of
    # test_landed_fix_error_plus_suspect_needs_both_flags at ~line 2127, but
    # for the closed-sibling probe. A slug carrying BOTH an ERROR row AND a
    # closed-sibling suspect row: --retry-errors alone re-runs the probe
    # (benign accumulation — another suspect row, no filing); BOTH
    # --retry-errors --retry-suspects re-drive it to a filed row.
    item = make_item("cs-both-rows", target="scripts/foo.py")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    d = make_filings_dir(tmp_path, [item])
    ddf.append_row(
        d,
        {
            "slug": "cs-both-rows",
            "outcome": "ERROR",
            "flag": "filer-failed",
            "id": None,
            "rc": 1,
            "fp": fp,
            "route": 2,
            "tail": "",
        },
    )
    ddf.append_row(
        d,
        {
            "slug": "cs-both-rows",
            "outcome": "landed-fix-suspect",
            "suspects": [_closed_sibling_hit(1600) | {"kind": "closed-sibling"}],
            "threshold": None,
            "window": "7.0 days",
            "paths": ["scripts/foo.py"],
            "fp": fp,
            "route": 2,
        },
    )
    # Under --retry-errors alone: the probe re-runs. Keep it returning a hit
    # to prove the second suspect row appends (benign accumulation).
    monkeypatch.setattr(
        ddf,
        "find_closed_sibling_suspects",
        lambda it: ([_closed_sibling_hit(1600)], []),
    )
    stub = make_stub(tmp_path, d)
    assert run_driver(d, tasks_root, stub, "--retry-errors") == 0
    assert filer_calls(d) == []
    assert [r["outcome"] for r in ledger_rows(d)][-1] == "landed-fix-suspect"
    # BOTH flags: `suspect_eyeballed` short-circuits the probe, filer runs.
    assert run_driver(d, tasks_root, stub, "--retry-errors", "--retry-suspects") == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)][-1] == "filed"


def test_find_closed_sibling_suspects_partitions_blocking_and_advisory(monkeypatch):
    # Unit-level (updated for #1735 composite arm rule):
    # find_closed_sibling_suspects partitions the helper's return into
    # (blocking, advisory) by the COMPOSITE arm rule — blocks ONLY when BOTH
    # a target-family arm (`target` / `infra-target`) AND a title-family arm
    # with at least one non-stopword informative token appear together. Every
    # bare-target / bare-title / target+all-stopword-title combination falls
    # to advisory.
    fake_helper_return = [
        _closed_sibling_hit(1, matched=["target"]),  # advisory (bare-target)
        _closed_sibling_hit(2, matched=["title:foo,bar"]),  # advisory (bare-title)
        _closed_sibling_hit(3, matched=["infra-target"]),  # advisory (bare-infra-target)
        _closed_sibling_hit(4, matched=["infra-title:baz,qux"]),  # advisory (bare-infra-title)
        _closed_sibling_hit(5, matched=["target", "title:z"]),  # blocking (composite, non-stopword)
        # composite with stopword-only title tokens — the exact FP shape #1735 exists for:
        _closed_sibling_hit(6, matched=["target", "title:main,runs"]),  # advisory (all stopwords)
        # composite where the title token list mixes a stopword + a non-stopword informative
        # token — informative wins, the hit blocks:
        _closed_sibling_hit(7, matched=["target", "title:main,planner"]),  # blocking
        # infra-title analogue of the composite blocking case:
        _closed_sibling_hit(
            8, matched=["infra-target", "infra-title:hub,upload"]
        ),  # blocking (composite)
    ]

    # Patch the imported symbol WHERE `find_closed_sibling_suspects` calls it
    # (module-scope import in the function body -> lives on task_workflow).
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "recent_closed_workflow_fix_tasks", lambda *a, **k: fake_helper_return)
    # Bypass the autouse-fixture neutralization to reach the REAL body.
    blocking, advisory = _REAL_FIND_CLOSED_SIBLING_SUSPECTS(
        make_item("part-test", target="scripts/x.py")
    )
    assert [h["id"] for h in blocking] == [5, 7, 8]
    assert [h["id"] for h in advisory] == [1, 2, 3, 4, 6]


# ── #1735: composite arm + driver-scoped stopword extension + SUMMARY ──────────


def test_composite_arm_target_only_is_advisory(tmp_path, tasks_root, monkeypatch, capsys):
    # A bare-target hit (path-family arm alone) is ADVISORY under the #1735
    # composite predicate — prints ONE CLOSED-SIBLING-ADVISORY stderr line,
    # NO CLOSED-SIBLING-SUSPECT on stdout, item files normally.
    item = make_item("cs-t-only", target="scripts/foo.py")
    hit = _closed_sibling_hit(1600, matched=["target"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], [hit]))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    out = capsys.readouterr()
    assert "CLOSED-SIBLING-ADVISORY cs-t-only -> #1600" in out.err
    assert "CLOSED-SIBLING-SUSPECT" not in out.out


def test_composite_arm_title_only_is_advisory(tmp_path, tasks_root, monkeypatch, capsys):
    # A bare-title hit (title-family arm alone, no path overlap) is ADVISORY —
    # unchanged from the pre-#1735 rule. Pins the sanity check.
    item = make_item("cs-title-only", target="scripts/foo.py")
    hit = _closed_sibling_hit(1601, matched=["title:planner,critic"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], [hit]))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    err = capsys.readouterr().err
    assert "CLOSED-SIBLING-ADVISORY cs-title-only -> #1601" in err


def test_composite_arm_target_plus_stopword_title_is_advisory(
    tmp_path, tasks_root, monkeypatch, capsys
):
    # The exact FP shape #1735 exists for: target + `title:main` (the top
    # false-positive reason on the 2026-07-26 batch, 9 hits). Under the
    # composite predicate + driver-scoped stopword filter this is ADVISORY,
    # NOT blocking — the item files normally.
    item = make_item("cs-t-stop", target=".claude/skills/issue/SKILL.md")
    hit = _closed_sibling_hit(1604, matched=["target", "title:main"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], [hit]))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert len(filer_calls(d)) == 1
    assert [r["outcome"] for r in ledger_rows(d)] == ["attempting", "filed"]
    out = capsys.readouterr()
    assert "CLOSED-SIBLING-ADVISORY cs-t-stop -> #1604" in out.err
    assert "CLOSED-SIBLING-SUSPECT" not in out.out


def test_composite_arm_target_plus_informative_title_blocks(
    tmp_path, tasks_root, monkeypatch, capsys
):
    # Composite predicate satisfied — target + title arm carrying a
    # non-stopword informative token. Expect CLOSED-SIBLING-SUSPECT on stdout
    # + a terminal `landed-fix-suspect` ledger row with kind=closed-sibling,
    # NO filer call.
    item = make_item("cs-block", target="scripts/foo.py")
    hit = _closed_sibling_hit(1700, matched=["target", "title:planner"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    (suspect,) = rows[0]["suspects"]
    assert suspect["kind"] == "closed-sibling"
    out = capsys.readouterr().out
    assert "CLOSED-SIBLING-SUSPECT cs-block -> #1700" in out


def test_composite_arm_pinned_1350_1329_shape_blocks(tmp_path, tasks_root, monkeypatch, capsys):
    # Pinned true positive: the #1350 vs #1329 shape — shared target +
    # shared informative title token (`workload-cmd`) that is NOT in the
    # driver-scoped stoplist. Composite predicate MUST still block this.
    # Regression coverage for the #1735 trade-off: bare-target duplicates
    # now downgrade to advisory (#1330, #1652), but a same-bug sibling with
    # ≥1 non-stopword title token still fires as blocking (the composite-
    # rule survivor class).
    item = make_item(
        "cs-1350-shape",
        target="src/explore_persona_space/backends/gcp.py",
        bug="workload-cmd env passthrough",
        change="fix workload-cmd env pin",
    )
    hit = _closed_sibling_hit(
        1329,
        title="workflow-fix: gcp workload-cmd env pin",
        matched=["target", "title:workload-cmd,env,pin"],
        target="src/explore_persona_space/backends/gcp.py",
    )
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    assert filer_calls(d) == []
    rows = ledger_rows(d)
    assert [r["outcome"] for r in rows] == ["landed-fix-suspect"]
    (suspect,) = rows[0]["suspects"]
    assert suspect["kind"] == "closed-sibling"
    assert suspect["id"] == 1329
    assert "CLOSED-SIBLING-SUSPECT cs-1350-shape -> #1329" in capsys.readouterr().out


def test_composite_arm_all_stopwords_extended_list(monkeypatch):
    # Every token in the driver-scoped stoplist, as the SOLE shared title
    # token accompanying a `target` arm hit, must partition into advisory
    # (all-stopword informative-title set is empty ⇒ composite predicate
    # not satisfied). Pins the stoplist membership + the composite rule.
    # find_closed_sibling_suspects caps advisory at CLOSED_SIBLING_MAX_HITS
    # (=5); asserting the cap-slice keeps the pin robust regardless of a
    # future stoplist size.
    stoplist = sorted(ddf.CLOSED_SIBLING_TITLE_STOPWORDS)
    assert stoplist, "stoplist must be non-empty (#1735 §4.2)"
    fake_helper_return = [
        _closed_sibling_hit(1000 + i, matched=["target", f"title:{tok}"])
        for i, tok in enumerate(stoplist)
    ]

    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "recent_closed_workflow_fix_tasks", lambda *a, **k: fake_helper_return)
    blocking, advisory = _REAL_FIND_CLOSED_SIBLING_SUSPECTS(
        make_item("stop-list-part", target="scripts/x.py")
    )
    assert blocking == []
    # The advisory list is capped at CLOSED_SIBLING_MAX_HITS by the helper;
    # every returned advisory hit must be a stoplist-token hit (never a
    # spurious blocker), and the surviving ids are the first-N of the fed
    # input in listing order.
    assert len(advisory) == min(len(stoplist), ddf.CLOSED_SIBLING_MAX_HITS)
    expected_ids = [1000 + i for i in range(len(advisory))]
    assert [h["id"] for h in advisory] == expected_ids


# ── #1735: terminal SUMMARY line + `daily-drive-summary` ledger row ────────────

_SUMMARY_KEYS = (
    "filed",
    "deduped",
    "already-tracked",
    "recovered",
    "skip",
    "error",
    "closed-sibling-suspects",
    "landed-fix-suspects",
)


def _last_summary_row(d: Path) -> dict | None:
    for row in reversed(ledger_rows_all(d)):
        if row.get("outcome") == "daily-drive-summary":
            return row
    return None


def test_summary_line_printed_at_end(tmp_path, tasks_root, capsys):
    # After a successful run, exactly ONE `SUMMARY dir=... filed=1 ...` line
    # appears on stderr with correct counts and no `held=H` column
    # (Statistics-critic MF2 — `held` is a row FIELD, not a returned outcome).
    item = make_item("sum-basic")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    err = capsys.readouterr().err
    summary_lines = [ln for ln in err.splitlines() if ln.startswith("SUMMARY dir=")]
    assert len(summary_lines) == 1
    line = summary_lines[0]
    assert "filed=1" in line
    assert "closed-sibling-suspects=0" in line
    assert "landed-fix-suspects=0" in line
    assert "held=" not in line  # MF2


def test_summary_row_appended_to_ledger(tmp_path, tasks_root):
    # Non-dry-run: filed.jsonl gains ONE terminal daily-drive-summary row
    # with the exact 8-key counts schema pinned by plan §4.4.
    item = make_item("sum-ledger")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    summary = _last_summary_row(d)
    assert summary is not None
    assert summary["slug"] is None
    assert summary["outcome"] == "daily-drive-summary"
    counts = summary["counts"]
    assert set(counts.keys()) == set(_SUMMARY_KEYS)
    # 8-key schema pin: the exact key set is exhaustive, so any accidental
    # additional field (a resurrected `held=H` column, a renamed suspect
    # counter) fails loud here.
    assert counts["filed"] == 1
    assert counts["closed-sibling-suspects"] == 0
    assert counts["landed-fix-suspects"] == 0
    assert "sliced" in summary
    assert "date" in summary


def test_summary_counter_splits_closed_sibling_vs_landed_fix(tmp_path, tasks_root, monkeypatch):
    # A fixture with ONE closed-sibling composite blocker AND ONE #1674
    # landed-fix-sha suspect must produce `closed-sibling-suspects=1` AND
    # `landed-fix-suspects=1` — never a single conflated `suspects=2`.
    # Statistics-critic MF3.
    item_cs = make_item("split-cs", target="scripts/foo.py")
    item_lf = make_item("split-lf", target="scripts/bar.py")
    d = make_filings_dir(tmp_path, [item_cs, item_lf])

    # Route the closed-sibling probe: returns a composite blocker for
    # item_cs, empty for item_lf.
    def fake_closed(item):
        if item["slug"] == "split-cs":
            return ([_closed_sibling_hit(1600, matched=["target", "title:planner"])], [])
        return ([], [])

    # Route the #1674 landed-fix-sha probe: returns a synthetic sha suspect
    # for item_lf only. Patched at the outcome layer so we mint the row
    # exactly like production (no `kind` field ⇒ landed-fix-suspects counter).
    def fake_lf_outcome(item, root, *, dirpath, fp, dry_run):
        if item["slug"] != "split-lf" or dry_run:
            return None
        ddf.append_row(
            dirpath,
            {
                "slug": item["slug"],
                "outcome": "landed-fix-suspect",
                "suspects": [{"sha": "deadbee", "subject": "prior fix", "shared": ["a", "b", "c"]}],
                "threshold": ddf.LANDED_FIX_MIN_SHARED_TOKENS,
                "window": ddf.LANDED_FIX_WINDOW,
                "paths": ["scripts/bar.py"],
                "fp": fp,
                "route": item["route"],
            },
        )
        return "landed-fix-suspect"

    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", fake_closed)
    monkeypatch.setattr(ddf, "_landed_fix_suspect_outcome", fake_lf_outcome)
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    summary = _last_summary_row(d)
    assert summary is not None
    counts = summary["counts"]
    assert counts["closed-sibling-suspects"] == 1
    assert counts["landed-fix-suspects"] == 1
    assert counts["filed"] == 0


def test_summary_row_skipped_on_dry_run(tmp_path, tasks_root, capsys):
    # --dry-run: the SUMMARY stderr line still prints, but NO ledger row is
    # appended (read-only by construction).
    item = make_item("sum-dry")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d), "--dry-run") == 0
    err = capsys.readouterr().err
    assert any(ln.startswith("SUMMARY dir=") for ln in err.splitlines())
    # A dry-run must not create filed.jsonl at all — no summary row either.
    assert not (d / "filed.jsonl").exists()


def test_summary_hint_when_suspects_nonzero(tmp_path, tasks_root, monkeypatch, capsys):
    # The SUMMARY line ends with the --retry-suspects hint iff
    # closed-sibling-suspects + landed-fix-suspects > 0.
    item = make_item("sum-hint", target="scripts/foo.py")
    hit = _closed_sibling_hit(1700, matched=["target", "title:planner"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d)) == 0
    err = capsys.readouterr().err
    (summary_line,) = [ln for ln in err.splitlines() if ln.startswith("SUMMARY dir=")]
    assert "re-run with --retry-suspects to file suspects" in summary_line

    # Clean batch — no suspects, no hint. Fresh dir + fresh stub.
    item2 = make_item("sum-clean-hint")
    d2 = make_filings_dir(tmp_path, [item2], date="2026-07-06")
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([], []))
    assert run_driver(d2, tasks_root, make_stub(tmp_path, d2, name="stub_clean.py")) == 0
    err2 = capsys.readouterr().err
    (line2,) = [ln for ln in err2.splitlines() if ln.startswith("SUMMARY dir=")]
    assert "--retry-suspects" not in line2


def test_retry_suspects_zero_match_notice(tmp_path, tasks_root, monkeypatch, capsys):
    # #1758: --retry-suspects on a fresh dir (no suspect rows anywhere) prints
    # ONE loud stderr NOTICE, the SUMMARY line carries the zero-match suffix,
    # and the daily-drive-summary ledger row keeps the exact 8-key schema.
    item = make_item("rs-zero-match")
    d = make_filings_dir(tmp_path, [item])
    assert run_driver(d, tasks_root, make_stub(tmp_path, d), "--retry-suspects") == 0
    err = capsys.readouterr().err
    notices = [ln for ln in err.splitlines() if ln.startswith("NOTICE: --retry-suspects")]
    assert len(notices) == 1
    assert "matched 0 recorded suspects" in notices[0]
    assert "nothing to retry (#1758)" in notices[0]
    (summary_line,) = [ln for ln in err.splitlines() if ln.startswith("SUMMARY dir=")]
    assert "--retry-suspects matched 0 recorded suspects (nothing retried)" in summary_line
    summary = _last_summary_row(d)
    assert summary is not None
    assert set(summary["counts"].keys()) == set(_SUMMARY_KEYS)

    # Coexistence pin: flag set, 0 pre-loop matches, but the run MINTS a new
    # suspect row (composite closed-sibling blocker) → the zero-match suffix
    # AND the re-run hint share ONE SUMMARY line, zero-match FIRST
    # (chronological: pre-loop state before this run's minted suspects).
    item2 = make_item("rs-zero-mints", target="scripts/foo.py")
    hit = _closed_sibling_hit(1800, matched=["target", "title:planner"])
    monkeypatch.setattr(ddf, "find_closed_sibling_suspects", lambda it: ([hit], []))
    d2 = make_filings_dir(tmp_path, [item2], date="2026-07-08")
    stub2 = make_stub(tmp_path, d2, name="stub_mint.py")
    assert run_driver(d2, tasks_root, stub2, "--retry-suspects") == 0
    err2 = capsys.readouterr().err
    assert any(ln.startswith("NOTICE: --retry-suspects") for ln in err2.splitlines())
    (line2,) = [ln for ln in err2.splitlines() if ln.startswith("SUMMARY dir=")]
    zero_i = line2.index("matched 0 recorded suspects (nothing retried)")
    hint_i = line2.index("re-run with --retry-suspects to file suspects")
    assert zero_i < hint_i


def test_retry_suspects_notice_absent_when_matched_or_flag_unset(tmp_path, tasks_root, capsys):
    # #1758 (a): a pre-seeded landed-fix-suspect row on a sliced slug makes
    # --retry-suspects match ≥1 — no NOTICE, no zero-match SUMMARY suffix.
    item = make_item("rs-matched")
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    d = make_filings_dir(tmp_path, [item])
    seed = {
        "slug": "rs-matched",
        "outcome": "landed-fix-suspect",
        "suspects": [_closed_sibling_hit(1300) | {"kind": "closed-sibling"}],
        "threshold": None,
        "window": "7.0 days",
        "paths": [".claude/skills/daily/SKILL.md"],
        "fp": fp,
        "route": 2,
    }
    with open(d / "filed.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seed) + "\n")
    assert run_driver(d, tasks_root, make_stub(tmp_path, d), "--retry-suspects") == 0
    err = capsys.readouterr().err
    assert "NOTICE: --retry-suspects" not in err
    (summary_line,) = [ln for ln in err.splitlines() if ln.startswith("SUMMARY dir=")]
    assert "matched 0 recorded suspects" not in summary_line

    # #1758 (b): flag unset on a fresh dir — no NOTICE and no zero-match
    # suffix anywhere on stderr (stderr behavior byte-unchanged).
    item2 = make_item("rs-flag-unset")
    d2 = make_filings_dir(tmp_path, [item2], date="2026-07-07")
    assert run_driver(d2, tasks_root, make_stub(tmp_path, d2, name="stub_unset.py")) == 0
    err2 = capsys.readouterr().err
    assert "NOTICE: --retry-suspects" not in err2
    assert "matched 0 recorded suspects" not in err2
