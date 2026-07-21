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


def _held_item(slug: str, title: str, bug: str) -> dict:
    return make_item(slug, route=3, title=title, bug=bug)


def test_route3_open_daily_held_overlap_dedups_no_filing(tmp_path, tasks_root, capsys):
    # Candidate tokens: {quota, decision, alpha} (title) | {widget, quota, exceeds,
    # gadget, budget, threshold} (bug). Task tokens: {widget, budget, review} (title) |
    # {decide, gadget, policy} (origin). Shared = {widget, budget, gadget} — EXACTLY 3,
    # pinning the >= boundary at ROUTE3_MIN_SHARED_TOKENS (a <-vs-<= off-by-one flips it).
    item = _held_item(
        "held-a",
        "daily-held: gpu quota decision alpha",
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
    assert row["route"] == 3 and "ts" in row
    assert "ALREADY-TRACKED held-a -> #77" in capsys.readouterr().out


def test_route3_two_shared_tokens_files_as_today(tmp_path, tasks_root):
    # Threshold boundary: shared = {widget, gadget} — exactly 2 < 3 — files normally.
    item = _held_item(
        "held-b",
        "daily-held: gpu quota decision alpha",
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
    item = _held_item(
        "held-d",
        "daily-held: gpu quota decision alpha",
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
    item = _held_item(
        "held-e",
        "daily-held: gpu quota decision alpha",
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
    # Quantitative acceptance (plan #1483 §6.1): the REAL #1140/#1472 pair replays
    # as a dedup hit with wide margin (measured 7 shared tokens vs threshold 3).
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
    assert len(row["shared"]) >= ddf.ROUTE3_MIN_SHARED_TOKENS  # measured 7 on the real texts
    assert {"codex", "quota"} <= set(row["shared"])
    assert "ALREADY-TRACKED codex-outage-refile -> #1140" in capsys.readouterr().out


def test_route3_already_tracked_is_terminal_on_resume(tmp_path, tasks_root, capsys):
    item = _held_item(
        "held-g",
        "daily-held: gpu quota decision alpha",
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
