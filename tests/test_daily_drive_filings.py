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

import json
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
) -> str:
    """Write a stub filer executable; return the --filer prefix string."""
    stub = tmp_path / name
    stub.write_text(
        STUB_TEMPLATE.format(
            dirpath=str(dirpath),
            output_line=output_line,
            exit_code=exit_code,
            fail_marker=fail_marker,
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
) -> Path:
    """Build one synthetic task dir through the REAL frontmatter writer (yaml.safe_dump)."""
    fm = {"title": title, "kind": "infra", "tags": tags or []}
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
