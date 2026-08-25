"""Tests for ``scripts/audit_artifact_supersession.py`` (#2568).

Fixtures are scratch git repos built with GENERIC artifact/consumer names
(no trigger-dense text), mirroring the #1901 plan-v15 record instance
semantically: refuted claim + refuted artifacts + evidence + replacement
artifacts + producers + acknowledged_pending + label_patterns.

Scratch repos use ``tempfile.mkdtemp`` (NOT ``tmp_path``): concurrent pytest
sessions prune ``/tmp/pytest-of-*`` numbered roots and have deleted live
subprocess-heavy scratch dirs mid-test.

Covered (plan #2568 §3):

- four consumer states — labeled -> PASS, consumes-replacement -> PASS,
  unlabeled -> violation with file:line, producer-listed -> excluded;
- ``acknowledged_pending`` -> WARN (exit 0);
- malformed record -> WARN + exit 0 (``--strict``: FAIL + exit 1);
- ``--record`` single-record mode (+ not-in-index -> exit 2);
- ``--json`` report written + parseable;
- ``--include-tasks`` flips the tasks/** exclusion;
- exclusion classes (tasks/** + eval_results/** data hits excluded at
  grep time; tree classes recorded at class level);
- empty ``replacement_artifacts`` arm (only the label arm can pass);
- NEGATIVE fixture: JSONs under a ``superseded_*`` DIRECTORY are not
  records (the issue_1482 collision shape) — including a basename-matching
  JSON nested under such a directory;
- SPARSE fixture: record (and consumer) in the index but absent on disk is
  still fully audited (all reads are index-based);
- ``test_analyzer_fold_in_duty_pins_audit_tool``: BOTH analyzer surfaces
  name the audit tool (the c31 prose-pin discipline), asserted against the
  WORKTREE copies via a repo-root-relative resolve.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
_SCRIPT = _SCRIPTS / "audit_artifact_supersession.py"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_artifact_supersession import (  # noqa: E402
    UsageError,
    discover_records,
    run_audit,
)

REFUTED = "alpha_metrics_v1.json"
REPLACEMENT = "alpha_metrics_v2.json"
RECORD_REL = "eval_results/issue_42/superseded_alpha_join.json"


def _record_dict(**overrides: object) -> dict:
    rec: dict = {
        "schema": "artifact-supersession-record-v1",
        "issue": 42,
        "refuted_claim": "the v1 metric join mixed rows from two different pools",
        "refuted_artifacts": [REFUTED],
        "evidence": ["epm:progress 2026-01-01T00:00:00Z on #42"],
        "replacement_artifacts": [REPLACEMENT],
        "consumers_at_record_time": ["scripts/consumer_labeled.py"],
        "producers": ["scripts/producer_gen.py"],
        "acknowledged_pending": [],
        "label_patterns": ["superseded", "different eval pool", "different-eval-pool"],
    }
    rec.update(overrides)
    return rec


@pytest.fixture()
def repo() -> Iterator[Path]:
    """Scratch git repo (mkdtemp — see module docstring), removed after."""
    d = Path(tempfile.mkdtemp(prefix="eps2568-audit-"))
    try:
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write(repo_dir: Path, rel: str, content: str) -> None:
    p = repo_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _git_add_all(repo_dir: Path) -> None:
    # Explicit-path staging discipline governs the SHARED repo root; a
    # single-owner scratch fixture repo stages wholesale for brevity.
    subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)


def _build_standard_repo(repo_dir: Path, *, record_overrides: dict | None = None) -> None:
    """Record + the four consumer states + excluded-tree data hits."""
    rec = _record_dict(**(record_overrides or {}))
    _write(repo_dir, RECORD_REL, json.dumps(rec, indent=2) + "\n")
    _write(
        repo_dir,
        "scripts/consumer_labeled.py",
        'DATA = "alpha_metrics_v1.json"  # superseded by the v2 rebuild\n',
    )
    _write(
        repo_dir,
        "scripts/consumer_replacement.py",
        'OLD = "alpha_metrics_v1.json"\nNEW = "alpha_metrics_v2.json"\n',
    )
    _write(
        repo_dir,
        "scripts/consumer_unlabeled.py",
        'x = 1\ny = 2\nDATA = "alpha_metrics_v1.json"\nz = 3\n',
    )
    _write(repo_dir, "scripts/producer_gen.py", 'OUT = "alpha_metrics_v1.json"\n')
    # Excluded-tree data hits: inert provenance strings, never consumers.
    _write(repo_dir, "tasks/running/42/notes.md", "the run wrote alpha_metrics_v1.json\n")
    _write(
        repo_dir,
        "eval_results/issue_42/summary_over_v1.md",
        "derived from alpha_metrics_v1.json\n",
    )


def _run_cli(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_negative_fixture_superseded_directory_contents_are_not_records(repo: Path) -> None:
    """The issue_1482 collision shape: JSONs living UNDER a ``superseded_*``
    directory are not records — including one whose BASENAME matches."""
    _build_standard_repo(repo)
    _write(
        repo,
        "eval_results/issue_9/superseded_panel_stuff/blinding_manifest.json",
        "{}\n",
    )
    _write(
        repo,
        "eval_results/issue_9/superseded_panel_stuff/superseded_nested.json",
        "{}\n",
    )
    _git_add_all(repo)
    assert discover_records(repo) == [RECORD_REL]


def test_discovery_is_index_based_not_worktree(repo: Path) -> None:
    """An UNTRACKED record file is not discovered (index-based reads)."""
    _build_standard_repo(repo)
    _git_add_all(repo)
    _write(repo, "eval_results/issue_43/superseded_untracked.json", "{}\n")
    assert discover_records(repo) == [RECORD_REL]


# ---------------------------------------------------------------------------
# The four consumer states + exclusion classes
# ---------------------------------------------------------------------------


def test_four_consumer_states(repo: Path) -> None:
    _build_standard_repo(repo)
    _git_add_all(repo)
    report = run_audit(repo)
    assert len(report.violations) == 1, report.violations
    v = report.violations[0]
    assert "consumer scripts/consumer_unlabeled.py:3" in v
    assert REFUTED in v
    assert RECORD_REL in v
    rec = report.records[0]
    assert rec["schema_ok"] is True
    assert rec["consumers"]["scripts/consumer_labeled.py"]["status"] == "pass-labeled"
    assert rec["consumers"]["scripts/consumer_replacement.py"]["status"] == "pass-replacement"
    assert rec["consumers"]["scripts/consumer_unlabeled.py"]["status"] == "violation"
    assert rec["excluded"]["files"]["scripts/producer_gen.py"] == "producer"


def test_exclusion_classes_tree_level_and_include_tasks(repo: Path) -> None:
    """tasks/** + eval_results/** hits are excluded at grep time (recorded at
    class level); ``include_tasks`` re-admits tasks/** as consumers."""
    _build_standard_repo(repo)
    _git_add_all(repo)
    report = run_audit(repo)
    rec = report.records[0]
    assert "tasks" in rec["excluded"]["tree_classes"]
    assert "eval_results" in rec["excluded"]["tree_classes"]
    assert not any(p.startswith("tasks/") for p in rec["consumers"])
    assert not any(p.startswith("eval_results/") for p in rec["consumers"])
    report_tasks = run_audit(repo, include_tasks=True)
    rec_tasks = report_tasks.records[0]
    assert "tasks" not in rec_tasks["excluded"]["tree_classes"]
    assert rec_tasks["consumers"]["tasks/running/42/notes.md"]["status"] == "violation"
    assert any("tasks/running/42/notes.md" in v for v in report_tasks.violations)


def test_empty_replacement_artifacts_only_label_arm_passes(repo: Path) -> None:
    _build_standard_repo(repo, record_overrides={"replacement_artifacts": []})
    _git_add_all(repo)
    report = run_audit(repo)
    rec = report.records[0]
    assert rec["schema_ok"] is True
    # The replacement-consuming file has no replacement to consume and no
    # label -> it now violates alongside the unlabeled consumer.
    assert rec["consumers"]["scripts/consumer_replacement.py"]["status"] == "violation"
    assert rec["consumers"]["scripts/consumer_labeled.py"]["status"] == "pass-labeled"
    assert any("record declares no replacement artifacts" in v for v in report.violations)


# ---------------------------------------------------------------------------
# acknowledged_pending + malformed records
# ---------------------------------------------------------------------------


def test_acknowledged_pending_is_warn_not_fail(repo: Path) -> None:
    _build_standard_repo(
        repo,
        record_overrides={"acknowledged_pending": ["scripts/consumer_unlabeled.py"]},
    )
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("acknowledged_pending" in w for w in report.warnings)
    rec = report.records[0]
    assert rec["consumers"]["scripts/consumer_unlabeled.py"]["status"] == "acknowledged_pending"
    proc = _run_cli(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARN:" in proc.stdout


def test_malformed_record_warns_default_fails_strict(repo: Path) -> None:
    _build_standard_repo(repo, record_overrides={"evidence": []})
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("evidence" in w and RECORD_REL in w for w in report.warnings)
    assert report.records[0]["schema_ok"] is False
    strict_report = run_audit(repo, strict=True)
    assert any("evidence" in v for v in strict_report.violations)
    assert _run_cli(repo).returncode == 0
    assert _run_cli(repo, "--strict").returncode == 1


def test_unrecognized_schema_warns_with_migration_hint(repo: Path) -> None:
    _build_standard_repo(repo, record_overrides={"schema": "some-other-schema-v9"})
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("migration recipe" in w for w in report.warnings)


def test_unparseable_json_record_warns(repo: Path) -> None:
    _build_standard_repo(repo)
    _write(repo, RECORD_REL, "{ not json\n")
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("unparseable JSON" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# CLI: exit codes, --json, --record, --include-tasks
# ---------------------------------------------------------------------------


def test_cli_violation_exit_1_and_json_report(repo: Path) -> None:
    _build_standard_repo(repo)
    _git_add_all(repo)
    out_json = repo / "audit_report.json"
    proc = _run_cli(repo, "--json", str(out_json))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL:" in proc.stdout
    assert "consumer scripts/consumer_unlabeled.py:3" in proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["schema"] == "artifact-supersession-audit-report-v1"
    assert len(data["violations"]) == 1
    assert data["records"][0]["path"] == RECORD_REL


def test_cli_clean_exit_0(repo: Path) -> None:
    _build_standard_repo(repo)
    (repo / "scripts" / "consumer_unlabeled.py").unlink()
    _git_add_all(repo)
    proc = _run_cli(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL:" not in proc.stdout


def test_record_mode_audits_only_the_named_record(repo: Path) -> None:
    _build_standard_repo(repo)
    other = _record_dict(
        issue=43,
        refuted_claim="an unrelated refuted join over the beta pool",
        refuted_artifacts=["beta_metrics_v1.json"],
        replacement_artifacts=["beta_metrics_v2.json"],
        producers=[],
    )
    _write(repo, "eval_results/issue_43/superseded_beta_join.json", json.dumps(other) + "\n")
    _git_add_all(repo)
    # Full run sees both records; --record narrows to one.
    assert len(run_audit(repo).records) == 2
    report = run_audit(repo, record="eval_results/issue_43/superseded_beta_join.json")
    assert [r["path"] for r in report.records] == [
        "eval_results/issue_43/superseded_beta_join.json"
    ]
    assert report.violations == []
    proc = _run_cli(repo, "--record", "eval_results/issue_43/superseded_beta_join.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc_bad = _run_cli(repo, "--record", RECORD_REL)
    assert proc_bad.returncode == 1  # the unlabeled consumer violates


def test_record_mode_not_in_index_is_usage_error(repo: Path) -> None:
    _build_standard_repo(repo)
    _git_add_all(repo)
    _write(repo, "eval_results/issue_44/superseded_unadded.json", json.dumps(_record_dict()))
    proc = _run_cli(repo, "--record", "eval_results/issue_44/superseded_unadded.json")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "git add" in proc.stderr
    with pytest.raises(UsageError):
        run_audit(repo, record="eval_results/issue_44/superseded_unadded.json")


def test_cli_include_tasks_flag(repo: Path) -> None:
    _build_standard_repo(repo)
    (repo / "scripts" / "consumer_unlabeled.py").unlink()
    _git_add_all(repo)
    assert _run_cli(repo).returncode == 0
    proc = _run_cli(repo, "--include-tasks")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "tasks/running/42/notes.md" in proc.stdout


# ---------------------------------------------------------------------------
# Sparse (index-only) fixture
# ---------------------------------------------------------------------------


def test_sparse_index_only_record_and_consumer_still_audited(repo: Path) -> None:
    """Record AND consumer present in the index but ABSENT on disk (the
    sparse-worktree shape) are still fully audited — every read is
    index-based, never a working-tree open()."""
    _build_standard_repo(repo)
    _git_add_all(repo)
    shutil.rmtree(repo / "eval_results")
    (repo / "scripts" / "consumer_unlabeled.py").unlink()
    report = run_audit(repo)
    assert len(report.violations) == 1
    assert "consumer scripts/consumer_unlabeled.py:3" in report.violations[0]


# ---------------------------------------------------------------------------
# Agent-prose durability pin (the c31 prose-pin discipline)
# ---------------------------------------------------------------------------


def test_analyzer_fold_in_duty_pins_audit_tool() -> None:
    """BOTH analyzer surfaces name the audit tool, so the item-(7) re-fold
    duty cannot be silently dropped by a later spec edit."""
    for rel in (
        ".claude/agents/analyzer.md",
        ".claude/rules/analyzer-section-reference.md",
    ):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "audit_artifact_supersession.py" in text, (
            f"{rel} no longer names audit_artifact_supersession.py — the analyzer "
            "re-fold checklist item (7) supersession duty was dropped (#2568)"
        )
