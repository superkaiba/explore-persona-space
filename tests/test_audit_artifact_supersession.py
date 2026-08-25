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
- degenerate matching tokens (#2568 round 2): ``replacement_artifacts=[""]``,
  an eight-space refuted name, a whitespace-only label pattern, path
  separators / control characters, and a bool ``issue`` — each a schema
  WARN (strict: FAIL) with ZERO consumer verdicts (no pass-replacement
  from a degenerate token);
- non-UTF-8 tracked record bytes (#2568 round 2) -> malformed-record WARN
  (strict: FAIL), never a UsageError/exit-2;
- round-3 token classes (#2568 round 3), each a schema WARN (strict: FAIL)
  with ZERO consumer verdicts: cross-list containment (replacement equal
  to / contained in a refuted name; label pattern — explicit or DEFAULT —
  case-insensitively contained in a refuted name), U+2028/U+2029
  line-boundary tokens, subprocess-unsafe tokens (lone surrogate;
  oversized), and the minimum-substance floors (replacement >= 8 non-ws
  chars, label pattern >= 3);
- ``_read_index_blob_bytes`` (#2568 round 3 nit): None ONLY on confirmed
  index absence; a genuine git failure raises UsageError with stderr
  preserved;
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
    _read_index_blob_bytes,
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
# Degenerate matching tokens (#2568 round 2, blocker
# `empty-replacement-universal-pass`): a blank/degenerate entry is a schema
# problem and NO consumer is ever audited against it — in particular no
# consumer may receive a pass-replacement verdict from an empty string.
# ---------------------------------------------------------------------------


def _assert_degenerate_record_semantics(repo: Path, *, field_token: str) -> None:
    """Default: WARN + exit 0, ZERO consumer verdicts; --strict: FAIL + 1."""
    report = run_audit(repo)
    assert report.violations == [], report.violations
    assert any(field_token in w and RECORD_REL in w for w in report.warnings), report.warnings
    rec = report.records[0]
    assert rec["schema_ok"] is False
    # NO consumer processing on a degenerate record — zero verdicts of any
    # kind, so a pass-replacement verdict from a degenerate token is
    # structurally unreachable.
    assert rec["consumers"] == {}, rec["consumers"]
    strict_report = run_audit(repo, strict=True)
    assert any(field_token in v for v in strict_report.violations)
    # The strict arm pins the FULL consumer skip too (Claude review round 2
    # Minor): a degenerate record's consumers are never audited under
    # EITHER mode — zero verdicts of any kind, not merely no
    # pass-replacement.
    assert all(r["consumers"] == {} for r in strict_report.records), strict_report.records
    assert _run_cli(repo).returncode == 0
    assert _run_cli(repo, "--strict").returncode == 1


def test_empty_replacement_entry_is_schema_warn_never_universal_pass(repo: Path) -> None:
    """``replacement_artifacts=[""]``: '"" in content' is True for every
    file, so an empty entry must be rejected at schema time — never reach
    the consumption arm."""
    _build_standard_repo(repo, record_overrides={"replacement_artifacts": [""]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="replacement_artifacts")


def test_whitespace_only_refuted_name_is_schema_warn(repo: Path) -> None:
    """Eight spaces satisfy a raw-length >= 8 floor while grep-matching most
    indented files — the floor counts NON-whitespace chars and the blank
    token is rejected outright."""
    _build_standard_repo(repo, record_overrides={"refuted_artifacts": [" " * 8]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="refuted_artifacts")


def test_whitespace_only_label_pattern_is_schema_warn(repo: Path) -> None:
    """A whitespace-only label pattern would satisfy the +/-5-line window on
    nearly any line — rejected at schema time."""
    _build_standard_repo(repo, record_overrides={"label_patterns": ["   "]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="label_patterns")


def test_path_separator_and_control_char_tokens_are_schema_warns(repo: Path) -> None:
    _build_standard_repo(
        repo,
        record_overrides={
            "refuted_artifacts": ["dir/alpha_metrics_v1.json"],
            "replacement_artifacts": ["alpha\tmetrics_v2.json"],
        },
    )
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("path separator" in w for w in report.warnings), report.warnings
    assert any("control character" in w for w in report.warnings), report.warnings
    assert report.records[0]["consumers"] == {}


def test_bool_issue_is_schema_warn(repo: Path) -> None:
    """bool subclasses int — ``issue: true`` must not validate."""
    _build_standard_repo(repo, record_overrides={"issue": True})
    _git_add_all(repo)
    report = run_audit(repo)
    assert report.violations == []
    assert any("issue" in w and RECORD_REL in w for w in report.warnings), report.warnings
    assert report.records[0]["schema_ok"] is False


# ---------------------------------------------------------------------------
# Round-3 token classes (#2568 round 3): cross-list containment
# (`cross-list-token-universal-pass`), line-boundary separators
# (`unicode-line-separator-silent-pass`), subprocess-unsafe tokens
# (`subprocess-unsafe-refuted-token-false-fail`), and minimum-substance
# floors (`short-token-universal-pass`) — each a schema WARN (strict: FAIL)
# with ZERO consumer verdicts.
# ---------------------------------------------------------------------------


def test_replacement_equal_to_refuted_is_schema_warn(repo: Path) -> None:
    """A replacement token EQUAL to a refuted name is present in every grep
    hit by construction — a silent universal pass-replacement."""
    _build_standard_repo(repo, record_overrides={"replacement_artifacts": [REFUTED]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="replacement_artifacts")


def test_replacement_substring_of_refuted_is_schema_warn(repo: Path) -> None:
    """A replacement token CONTAINED in a refuted name (here a >=8-non-ws
    token, so the substance floor cannot be the catching arm) is present in
    every grep hit by construction."""
    _build_standard_repo(repo, record_overrides={"replacement_artifacts": ["metrics_v1.json"]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="replacement_artifacts")


@pytest.mark.parametrize("pattern", ["json", "Metrics_V1"])
def test_label_pattern_contained_in_refuted_is_schema_warn(repo: Path, pattern: str) -> None:
    """A label pattern contained CASE-INSENSITIVELY in a refuted name
    self-labels every mention line (the +/-5-line window includes the
    mention line itself) — the round-2 live demo ``label_patterns=["json"]``
    plus a mixed-case sibling."""
    _build_standard_repo(repo, record_overrides={"label_patterns": [pattern]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="label_patterns")


def test_default_label_pattern_contained_in_refuted_is_schema_warn(repo: Path) -> None:
    """With ``label_patterns`` OMITTED the defaults are effective — a refuted
    filename containing ``superseded`` would self-label every mention line
    under the default label arm, so it is a schema problem too."""
    _build_standard_repo(repo)
    rec = _record_dict(refuted_artifacts=["old_superseded_alpha_v1.json"])
    del rec["label_patterns"]  # omitted -> DEFAULT_LABEL_PATTERNS are effective
    _write(repo, RECORD_REL, json.dumps(rec, indent=2) + "\n")
    _write(
        repo,
        "scripts/consumer_default_label.py",
        'DATA = "old_superseded_alpha_v1.json"\n',
    )
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="label_patterns")


@pytest.mark.parametrize("sep", ["\u2028", "\u2029"])
def test_line_separator_refuted_token_is_schema_warn_never_pass_labeled(
    repo: Path, sep: str
) -> None:
    """U+2028/U+2029 pass the numeric control ranges but ARE splitlines()
    boundaries: git grep matches the bytes, the line split erases the
    mention, and the consumer silently reads pass-labeled. The token is
    rejected at schema time instead — via the splitlines() probe itself, so
    the rejected set cannot drift from Python's boundary set."""
    name = f"alpha{sep}metrics_v1.json"
    _build_standard_repo(repo, record_overrides={"refuted_artifacts": [name]})
    _write(repo, "scripts/consumer_linesep.py", f'DATA = "{name}"\n')
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="refuted_artifacts")
    assert any("line-boundary" in w for w in run_audit(repo).warnings)


def test_lone_surrogate_refuted_token_is_schema_warn_not_crash(repo: Path) -> None:
    """A JSON-escaped lone surrogate (U+D800) decodes to a str that passes
    blank/separator/control checks but raises UnicodeEncodeError at git argv
    construction — rejected at schema time so the subprocess is never
    reached (WARN default / strict FAIL, zero consumer verdicts)."""
    _build_standard_repo(
        repo,
        record_overrides={"refuted_artifacts": ["alpha_metrics_\ud800_v1.json"]},
    )
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="refuted_artifacts")
    assert any("not encodable as strict UTF-8" in w for w in run_audit(repo).warnings)


def test_oversized_refuted_token_is_schema_warn_not_crash(repo: Path) -> None:
    """An oversized token (over the filename-sized 255-byte bound) must
    never reach execve as an overlong argument — schema WARN instead."""
    _build_standard_repo(repo, record_overrides={"refuted_artifacts": ["a" * 300 + ".json"]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="refuted_artifacts")
    assert any("UTF-8 bytes" in w for w in run_audit(repo).warnings)


def test_short_replacement_token_is_schema_warn(repo: Path) -> None:
    """The round-2 live demo ``replacement_artifacts=[".json"]``: schema-valid
    then, a universal pass-replacement (every consumer was grep-selected on
    refuted ``*.json`` names). The >=8 non-whitespace floor now mirrors the
    refuted-name floor."""
    _build_standard_repo(repo, record_overrides={"replacement_artifacts": [".json"]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="replacement_artifacts")


def test_short_label_pattern_is_schema_warn(repo: Path) -> None:
    """A sub-floor (<3 non-whitespace chars) label pattern label-matches
    nearly any window — rejected at schema time."""
    _build_standard_repo(repo, record_overrides={"label_patterns": ["ok"]})
    _git_add_all(repo)
    _assert_degenerate_record_semantics(repo, field_token="label_patterns")


# ---------------------------------------------------------------------------
# _read_index_blob_bytes (#2568 round 3 nit
# `cat-file-error-misclassified-index-absence`): None ONLY on confirmed index
# absence; any other git failure raises UsageError with stderr preserved.
# ---------------------------------------------------------------------------


def test_read_index_blob_bytes_none_only_on_confirmed_index_absence(repo: Path) -> None:
    _build_standard_repo(repo)
    _git_add_all(repo)
    assert _read_index_blob_bytes(repo, "scripts/consumer_labeled.py") is not None
    assert _read_index_blob_bytes(repo, "scripts/never_added.py") is None


def test_read_index_blob_bytes_genuine_git_error_raises_with_stderr() -> None:
    """A non-repo dir makes BOTH git calls fail with rc != 1 — a genuine git
    error must raise (stderr preserved), never read as "absent from the
    index" and be misreported as "git add it first"."""
    scratch = Path(tempfile.mkdtemp(prefix="eps2568-notarepo-"))
    try:
        with pytest.raises(UsageError) as excinfo:
            _read_index_blob_bytes(scratch, "some/file.py")
        msg = str(excinfo.value)
        assert "git cat-file blob :some/file.py failed" in msg
        assert "rc=" in msg
        # git's own stderr diagnostic is carried, not discarded.
        assert "repository" in msg or "fatal" in msg, msg
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Non-UTF-8 record bytes (#2568 round 2, blocker
# `malformed-nonutf8-record-false-fail`): a DISCOVERED tracked record that is
# undecodable is a MALFORMED record (WARN; --strict: FAIL) — never a
# UsageError/exit-2 (which the lint wrapper would convert into a
# fleet-blocking FAIL).
# ---------------------------------------------------------------------------


def test_nonutf8_record_is_malformed_warn_not_usage_error(repo: Path) -> None:
    _build_standard_repo(repo)
    p = repo / RECORD_REL
    p.write_bytes(b"\xff\xfe{ this is not utf-8 \x80\x81 }\n")
    _git_add_all(repo)
    report = run_audit(repo)  # must NOT raise UsageError
    assert report.violations == []
    assert any("not decodable as UTF-8" in w and RECORD_REL in w for w in report.warnings), (
        report.warnings
    )
    assert report.records[0]["schema_ok"] is False
    assert report.records[0]["consumers"] == {}
    strict_report = run_audit(repo, strict=True)
    assert any("not decodable as UTF-8" in v for v in strict_report.violations)
    assert _run_cli(repo).returncode == 0
    assert _run_cli(repo, "--strict").returncode == 1


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
