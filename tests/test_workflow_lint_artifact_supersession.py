"""Tests for ``workflow_lint.check_artifact_supersession`` (#2568).

Four pins:

1. Scoped-flag SUBPROCESS on a fixture tree (the
   ``test_workflow_lint_inline_round_duty_mirror.py`` Part-A pattern):
   ``workflow_lint.py --check-artifact-supersession --file <workflow.yaml>``
   against a violating scratch git repo (via
   ``EPS_WORKFLOW_LINT_REPO_ROOT``) exits nonzero and surfaces the
   ``check-artifact-supersession`` token — the flag exists, the dispatch
   calls the check, and the ``__file__``-relative load exercises the REAL
   audit core (not the load-failure branch).
2. Direct-call semantics: violations -> FAIL lines; ``acknowledged_pending``
   + schema problems -> ``WARN:`` stderr lines, empty FAIL list.
3. Core LOAD FAILURE -> a FAIL line (monkeypatched ``workflow_lint.__file__``
   so the ``__file__``-relative core path resolves nowhere) — never a
   silent skip (the new-fence-silent-pass class).
4. ``test_check_artifact_supersession_bundled_in_no_flags`` — derives
   main()'s no-flags dispatch set from the workflow_lint.py SOURCE (the c37
   satisfier shape / duty-mirror Part-B pattern) and asserts the check is
   bundled, so a later dispatcher refactor cannot silently unbundle it.
5. Non-UTF-8 tracked record (#2568 round 2) -> non-blocking ``WARN:`` line,
   empty FAIL list — never a wrapper except-Exception FAIL.
6. Degenerate ``replacement_artifacts=[""]`` (#2568 round 2) -> schema
   ``WARN:`` line, never a silent universal pass.
7. Files-mode scope (#2568 round 2, `artifact-supersession-files-scope-hole`):
   the check carries the ``REPO_WIDE_SURFACE`` sentinel, so a ``--files``
   payload under a location the old enumerated surface set omitted (papers/)
   still RUNS the check and FAILs on a planted unlabeled mention.

Scratch repos use ``tempfile.mkdtemp`` (NOT ``tmp_path``): concurrent pytest
sessions prune ``/tmp/pytest-of-*`` numbered roots and have deleted live
subprocess-heavy scratch dirs mid-test.
"""

from __future__ import annotations

import json
import os
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
_LINT_SCRIPT = _SCRIPTS / "workflow_lint.py"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint  # noqa: E402
from workflow_lint import check_artifact_supersession  # noqa: E402

REFUTED = "gamma_metrics_v1.json"
REPLACEMENT = "gamma_metrics_v2.json"
RECORD_REL = "eval_results/issue_77/superseded_gamma_join.json"


def _record_dict(**overrides: object) -> dict:
    rec: dict = {
        "schema": "artifact-supersession-record-v1",
        "issue": 77,
        "refuted_claim": "the v1 gamma join mixed rows from two different pools",
        "refuted_artifacts": [REFUTED],
        "evidence": ["epm:progress 2026-01-01T00:00:00Z on #77"],
        "replacement_artifacts": [REPLACEMENT],
        "producers": [],
        "acknowledged_pending": [],
    }
    rec.update(overrides)
    return rec


@pytest.fixture()
def fixture_repo() -> Iterator[Path]:
    """Scratch git repo with a record + one UNLABELED consumer (a violation)."""
    d = Path(tempfile.mkdtemp(prefix="eps2568-lint-"))
    try:
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        rec_path = d / RECORD_REL
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(_record_dict(), indent=2) + "\n", encoding="utf-8")
        consumer = d / "scripts" / "gamma_consumer.py"
        consumer.parent.mkdir(parents=True, exist_ok=True)
        consumer.write_text('DATA = "gamma_metrics_v1.json"\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# (1) Scoped-flag subprocess on a fixture tree (Part-A pattern)
# ---------------------------------------------------------------------------


def test_scoped_flag_subprocess_detects_fixture_violation(fixture_repo: Path) -> None:
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = fixture_repo / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(fixture_repo)}
    result = subprocess.run(
        [
            sys.executable,
            str(_LINT_SCRIPT),
            "--check-artifact-supersession",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "check-artifact-supersession" in combined, (
        "check-artifact-supersession token missing from output — the CLI flag "
        f"does not dispatch the check. exit={result.returncode}, output:\n{combined}"
    )
    assert "scripts/gamma_consumer.py:1" in combined, (
        f"the REAL audit core did not run against the fixture tree:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit on a violating fixture tree; got {result.returncode}:\n{combined}"
    )


# ---------------------------------------------------------------------------
# (2) Direct-call semantics: FAIL lines vs WARN lines
# ---------------------------------------------------------------------------


def test_direct_call_maps_violations_to_fail_lines(fixture_repo: Path) -> None:
    errors = check_artifact_supersession(repo_root=fixture_repo)
    assert len(errors) == 1, errors
    assert errors[0].startswith("check-artifact-supersession: ")
    assert "scripts/gamma_consumer.py:1" in errors[0]
    assert REFUTED in errors[0]


def test_direct_call_acknowledged_pending_is_warn_line(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _record_dict(acknowledged_pending=["scripts/gamma_consumer.py"])
    (fixture_repo / RECORD_REL).write_text(json.dumps(rec) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    errors = check_artifact_supersession(repo_root=fixture_repo)
    assert errors == [], errors
    err_stream = capsys.readouterr().err
    assert "WARN: check-artifact-supersession:" in err_stream
    assert "acknowledged_pending" in err_stream


def test_direct_call_schema_problem_is_warn_line(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _record_dict(evidence=[])
    (fixture_repo / RECORD_REL).write_text(json.dumps(rec) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    errors = check_artifact_supersession(repo_root=fixture_repo)
    assert errors == [], errors
    err_stream = capsys.readouterr().err
    assert "WARN: check-artifact-supersession:" in err_stream
    assert "evidence" in err_stream


def test_nonutf8_record_is_warn_line_not_fleet_fail(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#2568 round 2 (`malformed-nonutf8-record-false-fail`): a tracked
    non-UTF-8 record routes to a non-blocking ``WARN:`` line — the wrapper's
    except-Exception FAIL backstop must never fire on it (default lint stays
    exit 0; the strict AUDIT exit-1 arm is pinned in
    tests/test_audit_artifact_supersession.py)."""
    (fixture_repo / RECORD_REL).write_bytes(b"\xff\xfe{ not utf-8 \x80 }\n")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    errors = check_artifact_supersession(repo_root=fixture_repo)
    assert errors == [], errors
    err_stream = capsys.readouterr().err
    assert "WARN: check-artifact-supersession:" in err_stream
    assert "not decodable as UTF-8" in err_stream


def test_degenerate_empty_replacement_is_warn_line_no_pass(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#2568 round 2 (`empty-replacement-universal-pass`): a record carrying
    ``replacement_artifacts=[""]`` is a schema WARN at the lint surface —
    never a silent universal pass (and never a FAIL line)."""
    rec = _record_dict(replacement_artifacts=[""])
    (fixture_repo / RECORD_REL).write_text(json.dumps(rec) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    errors = check_artifact_supersession(repo_root=fixture_repo)
    assert errors == [], errors
    err_stream = capsys.readouterr().err
    assert "WARN: check-artifact-supersession:" in err_stream
    assert "replacement_artifacts" in err_stream


# ---------------------------------------------------------------------------
# (3) Core load failure -> FAIL line, never a silent skip
# ---------------------------------------------------------------------------


def test_core_load_failure_is_a_fail_line(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = Path(tempfile.mkdtemp(prefix="eps2568-loadfail-"))
    try:
        # Point the __file__-relative core resolution at a dir with NO core.
        monkeypatch.setattr(
            workflow_lint, "__file__", str(scratch / "nowhere" / "workflow_lint.py")
        )
        errors = check_artifact_supersession(repo_root=scratch)
        assert len(errors) == 1, errors
        assert "check-artifact-supersession" in errors[0]
        assert "failed to load" in errors[0]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# (4) No-flags bundling pin (source-derived, the c37 satisfier shape)
# ---------------------------------------------------------------------------


def test_check_artifact_supersession_bundled_in_no_flags() -> None:
    """The check is bundled into the no-flags default run: main()'s
    ``no_flags = not (...)`` OR-chain AND the dispatch ladder both name
    ``args.check_artifact_supersession`` (derived from SOURCE so a
    dispatcher refactor cannot silently unbundle it)."""
    lint_src = _LINT_SCRIPT.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_artifact_supersession" in or_chain_src, (
        "args.check_artifact_supersession is NOT in the no_flags OR-chain — a bare "
        "workflow_lint.py invocation would not fire this check.\n"
        f"OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_artifact_supersession or no_flags" in main_src, (
        "args.check_artifact_supersession is NOT dispatched under `or no_flags` — "
        "the flag is defined but not bundled into the no-flags default run."
    )
    # Files-mode completeness: the runtime completeness check refuses on an
    # unclassified dispatch-site check, so both registries must carry it.
    assert "check_artifact_supersession" in workflow_lint.CHECK_SCOPES
    assert "check_artifact_supersession" in workflow_lint._FILES_MODE_RUNNERS


# ---------------------------------------------------------------------------
# (5) Files-mode scope: repo-wide sentinel — an omitted-location payload
#     still runs the check (#2568 round 2, files-scope-hole blocker).
# ---------------------------------------------------------------------------


def test_files_mode_repo_wide_sentinel_declared() -> None:
    """The CHECK_SCOPES entry is ``global`` with the REPO_WIDE_SURFACE
    sentinel — a consumer can live at ANY tracked path, so an enumerated
    top-level surface set (the round-1 shape) structurally under-covers."""
    scope = workflow_lint.CHECK_SCOPES["check_artifact_supersession"]
    assert scope.kind == "global"
    assert workflow_lint.REPO_WIDE_SURFACE in scope.surfaces
    assert workflow_lint._surface_hit("papers/new_consumer.py", workflow_lint.REPO_WIDE_SURFACE)
    assert workflow_lint._surface_hit("README.md", workflow_lint.REPO_WIDE_SURFACE)


def test_files_mode_omitted_location_payload_runs_check_and_fails(
    fixture_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``--files`` payload adding an unlabeled consumer under papers/ — a
    location the pre-round-2 enumerated surface set OMITTED — must RUN the
    check (no SKIP) and FAIL on the planted mention (the violation names the
    payload path, so the in-scope attribution filter keeps it)."""
    consumer = fixture_repo / "papers" / "new_consumer.py"
    consumer.parent.mkdir(parents=True, exist_ok=True)
    consumer.write_text('DATA = "gamma_metrics_v1.json"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "-A"], check=True)
    monkeypatch.chdir(workflow_lint._REPO_ROOT)
    monkeypatch.setenv("EPS_WORKFLOW_LINT_REPO_ROOT", str(fixture_repo))
    rc = workflow_lint.main(["--files", "papers/new_consumer.py"])
    cap = capsys.readouterr()
    out = cap.out + "\n" + cap.err
    assert "SKIP check_artifact_supersession" not in out, out
    assert "check-artifact-supersession" in out, out
    assert "papers/new_consumer.py:1" in out, out
    assert rc == 1, out
