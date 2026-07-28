"""Step 9c gate degrade on collection ImportError (#1746).

One collection-broken test file on origin/main must not abort the entire
Step 9c gate run rc=2 (unclassifiable — MF-1b refuses; three 2026-07-27
sessions each paid ~38-45 min hand-deselecting + re-running the full gate).
With ``--continue-on-collection-errors`` in ``PYTEST_BASE_FLAGS`` pytest runs
the surviving collected tests, reports each collect error as a per-file junit
``<error>`` testcase, and exits rc=1 — inside compare's accepted {0,1} set —
so the existing NEW-vs-pre-existing node subtraction classifies it.

Pinned here:

- ``parse_junit`` absorbs the OBSERVED pytest 9.0.2 collect-error shape
  (``file`` attr present, ``classname=""``, dotted ``name``) via the normal
  path, PLUS the version-drift fallback (``<error>`` child, NO ``file`` attr,
  ``name`` a plausible ``.py`` path -> derived Node); every other
  missing-file shape keeps the hard ``JunitParseError``.
- Tiny-real e2e: ``run_pytest`` + ``PYTEST_BASE_FLAGS`` on {1 passing,
  1 ImportError-broken} file -> rc==1, exactly one Node for the broken file,
  the passing test executed; broken-only -> rc==1 with ``tests>=1`` (the
  pristine zero-collected guard stays a no-op, plan L1259 contingency).
- ``PYTEST_BASE_FLAGS`` carries the flag; ``_pristine_command`` is built from
  ``PYTEST_BASE_FLAGS`` (Must-Fix 1: no duplicate literal drops the flag).
- Compare classification: a ledger-known collect-error Node strips (exit 0);
  a branch-introduced one is NEW (exit 1); ``--pytest-rc 2`` still refuses to
  classify (exit 2, MF-1b regression).

Reuses the ``tests/test_step9c_baseline.py`` fixture machinery (and ITS
loaded ``sb`` module object, so monkeypatched fakes bind consistently).
"""

from __future__ import annotations

import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.test_step9c_baseline import _compare_env, _run_json, sb


@pytest.fixture(autouse=True)
def _gate_tmp_routing_disabled(monkeypatch):
    """Host-independent determinism (#1408): keep the real-subprocess cases off
    the data-disk temp routing (same convention as test_step9c_baseline.py)."""
    monkeypatch.setenv("EPM_STEP9C_TMPDIR", "")


# The OBSERVED pytest 9.0.2 collect-error Node shape (probe 2026-07-28):
# file attr present, classname empty, name = dotted module path.
COLLECT_NODE = sb.Node(
    file="tests/test_collect_broken.py", classname="", name="tests.test_collect_broken"
)


def _write_junit(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest" tests="1" failures="0" errors="1" skipped="0" time="0.1">'
        f"{body}</testsuite></testsuites>"
    )
    return path


# --- parse_junit: collect-error absorb --------------------------------------------


def test_parse_junit_absorbs_observed_collect_error_shape(tmp_path: Path):
    """The pytest 9.0.2 shape (file attr PRESENT) keys to a stable Node via the
    normal path — no fallback needed."""
    junit = _write_junit(
        tmp_path,
        '<testcase classname="" name="tests.test_collect_broken" '
        'file="tests/test_collect_broken.py" time="0.0">'
        '<error message="collection failure">ImportError</error></testcase>',
    )
    failing, summary = sb.parse_junit(junit)
    assert failing == [COLLECT_NODE]
    assert summary["tests"] == 1 and summary["errors"] == 1


def test_parse_junit_derives_file_from_py_name_when_file_attr_missing(tmp_path: Path):
    """Version-drift fallback: <error> child + NO file attr + a .py-path name
    derives a stable per-file Node instead of raising."""
    junit = _write_junit(
        tmp_path,
        '<testcase classname="" name="tests/test_collect_broken.py" time="0.0">'
        '<error message="collection failure">ImportError</error></testcase>',
    )
    failing, _summary = sb.parse_junit(junit)
    assert failing == [
        sb.Node(
            file="tests/test_collect_broken.py",
            classname="",
            name="tests/test_collect_broken.py",
        )
    ]


def test_parse_junit_still_raises_on_missing_file_with_non_path_name(tmp_path: Path):
    """An <error> row whose name is NOT a plausible .py path keeps the hard
    fail-loud JunitParseError (xunit1 contract violation)."""
    junit = _write_junit(
        tmp_path,
        '<testcase classname="" name="tests.test_collect_broken" time="0.0">'
        '<error message="collection failure">ImportError</error></testcase>',
    )
    with pytest.raises(sb.JunitParseError, match=r"no file attribute"):
        sb.parse_junit(junit)


def test_parse_junit_still_raises_on_missing_file_failure_child(tmp_path: Path):
    """A <failure> row (not <error>) without a file attr never takes the
    collect-error fallback, even with a .py-path name."""
    junit = _write_junit(
        tmp_path,
        '<testcase classname="" name="tests/test_x.py" time="0.0">'
        '<failure message="boom">x</failure></testcase>',
    )
    with pytest.raises(sb.JunitParseError, match=r"no file attribute"):
        sb.parse_junit(junit)


# --- flag plumbing pins -------------------------------------------------------------


def test_pytest_base_flags_carry_continue_on_collection_errors():
    assert "--continue-on-collection-errors" in sb.PYTEST_BASE_FLAGS


def test_pristine_command_flag_set_is_superset_of_base_flags():
    """Must-Fix 1: the printed manual-recovery command is built from
    PYTEST_BASE_FLAGS (single source), so it reproduces the oracle's flags —
    including the collect-error flag — instead of aborting rc=2."""
    cmd = sb._pristine_command(Path("/repo"), "tests/test_x.py")
    tokens = set(cmd.replace("(", " ").replace(")", " ").split())
    assert tokens >= set(sb.PYTEST_BASE_FLAGS)
    assert "--continue-on-collection-errors" in tokens


# --- tiny-real e2e: run_pytest with the base flags ----------------------------------


def _make_probe_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "probe"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test_ok.py").write_text(
        textwrap.dedent(
            """
            def test_passes():
                assert 1 + 1 == 2
            """
        )
    )
    (tree / "tests" / "test_broken.py").write_text(
        textwrap.dedent(
            """
            from nonexistent_module_i1746 import nothing  # noqa: F401


            def test_never_runs():
                assert True
            """
        )
    )
    return tree


def test_run_pytest_continues_past_collection_error(tmp_path: Path):
    """Real pytest subprocess: {1 passing, 1 ImportError-broken} file under
    PYTEST_BASE_FLAGS exits rc=1 (never the old abort rc=2), the broken file
    yields exactly ONE Node keyed to its path, and the passing test executed."""
    tree = _make_probe_tree(tmp_path)
    junit = tmp_path / "e2e-junit.xml"
    rc = sb.run_pytest(
        files=["tests/test_ok.py", "tests/test_broken.py"],
        cwd=tree,
        timeout_s=120.0,
        junit_path=junit,
        python_exe=sys.executable,
    )
    assert rc == 1
    failing, summary = sb.parse_junit(junit)
    assert len(failing) == 1
    assert failing[0].file == "tests/test_broken.py"
    # The passing test ran: 2 testcase rows total, 1 error, 0 failures.
    assert summary["tests"] == 2
    assert summary["errors"] == 1
    assert summary["failures"] == 0
    # Node identity is stable across runs (the compare/ledger join key):
    junit2 = tmp_path / "e2e-junit-2.xml"
    rc2 = sb.run_pytest(
        files=["tests/test_ok.py", "tests/test_broken.py"],
        cwd=tree,
        timeout_s=120.0,
        junit_path=junit2,
        python_exe=sys.executable,
    )
    failing2, _ = sb.parse_junit(junit2)
    assert rc2 == 1 and failing2 == failing


def test_run_pytest_broken_only_file_reads_tests_ge_1(tmp_path: Path):
    """Broken-only run: rc=1 and the junit counts the collect-error testcase
    ELEMENT (tests>=1) — the pristine zero-collected guard (plan L1259
    contingency) stays a no-op on a collection-red single file."""
    tree = _make_probe_tree(tmp_path)
    junit = tmp_path / "broken-only-junit.xml"
    rc = sb.run_pytest(
        files=["tests/test_broken.py"],
        cwd=tree,
        timeout_s=120.0,
        junit_path=junit,
        python_exe=sys.executable,
    )
    assert rc == 1
    failing, summary = sb.parse_junit(junit)
    assert [n.file for n in failing] == ["tests/test_broken.py"]
    assert summary["tests"] >= 1  # the guard `summary["tests"] == 0` cannot fire


def test_e2e_collect_error_junit_shape_matches_absorb_contract(tmp_path: Path):
    """The REAL junit row for a collect error carries the file attr (the
    observed pytest 9.0.2 shape the docstring documents) — if a future pytest
    drops it, this pin fails and the fallback branch becomes load-bearing."""
    tree = _make_probe_tree(tmp_path)
    junit = tmp_path / "shape-junit.xml"
    sb.run_pytest(
        files=["tests/test_broken.py"],
        cwd=tree,
        timeout_s=120.0,
        junit_path=junit,
        python_exe=sys.executable,
    )
    rows = [tc for tc in ET.parse(junit).getroot().iter("testcase") if tc.find("error") is not None]
    assert len(rows) == 1
    assert rows[0].get("file") == "tests/test_broken.py"


# --- compare classification ----------------------------------------------------------


def test_compare_strips_ledger_known_collect_error(tmp_path: Path, monkeypatch, capsys):
    """A KNOWN main-side collection-red file (its Node in the ledger) strips as
    pre-existing -> exit 0."""
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(COLLECT_NODE.file, COLLECT_NODE.classname, COLLECT_NODE.name, "error")],
        ledger_kw={"failing": (COLLECT_NODE,)},
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert {**COLLECT_NODE._asdict(), "via": "ledger"} in out["stripped"]
    assert out["new"] == []


def test_compare_flags_branch_new_collect_error_as_new(tmp_path: Path, monkeypatch, capsys):
    """A branch-INTRODUCED collection-red file (absent from the ledger AND from
    the main root) classifies NEW -> exit 1 (blocks)."""
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(COLLECT_NODE.file, COLLECT_NODE.classname, COLLECT_NODE.name, "error")],
        ledger_kw={"failing": ()},
        root_test_files=[],  # the broken file does not exist on main
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 1
    assert out["new"] == [COLLECT_NODE._asdict()]


def test_compare_pytest_rc_2_still_refuses_to_classify(tmp_path: Path, monkeypatch, capsys):
    """MF-1b regression: a genuinely aborted run (--pytest-rc 2) refuses to
    classify (exit 2) even when the junit parseably carries a collect-error
    row — rc=2 now means interruption/internal error, never a collect error."""
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(COLLECT_NODE.file, COLLECT_NODE.classname, COLLECT_NODE.name, "error")],
        ledger_kw={"failing": (COLLECT_NODE,)},
        pytest_rc=2,
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
