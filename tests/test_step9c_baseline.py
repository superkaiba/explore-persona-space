"""Unit tests for ``scripts/step9c_baseline.py`` (#1022).

The helper maintains the known-red-on-main baseline ledger the ``/issue``
Step 9c test-verdict gate diffs against. Most cases inject signature-conformant
fake runners (pytest / ruff / git probes / the pristine oracle) plus
``tmp_path`` ledger + junit fixtures, so no real git branch state is needed.

The plan #1022 §3.6 case list is implemented below; cases 2, 3, 6, 12 and
16-22 form the BINDING A2 fail-loud / never-mask set. Case 15 is the one
real-subprocess integration case (real pytest, bounded in-test timeout); the
"real body" tests at the bottom execute every production function the unit
cases stub, per the one-production-body-test-per-seam-stubbed-function rule
(code-style.md #906).
"""

from __future__ import annotations

import fcntl
import fnmatch
import importlib.util
import json
import os
import subprocess

# Import the helper by path (it lives under scripts/, not an importable package).
# sys.modules registration BEFORE exec_module is required: the module defines
# dataclasses, whose field-type resolution looks itself up in sys.modules.
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "step9c_baseline.py"
_spec = importlib.util.spec_from_file_location("step9c_baseline", _HELPER_PATH)
assert _spec and _spec.loader
sb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sb
_spec.loader.exec_module(sb)

# Captured BEFORE any fixture monkeypatches it — the missing-ruff case restores it.
_REAL_RUFF_ERROR_COUNT = sb.ruff_error_count

NODE_A = sb.Node(file="tests/test_known_red.py", classname="tests.test_known_red", name="test_red")
NODE_SCAN = sb.Node(
    file="tests/test_scan_thing.py", classname="tests.test_scan_thing", name="test_scan_red"
)


# --- Fixture builders ---------------------------------------------------------


def _junit_xml(cases: list[tuple[str, str, str, str]]) -> str:
    """Build an xunit1 junitxml string from (file, classname, name, status) rows."""
    n_fail = sum(1 for *_r, s in cases if s == "failed")
    n_err = sum(1 for *_r, s in cases if s == "error")
    n_skip = sum(1 for *_r, s in cases if s == "skipped")
    child = {
        "passed": "",
        "failed": '<failure message="boom">x</failure>',
        "error": '<error message="boom">x</error>',
        "skipped": "<skipped/>",
    }
    rows = "".join(
        f'<testcase classname="{cls}" name="{name}" file="{file}" time="0.01">'
        f"{child[status]}</testcase>"
        for file, cls, name, status in cases
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest" tests="{len(cases)}" failures="{n_fail}" errors="{n_err}" '
        f'skipped="{n_skip}" time="0.5">{rows}</testsuite></testsuites>'
    )


def _ledger_dict(
    *,
    main_sha: str = "a" * 40,
    age_h: float = 0.0,
    dirty: bool = False,
    dirty_paths: tuple[str, ...] = (),
    failing: tuple = (),
    ruff_count: int = 100,
    ruff_format_files: int = 0,
) -> dict:
    """A schema-valid ledger dict with the given knobs."""
    refreshed = datetime.now(UTC) - timedelta(hours=age_h)
    return {
        "schema_version": 1,
        "main_sha": main_sha,
        "refreshed_at": refreshed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dirty_code_paths": dirty,
        "dirty_paths": list(dirty_paths),
        "test_universe": ["tests/test_known_red.py"],
        "failing_tests": [n._asdict() for n in failing],
        "pytest_summary": {
            "tests": 10,
            "failures": len(failing),
            "errors": 0,
            "skipped": 0,
            "duration_s": 1.0,
        },
        "ruff_count": ruff_count,
        "ruff_format_files": ruff_format_files,
        "refresh_timeout_s": 1800,
        "generator": "step9c_baseline.py v1",
    }


def _write_ledger(root: Path, **kw) -> dict:
    """Write a schema-valid ledger under *root*'s cache dir; return the dict."""
    led = _ledger_dict(**kw)
    path = sb.ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led))
    return led


class _FakeSel:
    """Signature-conformant stand-in for the loaded selector module (compare seam)."""

    def __init__(self, touched, reasons, glob_scan):
        self._touched = list(touched)
        self._reasons = {k: sorted(v) for k, v in (reasons or {}).items()}
        self.GLOB_SCAN_TESTS = dict(glob_scan or {})
        self.WORKFLOW_INVARIANT: tuple[str, ...] = ()

    def compute_touched(self, base: str, work_root: Path, _runner=None) -> list[str]:
        return list(self._touched)

    def select_tests_with_reasons(
        self, touched: list[str], work_root: Path
    ) -> tuple[list[str], list[str], dict[str, list[str]]]:
        return sorted(self._reasons), [], dict(self._reasons)

    def _matches_any(self, path: str, globs: tuple[str, ...]) -> bool:
        return any(fnmatch.fnmatch(path, g) for g in globs)


def _materialize_compare_tree(
    tmp_path: Path,
    *,
    junit_cases,
    touched,
    root_test_files,
    ledger: bool,
    ledger_kw: dict | None,
    ledger_raw: str | None,
) -> tuple[Path, Path, Path]:
    """Build the compare fixture dirs + ledger + junit; return (root, wt, junit)."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True, exist_ok=True)  # some cases build the env twice
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True, exist_ok=True)
    files_at_root = (
        root_test_files if root_test_files is not None else sorted({c[0] for c in junit_cases})
    )
    for f in files_at_root:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")
    for f in touched:
        if f.endswith(".py"):
            p = wt / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("X = 1\n")
    if ledger_raw is not None:
        path = sb.ledger_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger_raw)
    elif ledger:
        _write_ledger(root, **(ledger_kw or {}))
    junit = tmp_path / "run-junit.xml"
    junit.write_text(_junit_xml(list(junit_cases)))
    return root, wt, junit


def _install_compare_fakes(
    monkeypatch,
    *,
    root: Path,
    fake_sel,
    changed_tests,
    live_dirty,
    pristine_failing,
    pristine_exc,
    base_ruff,
    wt_ruff,
    touched_ruff,
    sha_known,
    code_commits,
) -> dict[str, list]:
    """Monkeypatch signature-conformant fakes onto the module; return the call recorder."""
    calls: dict[str, list] = {"pristine": []}

    def fake_load_selector_module(root_: Path):
        return fake_sel

    def fake_changed_test_files_since(root_: Path, sha: str) -> set[str]:
        return set(changed_tests)

    def fake_dirty_code_paths(root_: Path) -> list[str]:
        return list(live_dirty)

    def fake_git_sha_known(root_: Path, sha: str) -> bool:
        return sha_known

    def fake_code_commits_since(root_: Path, sha: str) -> int:
        return code_commits

    def fake_run_single_file_pristine(test_file: str, cwd: Path, timeout_s: float) -> set:
        calls["pristine"].append(test_file)
        if pristine_exc is not None:
            raise pristine_exc
        return {n for n in pristine_failing if n.file == test_file}

    def fake_ruff_error_count(target: Path, paths: list[str] | None = None) -> int:
        if paths is not None:
            return touched_ruff[0]
        return base_ruff[0] if Path(target).resolve() == root.resolve() else wt_ruff[0]

    def fake_ruff_format_count(target: Path, paths: list[str] | None = None) -> int:
        if paths is not None:
            return touched_ruff[1]
        return base_ruff[1] if Path(target).resolve() == root.resolve() else wt_ruff[1]

    monkeypatch.setattr(sb, "load_selector_module", fake_load_selector_module)
    monkeypatch.setattr(sb, "changed_test_files_since", fake_changed_test_files_since)
    monkeypatch.setattr(sb, "dirty_code_paths", fake_dirty_code_paths)
    monkeypatch.setattr(sb, "git_sha_known", fake_git_sha_known)
    monkeypatch.setattr(sb, "code_commits_since", fake_code_commits_since)
    monkeypatch.setattr(sb, "run_single_file_pristine", fake_run_single_file_pristine)
    monkeypatch.setattr(sb, "ruff_error_count", fake_ruff_error_count)
    monkeypatch.setattr(sb, "ruff_format_count", fake_ruff_format_count)
    return calls


def _compare_env(
    tmp_path: Path,
    monkeypatch,
    *,
    junit_cases,
    pytest_rc: int = 1,
    ledger: bool = True,
    ledger_kw: dict | None = None,
    ledger_raw: str | None = None,
    touched=(),
    reasons=None,
    glob_scan=None,
    changed_tests=(),
    live_dirty=(),
    pristine_failing=(),
    pristine_exc: Exception | None = None,
    base_ruff=(100, 0),
    wt_ruff=(100, 0),
    touched_ruff=(0, 0),
    sha_known: bool = True,
    code_commits: int = 0,
    root_test_files=None,
    extra_args=(),
):
    """Set up a compare fixture tree + fakes; return (argv, calls, root, wt)."""
    root, wt, junit = _materialize_compare_tree(
        tmp_path,
        junit_cases=junit_cases,
        touched=touched,
        root_test_files=root_test_files,
        ledger=ledger,
        ledger_kw=ledger_kw,
        ledger_raw=ledger_raw,
    )
    calls = _install_compare_fakes(
        monkeypatch,
        root=root,
        fake_sel=_FakeSel(touched, reasons, glob_scan),
        changed_tests=changed_tests,
        live_dirty=live_dirty,
        pristine_failing=pristine_failing,
        pristine_exc=pristine_exc,
        base_ruff=base_ruff,
        wt_ruff=wt_ruff,
        touched_ruff=touched_ruff,
        sha_known=sha_known,
        code_commits=code_commits,
    )
    argv = [
        "compare",
        "--junitxml",
        str(junit),
        "--pytest-rc",
        str(pytest_rc),
        "--worktree",
        str(wt),
        "--repo-root",
        str(root),
        "--json",
        *extra_args,
    ]
    return argv, calls, root, wt


def _run_json(argv: list[str], capsys) -> tuple[int, dict | None, str]:
    """Run sb.main(argv); return (rc, parsed stdout JSON or None, stderr)."""
    rc = sb.main(argv)
    captured = capsys.readouterr()
    out = captured.out.strip()
    return rc, (json.loads(out) if out.startswith("{") else None), captured.err


def _refresh_env(
    tmp_path: Path, monkeypatch, *, junit_cases=None, pytest_rc=1, raise_timeout=False, venv=True
):
    """Refresh fixture: root tree + fake selector/pytest/git/ruff; returns (argv, root, seen)."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    if venv:
        # Existence satisfies resolve_root_python; the fake runner never execs it.
        venv_py = root / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.write_text("")
    invariants = ("tests/test_inv_a.py", "tests/test_inv_b.py", "tests/test_inv_missing.py")
    scans = {"tests/test_scan.py": ("scripts/issue*_*.py",)}
    for f in ("tests/test_inv_a.py", "tests/test_inv_b.py", "tests/test_scan.py"):
        (root / f).write_text("# stub\n")

    fake_sel = _FakeSel([], {}, scans)
    fake_sel.WORKFLOW_INVARIANT = invariants
    seen: dict = {}

    def fake_load_selector_module(root_: Path):
        return fake_sel

    cases = (
        junit_cases
        if junit_cases is not None
        else [
            ("tests/test_inv_a.py", "tests.test_inv_a", "test_ok", "passed"),
            ("tests/test_scan.py", "tests.test_scan", "test_red", "failed"),
        ]
    )

    def fake_run_pytest(
        files, cwd, timeout_s, junit_path, extra=sb.PYTEST_BASE_FLAGS, *, python_exe
    ) -> int:
        seen["stale_junit_gone"] = not Path(junit_path).exists()
        seen["files"] = list(files)
        seen["python_exe"] = python_exe
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=timeout_s)
        Path(junit_path).write_text(_junit_xml(cases))
        return pytest_rc

    def fake_git_head(root_: Path) -> str:
        return "b" * 40

    def fake_dirty_code_paths(root_: Path) -> list[str]:
        return []

    def fake_ruff_error_count(target: Path, paths: list[str] | None = None) -> int:
        return 2149

    def fake_ruff_format_count(target: Path, paths: list[str] | None = None) -> int:
        return 18

    monkeypatch.setattr(sb, "load_selector_module", fake_load_selector_module)
    monkeypatch.setattr(sb, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(sb, "git_head", fake_git_head)
    monkeypatch.setattr(sb, "dirty_code_paths", fake_dirty_code_paths)
    monkeypatch.setattr(sb, "ruff_error_count", fake_ruff_error_count)
    monkeypatch.setattr(sb, "ruff_format_count", fake_ruff_format_count)
    return ["refresh", "--repo-root", str(root)], root, seen


# --- Case 1: refresh writes a schema-valid ledger ------------------------------


def test_refresh_writes_schema_valid_ledger(tmp_path: Path, monkeypatch):
    argv, root, seen = _refresh_env(tmp_path, monkeypatch)
    # A stale junit from a prior refresh must be unlinked BEFORE pytest runs (MF-1a).
    junit = root / ".claude" / "cache" / "step9c-baseline-junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    junit.write_text("STALE GARBAGE")
    assert sb.main(argv) == 0
    assert seen["stale_junit_gone"] is True
    # Universe = sorted present-on-disk invariants + scan keys (absent entry filtered).
    assert seen["files"] == ["tests/test_inv_a.py", "tests/test_inv_b.py", "tests/test_scan.py"]
    ledger = json.loads(sb.ledger_path(root).read_text())
    assert set(ledger) == sb.REQUIRED_LEDGER_KEYS
    assert ledger["schema_version"] == 1
    assert ledger["main_sha"] == "b" * 40
    assert ledger["failing_tests"] == [
        {"file": "tests/test_scan.py", "classname": "tests.test_scan", "name": "test_red"}
    ]
    assert ledger["ruff_count"] == 2149 and ledger["ruff_format_files"] == 18
    assert ledger["dirty_code_paths"] is False
    assert ledger["test_universe"] == seen["files"]
    # The refresh pytest runs the ROOT's own venv interpreter, never the invoking
    # sys.executable (#1022 round-2 Critical, refresh sibling).
    assert seen["python_exe"] == str(root / ".venv" / "bin" / "python")
    assert seen["python_exe"] != sys.executable
    # Atomic write: no tmp residue in the cache dir.
    assert list((root / ".claude" / "cache").glob("*.tmp")) == []


# --- Case 2 [A2]: lock-busy refresh -> single-flight clean exit ----------------


def test_refresh_lock_busy_single_flight(tmp_path: Path, monkeypatch, capsys):
    argv, root, _seen = _refresh_env(tmp_path, monkeypatch)
    _write_ledger(root)
    before = sb.ledger_path(root).read_bytes()
    lock_file = root / ".claude" / "cache" / "step9c-baseline.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "wb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = sb.main(argv)
    assert rc == 0
    assert "single-flight" in capsys.readouterr().err
    assert sb.ledger_path(root).read_bytes() == before  # existing ledger byte-unchanged


# --- Case 3 [A2]: refresh timeout / rc not in {0,1} -> exit 2, NO ledger write --


@pytest.mark.parametrize("mode", ["timeout", 2, 3, 5])
def test_refresh_timeout_or_bad_rc_writes_no_ledger(tmp_path: Path, monkeypatch, mode):
    if mode == "timeout":
        argv, root, _ = _refresh_env(tmp_path, monkeypatch, raise_timeout=True)
    else:
        argv, root, _ = _refresh_env(tmp_path, monkeypatch, pytest_rc=mode)
    assert sb.main(argv) == 2
    assert not sb.ledger_path(root).exists()


# --- Case 4: refresh zero-collected junit -> exit 2, no write ------------------


def test_refresh_zero_collected_writes_no_ledger(tmp_path: Path, monkeypatch):
    argv, root, _ = _refresh_env(tmp_path, monkeypatch, junit_cases=[], pytest_rc=0)
    assert sb.main(argv) == 2
    assert not sb.ledger_path(root).exists()


# --- Case 5: blind-strip happy path ---------------------------------------------


def test_compare_blind_strip_happy_path(tmp_path: Path, monkeypatch, capsys):
    argv, calls, _root, _wt = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[
            (NODE_A.file, NODE_A.classname, NODE_A.name, "failed"),
            ("tests/test_fine.py", "tests.test_fine", "test_ok", "passed"),
        ],
        ledger_kw={"failing": (NODE_A,)},
        reasons={NODE_A.file: ["invariant"]},
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["new"] == []
    assert out["stripped"] == [{**NODE_A._asdict(), "via": "ledger"}]
    assert calls["pristine"] == []  # safe blind strip — no pristine run needed


# --- Case 6 [A2]: node granularity — NEW node inside a known-red FILE -----------


def test_compare_new_node_in_known_red_file_blocks(tmp_path: Path, monkeypatch, capsys):
    node_b = sb.Node(file=NODE_A.file, classname=NODE_A.classname, name="test_other_red")
    kwargs = dict(
        junit_cases=[(node_b.file, node_b.classname, node_b.name, "failed")],
        ledger_kw={"failing": (NODE_A,)},
        reasons={NODE_A.file: ["invariant"]},
        pristine_failing=(NODE_A,),  # node_b PASSES on main
    )
    # Without --run-pristine: indeterminate, exit 2 (never a silent strip).
    argv, _calls, _r, _w = _compare_env(tmp_path, monkeypatch, **kwargs)
    rc, _out, err = _run_json(argv, capsys)
    assert rc == 2
    assert "pristine" in err
    # With --run-pristine: node_b passes on main -> NEW -> exit 1.
    argv, calls, _r, _w = _compare_env(
        tmp_path, monkeypatch, extra_args=("--run-pristine",), **kwargs
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 1
    assert out["new"] == [node_b._asdict()]
    assert calls["pristine"] == [node_b.file]


# --- Case 7: unknown provenance without --run-pristine -> commands printed ------


def test_compare_unknown_provenance_prints_pristine_commands(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(file="tests/test_mystery.py", classname="tests.test_mystery", name="test_x")
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
    )
    rc, _out, err = _run_json(argv, capsys)
    assert rc == 2
    assert "tests/test_mystery.py" in err and "pytest" in err  # copy-pasteable command


# --- Case 8: --run-pristine strips fail-on-main, blocks pass-on-main ------------


@pytest.mark.parametrize("fails_on_main", [True, False])
def test_compare_run_pristine_strip_or_new(tmp_path: Path, monkeypatch, capsys, fails_on_main):
    node = sb.Node(file="tests/test_mystery.py", classname="tests.test_mystery", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        pristine_failing=(node,) if fails_on_main else (),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert calls["pristine"] == [node.file]
    if fails_on_main:
        assert rc == 0
        assert out["stripped"] == [{**node._asdict(), "via": "pristine"}]
    else:
        assert rc == 1
        assert out["new"] == [node._asdict()]


# --- Case 9: diff-linked known-red never blind-strips; pristine strip WARNs -----


def test_compare_diff_linked_known_red_pristine_routed(tmp_path: Path, monkeypatch, capsys):
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger_kw={"failing": (NODE_A,)},
        touched=("scripts/known_red.py",),
        reasons={NODE_A.file: ["invariant", "stem-map:scripts/known_red.py"]},
        pristine_failing=(NODE_A,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine"] == [NODE_A.file]  # NOT blind-stripped
    assert out["stripped"] == [{**NODE_A._asdict(), "via": "pristine"}]
    assert any("diff-linked" in w for w in out["warns"])


# --- Case 10: stale ledger routes everything to pristine ------------------------


@pytest.mark.parametrize("mode", ["age", "commits", "sha"])
def test_compare_stale_ledger_routes_pristine(tmp_path: Path, monkeypatch, capsys, mode):
    kwargs = dict(
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger_kw={"failing": (NODE_A,), "age_h": 48.0 if mode == "age" else 0.0},
        reasons={NODE_A.file: ["invariant"]},
        pristine_failing=(NODE_A,),
        code_commits=999 if mode == "commits" else 0,
        sha_known=mode != "sha",
        extra_args=("--run-pristine",),
    )
    argv, calls, _r, _w = _compare_env(tmp_path, monkeypatch, **kwargs)
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["stale"] is True and out["stale_reasons"]
    # Pristine-routed, never blind-stripped, despite the node being in the ledger.
    assert calls["pristine"] == [NODE_A.file]
    assert out["stripped"] == [{**NODE_A._asdict(), "via": "pristine"}]


# --- Case 11: branch-new failing test -> NEW without a pristine run -------------


def test_compare_branch_new_failing_test_is_new(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(
        file="tests/test_branch_new.py", classname="tests.test_branch_new", name="test_x"
    )
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        root_test_files=[],  # the file does NOT exist at the main root
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 1
    assert out["new"] == [node._asdict()]
    assert calls["pristine"] == []  # main cannot vouch for a file it does not have


# --- Case 12 [A2]: missing / empty / zero-case junit -> exit 2 -------------------


@pytest.mark.parametrize("mode", ["missing", "empty-file", "zero-cases"])
def test_compare_missing_or_empty_junit_exit2(tmp_path: Path, monkeypatch, capsys, mode):
    argv, _calls, _r, _w = _compare_env(
        tmp_path, monkeypatch, junit_cases=[], ledger_kw={"failing": ()}
    )
    junit = Path(argv[argv.index("--junitxml") + 1])
    if mode == "missing":
        junit.unlink()
    elif mode == "empty-file":
        junit.write_text("")
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2
    assert out is None
    assert "junit" in err.lower()  # missing / unparseable / ZERO-testcases all name it


# --- Case 13: lint verdict — delta vs live baseline + absolute-clean touched -----


@pytest.mark.parametrize(
    ("base_ruff", "wt_ruff", "touched_ruff", "expect_rc"),
    [
        ((100, 0), (101, 0), (0, 0), 1),  # worktree count above live baseline
        ((100, 0), (100, 0), (0, 0), 0),  # equal + touched clean
        ((100, 0), (99, 0), (0, 0), 0),  # below baseline is fine
        ((100, 2), (100, 3), (0, 0), 1),  # format-file count regression
        ((100, 0), (100, 0), (1, 0), 1),  # touched .py carries a diagnostic
        ((100, 0), (100, 0), (0, 1), 1),  # touched .py would reformat
    ],
)
def test_compare_lint_verdict(
    tmp_path: Path, monkeypatch, capsys, base_ruff, wt_ruff, touched_ruff, expect_rc
):
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=0,
        ledger_kw={"failing": ()},
        touched=("scripts/mychange.py",),
        base_ruff=base_ruff,
        wt_ruff=wt_ruff,
        touched_ruff=touched_ruff,
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == expect_rc
    assert out["lint"]["ok"] is (expect_rc == 0)
    assert out["lint"]["touched_py"] == ["scripts/mychange.py"]


# --- Case 14: determinism — identical inputs -> identical --json output ----------


def test_compare_json_deterministic(tmp_path: Path, monkeypatch, capsys):
    kwargs = dict(
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger_kw={"failing": (NODE_A,)},
        reasons={NODE_A.file: ["invariant"]},
    )
    argv, _c, _r, _w = _compare_env(tmp_path, monkeypatch, **kwargs)
    rc1, out1, _ = _run_json(argv, capsys)
    rc2, out2, _ = _run_json(argv, capsys)
    assert rc1 == rc2 == 0
    # ledger_age_h is wall-clock-derived; everything else must be identical.
    a1 = out1.pop("ledger_age_h")
    a2 = out2.pop("ledger_age_h")
    assert isinstance(a1, float) and isinstance(a2, float)
    assert out1 == out2


# --- Case 15: real-pytest integration (the one non-injected case) ----------------


def _write_python_shim(root: Path, marker: Path) -> Path:
    """Install ``<root>/.venv/bin/python`` as an exec-shim onto the test-runner's
    interpreter that first records its own invocation to *marker* — a REAL
    subprocess target for resolve_root_python in a tmp tree with no real venv."""
    shim = root / ".venv" / "bin" / "python"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(f'#!/bin/sh\necho "$0" >> "{marker}"\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    return shim


def test_real_pytest_single_file_pristine_extracts_failing_node(tmp_path: Path):
    """Runs REAL pytest (bounded) through run_single_file_pristine — executes the
    real bodies of run_pytest, parse_junit, thread_capped and
    run_single_file_pristine, and pins the xunit1 rootdir-relative ``file``
    attribute assumption against the installed pytest (plan A3 / case 15).
    The tree carries a ``.venv/bin/python`` exec-shim because the pristine
    runner now resolves the root's OWN interpreter (round-2 Critical fix)."""
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    _write_python_shim(tree, tmp_path / "case15-shim-invocations.txt")
    # A pyproject [tool.pytest.ini_options] table pins rootdir=tree, mirroring
    # the production repo layout (file attrs come out rootdir-relative).
    (tree / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = ""\n')
    (tree / "tests" / "test_probe.py").write_text(
        "def test_ok():\n    assert True\n\n\ndef test_bad():\n    assert False\n"
    )
    failing = sb.run_single_file_pristine("tests/test_probe.py", cwd=tree, timeout_s=180.0)
    assert failing == {
        sb.Node(file="tests/test_probe.py", classname="tests.test_probe", name="test_bad")
    }


# --- Round-2 Critical regression: pristine/refresh resolve the ROOT's interpreter -


def test_resolve_root_python_resolves_and_fails_loud(tmp_path: Path):
    """resolve_root_python returns <root>/.venv/bin/python; missing venv is a
    fail-loud ToolMissingError, never a silent sys.executable fallback."""
    root = tmp_path / "root"
    with pytest.raises(sb.ToolMissingError, match="venv interpreter"):
        sb.resolve_root_python(root)
    venv_py = root / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    assert sb.resolve_root_python(root) == str(venv_py)


def test_pristine_argv_interpreter_derives_from_root_not_sys_executable(tmp_path, monkeypatch):
    """run_single_file_pristine threads the ROOT's venv interpreter into the
    pytest subprocess argv — NOT sys.executable (#1022 round-2 Critical: from an
    issue worktree, sys.executable is the worktree venv whose editable .pth
    imports the WORKTREE's src/, so the 'pristine-main' oracle would execute
    branch library code and strip a branch-caused src/ regression as
    pre-existing)."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    venv_py = root / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    seen: dict = {}

    def fake_run_pytest(
        files, cwd, timeout_s, junit_path, extra=sb.PYTEST_BASE_FLAGS, *, python_exe
    ) -> int:
        seen["python_exe"] = python_exe
        Path(junit_path).write_text(
            _junit_xml([("tests/test_probe.py", "tests.test_probe", "test_x", "failed")])
        )
        return 1

    monkeypatch.setattr(sb, "run_pytest", fake_run_pytest)
    failing = sb.run_single_file_pristine("tests/test_probe.py", cwd=root, timeout_s=30.0)
    assert seen["python_exe"] == str(venv_py)
    assert seen["python_exe"] != sys.executable
    assert failing == {
        sb.Node(file="tests/test_probe.py", classname="tests.test_probe", name="test_x")
    }


def test_pristine_missing_root_venv_is_pristine_run_error(tmp_path: Path):
    """A root with no venv interpreter cannot vouch anything: PristineRunError
    (compare maps it to indeterminate exit 2), never a sys.executable run."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    with pytest.raises(sb.PristineRunError, match="venv interpreter"):
        sb.run_single_file_pristine("tests/test_probe.py", cwd=root, timeout_s=30.0)


def test_pristine_real_subprocess_executes_root_venv_interpreter(tmp_path: Path):
    """Real-subprocess regression for the round-2 Critical: the pristine pytest
    process IS <root>/.venv/bin/python — proven by the shim's invocation record.
    Pre-fix this fails: run_pytest launched sys.executable, so the shim was
    never invoked and the marker file does not exist."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = ""\n')
    (root / "tests" / "test_probe.py").write_text("def test_bad():\n    assert False\n")
    marker = tmp_path / "shim-invocations.txt"
    shim = _write_python_shim(root, marker)
    failing = sb.run_single_file_pristine("tests/test_probe.py", cwd=root, timeout_s=180.0)
    assert failing == {
        sb.Node(file="tests/test_probe.py", classname="tests.test_probe", name="test_bad")
    }
    assert marker.exists(), "the root-venv shim was never invoked — sys.executable leak"
    assert str(shim) in marker.read_text()


def test_refresh_missing_root_venv_exit2_no_ledger(tmp_path: Path, monkeypatch):
    """Refresh sibling of the round-2 Critical: interpreter resolution failure is
    fail-loud — exit 2, NO ledger write, and pytest is never invoked."""
    argv, root, seen = _refresh_env(tmp_path, monkeypatch, venv=False)
    assert sb.main(argv) == 2
    assert not sb.ledger_path(root).exists()
    assert "python_exe" not in seen  # failed BEFORE any pytest invocation


# --- Case 16 [A2]: --pytest-rc not in {0,1} -> exit 2 even with a clean junit ----


@pytest.mark.parametrize("rc_in", [2, 3, 137])
def test_compare_pytest_rc_guard(tmp_path: Path, monkeypatch, capsys, rc_in):
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=rc_in,
        ledger_kw={"failing": ()},
    )
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2
    assert out is None
    assert "refusing to classify" in err


# --- Case 17 [A2]: test file changed on main since ledger sha -> never blind-strip


def test_compare_changed_test_file_never_blind_stripped(tmp_path: Path, monkeypatch, capsys):
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger_kw={"failing": (NODE_A,)},
        reasons={NODE_A.file: ["invariant"]},  # NOT diff-linked
        changed_tests=(NODE_A.file,),  # but its file changed on main since the sha
        pristine_failing=(),  # and it PASSES at current main HEAD
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 1
    assert calls["pristine"] == [NODE_A.file]
    assert out["stripped"] == []
    assert out["new"] == [NODE_A._asdict()]


# --- Case 18 [A2]: dirty-rooted ledger never blind-strips ------------------------


def test_compare_dirty_ledger_never_blind_strips(tmp_path: Path, monkeypatch, capsys):
    kwargs = dict(
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger_kw={"failing": (NODE_A,), "dirty": True, "dirty_paths": ("scripts/wip.py",)},
        reasons={NODE_A.file: ["invariant"]},
        pristine_failing=(NODE_A,),
    )
    # Without --run-pristine the fresh-but-dirty ledger is unusable -> exit 2.
    argv, calls, _r, _w = _compare_env(tmp_path, monkeypatch, **kwargs)
    rc, _out, err = _run_json(argv, capsys)
    assert rc == 2 and "pristine" in err
    assert calls["pristine"] == []
    # With --run-pristine every node is pristine-routed; JSON reports the dirty flag.
    argv, calls, _r, _w = _compare_env(
        tmp_path, monkeypatch, extra_args=("--run-pristine",), **kwargs
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["ledger_dirty"] is True
    assert out["ledger_dirty_paths"] == ["scripts/wip.py"]
    assert calls["pristine"] == [NODE_A.file]
    assert out["stripped"] == [{**NODE_A._asdict(), "via": "pristine"}]


# --- Case 19 [A2]: dirty pristine oracle never vouches "pre-existing" ------------


@pytest.mark.parametrize("fails_on_main", [True, False])
def test_compare_dirty_pristine_oracle(tmp_path: Path, monkeypatch, capsys, fails_on_main):
    node = sb.Node(file="tests/test_mystery.py", classname="tests.test_mystery", name="test_x")
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/uncommitted.py",),
        pristine_failing=(node,) if fails_on_main else (),
        extra_args=("--run-pristine",),
    )
    rc, out, err = _run_json(argv, capsys)
    if fails_on_main:
        # A "pre-existing" verdict from a dirty oracle is untrustworthy -> exit 2.
        assert rc == 2
        assert out is None
        assert "scripts/uncommitted.py" in err
    else:
        # A PASS on a dirty root still classifies NEW (fail-closed) -> exit 1.
        assert rc == 1
        assert out["new"] == [node._asdict()]
        assert out["live_dirty_paths"] == ["scripts/uncommitted.py"]


# --- Case 20 [A2]: fail-loud census ----------------------------------------------


def test_compare_pristine_timeout_or_crash_exit2(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(file="tests/test_mystery.py", classname="tests.test_mystery", name="test_x")
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        pristine_exc=sb.PristineRunError("pristine run of tests/test_mystery.py timed out"),
        extra_args=("--run-pristine",),
    )
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2 and out is None and "indeterminate" in err


def test_compare_missing_ruff_binary_exit2(tmp_path: Path, monkeypatch, capsys):
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=0,
        ledger_kw={"failing": ()},
    )
    # Undo the ruff fake so the REAL ruff_error_count runs its _ruff_bin() branch,
    # with which() forced to None (MF-5).
    monkeypatch.setattr(sb, "ruff_error_count", _REAL_RUFF_ERROR_COUNT)
    monkeypatch.setattr(sb.shutil, "which", lambda _name: None)
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2 and out is None and "ruff" in err


@pytest.mark.parametrize(
    "ledger_raw",
    [
        None,
        "NOT JSON",
        '{"schema_version": 99}',
        # Top-level-valid but a failing_tests entry lacks file/name — must route to
        # the unusable-ledger indeterminate path, never a ledger_nodes KeyError
        # that Python turns into a misleading exit 1 (round-2 Minor).
        json.dumps({**_ledger_dict(), "failing_tests": [{"bogus": 1}]}),
    ],
)
def test_compare_unusable_ledger_with_failures_no_pristine_exit2(
    tmp_path: Path, monkeypatch, capsys, ledger_raw
):
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(NODE_A.file, NODE_A.classname, NODE_A.name, "failed")],
        ledger=False,
        ledger_raw=ledger_raw,
    )
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2 and out is None
    assert "pristine" in err  # unresolved bucket, no --run-pristine -> indeterminate


def test_try_load_ledger_rejects_malformed_failing_tests_entry(tmp_path: Path, capsys):
    """Entry-shape validation (round-2 Minor): a schema-keyed ledger whose
    failing_tests entries are not {file: str, name: str, ...} dicts is unusable
    (None + loud stderr), so downstream ledger_nodes can never KeyError."""
    root = tmp_path / "root"
    for bad_entries in ([{"bogus": 1}], ["not-a-dict"], [{"file": 3, "name": "x"}]):
        led = _ledger_dict()
        led["failing_tests"] = bad_entries
        path = sb.ledger_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(led))
        assert sb.try_load_ledger(root) is None
        assert "entry shape" in capsys.readouterr().err
    # The well-formed shape still loads.
    led = _ledger_dict(failing=(NODE_A,))
    sb.ledger_path(root).write_text(json.dumps(led))
    assert sb.try_load_ledger(root) == led


# --- Case 21 [A2]: scan-test strips ALWAYS carry a masking WARN -------------------


def test_compare_scan_test_blind_strip_warns(tmp_path: Path, monkeypatch, capsys):
    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(NODE_SCAN.file, NODE_SCAN.classname, NODE_SCAN.name, "failed")],
        ledger_kw={"failing": (NODE_SCAN,)},
        touched=("scripts/issue999_x.py",),
        # The fake reasons deliberately carry NO glob-scan entry, so the node is
        # NOT diff-linked and takes the BLIND strip path — pinning do_strip's
        # scan WARN in isolation (MF-6: the v2 gap was a silent blind strip).
        reasons={NODE_SCAN.file: ["invariant"]},
        glob_scan={NODE_SCAN.file: ("scripts/issue*_*.py",)},
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["stripped"] == [{**NODE_SCAN._asdict(), "via": "ledger"}]
    assert out["warns"], "blind strip of a scan-covered test MUST warn"
    assert any("scripts/issue999_x.py" in w for w in out["warns"])


def test_compare_scan_test_pristine_strip_warns(tmp_path: Path, monkeypatch, capsys):
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(NODE_SCAN.file, NODE_SCAN.classname, NODE_SCAN.name, "failed")],
        ledger_kw={"failing": (NODE_SCAN,)},
        touched=("scripts/issue999_x.py",),
        reasons={NODE_SCAN.file: ["glob-scan:scripts/issue999_x.py"]},  # diff-linked
        glob_scan={NODE_SCAN.file: ("scripts/issue*_*.py",)},
        pristine_failing=(NODE_SCAN,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine"] == [NODE_SCAN.file]
    assert out["stripped"] == [{**NODE_SCAN._asdict(), "via": "pristine"}]
    assert out["warns"], "pristine strip of a scan-covered test MUST warn too"
    assert any("scripts/issue999_x.py" in w for w in out["warns"])
    assert any("diff-linked" in w for w in out["warns"])


# --- Case 22 [A2]: pristine-file budget -> systemic main breakage exit 2 ----------


def test_compare_pristine_budget_exceeded_exit2(tmp_path: Path, monkeypatch, capsys):
    cases = [
        (f"tests/test_red_{i}.py", f"tests.test_red_{i}", "test_x", "failed") for i in range(6)
    ]
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=cases,
        ledger_kw={"failing": ()},
        extra_args=("--run-pristine", "--max-pristine-files", "5"),
    )
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2 and out is None
    assert "systemic main breakage" in err
    assert calls["pristine"] == []  # budget refusal happens BEFORE any pristine run


# --- status subcommand ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fresh", "stale", "missing"])
def test_status_exit_codes(tmp_path: Path, monkeypatch, capsys, mode):
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    if mode != "missing":
        _write_ledger(root, age_h=48.0 if mode == "stale" else 0.0)

    def fake_git_sha_known(root_: Path, sha: str) -> bool:
        return True

    def fake_code_commits_since(root_: Path, sha: str) -> int:
        return 0

    monkeypatch.setattr(sb, "git_sha_known", fake_git_sha_known)
    monkeypatch.setattr(sb, "code_commits_since", fake_code_commits_since)
    rc = sb.main(["status", "--repo-root", str(root)])
    captured = capsys.readouterr()
    if mode == "fresh":
        assert rc == 0 and "fresh" in captured.out
    elif mode == "stale":
        assert rc == 3 and "stale" in captured.out
    else:
        assert rc == 2


# --- Real-body coverage for the seam-stubbed helpers (code-style.md #906) ---------


def test_load_selector_module_real_body():
    """Loads the LIVE selector by path — the real body of the compare/refresh seam."""
    root = Path(sb.__file__).resolve().parents[1]
    mod = sb.load_selector_module(root)
    assert isinstance(mod.WORKFLOW_INVARIANT, tuple) and mod.WORKFLOW_INVARIANT
    assert isinstance(mod.GLOB_SCAN_TESTS, dict) and mod.GLOB_SCAN_TESTS
    assert callable(mod.select_tests_with_reasons) and callable(mod._matches_any)


def test_ruff_helpers_real_body(tmp_path: Path):
    """Real ruff run: a diagnostic-carrying, unformatted file counts on both probes."""
    (tmp_path / "bad.py").write_text("import os\nx=1\n")  # F401 + would-reformat
    assert sb.ruff_error_count(tmp_path) >= 1
    assert sb.ruff_format_count(tmp_path) >= 1


def _git(cwd: Path, *args: str) -> None:
    """Run git hermetically (no global/system config leaks), like the selector tests."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.run(
        ["git", *args], cwd=str(cwd), env=env, check=True, capture_output=True, text=True
    )


def test_git_helpers_real_body(tmp_path: Path):
    """Real-git coverage for git_head / git_sha_known / code_commits_since /
    changed_test_files_since / dirty_code_paths (all monkeypatched in the unit cases)."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (repo / "scripts" / "tool.py").write_text("X = 1\n")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    sha1 = sb.git_head(repo)
    assert len(sha1) == 40
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert 1 == 1\n")
    _git(repo, "add", "tests/test_a.py")
    _git(repo, "commit", "-m", "touch tests")
    assert sb.git_head(repo) != sha1
    assert sb.git_sha_known(repo, sha1) is True
    assert sb.git_sha_known(repo, "0" * 40) is False
    assert sb.code_commits_since(repo, sha1) == 1
    assert sb.changed_test_files_since(repo, sha1) == {"tests/test_a.py"}
    # Dirt probe: a modified .py counts; non-code churn (json) does not (MF-4a scope).
    (repo / "scripts" / "tool.py").write_text("X = 2\n")
    (repo / "state.json").write_text("{}")
    dirty = sb.dirty_code_paths(repo)
    assert "scripts/tool.py" in dirty
    assert all(not p.endswith(".json") for p in dirty)


def test_main_repo_root_and_resolve_work_root_real_body(monkeypatch):
    """The no-arg resolvers run real git against the invoking checkout."""
    here = Path(sb.__file__).resolve().parents[1]
    monkeypatch.chdir(here)
    wt_root = sb.resolve_work_root(None)
    assert wt_root == here
    main_root = sb.main_repo_root()
    assert (main_root / ".git").is_dir()  # the MAIN root owns the real .git dir
    # The override path bypasses git entirely.
    assert sb.resolve_work_root(str(here)) == here
