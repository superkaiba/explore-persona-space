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

The "#1077" section pins the dirty-oracle scratch-worktree fallback + the
JSON-on-every-exit contract (fake-based N1-N4b/N9-N12 in that section;
real-git N5-N8 + the _work_root_sparse_cones body test in the real-body
section at the bottom).

The "#1251" section pins the scratch PYTHONPATH src-shadow (in-package dirty
src/ no longer refuses): fake-based compare cases + the pure residual-split
unit; the real-subprocess mechanism proof (a fresh ``--without-pip`` venv with
a production-style single-line ``.pth``), the real ``run_pytest`` env branch,
and the live-venv ``assert_scratch_src_shadow`` durability pin (+ its
missing-scratch-src negative) live in the real-body section at the bottom.
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
import types
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
    contamination_paths=(),
    wt_cones=("tests",),
    scratch_exc: Exception | None = None,
    shadow_probe_exc: Exception | None = None,
) -> dict[str, list]:
    """Monkeypatch signature-conformant fakes onto the module; return the call recorder.

    #1077 scratch-fallback knobs: ``contamination_paths`` fakes
    ``scratch_contamination_probe`` (a list, or a zero-arg callable for the
    per-call mid-loop-transition case); ``wt_cones`` fakes
    ``_work_root_sparse_cones`` (None = non-sparse work root); ``scratch_exc``
    makes the fake ``create_scratch_worktree`` raise instead of returning a
    fake ``_ScratchTree``. #1251 knob: ``shadow_probe_exc`` makes the fake
    ``assert_scratch_src_shadow`` raise (the probe-failure fail-closed case).
    """
    calls: dict[str, list] = {
        "pristine": [],
        "pristine_detail": [],  # (test_file, cwd, venv_root) per pristine call
        "pristine_timeout": [],  # timeout_s per pristine call (#1129 derived-default pin)
        "pristine_pythonpath": [],  # pythonpath kwarg per pristine call (#1251 shadow pin)
        "scratch_created": [],
        "scratch_removed": [],
        "shadow_probe": [],  # (root, scratch_path) per assert_scratch_src_shadow call (#1251)
    }
    _install_scratch_fakes(
        monkeypatch,
        calls,
        root=root,
        contamination_paths=contamination_paths,
        wt_cones=wt_cones,
        scratch_exc=scratch_exc,
        shadow_probe_exc=shadow_probe_exc,
    )

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

    def fake_run_single_file_pristine(
        test_file: str,
        cwd: Path,
        timeout_s: float,
        *,
        venv_root: Path | None = None,
        pythonpath: str | None = None,
    ) -> set:
        calls["pristine"].append(test_file)
        calls["pristine_detail"].append((test_file, cwd, venv_root))
        calls["pristine_timeout"].append(timeout_s)
        calls["pristine_pythonpath"].append(pythonpath)
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


def _install_scratch_fakes(
    monkeypatch,
    calls: dict[str, list],
    *,
    root: Path,
    contamination_paths,
    wt_cones,
    scratch_exc,
    shadow_probe_exc=None,
) -> None:
    """Install the #1077 scratch-fallback (+ #1251 shadow-probe) fakes."""
    fake_scratch = sb._ScratchTree(
        parent=root / "scratch-parent", path=root / "scratch-fake", sha="f" * 40
    )

    def fake_scratch_contamination_probe(root_: Path) -> list[str]:
        if callable(contamination_paths):
            return list(contamination_paths())
        return list(contamination_paths)

    def fake_work_root_sparse_cones(wt_: Path) -> list[str] | None:
        return list(wt_cones) if wt_cones is not None else None

    def fake_create_scratch_worktree(root_: Path, cones: list[str], timeout_s: float):
        calls["scratch_created"].append((root_, tuple(cones), timeout_s))
        if scratch_exc is not None:
            raise scratch_exc
        fake_scratch.path.mkdir(parents=True, exist_ok=True)
        return fake_scratch

    def fake_remove_scratch_worktree(root_: Path, scratch) -> None:
        calls["scratch_removed"].append(scratch)

    def fake_assert_scratch_src_shadow(root_: Path, scratch_path: Path, timeout_s: float) -> None:
        calls["shadow_probe"].append((root_, scratch_path))
        if shadow_probe_exc is not None:
            raise shadow_probe_exc

    monkeypatch.setattr(sb, "scratch_contamination_probe", fake_scratch_contamination_probe)
    monkeypatch.setattr(sb, "_work_root_sparse_cones", fake_work_root_sparse_cones)
    monkeypatch.setattr(sb, "create_scratch_worktree", fake_create_scratch_worktree)
    monkeypatch.setattr(sb, "remove_scratch_worktree", fake_remove_scratch_worktree)
    monkeypatch.setattr(sb, "assert_scratch_src_shadow", fake_assert_scratch_src_shadow)


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
    contamination_paths=(),
    wt_cones=("tests",),
    scratch_exc: Exception | None = None,
    shadow_probe_exc: Exception | None = None,
    sel_attrs: dict | None = None,
    extra_args=(),
):
    """Set up a compare fixture tree + fakes; return (argv, calls, root, wt).

    ``sel_attrs`` setattrs extra attributes onto the ``_FakeSel`` post-construction
    (the line-level ``fake_sel.WORKFLOW_INVARIANT = ...`` pattern) — e.g. the #1046
    timeout constants for the #1129 derived-pristine-timeout cases.
    """
    root, wt, junit = _materialize_compare_tree(
        tmp_path,
        junit_cases=junit_cases,
        touched=touched,
        root_test_files=root_test_files,
        ledger=ledger,
        ledger_kw=ledger_kw,
        ledger_raw=ledger_raw,
    )
    fake_sel = _FakeSel(touched, reasons, glob_scan)
    for _k, _v in (sel_attrs or {}).items():
        setattr(fake_sel, _k, _v)
    calls = _install_compare_fakes(
        monkeypatch,
        root=root,
        fake_sel=fake_sel,
        changed_tests=changed_tests,
        live_dirty=live_dirty,
        pristine_failing=pristine_failing,
        pristine_exc=pristine_exc,
        base_ruff=base_ruff,
        wt_ruff=wt_ruff,
        touched_ruff=touched_ruff,
        sha_known=sha_known,
        code_commits=code_commits,
        contamination_paths=contamination_paths,
        wt_cones=wt_cones,
        scratch_exc=scratch_exc,
        shadow_probe_exc=shadow_probe_exc,
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
    assert out["indeterminate"] is False
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
    rc, out, err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert "tests/test_mystery.py" in out["reason"] and "pytest" in out["reason"]
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


# --- #1289: compare --base default resolves via the worktree selector ------------


def test_compare_base_default_resolves_via_selector(tmp_path: Path, monkeypatch, capsys):
    """With no --base, compare resolves via the WORKTREE selector's
    resolve_base(DEFAULT_BASE, wt, fetch=False) — same-mapping-logic (#1022)
    with fetched-origin/main semantics and NO second fetch (#1289) — and
    threads the RESOLVED ref into compute_touched."""
    seen: dict[str, object] = {}

    def _resolve(base: str, work_root: Path, *, fetch: bool = True) -> str:
        seen["resolve"] = (base, fetch)
        return "resolved-ref"

    def _ct(base: str, work_root: Path, _runner=None) -> list[str]:
        seen["ct_base"] = base
        return []

    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=0,
        ledger_kw={"failing": ()},
        sel_attrs={"resolve_base": _resolve, "DEFAULT_BASE": "origin/main", "compute_touched": _ct},
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert seen["resolve"] == ("origin/main", False)  # DEFAULT_BASE threaded, no second fetch
    assert seen["ct_base"] == "resolved-ref"  # compute_touched got resolve_base's RETURN


def test_compare_base_default_pre1289_selector_falls_back_to_main(
    tmp_path: Path, monkeypatch, capsys
):
    """A pre-#1289 worktree selector (no resolve_base — the default _FakeSel)
    keeps that era's self-consistent behavior: compare diffs against local
    'main' (the getattr transition guard)."""
    seen: dict[str, object] = {}

    def _ct(base: str, work_root: Path, _runner=None) -> list[str]:
        seen["ct_base"] = base
        return []

    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=0,
        ledger_kw={"failing": ()},
        sel_attrs={"compute_touched": _ct},  # _FakeSel has NO resolve_base
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert seen["ct_base"] == "main"


def test_compare_base_explicit_ref_used_verbatim(tmp_path: Path, monkeypatch, capsys):
    """An explicit --base REF bypasses resolution entirely (used verbatim)."""
    seen: dict[str, object] = {}

    def _resolve(base: str, work_root: Path, *, fetch: bool = True) -> str:
        raise AssertionError("resolve_base must not be called for an explicit --base")

    def _ct(base: str, work_root: Path, _runner=None) -> list[str]:
        seen["ct_base"] = base
        return []

    argv, _calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[("tests/test_fine.py", "tests.test_fine", "test_ok", "passed")],
        pytest_rc=0,
        ledger_kw={"failing": ()},
        sel_attrs={"resolve_base": _resolve, "compute_touched": _ct},
        extra_args=("--base", "feature-x"),
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert seen["ct_base"] == "feature-x"


# --- #1129: derived per-file pristine timeout ------------------------------------


def test_derive_pristine_timeout_from_selector_constants():
    """Pure unit: BASE + PER_FILE + 2x surcharge for slow files; floor 600 for the rest."""
    sel = types.SimpleNamespace(
        TIMEOUT_BASE_S=120,
        TIMEOUT_PER_FILE_S=30,
        SLOW_TESTS={"tests/test_workflow_lint.py": 900},
    )
    assert sb.derive_pristine_timeout_s(sel, "tests/test_workflow_lint.py") == 1950.0
    assert sb.derive_pristine_timeout_s(sel, "tests/test_other.py") == 600.0


def test_derive_pristine_timeout_live_selector_covers_incident_1098():
    """Live-tree drift pin: the derived bound must keep covering the #1098 incident.

    Dual grounding: 1200 s is the demonstrated-sufficient manual rerun bound
    (`--pristine-timeout-s 1200` succeeded — the BINDING incident floor), and the
    measured pristine runtime of tests/test_workflow_lint.py is bracketed at
    ~640-780 s (the #1129 filing says ~13 min ~= 780 s; #1098's events record
    "~640s+"). The `>= 2 * 780 = 1560` threshold gives >=2x headroom over the
    bracket top; a future legitimate SLOW_TESTS re-measurement that lands the
    derived value in [1200, 1560) should be reconciled against the 1200 s
    incident floor rather than misread as an incident-coverage regression.
    """
    real_sel = sb.load_selector_module(Path(__file__).resolve().parents[1])
    assert sb.derive_pristine_timeout_s(real_sel, "tests/test_workflow_lint.py") >= 2 * 780


def test_derive_pristine_timeout_selector_skew_falls_back_to_floor():
    """A selector copy lacking the #1046 constants (version skew) degrades to 600 s."""
    skewed = _FakeSel([], {}, {})  # _FakeSel deliberately lacks the #1046 constants
    assert sb.derive_pristine_timeout_s(skewed, "tests/test_workflow_lint.py") == 600.0


def test_compare_pristine_timeout_derived_and_override_wins(tmp_path: Path, monkeypatch, capsys):
    """Integration through sb.main: per-file derivation at default flags; explicit flag wins."""
    node_wl = sb.Node(
        file="tests/test_workflow_lint.py", classname="tests.test_workflow_lint", name="test_x"
    )
    # tests/test_zz_plain.py sorts AFTER test_workflow_lint.py -> recorded order is
    # [surcharge file, plain file], pinning the PER-FILE (in-loop) derivation.
    node_plain = sb.Node(
        file="tests/test_zz_plain.py", classname="tests.test_zz_plain", name="test_y"
    )
    sel_1046 = {
        "TIMEOUT_BASE_S": 120,
        "TIMEOUT_PER_FILE_S": 30,
        "SLOW_TESTS": {"tests/test_workflow_lint.py": 900},
    }

    # (a) default flags + default _FakeSel (no #1046 constants) -> skew fallback 600.0
    #     through the real _resolve_pristine_bucket body.
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node_plain.file, node_plain.classname, node_plain.name, "failed")],
        ledger_kw={"failing": ()},
        pristine_failing=(node_plain,),
        extra_args=("--run-pristine",),
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine_timeout"] == [600.0]

    # (b) default flags + #1046 constants on the selector -> derived 1950.0 for the
    #     surcharge file.
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node_wl.file, node_wl.classname, node_wl.name, "failed")],
        ledger_kw={"failing": ()},
        pristine_failing=(node_wl,),
        sel_attrs=sel_1046,
        extra_args=("--run-pristine",),
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine_timeout"] == [1950.0]

    # (c) explicit --pristine-timeout-s wins verbatim, even with the constants present.
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node_wl.file, node_wl.classname, node_wl.name, "failed")],
        ledger_kw={"failing": ()},
        pristine_failing=(node_wl,),
        sel_attrs=sel_1046,
        extra_args=("--run-pristine", "--pristine-timeout-s", "1200"),
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine_timeout"] == [1200.0]

    # (d) mixed two-file bucket -> per-file (in-loop) derivation: [1950.0, 600.0].
    #     An implementation hoisting the derivation above the loop (deriving once
    #     from the first file) cannot produce two distinct values.
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[
            (node_wl.file, node_wl.classname, node_wl.name, "failed"),
            (node_plain.file, node_plain.classname, node_plain.name, "failed"),
        ],
        ledger_kw={"failing": ()},
        pristine_failing=(node_wl, node_plain),
        sel_attrs=sel_1046,
        extra_args=("--run-pristine",),
    )
    rc, _out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert calls["pristine"] == [node_wl.file, node_plain.file]
    assert calls["pristine_timeout"] == [1950.0, 600.0]


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
    assert out["indeterminate"] is True
    assert "junit" in out["reason"].lower()
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
    assert out1["indeterminate"] is False
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
        files,
        cwd,
        timeout_s,
        junit_path,
        extra=sb.PYTEST_BASE_FLAGS,
        *,
        python_exe,
        pythonpath=None,
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
    assert out["indeterminate"] is True
    assert "refusing to classify" in out["reason"]
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


# --- Case 19 [A2]: RESIDUAL contaminating dirty oracle never vouches "pre-existing"
# (#1077: decontaminable dirt auto-falls back to a scratch oracle — see N1; since
# #1251 in-package src/ dirt is shadow-neutralized too — see the #1251 section —
# so this exit-2 pin uses a RESIDUAL pyproject.toml leg, which keeps the MF-4c raise.)


@pytest.mark.parametrize("fails_on_main", [True, False])
def test_compare_dirty_pristine_oracle(tmp_path: Path, monkeypatch, capsys, fails_on_main):
    node = sb.Node(file="tests/test_mystery.py", classname="tests.test_mystery", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("pyproject.toml",),  # pyproject.toml IS in DIRTY_CODE_PATHSPEC
        contamination_paths=("pyproject.toml",),
        pristine_failing=(node,) if fails_on_main else (),
        extra_args=("--run-pristine",),
    )
    rc, out, err = _run_json(argv, capsys)
    assert calls["scratch_created"] == []  # residual contaminating dirt never falls back
    if fails_on_main:
        # A "pre-existing" verdict from a contaminated oracle is untrustworthy -> exit 2.
        assert rc == 2
        assert out["indeterminate"] is True
        assert "pyproject.toml" in out["reason"]
        assert "pyproject.toml" in err
    else:
        # A PASS on a dirty root still classifies NEW (fail-closed) -> exit 1.
        assert rc == 1
        assert out["new"] == [node._asdict()]
        assert out["live_dirty_paths"] == ["pyproject.toml"]
        assert out["pristine_oracle"] == "root"


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
    assert rc == 2 and out["indeterminate"] is True and "indeterminate" in err


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
    assert rc == 2 and out["indeterminate"] is True and "ruff" in err


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
    assert rc == 2 and out["indeterminate"] is True
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
    assert rc == 2 and out["indeterminate"] is True
    assert "systemic main breakage" in err
    assert "systemic main breakage" in out["reason"]
    assert calls["pristine"] == []  # budget refusal happens BEFORE any pristine run


# --- #1077: dirty-oracle scratch-worktree fallback + JSON-on-every-exit -----------


@pytest.mark.parametrize("fails_on_main", [True, False])
def test_compare_dirty_oracle_scratch_fallback_strip_or_new(
    tmp_path: Path, monkeypatch, capsys, fails_on_main
):
    """N1: decontaminable dirt + sparse work root + non-scan node -> scratch oracle.

    The node is deliberately DIFF-LINKED so the strip case also pins the MF-6
    diff-linked masking WARN under ``via="pristine-scratch"`` (the
    ``via.startswith`` change).
    """
    node = sb.Node(file="tests/test_linked.py", classname="tests.test_linked", name="test_x")
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        touched=("scripts/known_red.py",),
        reasons={node.file: ["stem-map:scripts/known_red.py"]},  # diff-linked (MF-6)
        live_dirty=("scripts/issue_642/i642_figures_v4.py",),  # the incident dirt class
        pristine_failing=(node,) if fails_on_main else (),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert len(calls["scratch_created"]) == 1
    # The oracle ran IN the scratch tree with the MAIN root's venv interpreter.
    assert calls["pristine_detail"] == [(node.file, root / "scratch-fake", root)]
    assert calls["scratch_removed"], "finally teardown must run"
    assert out["pristine_oracle"] == "scratch-worktree"
    assert out["scratch_sha"] == "f" * 40
    assert any("SCRATCH-ORACLE WARN" in w for w in out["warns"])
    if fails_on_main:
        assert rc == 0
        assert out["indeterminate"] is False
        assert out["stripped"] == [{**node._asdict(), "via": "pristine-scratch"}]
        # Diff-linked scratch strips keep the MF-6 masking WARN (via.startswith).
        assert any("diff-linked" in w for w in out["warns"])
    else:
        assert rc == 1
        assert out["new"] == [node._asdict()]


def test_compare_scratch_created_once_for_multi_file_bucket(tmp_path: Path, monkeypatch, capsys):
    n1 = sb.Node(file="tests/test_m1.py", classname="tests.test_m1", name="test_x")
    n2 = sb.Node(file="tests/test_m2.py", classname="tests.test_m2", name="test_x")
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(n.file, n.classname, n.name, "failed") for n in (n1, n2)],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/wip.py",),
        pristine_failing=(n1, n2),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert len(calls["scratch_created"]) == 1  # ONE scratch, reused across the bucket
    assert [d[1] for d in calls["pristine_detail"]] == [root / "scratch-fake"] * 2
    assert {s["via"] for s in out["stripped"]} == {"pristine-scratch"}
    assert len(calls["scratch_removed"]) == 1


def test_compare_scratch_removed_on_pristine_crash(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/wip.py",),
        pristine_exc=sb.PristineRunError("pristine run of tests/test_m.py timed out (600.0s)"),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert calls["scratch_removed"], "finally teardown must run on a mid-loop crash"
    # Scratch provenance is never dropped on a mid-loop failure (exit-2 warns).
    assert any("SCRATCH-ORACLE WARN" in w for w in out["warns"])


def test_compare_scratch_creation_failure_indeterminate(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/wip.py",),
        scratch_exc=subprocess.TimeoutExpired(cmd=["git"], timeout=120.0),
        pristine_failing=(node,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert "scratch-worktree fallback failed" in out["reason"]
    assert out["live_dirty_paths"] == ["scripts/wip.py"]
    assert calls["pristine"] == []  # creation failed BEFORE any oracle run


def test_compare_no_scratch_fallback_flag_restores_old_raise(tmp_path: Path, monkeypatch, capsys):
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/wip.py",),
        pristine_failing=(node,),
        extra_args=("--run-pristine", "--no-scratch-fallback"),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert calls["scratch_created"] == []  # the kill switch restores the pre-#1077 raise


def test_compare_mixed_dirt_residual_uv_lock_blocks_scratch(tmp_path: Path, monkeypatch, capsys):
    """N9': dirty scripts/*.py (the visible trigger) + dirty uv.lock (a RESIDUAL
    leg the #1251 shadow cannot neutralize — installed deps derive from it) MUST
    stay exit-2 — the mixed-residual case is never a scratch strip. (The old N9
    premise — dirty src/*.json blocks — is superseded by the #1251 shadow test.)"""
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/x.py",),
        contamination_paths=("uv.lock",),
        pristine_failing=(node,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert out["contaminating_paths"] == ["uv.lock"]
    assert out["residual_contaminating_paths"] == ["uv.lock"]
    assert calls["scratch_created"] == []


def test_compare_mid_loop_dirt_transition_fail_closed(tmp_path: Path, monkeypatch, capsys):
    """N10: RESIDUAL contamination appearing MID-LOOP reverts later files to the
    root oracle; a fail-on-main node there goes indeterminate (fail-closed),
    while file1's scratch provenance rides the exit-2 warns. (#1251 repointed
    the transition dirt from src/*.json — now shadowable — to uv.lock.)"""
    n1 = sb.Node(file="tests/test_a1.py", classname="tests.test_a1", name="test_x")
    n2 = sb.Node(file="tests/test_a2.py", classname="tests.test_a2", name="test_x")
    contamination_seq = iter([[], ["uv.lock"]])
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(n.file, n.classname, n.name, "failed") for n in (n1, n2)],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/x.py",),
        contamination_paths=lambda: next(contamination_seq),
        pristine_failing=(n1, n2),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    # file1 was scratch-resolved; file2 reverted to the ROOT oracle (never scratch-stripped).
    assert len(calls["scratch_created"]) == 1
    assert calls["pristine_detail"][0] == (n1.file, root / "scratch-fake", root)
    assert calls["pristine_detail"][1] == (n2.file, root, None)
    assert out["contaminating_paths"] == ["uv.lock"]
    assert any("SCRATCH-ORACLE WARN" in w for w in out["warns"])
    assert calls["scratch_removed"], "finally teardown must run"


def test_compare_scan_set_node_never_scratch_stripped(tmp_path: Path, monkeypatch, capsys):
    """N11 (R-F): a GLOB_SCAN_TESTS node is never scratch-resolved — live-tree
    scanners read the MAIN root via repo_root() from any cwd, so a scratch cannot
    decontaminate them; the node keeps the MF-4c indeterminate."""
    node = sb.Node(
        file="tests/test_scan_thing.py", classname="tests.test_scan_thing", name="test_x"
    )
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        glob_scan={node.file: ("scripts/issue*_*.py",)},
        live_dirty=("scripts/wip.py",),
        pristine_failing=(node,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert calls["scratch_created"] == []
    assert "scan_set=True" in out["reason"]
    # The oracle ran at the ROOT (per-file granularity: only THIS node is barred).
    assert calls["pristine_detail"] == [(node.file, root, None)]


def test_compare_non_sparse_work_root_ineligible(tmp_path: Path, monkeypatch, capsys):
    """N12 (R-G): a non-sparse work root cannot be superset-matched -> no fallback."""
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        wt_cones=None,  # non-sparse work root
        live_dirty=("scripts/wip.py",),
        pristine_failing=(node,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert calls["scratch_created"] == []
    assert "sparse_wt=False" in out["reason"]


# --- #1251: scratch PYTHONPATH src-shadow (dirty src/ no longer refuses) -----------


@pytest.mark.parametrize("fails_on_main", [True, False])
def test_compare_src_dirt_shadowed_scratch_strip_or_new(
    tmp_path: Path, monkeypatch, capsys, fails_on_main
):
    """#1251 (the #1190 regression pin): unrelated concurrent dirty IN-PACKAGE
    src/ at the shared root no longer refuses (rc=2) — the scratch oracle arms
    with a probe-verified PYTHONPATH=<scratch>/src shadow and resolves the node
    strip-or-NEW (rc 0/1). Covers .py AND package-data .json dirt (the #1077
    mixed case the shadow now neutralizes via the package __path__)."""
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("src/explore_persona_space/wip.py",),  # the #1190 dirt class
        contamination_paths=(
            "src/explore_persona_space/wip.py",
            "src/explore_persona_space/artifacts/query_banks/x.json",
        ),
        pristine_failing=(node,) if fails_on_main else (),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert len(calls["scratch_created"]) == 1
    assert calls["shadow_probe"] == [(root, root / "scratch-fake")]  # probe ran ONCE
    assert calls["pristine_pythonpath"] == [str(root / "scratch-fake" / "src")]
    assert out["pristine_oracle"] == "scratch-worktree"
    assert out["scratch_src_shadow"] is True
    # The WARN names the neutralized src paths.
    assert any("src/explore_persona_space/wip.py" in w and "neutralized" in w for w in out["warns"])
    if fails_on_main:
        assert rc == 0
        assert out["indeterminate"] is False
        assert out["stripped"] == [{**node._asdict(), "via": "pristine-scratch"}]
    else:
        assert rc == 1
        assert out["new"] == [node._asdict()]


def test_compare_no_src_shadow_flag_restores_1077_eligibility(tmp_path: Path, monkeypatch, capsys):
    """--no-src-shadow restores the #1077 rule: ANY dirty src/ path keeps the
    fail-closed exit 2 (no scratch); scripts-only dirt stays scratch-eligible
    with the shadow DISARMED (scratch_src_shadow false, pre-#1251 WARN text)."""
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("src/explore_persona_space/wip.py",),
        contamination_paths=("src/explore_persona_space/wip.py",),
        pristine_failing=(node,),
        extra_args=("--run-pristine", "--no-src-shadow"),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert calls["scratch_created"] == []
    assert "src/explore_persona_space/wip.py" in out["reason"]
    # Scripts-only dirt under --no-src-shadow: the #1077 fallback still fires,
    # shadow disarmed — no probe, no PYTHONPATH, scratch_src_shadow false.
    argv, calls, _root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/wip.py",),
        pristine_failing=(node,),
        extra_args=("--run-pristine", "--no-src-shadow"),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["stripped"] == [{**node._asdict(), "via": "pristine-scratch"}]
    assert out["scratch_src_shadow"] is False
    assert calls["shadow_probe"] == []
    assert calls["pristine_pythonpath"] == [None]
    assert any(
        "contamination probe src//pyproject.toml/uv.lock was clean" in w for w in out["warns"]
    )


def test_compare_shadow_probe_failure_indeterminate(tmp_path: Path, monkeypatch, capsys):
    """A failing src-shadow probe is fail-closed: exit 2, the scratch is torn
    down, and NO oracle run rests on the unverified shadow."""
    node = sb.Node(file="tests/test_m.py", classname="tests.test_m", name="test_x")
    argv, calls, _r, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(node.file, node.classname, node.name, "failed")],
        ledger_kw={"failing": ()},
        live_dirty=("src/explore_persona_space/wip.py",),
        contamination_paths=("src/explore_persona_space/wip.py",),
        shadow_probe_exc=sb.PristineRunError(
            "src-shadow probe rc=3: PYTHONPATH did NOT win over the root venv's editable install"
        ),
        pristine_failing=(node,),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 2
    assert out["indeterminate"] is True
    assert "src-shadow probe" in out["reason"]
    assert calls["scratch_removed"], "finally teardown must run on a probe failure"
    assert calls["pristine"] == []  # no verdict rested on an unverified shadow


def test_compare_mid_loop_src_dirt_does_not_revert(tmp_path: Path, monkeypatch, capsys):
    """The N10 contrast pin: IN-PACKAGE src dirt appearing MID-LOOP does NOT
    revert later files to the root oracle — the shadow is armed uniformly on
    every scratch pristine call, so both files stay scratch-resolved (rc 0)."""
    n1 = sb.Node(file="tests/test_a1.py", classname="tests.test_a1", name="test_x")
    n2 = sb.Node(file="tests/test_a2.py", classname="tests.test_a2", name="test_x")
    contamination_seq = iter([[], ["src/explore_persona_space/late.py"]])
    argv, calls, root, _w = _compare_env(
        tmp_path,
        monkeypatch,
        junit_cases=[(n.file, n.classname, n.name, "failed") for n in (n1, n2)],
        ledger_kw={"failing": ()},
        live_dirty=("scripts/x.py",),  # non-empty at file 1 or use_scratch never arms
        contamination_paths=lambda: next(contamination_seq),
        pristine_failing=(n1, n2),
        extra_args=("--run-pristine",),
    )
    rc, out, _err = _run_json(argv, capsys)
    assert rc == 0
    assert out["indeterminate"] is False
    assert len(calls["scratch_created"]) == 1
    # BOTH files scratch-resolved under the shadow (hermetic against the transition).
    assert [d[1] for d in calls["pristine_detail"]] == [root / "scratch-fake"] * 2
    assert calls["pristine_pythonpath"] == [str(root / "scratch-fake" / "src")] * 2
    assert {s["via"] for s in out["stripped"]} == {"pristine-scratch"}
    assert out["scratch_src_shadow"] is True


def test_resolve_pristine_threads_pythonpath(tmp_path: Path, monkeypatch):
    """run_single_file_pristine threads the pythonpath kwarg verbatim into
    run_pytest (default None) — the #1251 sibling of the interpreter-threading
    pin above."""
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    venv_py = root / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    seen: list = []

    def fake_run_pytest(
        files,
        cwd,
        timeout_s,
        junit_path,
        extra=sb.PYTEST_BASE_FLAGS,
        *,
        python_exe,
        pythonpath=None,
    ) -> int:
        seen.append(pythonpath)
        Path(junit_path).write_text(
            _junit_xml([("tests/test_probe.py", "tests.test_probe", "test_x", "failed")])
        )
        return 1

    monkeypatch.setattr(sb, "run_pytest", fake_run_pytest)
    sb.run_single_file_pristine("tests/test_probe.py", cwd=root, timeout_s=30.0, pythonpath="X")
    sb.run_single_file_pristine("tests/test_probe.py", cwd=root, timeout_s=30.0)
    assert seen == ["X", None]


def test_residual_scratch_contamination_split():
    """Pure unit: only in-package src/explore_persona_space/ paths are shadowable;
    pyproject.toml / uv.lock / out-of-package src/ / oddballs stay residual."""
    assert sb.residual_scratch_contamination(
        [
            "src/explore_persona_space/a.py",
            "src/explore_persona_space/d/x.json",
            "pyproject.toml",
            "uv.lock",
            "srcfile",
            "src/rogue.py",
        ]
    ) == ["pyproject.toml", "uv.lock", "srcfile", "src/rogue.py"]
    assert sb.residual_scratch_contamination([]) == []


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


def _scratch_repo(repo: Path) -> None:
    """git init a tmp repo mirroring the production shared-config shape (plan A4):
    ``extensions.worktreeConfig=true`` pre-set, so a scratch worktree's
    ``sparse-checkout init`` writes only worktree-local config and the shared
    ``.git/config`` stays byte-identical (pinned by the N5 roundtrip)."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "core.repositoryformatversion", "1")
    _git(repo, "config", "extensions.worktreeConfig", "true")


def test_scratch_worktree_real_git_roundtrip(tmp_path: Path):
    """N5: real create/remove roundtrip — the scratch materializes HEAD (root dirt
    ABSENT), and teardown leaves the root's HEAD + porcelain + shared config
    byte-identical (R-D no-mutation pin; makes kill-criterion 2 decidable).
    Executes the real bodies of create/remove_scratch_worktree + _scratch_cones
    + _git_bounded (all seam-stubbed in the compare unit cases, #906)."""
    repo = tmp_path / "repo"
    _scratch_repo(repo)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (repo / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = ""\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "dirt.py").write_text("X = 1\n")  # UNCOMMITTED root dirt

    def _porcelain() -> str:
        return subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    # SNAPSHOT the root state the fallback must never mutate (R-D).
    head_before = sb.git_head(repo)
    porcelain_before = _porcelain()
    config_before = (repo / ".git" / "config").read_bytes()
    scratch = sb.create_scratch_worktree(repo, ["tests"], 120.0)
    try:
        assert scratch.sha == head_before
        assert (scratch.path / "tests" / "test_a.py").exists()  # committed file materialized
        assert not (scratch.path / "scripts" / "dirt.py").exists()  # root dirt ABSENT
    finally:
        sb.remove_scratch_worktree(repo, scratch)
    assert not scratch.path.exists() and not scratch.parent.exists()
    wt_lines = subprocess.run(
        ["git", "worktree", "list"], cwd=str(repo), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert len([ln for ln in wt_lines if ln.strip()]) == 1  # only the root itself remains
    # Root-state no-mutation pin (R-D).
    assert sb.git_head(repo) == head_before
    assert _porcelain() == porcelain_before
    assert (repo / ".git" / "config").read_bytes() == config_before


def test_real_pristine_in_scratch_worktree_nodes_match_root_relative(tmp_path: Path):
    """N6 (the A2/kill-criterion pin): junit ``file`` attrs stay rootdir-relative
    when pytest runs with cwd=scratch (committed pyproject.toml anchors rootdir),
    so Node keys match the gate junit's — AND the MAIN root's interpreter ran
    (the venv_root split; the scratch has no venv)."""
    root = tmp_path / "root"
    _scratch_repo(root)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = ""\n')
    (root / "tests" / "test_probe.py").write_text("def test_bad():\n    assert False\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "baseline")
    marker = tmp_path / "scratch-shim-invocations.txt"
    shim = _write_python_shim(root, marker)  # the root venv shim is NOT committed
    scratch = sb.create_scratch_worktree(root, ["tests"], 120.0)
    try:
        assert not (scratch.path / ".venv").exists()  # the scratch has no venv of its own
        failing = sb.run_single_file_pristine(
            "tests/test_probe.py", cwd=scratch.path, timeout_s=180.0, venv_root=root
        )
    finally:
        sb.remove_scratch_worktree(root, scratch)
    assert failing == {
        sb.Node(file="tests/test_probe.py", classname="tests.test_probe", name="test_bad")
    }
    assert marker.exists(), "the MAIN root's venv shim was never invoked (venv_root split)"
    assert str(shim) in marker.read_text()


def test_scratch_contamination_probe_real_git(tmp_path: Path):
    """N7: the src/-wide (ALL files, not just .py) contamination probe — the
    non-Python src/*.json hole, the top-level-only src/ pathspec, and the
    pyproject/uv.lock legs, against real git."""
    repo = tmp_path / "repo"
    _scratch_repo(repo)
    for rel, text in [
        ("src/pkg/data.json", "{}\n"),
        ("src/pkg/mod.py", "X = 1\n"),
        ("scripts/x.py", "Y = 1\n"),
        ("dashboard/src/z.py", "Z = 1\n"),
        ("pyproject.toml", "[project]\n"),
        ("uv.lock", "# lock\n"),
    ]:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    assert sb.scratch_contamination_probe(repo) == []  # clean repo
    (repo / "scripts" / "x.py").write_text("Y = 2\n")
    assert sb.scratch_contamination_probe(repo) == []  # scripts/ dirt is decontaminable
    (repo / "dashboard" / "src" / "z.py").write_text("Z = 2\n")
    assert sb.scratch_contamination_probe(repo) == []  # pathspec matches only TOP-LEVEL src/
    (repo / "src" / "pkg" / "data.json").write_text('{"a": 1}\n')  # NON-Python src/ dirt
    assert sb.scratch_contamination_probe(repo) == ["src/pkg/data.json"]
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (repo / "uv.lock").write_text("# lock2\n")
    assert set(sb.scratch_contamination_probe(repo)) == {
        "src/pkg/data.json",
        "pyproject.toml",
        "uv.lock",
    }
    # The .py-scoped dirt TRIGGER still sees the scripts/ dirt the probe ignores.
    assert "scripts/x.py" in sb.dirty_code_paths(repo)


def test_scratch_cones_union_head_registry_and_wt_list(tmp_path: Path):
    """N8: scratch profile = work-root cones (union) HEAD-pinned registry (union)
    top-dirs floor minus excludes; a DIRTY live registry line is never read;
    a registry absent at HEAD does not raise."""
    repo = tmp_path / "repo"
    _scratch_repo(repo)
    for rel in ("tests/t.py", "scripts/s.py", "docs/d.md", "eval_results/e.json"):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    # Registry ABSENT at HEAD: floor (top dirs minus excludes) + wt cones, no raise.
    assert sb._scratch_cones(repo, ["eval_results/issue_99"]) == sorted(
        {"tests", "scripts", "docs", "eval_results/issue_99"}
    )
    (repo / "tests" / "sparse_cones.txt").write_text(
        "# fixture cones\n\neval_results/issue_5\nfigures/issue_5\n"
    )
    _git(repo, "add", "tests/sparse_cones.txt")
    _git(repo, "commit", "-m", "registry")
    # DIRTY the live registry — the HEAD-pinned read must ignore the bogus line.
    with (repo / "tests" / "sparse_cones.txt").open("a") as fh:
        fh.write("eval_results/issue_BOGUS\n")
    cones = sb._scratch_cones(repo, ["eval_results/issue_99"])
    assert "eval_results/issue_5" in cones and "figures/issue_5" in cones  # HEAD-pinned rows
    assert "eval_results/issue_99" in cones  # work-root cone unioned in
    assert "eval_results/issue_BOGUS" not in cones  # live-file dirt never read
    assert {"tests", "scripts", "docs"} <= set(cones)  # top-dir floor
    assert "eval_results" not in cones  # SCRATCH_EXCLUDES floor exclusion


def test_work_root_sparse_cones_real_git(tmp_path: Path):
    """Real-git body for _work_root_sparse_cones (seam-stubbed in compare cases):
    a NON-sparse tree maps to None — on git 2.34 ``sparse-checkout list`` exits 0
    with EMPTY stdout there, so the empty list MUST fold to None or R-G's
    non-sparse ineligibility silently breaks — and a sparse cone-mode tree
    returns its cone list; a non-git dir maps to None too."""
    repo = tmp_path / "repo"
    _scratch_repo(repo)
    for rel in ("tests/t.py", "scripts/s.py"):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "baseline")
    assert sb._work_root_sparse_cones(repo) is None  # non-sparse -> ineligible
    _git(repo, "sparse-checkout", "init", "--cone")
    _git(repo, "sparse-checkout", "set", "tests")
    assert sb._work_root_sparse_cones(repo) == ["tests"]
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    assert sb._work_root_sparse_cones(not_a_repo) is None


# --- #1251 real-subprocess mechanism + live-venv durability pins -------------------


def _make_probe_pkg(base: Path, where: str) -> Path:
    """Write <base>/src/probe_pkg/__init__.py with WHERE=<where>; return <base>/src."""
    pkg = base / "src" / "probe_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(f'WHERE = "{where}"\n')
    return base / "src"


def test_real_pythonpath_shadow_wins_over_static_editable_pth(tmp_path: Path):
    """Mechanism proof (real venv machinery): a single-line static ``.pth``
    (byte-style-identical to the production ``__editable__.*.pth``) resolves the
    root copy WITHOUT PYTHONPATH (control arm — proves the .pth engages, guards
    a vacuous pass), and PYTHONPATH=<scratch>/src WINS over it (the #1251
    shadow's load-bearing sys.path-precedence claim)."""
    venvroot = tmp_path / "venvroot"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venvroot)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    py = str(venvroot / "bin" / "python")
    # Locate the tmp venv's site-packages portably (sysconfig, not a hardcoded pythonX.Y).
    site_pkgs = Path(
        subprocess.run(
            [py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
    )
    root_src = _make_probe_pkg(tmp_path / "root", "root")
    scratch_src = _make_probe_pkg(tmp_path / "scratch", "scratch")
    (site_pkgs / "__editable__.probe_pkg.pth").write_text(f"{root_src}\n")
    code = "import probe_pkg, sys; sys.stdout.write(probe_pkg.WHERE)"
    env_base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # (a) control: WITHOUT PYTHONPATH the .pth engages -> the root copy resolves.
    proc_a = subprocess.run(
        [py, "-c", code], env=env_base, capture_output=True, text=True, timeout=60
    )
    assert proc_a.returncode == 0, proc_a.stderr
    assert proc_a.stdout == "root"
    # (b) WITH PYTHONPATH=<scratch>/src the shadow wins over the .pth entry.
    proc_b = subprocess.run(
        [py, "-c", code],
        env={**env_base, "PYTHONPATH": str(scratch_src)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc_b.returncode == 0, proc_b.stderr
    assert proc_b.stdout == "scratch"


def test_real_run_pytest_pythonpath_env(tmp_path: Path, monkeypatch):
    """Real-subprocess coverage of run_pytest's #1251 env branch: an explicitly
    passed ``pythonpath`` reaches the pytest child (the probe test imports the
    scratch copy and passes), while an AMBIENT PYTHONPATH is still stripped
    (the #1022 vector: the same run WITHOUT the kwarg cannot import probe_pkg
    even with PYTHONPATH exported in the parent env)."""
    scratch_src = _make_probe_pkg(tmp_path / "scratch", "scratch")
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = ""\n')
    (tree / "tests" / "test_probe_env.py").write_text(
        "import probe_pkg\n\n\ndef test_where():\n    assert probe_pkg.WHERE == 'scratch'\n"
    )
    junit = tmp_path / "junit-shadow.xml"
    rc = sb.run_pytest(
        files=["tests/test_probe_env.py"],
        cwd=tree,
        timeout_s=180.0,
        junit_path=junit,
        python_exe=sys.executable,
        pythonpath=str(scratch_src),
    )
    assert rc == 0
    failing, summary = sb.parse_junit(junit)
    assert failing == []  # the scratch copy resolved: probe_pkg.WHERE == 'scratch'
    assert summary["tests"] == 1
    # Strip pin: ambient PYTHONPATH (no kwarg) never reaches the child (#1022).
    monkeypatch.setenv("PYTHONPATH", str(scratch_src))
    junit2 = tmp_path / "junit-strip.xml"
    rc2 = sb.run_pytest(
        files=["tests/test_probe_env.py"],
        cwd=tree,
        timeout_s=180.0,
        junit_path=junit2,
        python_exe=sys.executable,
    )
    assert rc2 != 0  # probe_pkg unimportable -> the inherited PYTHONPATH was stripped


def test_live_venv_src_shadow_probe_passes(tmp_path: Path):
    """Durability pin: the REAL assert_scratch_src_shadow body against THIS
    repo's actual venv + editable install — goes red the day the editable
    style changes to a PYTHONPATH-preempting mechanism (finder hook /
    reordering .pth), the exact early warning #1251 wants."""
    scratch = tmp_path / "scratch"
    pkg = scratch / "src" / "explore_persona_space"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    sb.assert_scratch_src_shadow(sb.main_repo_root(), scratch, timeout_s=60.0)  # no raise


def test_live_venv_src_shadow_probe_fails_on_missing_scratch_src(tmp_path: Path):
    """Companion negative (real body): a scratch MISSING src/explore_persona_space
    makes the probe resolve the root package instead -> PristineRunError (the
    fail-closed missing-scratch-src path)."""
    scratch = tmp_path / "empty-scratch"
    scratch.mkdir()
    with pytest.raises(sb.PristineRunError, match="src-shadow probe"):
        sb.assert_scratch_src_shadow(sb.main_repo_root(), scratch, timeout_s=60.0)


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
