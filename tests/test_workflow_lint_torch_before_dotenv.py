"""Tests for ``workflow_lint --check-torch-before-dotenv`` (#2650).

The check is the LINT-time twin of
``tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints``
(#847). It exists because the pytest guard fires only when some LATER session
runs its Step 9c gate, by which point a violator is already on ``main`` and
reds that gate for every session that did not cause it — the #1388 fleet-red
shape, which ``tasks/REGISTRY.json`` had recorded nine times before #2650 made
it ten.

The load-bearing property is ANTI-DRIFT: the lint must not carry its own copy
of the rule. It imports the guard module by path and calls the guard's own
``_scan_targets`` / ``_first_heavy_import_line`` / ``_first_load_dotenv_line``
/ ``GRANDFATHERED_TORCH_BEFORE_DOTENV``. ``test_live_tree_matches_pytest_guard``
is the test that pins that property: it recomputes the guard's verdict
independently and asserts SET equality with the lint's, so a future
re-implementation of the predicate inside the lint fails here.

Coverage:
  (i)   set-equality with the guard on the live tree (anti-drift);
  (ii)  a synthetic violator FAILs and its preamble-fixed twin PASSes;
  (iii) the guard's allowlist is honoured (no waiver of the lint's own);
  (iv)  every wiring row exists — argparse flag, ``_FILES_MODE_RUNNERS``,
        ``CHECK_SCOPES``, and dispatch from the no-flags default run;
  (v)   a stale lint FAILs loudly (guard module missing / symbol renamed)
        rather than silently passing;
  (vi)  a missing ``pytest`` SKIPs with a note instead of reddening the run.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402

_REPO = _HERE.parent
_GUARD_REL = "tests/test_shared_vm_thread_caps.py"

PREAMBLE = "from explore_persona_space.orchestrate.env import load_dotenv\n\nload_dotenv()\n\n"


def _load_guard():
    """Independent load of the guard module (the test's own, not the lint's)."""
    spec = importlib.util.spec_from_file_location("_guard_under_test", _REPO / _GUARD_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_violations(root: Path) -> set[str]:
    """Recompute the pytest guard's verdict directly, without the lint."""
    guard = _load_guard()
    out: set[str] = set()
    for path in guard._scan_targets(root):
        src = path.read_text()
        heavy = guard._first_heavy_import_line(ast.parse(src))
        if heavy is None:
            continue
        rel = path.relative_to(root).as_posix()
        dotenv = guard._first_load_dotenv_line(src)
        if (
            dotenv is None or heavy < dotenv
        ) and rel not in guard.GRANDFATHERED_TORCH_BEFORE_DOTENV:
            out.add(rel)
    return out


def _lint_violations(root: Path | None = None) -> set[str]:
    """Paths named by the lint's FAIL rows (rows are ``<rel>:<line>: ...``)."""
    rows = wl.check_torch_before_dotenv(repo_root=root) if root else wl.check_torch_before_dotenv()
    return {row.split(":", 1)[0] for row in rows}


# ---------------------------------------------------------------------------
# (i) anti-drift — the whole point of loading the guard instead of copying it
# ---------------------------------------------------------------------------


def test_live_tree_matches_pytest_guard() -> None:
    """The lint and the pytest guard must report the SAME violation set.

    Set equality, not counts: a lint that drifted to a different predicate
    could easily agree on the count while disagreeing on which files. This is
    the test that fails if someone later re-implements the rule inside
    ``workflow_lint.py`` instead of loading it from the guard.
    """
    assert _lint_violations() == _guard_violations(_REPO)


def test_live_tree_is_clean() -> None:
    """#2650's own acceptance: the tree carries no unallowlisted violator."""
    assert _lint_violations() == set()


# ---------------------------------------------------------------------------
# (ii)/(iii) synthetic violator, its fixed twin, and the allowlist
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, body: str, *, rel: str = "scripts/probe.py") -> Path:
    """Tmp git repo carrying a copy of the guard module and one committed script."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / _GUARD_REL).write_text((_REPO / _GUARD_REL).read_text(), encoding="utf-8")
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", "--", rel, _GUARD_REL], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
    )
    return repo


def test_synthetic_violator_fails(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, "import numpy as np\n\nprint(np)\n")
    rows = wl.check_torch_before_dotenv(repo_root=repo)
    assert len(rows) == 1, rows
    assert rows[0].startswith("scripts/probe.py:1: check-torch-before-dotenv:")
    # The row must point at the fix, not merely announce the failure.
    assert "orchestrate.env.load_dotenv()" in rows[0]
    assert "issue778_null_battery.py" in rows[0]


def test_fixed_twin_passes(tmp_path: Path) -> None:
    """Same file with the canonical preamble: no FAIL row."""
    repo = _make_repo(tmp_path, PREAMBLE + "import numpy as np\n\nprint(np)\n")
    assert wl.check_torch_before_dotenv(repo_root=repo) == []


def test_heavy_free_script_passes(tmp_path: Path) -> None:
    """A script with no module-top heavy import needs no preamble."""
    repo = _make_repo(tmp_path, "import json\n\nprint(json)\n")
    assert wl.check_torch_before_dotenv(repo_root=repo) == []


def test_allowlisted_violator_is_not_reported(tmp_path: Path, monkeypatch) -> None:
    """A path in the guard's allowlist is skipped — the lint adds no waiver of
    its own, so grandfathering keeps going through the guard's shrink-only
    dict and the currency tests that pin it."""
    repo = _make_repo(tmp_path, "import numpy as np\n\nprint(np)\n")
    assert wl.check_torch_before_dotenv(repo_root=repo)  # violating before allowlisting

    real_loader = wl._load_thread_caps_guard

    def _allowlisting_loader(root: Path):
        guard, err = real_loader(root)
        if guard is not None:
            guard.GRANDFATHERED_TORCH_BEFORE_DOTENV = dict(
                guard.GRANDFATHERED_TORCH_BEFORE_DOTENV, **{"scripts/probe.py": "#2650 test pin"}
            )
        return guard, err

    monkeypatch.setattr(wl, "_load_thread_caps_guard", _allowlisting_loader)
    assert wl.check_torch_before_dotenv(repo_root=repo) == []


# ---------------------------------------------------------------------------
# (iv) wiring — the check is useless if it is not actually dispatched
# ---------------------------------------------------------------------------


def test_registered_in_files_mode_runners() -> None:
    assert "check_torch_before_dotenv" in wl._FILES_MODE_RUNNERS


def test_has_a_path_local_scope_covering_new_scripts_and_the_guard() -> None:
    scope = wl.CHECK_SCOPES["check_torch_before_dotenv"]
    assert scope.kind == "path-local"
    # A new violator arrives under scripts/ or src/; a rule/allowlist change
    # arrives as an edit to the guard module the check loads its rule from.
    assert "scripts/" in scope.surfaces
    assert "src/" in scope.surfaces
    assert any("test_shared_vm_thread_caps" in p for p in scope.surfaces)


def test_flag_is_accepted_and_bundled_into_the_no_flags_run() -> None:
    src = (_SCRIPTS / "workflow_lint.py").read_text()
    assert '"--check-torch-before-dotenv"' in src
    assert "if args.check_torch_before_dotenv or no_flags:" in src
    assert "or args.check_torch_before_dotenv" in src  # the any-flag gate


def test_scoped_cli_invocation_exits_zero_on_the_live_tree() -> None:
    """The plan's scoped exit-0 success criterion (S7), executed."""
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "workflow_lint.py"), "--check-torch-before-dotenv"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# (v)/(vi) staleness fails loudly; a missing dev dep only skips
# ---------------------------------------------------------------------------


def test_missing_guard_module_fails_loudly(tmp_path: Path) -> None:
    """If the guard moves, the lint must say so rather than pass vacuously."""
    empty = tmp_path / "empty"
    (empty / "scripts").mkdir(parents=True)
    rows = wl.check_torch_before_dotenv(repo_root=empty)
    assert len(rows) == 1 and "is MISSING" in rows[0]


def test_renamed_guard_symbol_fails_loudly(tmp_path: Path) -> None:
    """A guard that no longer defines a symbol the lint calls is a stale lint."""
    repo = _make_repo(tmp_path, "import json\n")
    guard_src = (repo / _GUARD_REL).read_text()
    (repo / _GUARD_REL).write_text(
        guard_src.replace("def _first_load_dotenv_line(", "def _renamed_away("), encoding="utf-8"
    )
    rows = wl.check_torch_before_dotenv(repo_root=repo)
    assert len(rows) == 1, rows
    assert "no longer defines" in rows[0] and "_first_load_dotenv_line" in rows[0]


def test_missing_pytest_skips_rather_than_failing(tmp_path: Path, monkeypatch, capsys) -> None:
    """``pytest`` is a dev dependency; its absence must not red the no-flags run."""
    repo = _make_repo(tmp_path, "import numpy as np\n")
    real_exec = importlib.util.module_from_spec

    def _boom(spec):
        module = real_exec(spec)
        if spec.name == "_eps_thread_caps_guard":
            raise ModuleNotFoundError("No module named 'pytest'", name="pytest")
        return module

    monkeypatch.setattr(wl.importlib.util, "module_from_spec", _boom)
    assert wl.check_torch_before_dotenv(repo_root=repo) == []
    assert "skipped" in capsys.readouterr().err


def test_unparseable_file_is_reported_not_crashed(tmp_path: Path) -> None:
    """A syntactically broken in-class file yields a FAIL row, not a traceback."""
    repo = _make_repo(tmp_path, "import numpy as np\ndef (:\n")
    rows = wl.check_torch_before_dotenv(repo_root=repo)
    assert len(rows) == 1 and "could not read/parse" in rows[0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
