"""Tests for ``workflow_lint --check-no-repo-root-syspath-in-tests`` (#2181).

The check FAILs any ``sys.path.insert``/``sys.path.append`` (or
``monkeypatch.syspath_prepend``) under ``tests/**/*.py`` whose argument
derives from the branch-guarded ``task_workflow`` resolvers
(``repo_root``/``tasks_dir``/``registry_path``) — directly, via a
module-scope one-hop constant, or via a module-scope import alias.
``repo_root()`` resolves to the MAIN checkout, so a worktree pytest run
imports main's copy of the module under test (a branch regression can pass
its own test on the branch) and leaks a foreign checkout's dir onto
``sys.path`` for the whole session (incident #2164; silently defeats the
#1296 negative control in ``tests/test_backend_poll.py``).

Covers cases 1-14 from the plan §4.3:

1.  non-vacuity: the VERBATIM pre-fix incident form
    (``test_issue1482_densesae_fullwidth.py`` pre-``69a58d6e5b``) fires with
    exactly one file:lineno error naming the resolver + #2164;
2.  attribute callee ``tw.repo_root()`` (``sys.path.append``) fires;
3.  one-hop module constant ``_REPO = repo_root()`` fires;
4.  module-scope import alias ``import repo_root as rr`` fires;
5.  sibling resolver ``tasks_dir()`` fires (all three branch-guard alike);
6.  resolver-derived ``monkeypatch.syspath_prepend`` inside a test fn fires
    (teardown-restore does not undo the in-session main-copy import);
7.  GREEN naming-collision trap: ``REPO_ROOT`` bound to a
    ``__file__``-derived value never taints — taint keys on the binding's
    VALUE, never its NAME (dozens of such constants live under ``tests/``);
8.  GREEN: tree-local ``monkeypatch.syspath_prepend`` passes;
9.  GREEN: a resolver name appearing only inside a string is invisible to
    the AST predicate (value-based, not text-based);
10. semantics pin: the one-hop taint is ORDER-INSENSITIVE — a name bound to
    ``repo_root()`` then REBOUND to a ``__file__``-derived value before the
    insert still fires (documented over-match (c) in the check docstring);
11. an unparseable file is skipped with a one-line stderr notice — never an
    exception, never a flag;
12. the LIVE tree is green (fleet-red guard: the check rides the no-flags
    default bundle, which gates every Step 9c compare + Step 10d pre-push
    merge — the ~535 sanctioned tree-local sys.path sites must not flag);
13. scope pin: the same banned form under ``scripts/`` is OUT of scope for
    THIS check (19 live one-hop offenders there — plan §8 defers them);
14. bundle-membership pin: the no-flags default run DISPATCHES the check
    (mutation-visible ``wl.main([])`` run, the
    ``test_workflow_lint_scripts_import_guard.py`` row-16 pattern) plus a
    source-level pin on the ``no_flags`` or-chain membership (the dispatch
    run alone cannot see the or-chain: with the flag absent from the
    disjunction, a bare ``main([])`` still dispatches via ``no_flags``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import check_no_repo_root_syspath_in_tests  # noqa: E402

# The verbatim pre-fix incident form (test_issue1482_densesae_fullwidth.py
# before fix commit 69a58d6e5b): direct repo_root()-derived insert.
_INCIDENT_BODY = (
    "import sys\n"
    "\n"
    "from explore_persona_space.task_workflow import repo_root\n"
    "\n"
    'sys.path.insert(0, str(repo_root() / "scripts"))\n'
)
_INCIDENT_LINENO = 5


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _run_on(tmp_path: Path) -> list[str]:
    """Run the check against a planted tmp tree via the documented
    ``repo_root`` unit-test override hook."""
    return check_no_repo_root_syspath_in_tests(repo_root=tmp_path)


# --------------------------------------------------------------------------
# case 1: negative control (non-vacuity) — the verbatim incident form fires
# --------------------------------------------------------------------------


def test_verbatim_incident_form_fires(tmp_path) -> None:
    offender = _plant(tmp_path, "tests/offender.py", _INCIDENT_BODY)
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{offender}:{_INCIDENT_LINENO}:"), errors[0]
    assert "repo_root" in errors[0]
    assert "#2164" in errors[0]


# --------------------------------------------------------------------------
# case 2: attribute callee (tw.repo_root()) via sys.path.append
# --------------------------------------------------------------------------


def test_attribute_callee_append_fires(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/attr_callee.py",
        "import sys\n"
        "\n"
        "from explore_persona_space import task_workflow as tw\n"
        "\n"
        'sys.path.append(str(tw.repo_root() / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "repo_root" in errors[0]


# --------------------------------------------------------------------------
# case 3: one-hop module constant
# --------------------------------------------------------------------------


def test_one_hop_module_constant_fires(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/one_hop.py",
        "import sys\n"
        "\n"
        "from explore_persona_space.task_workflow import repo_root\n"
        "\n"
        "_REPO = repo_root()\n"
        'sys.path.insert(0, str(_REPO / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "repo_root" in errors[0]


# --------------------------------------------------------------------------
# case 4: module-scope import alias
# --------------------------------------------------------------------------


def test_import_alias_fires(tmp_path) -> None:
    """The diagnostic names the MATCHED callee — for an aliased import that
    is the alias (``rr``), not the canonical resolver name (the plan's §4.1
    predicate returns the matched name; case 4 requires only that it fires)."""
    _plant(
        tmp_path,
        "tests/alias.py",
        "import sys\n"
        "\n"
        "from explore_persona_space.task_workflow import repo_root as rr\n"
        "\n"
        'sys.path.insert(0, str(rr() / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "`rr()`" in errors[0], errors[0]


# --------------------------------------------------------------------------
# case 5: sibling resolver (tasks_dir) — all three branch-guard identically
# --------------------------------------------------------------------------


def test_sibling_resolver_tasks_dir_fires(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/sibling.py",
        "import sys\n"
        "\n"
        "from explore_persona_space.task_workflow import tasks_dir\n"
        "\n"
        'sys.path.insert(0, str(tasks_dir().parent / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "tasks_dir" in errors[0]


# --------------------------------------------------------------------------
# case 6: resolver-derived monkeypatch.syspath_prepend fires
# --------------------------------------------------------------------------


def test_resolver_derived_syspath_prepend_fires(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/prepend.py",
        "from explore_persona_space.task_workflow import repo_root\n"
        "\n"
        "\n"
        "def test_x(monkeypatch):\n"
        '    monkeypatch.syspath_prepend(str(repo_root() / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "repo_root" in errors[0]


# --------------------------------------------------------------------------
# case 7: GREEN — naming-collision trap (VALUE-based taint, never NAME-based)
# --------------------------------------------------------------------------


def test_file_derived_repo_root_name_is_green(tmp_path) -> None:
    """A constant NAMED ``REPO_ROOT`` bound to the sanctioned
    ``__file__``-derived form must never taint — dozens of such constants
    live under the real ``tests/``; matching on identifier names here would
    red the fleet (the plan's D1 'taint keys on the binding's VALUE' pin)."""
    _plant(
        tmp_path,
        "tests/green_name.py",
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(REPO_ROOT / "scripts"))\n',
    )
    assert _run_on(tmp_path) == []


# --------------------------------------------------------------------------
# case 8: GREEN — tree-local monkeypatch.syspath_prepend
# --------------------------------------------------------------------------


def test_tree_local_syspath_prepend_is_green(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/green_prepend.py",
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.syspath_prepend(\n"
        '        str(Path(__file__).resolve().parents[1] / "scripts")\n'
        "    )\n",
    )
    assert _run_on(tmp_path) == []


# --------------------------------------------------------------------------
# case 9: GREEN — resolver name only inside a string (AST-based, not text)
# --------------------------------------------------------------------------


def test_resolver_name_in_string_is_green(tmp_path) -> None:
    _plant(
        tmp_path,
        "tests/green_string.py",
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        'x = "call repo_root() later"\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))\n',
    )
    assert _run_on(tmp_path) == []


# --------------------------------------------------------------------------
# case 10: semantics pin — order-insensitive taint (documented over-match)
# --------------------------------------------------------------------------


def test_order_insensitive_taint_fires(tmp_path) -> None:
    """A name bound to ``repo_root()`` and later REBOUND to a
    ``__file__``-derived value before the insert still fires: the one-hop
    taint is deliberately order-insensitive (over-match (c), enumerated in
    the ``check_no_repo_root_syspath_in_tests`` docstring; zero live hits)."""
    _plant(
        tmp_path,
        "tests/rebound.py",
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from explore_persona_space.task_workflow import repo_root\n"
        "\n"
        "_R = repo_root()\n"
        "_R = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(_R / "scripts"))\n',
    )
    errors = _run_on(tmp_path)
    assert len(errors) == 1, errors
    assert "repo_root" in errors[0]


# --------------------------------------------------------------------------
# case 11: unparseable file — skipped with a stderr notice, never a crash
# --------------------------------------------------------------------------


def test_unparseable_file_skipped_with_notice(tmp_path, capsys) -> None:
    _plant(tmp_path, "tests/broken.py", "def broken(:\n")
    assert _run_on(tmp_path) == []
    err = capsys.readouterr().err
    assert "skipped unparseable" in err, err
    assert "broken.py" in err, err


# --------------------------------------------------------------------------
# case 12: the LIVE tree is green (fleet-red guard)
# --------------------------------------------------------------------------


def test_live_tree_is_green() -> None:
    """The check rides the no-flags default bundle, which gates EVERY task's
    Step 9c compare and Step 10d pre-push merge — a single false positive on
    the real tree reds the fleet. Runs against the LIVE repo (production
    ``_REPO_ROOT`` default, no tmp_path): the ~535 sanctioned tree-local /
    ``__file__``-derived ``sys.path`` sites under ``tests/`` must not flag
    (the ``test_workflow_lint_gotchas_size.py`` live-tree pattern)."""
    assert check_no_repo_root_syspath_in_tests() == []


# --------------------------------------------------------------------------
# case 13: scope pin — tests/ only (scripts/ offenders deferred, plan §8)
# --------------------------------------------------------------------------


def test_scripts_dir_is_out_of_scope(tmp_path) -> None:
    """The same banned form under ``scripts/`` must NOT flag: widening would
    land the check with 19 live one-hop offenders (17 ``issue1482_*.py`` +
    2 ``issue1738_*.py``) — an instant fleet-wide red. Deferred to a
    separate remediation task (plan §8)."""
    _plant(tmp_path, "scripts/offender.py", _INCIDENT_BODY)
    # A sanctioned tests/ file so the tests/ scan genuinely runs (non-empty).
    _plant(
        tmp_path,
        "tests/green.py",
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))\n',
    )
    assert _run_on(tmp_path) == []


# --------------------------------------------------------------------------
# case 14: bundle-membership pin — the no-flags default run DISPATCHES the
# check (mutation-visible; the test_workflow_lint_scripts_import_guard.py
# row-16 pattern) + source-level no_flags or-chain membership
# --------------------------------------------------------------------------


def test_check_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #2181 check —
    deleting its ``or no_flags`` dispatch branch must fail this test
    (mutation-visible). Other bundled checks contribute unrelated errors on
    the minimal tree, so the assertion keys on the check's own diagnostic
    (the #2164 incident cite) + the offending path. The or-chain membership
    is pinned at SOURCE level: the dispatch run alone cannot see it, because
    with the flag absent from the ``no_flags`` disjunction a bare
    ``main([])`` still dispatches via ``no_flags``."""
    _plant(tmp_path, "tests/syspath_offender.py", _INCIDENT_BODY)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "#2164" in err and "syspath_offender.py" in err, (
        f"the no-repo-root-syspath diagnostic (naming syspath_offender.py) is "
        f"missing from the no-flags default run's stderr — the check is not "
        f"bundled into no_flags:\n{err}"
    )
    src = Path(wl.__file__).read_text(encoding="utf-8")
    assert "or args.check_no_repo_root_syspath_in_tests" in src, (
        "the no_flags disjunction lost `or args.check_no_repo_root_syspath_in_tests` — "
        "a scoped `--check-no-repo-root-syspath-in-tests` invocation would then "
        "run the WHOLE bundle instead of just this check"
    )
    assert "if args.check_no_repo_root_syspath_in_tests or no_flags:" in src, (
        "the dispatch ladder lost the "
        "`if args.check_no_repo_root_syspath_in_tests or no_flags:` line"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
