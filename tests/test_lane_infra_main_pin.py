"""Tests for the #987 lane-infra main-checkout pin in the dispatch/poll entrypoints.

``scripts/dispatch_issue.py`` and ``scripts/backend_poll.py`` carry a
module-top, ``__main__``-guarded bootstrap (``_resolve_main_checkout_root`` +
``_pin_main_lane_infra``, duplicated into both files by design) that inserts
``<main>/src`` + ``<main>`` at the FRONT of ``sys.path`` BEFORE any
``explore_persona_space`` import, so a script-mode invocation from ANY cwd
(repo root, an issue worktree, /tmp) sources the lane infra (``backends/``,
incl. the GCE startup template in ``gcp.py``) from the MAIN checkout while
``--repo-branch`` keeps cloning the issue branch for the remote workload.

Pinned invariants (plan #987 §6 tests 1-7 + the review-round additions):

1. the git-common-dir resolver maps a WORKTREE anchor to the MAIN root;
2. the pin's sys.path order (``[<main>/src, <main>, ...]``) + idempotency;
3. end-to-end: a front-of-path insert beats the ambient editable install
   (the exact production mechanism, plan §10 probe 2);
4. fail-loud outside a git checkout — plus the shape-check and missing-src
   RuntimeError branches (no silent fallback to the ambient package);
5. fail-loud when a stale ``explore_persona_space`` is already imported;
6. call-site shape: per file, EXACTLY ONE module-level
   ``_pin_main_lane_infra()`` call, it sits under ``if __name__ ==
   "__main__":``, and NO unguarded module-level call exists — plus the
   behavioral twin: importing the modules under BOTH import identities the
   suite uses (``scripts.backend_poll`` AND bare ``backend_poll``) does NOT
   apply the pin, so worktree pytest keeps testing branch code;
7. the two duplicated bootstraps stay source-identical (drift guard);
8. integration: each shipped script still runs as ``__main__`` (``--help``)
   from a non-main cwd with exit code 0 — the module-top pin runs before
   argparse, so this executes the shipped bootstrap end to end.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import scripts.backend_poll as backend_poll
import scripts.dispatch_issue as dispatch_issue

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

_MODULES = [dispatch_issue, backend_poll]
_MODULE_IDS = ["dispatch_issue", "backend_poll"]
_SCRIPT_PATHS = [_SCRIPTS_DIR / "dispatch_issue.py", _SCRIPTS_DIR / "backend_poll.py"]
_PIN_FN_NAMES = {"_resolve_main_checkout_root", "_pin_main_lane_infra"}


def _subprocess_env() -> dict[str, str]:
    """os.environ minus PYTHONPATH (flake guard: no inherited resolution paths)."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _git(*args: str, cwd: Path) -> str:
    """Run git in ``cwd`` with hermetic identity/config; returns stripped stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={
            **_subprocess_env(),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )
    return proc.stdout.strip()


@pytest.fixture()
def tmp_checkouts(tmp_path: Path) -> tuple[Path, Path]:
    """A tmp 'main' git checkout + a linked worktree on branch issue-999.

    ``main/src/explore_persona_space/__init__.py`` carries ``ORIGIN='main'``;
    the worktree copy is overwritten to ``ORIGIN='worktree'`` so an import's
    ``ORIGIN`` reveals WHICH checkout's package resolved. Returns
    ``(main_root, worktree_root)``, both resolved.
    """
    main = tmp_path / "main"
    (main / "src" / "explore_persona_space").mkdir(parents=True)
    (main / "scripts").mkdir()
    (main / "src" / "explore_persona_space" / "__init__.py").write_text("ORIGIN = 'main'\n")
    (main / "scripts" / ".keep").write_text("")
    _git("init", "-b", "main", cwd=main)
    _git("add", "-A", cwd=main)
    _git("commit", "-m", "init", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-b", "issue-999", str(wt), cwd=main)
    (wt / "src" / "explore_persona_space" / "__init__.py").write_text("ORIGIN = 'worktree'\n")
    return main.resolve(), wt.resolve()


# ---------------------------------------------------------------------------
# 1. Resolver maps a worktree anchor to the main root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolver_maps_worktree_anchor_to_main_root(mod, tmp_checkouts):
    main, wt = tmp_checkouts
    assert mod._resolve_main_checkout_root(wt / "scripts").resolve() == main
    # From the main checkout itself the resolver is the identity.
    assert mod._resolve_main_checkout_root(main / "scripts").resolve() == main


# ---------------------------------------------------------------------------
# 2. Pin order + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_pin_prepends_main_src_then_root_and_is_idempotent(mod, tmp_checkouts, monkeypatch):
    main, wt = tmp_checkouts
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The suite's own (main-resolved) package would trip the stale-preimport
    # guard against the TMP main root — remove it for the duration.
    monkeypatch.delitem(sys.modules, "explore_persona_space", raising=False)
    got = mod._pin_main_lane_infra(anchor=wt / "scripts")
    assert got.resolve() == main
    assert sys.path[0] == str(got / "src")
    assert sys.path[1] == str(got)
    before = list(sys.path)
    mod._pin_main_lane_infra(anchor=wt / "scripts")
    assert sys.path == before, "re-entrant pin must not duplicate or reorder entries"


# ---------------------------------------------------------------------------
# 3. End-to-end: front-of-path insert beats the ambient editable install
# ---------------------------------------------------------------------------


def test_pin_beats_editable_install_end_to_end(tmp_checkouts):
    main, wt = tmp_checkouts
    env = _subprocess_env()
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(main)!r})\n"
        f"sys.path.insert(0, {str(main / 'src')!r})\n"
        "import explore_persona_space as eps\n"
        "print(getattr(eps, 'ORIGIN', 'ambient'))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(wt),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert proc.stdout.strip() == "main"
    # Vacuity guard (review concern 5): WITHOUT the insert the same
    # interpreter resolves a DIFFERENT package (the test venv's editable
    # .pth), proving the tmp-main insert genuinely shadowed it rather than
    # being the only candidate on the path.
    control = subprocess.run(
        [
            sys.executable,
            "-c",
            "import explore_persona_space as eps; print(getattr(eps, 'ORIGIN', 'ambient'))",
        ],
        cwd=str(wt),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert control.stdout.strip() != "main"


# ---------------------------------------------------------------------------
# 4. Fail-loud resolver branches (no silent fallback to the ambient package)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolver_fails_loud_outside_git(mod, tmp_path):
    bare = tmp_path / "not-a-checkout"
    bare.mkdir()
    with pytest.raises(RuntimeError, match="#987"):
        mod._resolve_main_checkout_root(bare)


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolver_fails_loud_on_non_dot_git_common_dir_shape(mod, tmp_path):
    # A bare repo's --git-common-dir is the repo dir itself (name != ".git").
    bare_repo = tmp_path / "barerepo"
    bare_repo.mkdir()
    _git("init", "--bare", cwd=bare_repo)
    with pytest.raises(RuntimeError, match="does not look like"):
        mod._resolve_main_checkout_root(bare_repo)


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_resolver_fails_loud_when_main_lacks_src_package(mod, tmp_path):
    repo = tmp_path / "nosrc"
    repo.mkdir()
    (repo / "f.txt").write_text("x")
    _git("init", "-b", "main", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)
    with pytest.raises(RuntimeError, match="has no src/explore_persona_space"):
        mod._resolve_main_checkout_root(repo)


# ---------------------------------------------------------------------------
# 5. Fail-loud on a pre-imported stale package
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _MODULES, ids=_MODULE_IDS)
def test_preimported_stale_package_raises(mod, tmp_checkouts, monkeypatch):
    _main, wt = tmp_checkouts
    fake = types.ModuleType("explore_persona_space")
    fake.__file__ = "/somewhere/stale/explore_persona_space/__init__.py"
    monkeypatch.setitem(sys.modules, "explore_persona_space", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(RuntimeError, match="already imported from"):
        mod._pin_main_lane_infra(anchor=wt / "scripts")


# ---------------------------------------------------------------------------
# 6. Call-site shape: exactly one module-level call, __main__-guarded
# ---------------------------------------------------------------------------


def _is_pin_call_stmt(stmt: ast.stmt) -> bool:
    """True iff ``stmt`` is a bare expression-statement ``_pin_main_lane_infra(...)``."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "_pin_main_lane_infra"
    )


def _is_main_guard(node: ast.If) -> bool:
    """True iff ``node`` tests ``__name__ == "__main__"``."""
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
        and len(t.ops) == 1
        and isinstance(t.ops[0], ast.Eq)
        and isinstance(t.comparators[0], ast.Constant)
        and t.comparators[0].value == "__main__"
    )


@pytest.mark.parametrize("script_path", _SCRIPT_PATHS, ids=_MODULE_IDS)
def test_pin_call_exists_and_is_main_guarded(script_path):
    """Per file: >=1 guarded call, exactly one total, zero unguarded (AST walk)."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    guarded_calls = 0
    unguarded_calls = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue  # a call inside a def only runs when invoked, not at import
        if isinstance(stmt, ast.If) and _is_main_guard(stmt):
            guarded_calls += sum(1 for n in ast.walk(stmt) if _is_pin_call_stmt(n))
            continue
        # Any other module-level statement executing the pin runs on IMPORT —
        # the banned unconditional-pin shape (worktree pytest would silently
        # test MAIN's package).
        unguarded_calls += sum(1 for n in ast.walk(stmt) if _is_pin_call_stmt(n))
    assert guarded_calls == 1, f"{script_path.name}: expected exactly one guarded pin call"
    assert unguarded_calls == 0, f"{script_path.name}: unguarded module-level pin call found"


@pytest.mark.parametrize("script_name", _MODULE_IDS)
@pytest.mark.parametrize("identity", ["scripts-package", "bare"])
def test_import_under_both_suite_identities_does_not_apply_pin(script_name, identity):
    """Importing the module (either suite identity) must NOT apply the pin.

    The suite imports these scripts as ``scripts.<name>`` (test_backend_poll,
    test_dispatch_issue_cli) AND as bare ``<name>`` after inserting scripts/
    (test_runpod_wedge_detection, test_autonomous_session_watch_wedge). Under
    BOTH identities ``__name__ != "__main__"``, so the pin signature
    ``sys.path[:2] == [<main>/src, <main>]`` must be absent after a fresh
    import — worktree pytest keeps testing branch code.
    """
    if identity == "scripts-package":
        setup = f"sys.path.insert(0, {str(_REPO_ROOT)!r})\nimport scripts.{script_name} as m\n"
    else:
        setup = f"sys.path.insert(0, {str(_SCRIPTS_DIR)!r})\nimport {script_name} as m\n"
    code = (
        "import sys\n"
        f"{setup}"
        "from pathlib import Path\n"
        "root = m._resolve_main_checkout_root(Path(m.__file__).resolve().parent)\n"
        "pin_sig = [str(root / 'src'), str(root)]\n"
        "assert sys.path[:2] != pin_sig, f'pin applied on import: {sys.path[:4]}'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 7. The two duplicated bootstraps stay source-identical
# ---------------------------------------------------------------------------


def test_two_script_copies_in_sync():
    """AST-normalized source of both pin functions is identical across the files."""
    per_file: dict[str, dict[str, str]] = {}
    for path in _SCRIPT_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fns = {
            node.name: ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in _PIN_FN_NAMES
        }
        assert set(fns) == _PIN_FN_NAMES, f"{path.name}: missing pin function(s)"
        per_file[path.name] = fns
    assert per_file["dispatch_issue.py"] == per_file["backend_poll.py"], (
        "the #987 bootstrap is duplicated BY DESIGN — keep the two copies "
        "source-identical (edit both files together)"
    )


# ---------------------------------------------------------------------------
# 8. Integration: the shipped scripts run as __main__ from a non-main cwd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_path", _SCRIPT_PATHS, ids=_MODULE_IDS)
def test_script_help_runs_as_main_from_non_main_cwd(script_path, tmp_path):
    """``python <script> --help`` from a bare tmp cwd exits 0.

    The module-top pin runs BEFORE argparse, so rc 0 proves the shipped
    ``__main__`` bootstrap executes cleanly (anchor = the script file's own
    dir, never the cwd) — closing the 'no automated test runs the shipped
    bootstrap' gap.
    """
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(tmp_path),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
