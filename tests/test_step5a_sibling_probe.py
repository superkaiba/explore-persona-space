"""Unit tests for ``scripts/step5a_sibling_probe.py`` (#2208, #2412).

Hermetic scratch git fixtures (no ``uv`` needed: the collection path injects
``--collect-cmd``). Each fixture builds a scratch repo whose ``origin/main``
ref diverges from HEAD/worktree exactly like a fork-era issue worktree, then
replays the Step 5a sync (``git checkout origin/main -- <path>``) and runs the
helper by CONSTRUCTED PATH (subprocess CLI + importlib — the
``TRANSITIVE_CONSUMER_TESTS`` registration rationale in the Step 9c test
selector: no text-scan selector arm reaches a constructed-path consumer;
the contiguous selector path is deliberately NOT written here, so the
selector's literal dependency arm does not adopt this file into the
case-86 live-tree pin).

Scratch dirs use ``tempfile.mkdtemp`` (NOT ``tmp_path``): concurrent pytest
sessions prune ``/tmp/pytest-of-*`` numbered roots mid-test, which races the
subprocess-heavy scratch repos here.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_PATH = REPO_ROOT / "scripts" / "step5a_sibling_probe.py"

_EXP = "src/explore_persona_space/experiments"
_TRIVIAL_TEST = "def test_ok():\n    assert True\n"

# Structural MF1 fixture: the immediate child (the ``uv`` analog) spawns a
# grandchild (the ``pytest`` analog) that INHERITS stdout — both ignore
# SIGTERM, so only a process-GROUP SIGKILL escalation can close the pipe.
_HANG_WITH_GRANDCHILD = """\
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
# The grandchild inherits this process's stdout (the helper's pipe) — the
# structural `uv run pytest` process-tree shape (MF1).
subprocess.Popen(
    [
        sys.executable,
        "-c",
        (
            "import os, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open(sys.argv[1], 'w').write(str(os.getpid())); "
            "time.sleep(600)"
        ),
        {pidfile!r},
    ]
)
time.sleep(600)
"""


@pytest.fixture()
def scratch():
    d = Path(tempfile.mkdtemp(prefix="eps2412-probe-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _git_rc(repo: Path, *args: str) -> int:
    return _git(repo, *args, check=False).returncode


def _write(repo: Path, rel: str, content: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _make_repo(
    root: Path,
    origin_files: dict[str, str],
    branch_overrides: dict[str, str | None],
    synced: list[str],
) -> Path:
    """Scratch repo: origin/main state, branch-era drift, then the Step 5a sync.

    ``branch_overrides``: rel -> new content, or None to make the file ABSENT
    at branch era (so a synced copy of it is main-NEW: staged + present in the
    tree but absent from HEAD, exactly what ``git checkout origin/main -- f``
    produces in the real arm).
    """
    repo = root
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], capture_output=True, check=True)
    _git(repo, "config", "user.email", "probe-test@example.com")
    _git(repo, "config", "user.name", "probe test")
    _git(repo, "config", "commit.gpgsign", "false")
    for rel, content in origin_files.items():
        _write(repo, rel, content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "origin state")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    if branch_overrides:
        for rel, content in branch_overrides.items():
            if content is None:
                (repo / rel).unlink()
            else:
                _write(repo, rel, content)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "branch-era drift")
    for rel in synced:
        _git(repo, "checkout", "origin/main", "--", rel)
    return repo


def _run_helper(
    repo: Path,
    synced: list[str],
    *,
    collect_cmd: str = "true",
    warmup_cmd: str = "true",
    collect_timeout: float = 30,
    kill_after: float = 5,
    kept_out: Path | None = None,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(HELPER_PATH), "--worktree", str(repo)]
    if kept_out is not None:
        cmd += ["--kept-out", str(kept_out)]
    cmd += [
        "--collect-cmd",
        collect_cmd,
        "--warmup-cmd",
        warmup_cmd,
        "--collect-timeout",
        str(collect_timeout),
        "--kill-after",
        str(kill_after),
        "--",
        *synced,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("step5a_sibling_probe_under_test", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_reverted_main_new(repo: Path, rel: str) -> None:
    """A main-NEW synced file was dropped from tree AND index (``git rm``)."""
    assert not (repo / rel).exists(), f"{rel} still in the working tree"
    ls = _git(repo, "ls-files", "--", rel)
    assert ls.stdout.strip() == "", f"{rel} still in the index"


def _assert_reverted_branch_era(repo: Path, rel: str, head_content: str) -> None:
    """A branch-era synced file was restored to HEAD content, tree + index clean."""
    assert (repo / rel).read_text() == head_content
    assert _git_rc(repo, "diff", "--quiet", "HEAD", "--", rel) == 0
    assert _git_rc(repo, "diff", "--cached", "--quiet", "HEAD", "--", rel) == 0


def _assert_kept_synced(repo: Path, rel: str) -> None:
    """A kept synced file still carries origin/main content and stays staged."""
    assert (repo / rel).exists(), f"{rel} was reverted but should be KEPT"
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", rel) == 0
    ls = _git(repo, "ls-files", "--", rel)
    assert ls.stdout.strip() == rel, f"{rel} not in the index"


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state in ("Z", "X")


# ---------------------------------------------------------------------------
# importlib-level pins (constructed-path load — the TRANSITIVE_CONSUMER_TESTS
# rationale) for the m-extraction and the N8 classifier anchoring.
# ---------------------------------------------------------------------------


def test_issue_number_extraction_uses_full_path():
    mod = _load_helper_module()
    # The latent basename bug the plan names: `arms.py` has a digit-free
    # basename — extraction must read the FULL relative path.
    assert mod.issue_number("src/explore_persona_space/experiments/issue_1739/arms.py") == "1739"
    assert mod.issue_number("src/explore_persona_space/experiments/issue2203/caphook.py") == "2203"
    assert mod.issue_number("tests/test_issue99_foo.py") == "99"
    assert mod.issue_number("scripts/issue99_run.sh") == "99"
    assert mod.issue_number("scripts/helper.py") is None
    assert mod.is_synced_test("tests/test_issue99_foo.py")
    assert not mod.is_synced_test("scripts/issue99_run.py")


def test_classifier_anchoring_pin_owning_unit():
    """N8 anchoring: FULLMATCH on component stems, slug arm experiments/-scoped."""
    mod = _load_helper_module()
    unit = mod.owning_strict_unit
    assert (
        unit("src/explore_persona_space/experiments/issue_1739/fits.py")
        == "src/explore_persona_space/experiments/issue_1739"
    )
    # Trailing-slug convention: strict, at DIRECTORY grain (MF3), under
    # experiments/ only.
    assert (
        unit("src/explore_persona_space/experiments/behavior_testbed_545/corpora.py")
        == "src/explore_persona_space/experiments/behavior_testbed_545"
    )
    # issue_?\d+ fullmatch fires anywhere under src/ (dir grain).
    assert (
        unit("src/explore_persona_space/analysis/issue667/mod.py")
        == "src/explore_persona_space/analysis/issue667"
    )
    # `issue_763_cofit` does NOT fullmatch issue_?\d+ -> LENIENT routing.
    assert unit("src/explore_persona_space/analysis/issue_763_cofit.py") is None
    # A final .py component that fullmatches is strict at FILE grain.
    assert (
        unit("src/explore_persona_space/analysis/issue_763.py")
        == "src/explore_persona_space/analysis/issue_763.py"
    )
    # The trailing-slug arm does NOT apply outside experiments/.
    assert unit("src/explore_persona_space/analysis/foo_545/mod.py") is None
    # Legacy loose experiments files (i406_conditions.py) fullmatch neither arm.
    assert unit("src/explore_persona_space/experiments/i406_conditions.py") is None


# ---------------------------------------------------------------------------
# Static-scan verdict arms (end-to-end through the CLI on scratch repos).
# ---------------------------------------------------------------------------


def test_missing_module_fails_and_reverts_pair(scratch):
    """Module present at origin/main but MISSING in WT -> FAIL + pair revert."""
    test_rel = "tests/test_issue99_probe.py"
    script_rel = "scripts/issue99_run.py"
    fits_rel = f"{_EXP}/issue_99/fits.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{_EXP}/issue_99/__init__.py": "",
            fits_rel: "def gather():\n    return 1\n",
            test_rel: (
                "def test_uses_fits():\n"
                "    from explore_persona_space.experiments.issue_99.fits import gather\n"
                "    assert gather() == 1\n"
            ),
            script_rel: "print('run')\n",
        },
        branch_overrides={
            f"{_EXP}/issue_99/__init__.py": None,
            fits_rel: None,
            test_rel: None,
            script_rel: None,
        },
        synced=[test_rel, script_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel, script_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "static import scan" in cp.stdout
    assert "MISSING from worktree src" in cp.stdout
    assert "#2206" in cp.stdout
    assert "— reverting its issue-99 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    _assert_reverted_main_new(repo, script_rel)
    assert kept.read_text() == ""


def test_issue_src_skew_strict_fail_reverts_whole_issue(scratch):
    """Issue-namespaced src differing from origin/main -> strict FAIL, all
    three synced file types (test + script + src) reverted pair-atomically."""
    test_rel = "tests/test_issue99_skew.py"
    script_rel = "scripts/issue99_old.py"
    arms_rel = f"{_EXP}/issue_99/arms.py"  # digit-free basename: grouping is path-based
    fits_rel = f"{_EXP}/issue_99/fits.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{_EXP}/issue_99/__init__.py": "",
            fits_rel: "def gather():\n    return 2\n",
            arms_rel: "ARMS = [1]\n",
            test_rel: (
                "def test_uses_fits():\n"
                "    from explore_persona_space.experiments.issue_99.fits import gather\n"
                "    assert gather() == 2\n"
            ),
            script_rel: "VERSION = 2\n",
        },
        branch_overrides={
            # fork-era skew: symbol STILL PRESENT, only content differs — the
            # symbol-existence rule would KEEP; only the strict identity arm fires.
            fits_rel: "def gather():\n    return 999  # fork-era\n",
            arms_rel: None,
            test_rel: None,
            script_rel: "VERSION = 1\n",
        },
        synced=[test_rel, script_rel, arms_rel],  # fits.py NOT synced (sync skipped)
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel, script_rel, arms_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert f"issue-namespaced unit {_EXP}/issue_99 differs from origin/main" in cp.stdout
    assert "unsatisfiable" not in cp.stdout  # identity arm, not the symbol arm
    assert "— reverting its issue-99 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    _assert_reverted_main_new(repo, arms_rel)
    _assert_reverted_branch_era(repo, script_rel, "VERSION = 1\n")
    assert kept.read_text() == ""


def test_mf3_slug_submodule_skew_dir_grain_fails(scratch):
    """MF3: identical ``__init__.py`` + skewed sibling submodule + function-body
    ``from pkg import corpora`` -> the DIRECTORY-grain rule FAILs (reverts).

    Discriminating property: the resolved module path is the byte-identical
    ``__init__.py``, so the retired v2 FILE-grain rule would silently KEEP
    exactly this shape (the measured issue-699 state).
    """
    test_rel = "tests/test_issue545_reader.py"
    pkg = f"{_EXP}/behavior_testbed_545"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{pkg}/__init__.py": "",
            f"{pkg}/corpora.py": "def load():\n    return 'main'\n",
            test_rel: (
                "def test_reads_corpora():\n"
                "    from explore_persona_space.experiments.behavior_testbed_545"
                " import corpora\n"
                "    assert corpora.load()\n"
            ),
        },
        branch_overrides={
            f"{pkg}/corpora.py": "def load():\n    return 'fork'\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    # The silent-KEEP hole the dir grain closes: the resolved __init__.py is
    # byte-identical to origin/main while the sibling submodule skews.
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", f"{pkg}/__init__.py") == 0
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", pkg) == 1
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert f"issue-namespaced unit {pkg} differs from origin/main" in cp.stdout
    assert "— reverting its issue-545 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""


def test_slug_dir_identical_kept(scratch):
    """The same slug-package import with the dir content-identical -> KEPT."""
    test_rel = "tests/test_issue545_reader.py"
    pkg = f"{_EXP}/behavior_testbed_545"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{pkg}/__init__.py": "",
            f"{pkg}/corpora.py": "def load():\n    return 'main'\n",
            test_rel: (
                "def test_reads_corpora():\n"
                "    from explore_persona_space.experiments.behavior_testbed_545"
                " import corpora\n"
                "    assert corpora.load()\n"
            ),
        },
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)
    assert kept.read_text() == f"{test_rel}\n"


def test_mf3_parent_package_alias_child_skew_fails(scratch):
    """#2412 r2 (parent-package-strict-bypass): ``from ...experiments import
    behavior_testbed_545 as pkg`` resolves the SHARED ``experiments/__init__.py``,
    so the module-level strict arm never fires; the CHILD satisfied via
    submodule existence must itself route through the strict owning-unit
    identity arm -> a skewed slug child directory FAILs (reverts).

    Discriminating property (verified fail-pre-fix): round-1 code accepted the
    child on ``submodule_exists()`` existence ALONE — never diffed — so this
    exact topology was silently KEPT; the resolved parent ``__init__.py`` and
    the child's own ``__init__.py`` are both byte-identical, so neither the
    module-level strict arm nor the symbol arm can catch it.
    """
    test_rel = "tests/test_issue545_alias.py"
    pkg = f"{_EXP}/behavior_testbed_545"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{_EXP}/__init__.py": "",
            f"{pkg}/__init__.py": "",
            f"{pkg}/corpora.py": "def load():\n    return 'main'\n",
            test_rel: (
                "def test_alias_use():\n"
                "    from explore_persona_space.experiments"
                " import behavior_testbed_545 as pkg\n"
                "    assert pkg is not None\n"
            ),
        },
        branch_overrides={
            f"{pkg}/corpora.py": "def load():\n    return 'fork'\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    # The bypass topology: the resolved parent __init__.py AND the child's own
    # __init__.py are byte-identical; only a sibling submodule inside the
    # child dir skews.
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", f"{_EXP}/__init__.py") == 0
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", f"{pkg}/__init__.py") == 0
    assert _git_rc(repo, "diff", "--quiet", "origin/main", "--", pkg) == 1
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert f"issue-namespaced unit {pkg} differs from origin/main" in cp.stdout
    assert "— reverting its issue-545 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""


def test_parent_package_alias_child_identical_kept(scratch):
    """Control for the parent-package strict routing: the same alias-form
    import with the child dir content-identical to origin/main -> KEPT (the
    child routing must not over-revert healthy syncs)."""
    test_rel = "tests/test_issue545_alias.py"
    pkg = f"{_EXP}/behavior_testbed_545"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{_EXP}/__init__.py": "",
            f"{pkg}/__init__.py": "",
            f"{pkg}/corpora.py": "def load():\n    return 'main'\n",
            test_rel: (
                "def test_alias_use():\n"
                "    from explore_persona_space.experiments"
                " import behavior_testbed_545 as pkg\n"
                "    assert pkg is not None\n"
            ),
        },
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)
    assert kept.read_text() == f"{test_rel}\n"


def test_type_checking_guarded_import_kept(scratch):
    """N2 alpha: a TYPE_CHECKING-guarded import never executes -> KEPT even
    when the module is missing from the worktree (present at origin/main)."""
    test_rel = "tests/test_issue77_tc.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            "src/explore_persona_space/ghost_mod.py": "class GhostThing:\n    pass\n",
            "src/explore_persona_space/other_ghost.py": "class OtherThing:\n    pass\n",
            test_rel: (
                "import typing\n"
                "from typing import TYPE_CHECKING\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from explore_persona_space.ghost_mod import GhostThing\n"
                "if typing.TYPE_CHECKING:\n"
                "    from explore_persona_space.other_ghost import OtherThing\n"
                "\n" + _TRIVIAL_TEST
            ),
        },
        branch_overrides={
            "src/explore_persona_space/ghost_mod.py": None,
            "src/explore_persona_space/other_ghost.py": None,
            test_rel: None,
        },
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)
    assert kept.read_text() == f"{test_rel}\n"


def test_import_error_guarded_absent_module_kept(scratch):
    """N2 beta: try/except ImportError (tuple form) around a worktree-missing
    module handles absence gracefully -> KEPT (the MISSING arm is exempt)."""
    test_rel = "tests/test_issue77_guarded.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            "src/explore_persona_space/optional_mod.py": "def maybe():\n    return 1\n",
            test_rel: (
                "try:\n"
                "    from explore_persona_space.optional_mod import maybe\n"
                "except (RuntimeError, ImportError):\n"
                "    maybe = None\n"
                "\n" + _TRIVIAL_TEST
            ),
        },
        branch_overrides={
            "src/explore_persona_space/optional_mod.py": None,
            test_rel: None,
        },
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


def test_import_error_guard_does_not_exempt_skewed_slug_package(scratch):
    """N2 split pin: the SAME guard around a PRESENT-but-skewed slug package
    still FAILs strictly — the import succeeds, so the guard never fires and
    cannot protect the #2204 skew class."""
    test_rel = "tests/test_issue545_guarded.py"
    pkg = f"{_EXP}/behavior_testbed_545"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{pkg}/__init__.py": "",
            f"{pkg}/corpora.py": "def load():\n    return 'main'\n",
            test_rel: (
                "def test_guarded_use():\n"
                "    try:\n"
                "        from explore_persona_space.experiments.behavior_testbed_545"
                " import corpora\n"
                "    except ImportError:\n"
                "        corpora = None\n"
                "    assert corpora is None or corpora.load()\n"
            ),
        },
        branch_overrides={
            f"{pkg}/corpora.py": "def load():\n    return 'fork'\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert f"issue-namespaced unit {pkg} differs from origin/main" in cp.stdout
    _assert_reverted_main_new(repo, test_rel)


@pytest.mark.parametrize("defkw", ["def", "async def"])
def test_import_error_guard_not_inherited_by_deferred_function_body(scratch, defkw):
    """#2412 r2 (deferred-importerror-guard-leak): a module-level try/except
    ImportError around a ``def`` guards only function CREATION — the body's
    import runs later, at call time, with the handler out of scope — so the
    MISSING-arm exemption must NOT leak into the deferred body (FAIL +
    pair-atomic revert).

    Discriminating property (verified fail-pre-fix): round-1 recursion carried
    ``ie`` unchanged into FunctionDef/AsyncFunctionDef bodies, so this exact
    topology consumed the beta exemption and was silently KEPT.
    """
    test_rel = "tests/test_issue77_deferred.py"
    mod_rel = "src/explore_persona_space/optional_mod.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            mod_rel: "def maybe():\n    return 1\n",
            test_rel: (
                "try:\n"
                f"    {defkw} use_optional():\n"
                "        from explore_persona_space.optional_mod import maybe\n"
                "        return maybe()\n"
                "except ImportError:\n"
                "    use_optional = None\n"
                "\n" + _TRIVIAL_TEST
            ),
        },
        branch_overrides={mod_rel: None, test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert (
        "module explore_persona_space.optional_mod present at origin/main"
        " but MISSING from worktree src"
    ) in cp.stdout
    assert "— reverting its issue-77 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""


def test_type_checking_guard_inherited_by_function_body(scratch):
    """The deliberate ASYMMETRY of the #2412 r2 ie reset: ``tc``
    (TYPE_CHECKING) inheritance INTO a def stays — a TYPE_CHECKING-guarded def
    never exists at runtime, so its body imports never execute -> KEPT even
    when the imported module is missing from the worktree."""
    test_rel = "tests/test_issue77_tcdef.py"
    mod_rel = "src/explore_persona_space/ghost_mod.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            mod_rel: "class GhostThing:\n    pass\n",
            test_rel: (
                "from typing import TYPE_CHECKING\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    def helper():\n"
                "        from explore_persona_space.ghost_mod import GhostThing\n"
                "        return GhostThing\n"
                "\n" + _TRIVIAL_TEST
            ),
        },
        branch_overrides={mod_rel: None, test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)
    assert kept.read_text() == f"{test_rel}\n"


def test_origin_ref_error_fails_closed(scratch):
    """#2412 r2 (origin-has-git-error-fail-open): a broken/missing origin/main
    ref is UNDECIDABLE, never "absent at origin" -> FAIL + pair-atomic revert.

    Discriminating property (verified fail-pre-fix): round-1 ``origin_has``
    read ANY nonzero git rc as absent-at-origin, so under a broken ref a
    project-src missing-module import read as third-party (N7 skip) and the
    pair was silently KEPT.
    """
    test_rel = "tests/test_issue77_brokenref.py"
    mod_rel = "src/explore_persona_space/gone_mod.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            mod_rel: "def gone():\n    return 1\n",
            test_rel: (
                "def test_uses_gone():\n"
                "    from explore_persona_space.gone_mod import gone\n"
                "    assert gone()\n"
            ),
        },
        branch_overrides={mod_rel: None, test_rel: None},
        synced=[test_rel],
    )
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "origin/main probe failed for module explore_persona_space.gone_mod" in cp.stdout
    assert "— reverting its issue-77 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""


def test_origin_has_three_way_discrimination(scratch):
    """Unit pin for the #2412 r2 origin_has redesign: present -> True;
    absent-on-a-healthy-ref -> False; broken ref -> OriginProbeError
    (undecidable, never absent). Empirical basis (git 2.34): ``cat-file -e``
    exits 128 for BOTH an absent path and a broken ref, so the probe uses
    ``ls-tree`` (rc 0 + empty stdout = absent; rc != 0 = git error)."""
    mod = _load_helper_module()
    repo = _make_repo(
        scratch / "wt",
        origin_files={"src/present_mod.py": "X = 1\n"},
        branch_overrides={},
        synced=[],
    )
    ctx = mod._ScanContext(repo)
    assert ctx.origin_has("src/present_mod.py") is True
    assert ctx.origin_has("src/absent_mod.py") is False
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    ctx2 = mod._ScanContext(repo)
    with pytest.raises(mod.OriginProbeError):
        ctx2.origin_has("src/present_mod.py")


def test_both_absent_module_skipped_as_third_party(scratch):
    """N7: a module absent from BOTH the worktree and origin/main is a
    third-party/stdlib import -> SKIP, pair KEPT.

    Why the skip must never be "hardened" into a strict FAIL: a project-src
    module absent from BOTH trees would be a pre-existing origin/main red,
    not a sync artifact — while genuine third-party imports (numpy, pytest,
    ...) land in exactly this branch, so a strict FAIL here would revert
    every healthy sync that imports a third-party package.
    """
    test_rel = "tests/test_issue77_thirdparty.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            test_rel: (
                "import totally_thirdparty_pkg_2412\n"
                "from another_thirdparty_2412 import thing\n"
                "\n" + _TRIVIAL_TEST
            ),
        },
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


def test_shared_src_content_diff_with_symbol_kept(scratch):
    """Shared src differing in content but carrying the imported symbol -> KEPT
    (content diffs alone never fail shared src — old branches always differ)."""
    test_rel = "tests/test_issue88_shared.py"
    mod_rel = "src/explore_persona_space/util_shared.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            mod_rel: "def helper_fn():\n    return 'v2'\n\n\nEXTRA = 2\n",
            test_rel: (
                "def test_uses_helper():\n"
                "    from explore_persona_space.util_shared import helper_fn\n"
                "    assert helper_fn()\n"
            ),
        },
        branch_overrides={
            mod_rel: "def helper_fn():\n    return 'v1'\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


def test_shared_src_missing_symbol_fails(scratch):
    """The #2204 row-4 shape: a function-body import of a symbol absent from
    the branch-era shared module (present at origin/main) -> FAIL + revert."""
    test_rel = "tests/test_issue88_symbol.py"
    mod_rel = "src/explore_persona_space/util_shared.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            mod_rel: "def helper_fn():\n    return 'v2'\n",
            test_rel: (
                "def test_uses_helper():\n"
                "    from explore_persona_space.util_shared import helper_fn\n"
                "    assert helper_fn()\n"
            ),
        },
        branch_overrides={
            mod_rel: "def other_fn():\n    return 'v1'\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert "symbol helper_fn unsatisfiable in module explore_persona_space.util_shared" in cp.stdout
    assert "— reverting its issue-88 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)


def test_shared_package_submodule_import_satisfied(scratch):
    """``from pkg import helpers`` on a SHARED package binds nothing in
    ``__init__.py`` but resolves via submodule file existence -> KEPT."""
    test_rel = "tests/test_issue88_toolbox.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            "src/explore_persona_space/toolbox/__init__.py": "",
            "src/explore_persona_space/toolbox/helpers.py": "def h():\n    return 1\n",
            test_rel: (
                "def test_uses_toolbox():\n"
                "    from explore_persona_space.toolbox import helpers\n"
                "    assert helpers.h() == 1\n"
            ),
        },
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


def test_issue_namespaced_submodule_import_identical_kept(scratch):
    """The dominant real shape ``from ...experiments.issue_X import arms`` with
    the owning dir identical to origin/main -> KEPT (strict arm passes)."""
    test_rel = "tests/test_issue77_arms.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            f"{_EXP}/issue_77/__init__.py": "",
            f"{_EXP}/issue_77/arms.py": "ARMS = [1]\n",
            test_rel: (
                "def test_uses_arms():\n"
                "    from explore_persona_space.experiments.issue_77 import arms\n"
                "    assert arms.ARMS\n"
            ),
        },
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


@pytest.mark.parametrize("skewed_kind", ["lenient_loose_file", "strict_issue_dir"])
def test_classifier_anchoring_routes_behaviorally(scratch, skewed_kind):
    """N8 behavioral pin: an ``issue_763_cofit.py``-shaped loose file routes
    LENIENT (skew with symbol present -> KEPT); an ``analysis/issue667/``-shaped
    dir routes STRICT (same skew -> FAIL)."""
    if skewed_kind == "lenient_loose_file":
        mod_rel = "src/explore_persona_space/analysis/issue_763_cofit.py"
        import_line = "    from explore_persona_space.analysis.issue_763_cofit import cofit\n"
        test_rel = "tests/test_issue763_reader.py"
    else:
        mod_rel = "src/explore_persona_space/analysis/issue667/mod.py"
        import_line = "    from explore_persona_space.analysis.issue667.mod import cofit\n"
        test_rel = "tests/test_issue667_reader.py"
    origin_files = {
        mod_rel: "def cofit():\n    return 2\n",
        test_rel: "def test_uses_cofit():\n" + import_line + "    assert cofit()\n",
    }
    if skewed_kind == "strict_issue_dir":
        origin_files["src/explore_persona_space/analysis/issue667/__init__.py"] = ""
    repo = _make_repo(
        scratch / "wt",
        origin_files=origin_files,
        branch_overrides={
            # skew with the symbol PRESENT: only a strict routing can FAIL
            mod_rel: "def cofit():\n    return 1\n",
            test_rel: None,
        },
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    if skewed_kind == "lenient_loose_file":
        assert "reverting" not in cp.stdout
        _assert_kept_synced(repo, test_rel)
    else:
        assert "issue-namespaced unit src/explore_persona_space/analysis/issue667" in cp.stdout
        _assert_reverted_main_new(repo, test_rel)


@pytest.mark.parametrize("where", ["test", "module"])
def test_syntax_error_reverts(scratch, where):
    """AST SyntaxError on the test or a scanned module is undecidable -> revert.
    A module-side SyntaxError is NOT beta-exempt (an except-ImportError guard
    does not catch SyntaxError at import time)."""
    test_rel = "tests/test_issue88_syntax.py"
    mod_rel = "src/explore_persona_space/util_shared.py"
    if where == "test":
        origin_files = {test_rel: "def broken(:\n    pass\n"}
        branch_overrides: dict[str, str | None] = {test_rel: None}
    else:
        origin_files = {
            mod_rel: "def helper_fn():\n    return 'v2'\n",
            test_rel: (
                "try:\n"
                "    from explore_persona_space.util_shared import helper_fn\n"
                "except ImportError:\n"
                "    helper_fn = None\n"
                "\n" + _TRIVIAL_TEST
            ),
        }
        branch_overrides = {mod_rel: "def broken(:\n    pass\n", test_rel: None}
    repo = _make_repo(
        scratch / "wt",
        origin_files=origin_files,
        branch_overrides=branch_overrides,
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel])
    assert cp.returncode == 0, cp.stderr
    assert "syntax error" in cp.stdout
    assert "— reverting its issue-88 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)


# ---------------------------------------------------------------------------
# Collection-probe arm (injected --collect-cmd; no uv anywhere).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_cmd", ["false", "/nonexistent/definitely_missing_bin_2412"])
def test_collect_cmd_failure_reverts(scratch, bad_cmd):
    """Nonzero collection rc AND spawn error both FAIL -> revert (fail-safe)."""
    test_rel = "tests/test_issue99_probe.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={test_rel: _TRIVIAL_TEST},
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [test_rel], collect_cmd=bad_cmd, kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert "fails collection" in cp.stdout
    assert "— reverting its issue-99 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""


def test_childless_collect_timeout_reverts(scratch):
    """A hanging (childless) collection is killed at the fence -> FAIL/revert."""
    test_rel = "tests/test_issue99_probe.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={test_rel: _TRIVIAL_TEST},
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    hang_cmd = f'{sys.executable} -c "import time; time.sleep(600)"'
    t0 = time.monotonic()
    cp = _run_helper(repo, [test_rel], collect_cmd=hang_cmd, collect_timeout=2, kill_after=1)
    elapsed = time.monotonic() - t0
    assert cp.returncode == 0, cp.stderr
    assert elapsed <= 30, f"childless timeout took {elapsed:.1f}s"
    assert "fails collection" in cp.stdout
    _assert_reverted_main_new(repo, test_rel)


def test_mf1_grandchild_hang_fenced_kill(scratch):
    """MF1: a stdout-inheriting GRANDCHILD (the ``uv run pytest`` tree shape)
    hanging past the fence must not wedge the helper.

    Both fixture processes ignore SIGTERM (forcing the SIGKILL escalation) and
    sleep 600 s. A naive child-only kill (``subprocess.run(timeout=...)``)
    terminates only the immediate child; the grandchild keeps the stdout pipe
    open and the post-kill ``communicate()`` blocks until IT exits (~600 s) —
    mechanically failing the <=30 s wall-clock bound below. The process-group
    SIGTERM -> ``--kill-after`` -> SIGKILL fence returns in seconds.
    """
    test_rel = "tests/test_issue99_probe.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={test_rel: _TRIVIAL_TEST},
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    pidfile = scratch / "grandchild.pid"
    hang_script = scratch / "hang_with_grandchild.py"
    hang_script.write_text(_HANG_WITH_GRANDCHILD.format(pidfile=str(pidfile)))
    kept = scratch / "kept.txt"
    t0 = time.monotonic()
    cp = _run_helper(
        repo,
        [test_rel],
        collect_cmd=f"{sys.executable} {hang_script}",
        collect_timeout=2,
        kill_after=1,
        kept_out=kept,
        timeout=120,  # test-side safety net only; the bound below is the assert
    )
    elapsed = time.monotonic() - t0
    assert cp.returncode == 0, cp.stderr
    assert elapsed <= 30, (
        f"helper took {elapsed:.1f}s — the MF1 wedge: a child-only kill leaves the"
        " grandchild holding the stdout pipe"
    )
    assert "fails collection" in cp.stdout
    assert "— reverting its issue-99 synced pair (#2208)." in cp.stdout
    _assert_reverted_main_new(repo, test_rel)
    assert kept.read_text() == ""
    assert pidfile.exists(), "fixture grandchild never started"
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not _pid_dead(pid):
        time.sleep(0.2)
    assert _pid_dead(pid), f"grandchild {pid} survived the process-group fence"


def test_warmup_failure_is_not_a_fail(scratch):
    """The warm-up is best-effort: its own failure never FAILs an issue."""
    test_rel = "tests/test_issue99_probe.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={test_rel: _TRIVIAL_TEST},
        branch_overrides={test_rel: None},
        synced=[test_rel],
    )
    cp = _run_helper(repo, [test_rel], warmup_cmd="false")
    assert cp.returncode == 0, cp.stderr
    assert "reverting" not in cp.stdout
    _assert_kept_synced(repo, test_rel)


# ---------------------------------------------------------------------------
# Revert mechanics, kept-out ordering, rc propagation (N6 / MF2 support).
# ---------------------------------------------------------------------------


def test_kept_out_lists_survivors_in_input_order(scratch):
    """One issue FAILs, one passes: the kept list carries exactly the
    survivors in input order, and kept files are never mutated."""
    t22 = "tests/test_issue22_ok.py"
    s22 = "scripts/issue22_util.py"
    t11 = "tests/test_issue11_bad.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            "src/explore_persona_space/gone11.py": "X = 1\n",
            t11: (
                "def test_uses_gone():\n"
                "    from explore_persona_space.gone11 import X\n"
                "    assert X\n"
            ),
            t22: _TRIVIAL_TEST,
            s22: "print('util')\n",
        },
        branch_overrides={
            "src/explore_persona_space/gone11.py": None,
            t11: None,
            t22: None,
            s22: None,
        },
        synced=[t22, t11, s22],
    )
    kept = scratch / "kept.txt"
    cp = _run_helper(repo, [t22, t11, s22], kept_out=kept)
    assert cp.returncode == 0, cp.stderr
    assert kept.read_text() == f"{t22}\n{s22}\n"
    _assert_reverted_main_new(repo, t11)
    _assert_kept_synced(repo, t22)
    _assert_kept_synced(repo, s22)


def test_failed_revert_propagates_nonzero_and_skips_kept_out(scratch):
    """N6: a failing revert git op raises (exit != 0) and the kept list is
    NEVER written — a partially-reverted state must not present a kept-list
    to the arm (the arm's else-branch then reverts everything, MF2)."""
    t11 = "tests/test_issue11_bad.py"
    ghost11 = "scripts/issue11_ghost_never_created.py"  # in no tree, no index
    t22 = "tests/test_issue22_ok.py"
    repo = _make_repo(
        scratch / "wt",
        origin_files={
            "src/explore_persona_space/gone11.py": "X = 1\n",
            t11: (
                "def test_uses_gone():\n"
                "    from explore_persona_space.gone11 import X\n"
                "    assert X\n"
            ),
            t22: _TRIVIAL_TEST,
        },
        branch_overrides={
            "src/explore_persona_space/gone11.py": None,
            t11: None,
            t22: None,
        },
        synced=[t11, t22],
    )
    kept = scratch / "kept.txt"
    kept.write_text("SENTINEL\n")
    # The ghost path makes the issue-11 revert loop's `git rm -f -q` fail
    # (pathspec matches nothing): the helper must fail LOUD, not masquerade
    # as success with a kept list.
    cp = _run_helper(repo, [t11, ghost11, t22], kept_out=kept)
    assert cp.returncode != 0
    assert "revert git op failed" in cp.stderr
    assert kept.read_text() == "SENTINEL\n", "kept-out written despite a failed revert"
