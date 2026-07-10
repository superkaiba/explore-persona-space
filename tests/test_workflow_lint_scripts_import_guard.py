"""Tests for ``workflow_lint --check-scripts-import-guard`` (#823/#853).

The check FAILs any ``scripts.*`` import under
``src/explore_persona_space/experiments/**`` AND ``scripts/**`` (the latter
widened by #1229) — deferred (function-body) AND
module-top-level — lacking a repo-root ``sys.path`` guard: in script mode
(``python /abs/path/driver.py``) ``sys.path[0]`` is the script's own
directory, so an unguarded import raises ``ModuleNotFoundError`` pod/GCE-side
(incident #823 Phase-3: a deferred ``from scripts.issue779_collect import``
killed a full GCE launch after ~30 min of paid work; the #853 fix was
documentation-only — this check is the mechanical enforcement).

Covers, per plan §6 (test matrix rows 1-21): (i) deferred + top-level
offenders firing with position-specific diagnostics (rows 1-2, 19); (ii) the
guard-position rules — same-innermost-function PRECEDING guard, module-level
guard covering deferred imports, module-level guard NOT covering an earlier
top-level import, guard-after-import still firing (rows 3-7); (iii) the
pruned scope walk — a guard call inside a nested def is NOT module evidence
(row 8), and module-executing offender detection uses the SAME pruned walk so
a ``try/except ImportError``-wrapped module import and the
``if __name__ == "__main__":`` main-block hoist evasion both fire (rows
17-18, the round-1 critic Must-Fix fixtures); (iv) ``TYPE_CHECKING``
body-only skip precision (rows 9, 20); (v) the
``# SCRIPTS_IMPORT_GUARD_EXEMPT`` waiver in BOTH placements + the ≥10-char
reason boundary (rows 10-11, 21); (vi) out-of-scope files (``backends/``
only — ``scripts/`` is IN scope as of #1229) and prefix non-matches ignored
(rows 12-13); (vii) an
unparseable file is skipped with a printed notice, never a crash or a flag
(row 14); (viii) the live tree passes — locks today's tree, where
``run_823.py:1191`` and ``run_952.py:1965`` are both guarded (row 15); (ix)
the MUTATION-VISIBLE no-flags DISPATCH test (the
``tests/test_workflow_lint.py:3455`` pattern) — a direct call of the check
function is NOT sufficient evidence of bundling (row 16); (x) the #1229
``scripts/`` scan-root widening — offenders under ``scripts/`` fire, the
module-top-bootstrap convention + the waiver pass there, BOTH default roots
are mutation-visible, and the AST-presence fast path returns ``[]`` (not
None / crash) on import-free + TYPE_CHECKING-only scripts/ files.
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
from workflow_lint import (  # noqa: E402
    SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS,
    check_scripts_import_guard,
)

_EXP = "src/explore_persona_space/experiments"


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _run_on(monkeypatch, tmp_path: Path) -> list[str]:
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    return check_scripts_import_guard()


# --------------------------------------------------------------------------
# rows 1-2: unguarded deferred + top-level offenders fire
# --------------------------------------------------------------------------


def test_unguarded_deferred_from_import_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/offender.py",
        "def run():\n    from scripts.foo import bar\n    return bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "offender.py:2" in errors[0], errors
    assert "scripts-import-guard" in errors[0], errors
    assert "deferred" in errors[0], errors


def test_unguarded_toplevel_import_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/offender_top.py",
        "import scripts.foo\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "offender_top.py:1" in errors[0], errors
    assert "module-top-level" in errors[0], errors


# --------------------------------------------------------------------------
# rows 3-7: guard-position rules
# --------------------------------------------------------------------------


def test_guard_after_import_still_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/late_guard.py",
        "def run():\n"
        "    from scripts.foo import bar\n"
        "    _ensure_repo_root_on_syspath()\n"
        "    return bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "late_guard.py:2" in errors[0], errors


def test_guarded_same_function_passes(tmp_path, monkeypatch) -> None:
    """The run_823.py/run_952.py exemplar shape: a syspath-named guard call
    immediately before the deferred import, same function."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/guarded.py",
        "def _ensure_repo_root_on_syspath():\n"
        "    import sys\n"
        '    sys.path.insert(0, "/repo")\n'
        "\n"
        "def run():\n"
        "    _ensure_repo_root_on_syspath()\n"
        "    from scripts.foo import bar\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_literal_syspath_insert_passes(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/literal_insert.py",
        "import sys\n"
        "def run():\n"
        '    sys.path.insert(0, "/repo")\n'
        "    from scripts.foo import bar\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_module_level_guard_covers_deferred_passes(tmp_path, monkeypatch) -> None:
    """A module-top bootstrap (the issue_331_phase0_panel.py convention)
    covers every deferred import regardless of line order — the module body
    executes fully before any post-import function call."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/module_guard.py",
        "import sys\n"
        'sys.path.insert(0, "/repo")\n'
        "\n"
        "def run():\n"
        "    from scripts.foo import bar\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_module_guard_after_toplevel_import_fires(tmp_path, monkeypatch) -> None:
    """A top-level import needs a PRECEDING module-scope guard (the module
    body executes in order)."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/guard_too_late.py",
        'import sys\nimport scripts.foo\n\n\nsys.path.insert(0, "/repo")\n',
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "guard_too_late.py:2" in errors[0], errors
    assert "module-top-level" in errors[0], errors


# --------------------------------------------------------------------------
# row 8: pruned scope walk — a guard inside a nested def is NOT module
# evidence
# --------------------------------------------------------------------------


def test_guard_inside_nested_def_not_module_evidence(tmp_path, monkeypatch) -> None:
    """A naive ast.walk over the module body would count the guard call
    inside ``helper`` as module-level evidence — wrong: it does not execute
    when the def statement runs."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/nested_guard.py",
        "import sys\n"
        "\n"
        "def helper():\n"
        '    sys.path.insert(0, "/repo")\n'
        "\n"
        "def run():\n"
        "    from scripts.foo import bar\n"
        "    return bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "nested_guard.py:7" in errors[0], errors


# --------------------------------------------------------------------------
# rows 9 + 20: TYPE_CHECKING skip — body only, orelse + `if not` stay in
# scope
# --------------------------------------------------------------------------


def test_type_checking_import_passes(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/tc_only.py",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from scripts.foo import Bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_type_checking_orelse_fires(tmp_path, monkeypatch) -> None:
    """Only the ``if TYPE_CHECKING:`` BODY is skipped — the ``else`` branch
    DOES execute at runtime and stays in scope."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/tc_orelse.py",
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    pass\n"
        "else:\n"
        "    from scripts.foo import bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "tc_orelse.py:6" in errors[0], errors


def test_not_type_checking_body_fires(tmp_path, monkeypatch) -> None:
    """``if not TYPE_CHECKING:`` does not match the test predicate, so its
    body correctly stays in scope."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/tc_not.py",
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if not TYPE_CHECKING:\n"
        "    from scripts.foo import bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "tc_not.py:4" in errors[0], errors


# --------------------------------------------------------------------------
# rows 10-11 + 21: the waiver, both placements + the reason-length boundary
# --------------------------------------------------------------------------


def test_waiver_same_line_passes(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/waived_same_line.py",
        "def run():\n"
        "    from scripts.foo import bar"
        "  # SCRIPTS_IMPORT_GUARD_EXEMPT: launcher inserts repo root pre-exec\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_waiver_preceding_line_passes(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/waived_preceding.py",
        "def run():\n"
        "    # SCRIPTS_IMPORT_GUARD_EXEMPT: launcher inserts repo root pre-exec\n"
        "    from scripts.foo import bar\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_short_reason_waiver_still_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/short_waiver.py",
        "def run():\n"
        "    from scripts.foo import bar  # SCRIPTS_IMPORT_GUARD_EXEMPT: short\n"
        "    return bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "short_waiver.py:2" in errors[0], errors


def test_waiver_exact_min_reason_passes(tmp_path, monkeypatch) -> None:
    """The >= boundary: a reason of EXACTLY
    SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS chars passes."""
    reason = "x" * SCRIPTS_IMPORT_GUARD_WAIVER_MIN_REASON_CHARS
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/exact_waiver.py",
        "def run():\n"
        f"    from scripts.foo import bar  # SCRIPTS_IMPORT_GUARD_EXEMPT: {reason}\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# rows 12-13: scope + prefix non-matches
# --------------------------------------------------------------------------


def test_backends_out_of_scope_ignored(tmp_path, monkeypatch) -> None:
    """backends/ (entrypoint bootstraps, #987) stays out of scope — same
    offender body, no flag. (``scripts/`` is IN scope as of #1229 — see the
    scripts/-root tests below.)"""
    offender = "def run():\n    from scripts.foo import bar\n    return bar\n"
    _plant(tmp_path, "src/explore_persona_space/backends/x.py", offender)
    assert _run_on(monkeypatch, tmp_path) == []


def test_non_scripts_deferred_import_passes(tmp_path, monkeypatch) -> None:
    """A prefix non-match (``scripts_helper``) is NOT a scripts import; nor
    is any other deferred import."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/benign.py",
        "def run():\n"
        "    from pathlib import Path\n"
        "    import scripts_helper\n"
        "    return Path, scripts_helper\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# #1229: the scripts/ scan root — offenders fire, the module-top-bootstrap
# convention + the waiver pass, both default roots are mutation-visible,
# and the AST-presence fast path returns [] on import-free files
# --------------------------------------------------------------------------


def test_scripts_root_deferred_offender_fires(tmp_path, monkeypatch) -> None:
    """An unguarded deferred offender directly under ``scripts/`` fires with
    the same position-specific diagnostic as the experiments/ root."""
    _plant(
        tmp_path,
        "scripts/x.py",
        "def run():\n    from scripts.foo import bar\n    return bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "scripts/x.py:2" in errors[0], errors
    assert "deferred" in errors[0], errors


def test_scripts_root_module_top_bootstrap_passes(tmp_path, monkeypatch) -> None:
    """The scripts/ module-top-bootstrap convention (the
    ``issue_331_phase0_panel.py`` exemplar shape) is accepted as guard
    evidence for BOTH a later top-level import and a deferred one — the
    #1175 position rules already handle it, which is why #1229's widening
    needed no predicate change."""
    _plant(
        tmp_path,
        "scripts/guarded_tool.py",
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
        "\n"
        "from scripts.foo import bar\n"
        "\n"
        "def run():\n"
        "    from scripts.baz import qux\n"
        "    return bar, qux\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_waiver_under_scripts_root_passes(tmp_path, monkeypatch) -> None:
    """The ``# SCRIPTS_IMPORT_GUARD_EXEMPT`` waiver works identically under
    the scripts/ root."""
    _plant(
        tmp_path,
        "scripts/waived.py",
        "def run():\n"
        "    from scripts.foo import bar"
        "  # SCRIPTS_IMPORT_GUARD_EXEMPT: launcher inserts repo root pre-exec\n"
        "    return bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_default_roots_cover_both_trees(tmp_path, monkeypatch) -> None:
    """Mutation-visible for the default scan-root tuple: one unguarded
    DEFERRED offender under experiments/ + one unguarded TOP-LEVEL
    (module-executing) offender under scripts/ — dropping EITHER root from
    the production default fails this test, and the scripts/ plant pins the
    module-executing offender class under that root directly."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/offender.py",
        "def run():\n    from scripts.foo import bar\n    return bar\n",
    )
    _plant(tmp_path, "scripts/offender2.py", "import scripts.foo\n")
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 2, errors
    joined = "\n".join(errors)
    assert "offender.py:2" in joined, errors
    assert "scripts/offender2.py:1" in joined, errors
    assert "module-top-level" in joined and "deferred" in joined, errors


def test_import_free_scripts_file_passes(tmp_path, monkeypatch) -> None:
    """Fast-path smoke (#1229): a scripts/ file with NO scripts.* import
    node takes the AST-presence early return and yields ``[]`` (never None,
    never a crash); a TYPE_CHECKING-ONLY scripts/ file HAS a matching import
    node, so it passes the fast path and the full scan still yields ``[]``
    (the pruned-region soundness case)."""
    _plant(
        tmp_path,
        "scripts/no_imports.py",
        "import os\n\n# scripts token: force past the cheap text pre-gate\nprint(os.sep)\n",
    )
    _plant(
        tmp_path,
        "scripts/tc_only_tool.py",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from scripts.foo import Bar\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# row 14: unparseable file — skip-with-report, never crash, never flag
# --------------------------------------------------------------------------


def test_unparseable_file_skipped_with_notice(tmp_path, monkeypatch, capsys) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/broken.py",
        "def broken(:\n    from scripts.foo import bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert errors == []
    err = capsys.readouterr().err
    assert "--check-scripts-import-guard skipped unparseable" in err, err
    assert "broken.py" in err and "SyntaxError" in err, err


# --------------------------------------------------------------------------
# rows 17-18: module-executing detection uses the SAME pruned scope walk as
# guard detection (round-1 critic Must-Fix fixtures)
# --------------------------------------------------------------------------


def test_try_except_wrapped_module_import_fires(tmp_path, monkeypatch) -> None:
    """A try/except ImportError-wrapped module import EXECUTES at module
    scope and is IN scope; the try/except is NOT a guard (the fallback
    silently takes the wrong path pod-side — fail-fast rule). A flat
    ``tree.body`` scan would let this shape pass silently."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/try_wrapped.py",
        "try:\n    from scripts.foo import bar\nexcept ImportError:\n    bar = None\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "try_wrapped.py:2" in errors[0], errors
    assert "module-top-level" in errors[0], errors


def test_main_block_import_fires(tmp_path, monkeypatch) -> None:
    """The ``if __name__ == "__main__":`` main-block hoist evasion is
    closed — a main-block import is module-executing."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/main_block.py",
        'if __name__ == "__main__":\n    import scripts.foo\n',
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "main_block.py:2" in errors[0], errors


def test_main_block_import_after_module_guard_passes(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/main_block_guarded.py",
        "import sys\n"
        'sys.path.insert(0, "/repo")\n'
        "\n"
        'if __name__ == "__main__":\n'
        "    import scripts.foo\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# row 19: node-type x position cross-cells + the bare-name case
# --------------------------------------------------------------------------


def test_deferred_plain_import_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/deferred_plain.py",
        "def run():\n    import scripts.foo\n    return scripts.foo\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "deferred_plain.py:2" in errors[0], errors
    assert "deferred" in errors[0], errors


def test_toplevel_from_import_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/top_from.py",
        "from scripts.foo import bar\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "top_from.py:1" in errors[0], errors
    assert "module-top-level" in errors[0], errors


def test_bare_import_scripts_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/bare_scripts.py",
        "def run():\n    import scripts\n    return scripts\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "bare_scripts.py:2" in errors[0], errors
    assert "deferred" in errors[0], errors


def test_multi_alias_import_fires(tmp_path, monkeypatch) -> None:
    """``import os, scripts.foo`` lacks the "import scripts" bigram — the
    textual pre-gate must not skip it (code-review r1 Minor: gate on the
    bare "scripts" token, not the two verbatim bigrams)."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/multi_alias.py",
        "def run():\n    import os, scripts.foo\n    return os\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "multi_alias.py:2" in errors[0], errors
    assert "deferred" in errors[0], errors


# --------------------------------------------------------------------------
# row 15: live tree passes
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """The real repo carries zero un-waived offenders — locks today's tree
    (run_823.py:1191 and run_952.py:1965 are both guarded by
    ``_ensure_repo_root_on_syspath()``). A new unguarded site must be
    GUARDED (the canonical fix) or explicitly waived, never allowlisted."""
    assert check_scripts_import_guard() == []


# --------------------------------------------------------------------------
# row 16: the MUTATION-VISIBLE no-flags DISPATCH test (the
# tests/test_workflow_lint.py:3455 pattern)
# --------------------------------------------------------------------------


def test_check_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #823/#853 check —
    deleting its ``or no_flags`` branch must fail this test
    (mutation-visible), closing the dead-tripwire gap where all direct-call
    tests stay green while the CLI never runs the check. Other bundled
    checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + the offending
    path."""
    _plant(
        tmp_path,
        f"{_EXP}/issue_x/offender.py",
        "def run():\n    from scripts.foo import bar\n    return bar\n",
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "scripts-import-guard" in err and "offender.py" in err, (
        f"the scripts-import-guard diagnostic (naming offender.py) is missing "
        f"from the no-flags default run's stderr — the check is not bundled "
        f"into no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
