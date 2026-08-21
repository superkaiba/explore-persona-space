"""Tests for ``workflow_lint --check-prod-import-lockfile`` (#2253/#2223).

The check AST-scans every ``*.py`` under ``scripts/`` and ``src/`` and FAILs
any third-party import root not resolvable from ``uv.lock`` /
``pyproject.toml`` (incident #2223: an import of a package absent from the
lockfile killed a pod run at launch; #1336: the module-local-helper import
shape a smoke never executes).

Covers plan v3 §6: (i) branch-agnostic retention — smoke-conditional branches
are scanned, and adding the dists to the lock clears the FAILs; (ii) the
try/except ImportError body exemption in all its polarity details (Name /
tuple handler forms exempt; bare ``except:`` / ``except Exception:`` /
handler / orelse / finalbody / def-in-try-body NOT exempt); (iii) the
``TYPE_CHECKING`` body exclusion (orelse still scanned; ``typing.``
attribute form recognized); (iv) relative / stdlib / ``__future__`` /
first-party skips; (v) the :data:`IMPORT_TO_DIST` alias table incl. the
loud table-drift FAIL and the live-manifest subset pin (every table VALUE
present in the committed ``uv.lock``/``pyproject.toml`` universe, via the
check's own loader); (vi) the ``# PROD_IMPORT_LINT_EXEMPT`` waiver in both
placements, the ≥10-char reason boundary, and its PER-SITE (never per
``(file, root)``) scope; (vii) FAIL dedup to the first unexempt
``(file, root)`` site; (viii) the two WARN tiers (dangling issue-stem
first-party roots; extras/group-only dists) with the ``inline_lint_gate``
``NON_RED_PREFIXES`` leading-``WARN`` + path-free contract; (ix) PEP-508
name-prefix parsing, PEP-735 ``{include-group = ...}`` skip, and
``tool.uv.default-groups`` handling; (x) fail-loud missing/unparseable
manifests — all four legs of plan §6 row 14 (missing/malformed x
``uv.lock``/``pyproject.toml``, the malformed pair parameterized);
(xi) unparseable ``*.py`` skipped with a stderr note; (xii) the
live-tree stem-shadowing pin promised at
:func:`workflow_lint._first_party_import_roots`; (xiii) the
mutation-visible no-flags DISPATCH test (the
``test_workflow_lint_scripts_import_guard.py::test_check_bundled_in_no_flags``
pattern); (xiv) files-mode registry membership + chain position; (xv) the
live tree passes clean (post-#2253 waiver sweep).
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
    IMPORT_TO_DIST,
    PROD_IMPORT_LINT_WAIVER_MIN_REASON_CHARS,
    check_prod_import_lockfile,
)

_LOCK = """\
version = 1

[[package]]
name = "locked-dep"
version = "1.0.0"

[[package]]
name = "lock-only-dep"
version = "1.0.0"

[[package]]
name = "dev-dep"
version = "1.0.0"

[[package]]
name = "extra-only-dep"
version = "1.0.0"

[[package]]
name = "group-only-dep"
version = "1.0.0"
"""

_PYPROJECT = """\
[project]
name = "fixture"
version = "0.0.0"
dependencies = ["locked-dep>=1"]

[project.optional-dependencies]
viz = ["extra-only-dep"]

[dependency-groups]
dev = ["dev-dep"]
nondefault = ["group-only-dep"]
"""


def _run(
    tmp_path: Path,
    *files: tuple[str, str],
    lock: str = _LOCK,
    pyproject: str = _PYPROJECT,
) -> list[str]:
    """Plant *files* (rel-path, text) plus fixture manifests under
    ``tmp_path`` and run the check with every override hook pointed there."""
    for rel, text in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return check_prod_import_lockfile(
        scan_roots=(tmp_path / "scripts", tmp_path / "src"),
        lock_path=tmp_path / "uv.lock",
        pyproject_path=tmp_path / "pyproject.toml",
        repo_root=tmp_path,
    )


# ---------------------------------------------------------------- FAIL core


def test_unresolvable_import_fails(tmp_path: Path) -> None:
    errors = _run(tmp_path, ("scripts/mod.py", "import wickedlib_a2253\n"))
    assert len(errors) == 1, errors
    assert errors[0].startswith("scripts/mod.py:1:"), errors[0]
    assert "not resolvable from uv.lock/pyproject.toml" in errors[0]
    assert "wickedlib_a2253" in errors[0]


def test_src_root_scanned_too(tmp_path: Path) -> None:
    errors = _run(tmp_path, ("src/pkg/x.py", "import wickedlib_b2253\n"))
    assert len(errors) == 1, errors
    assert errors[0].startswith("src/pkg/x.py:1:"), errors[0]


def test_branch_agnostic_smoke_branch_scanned_then_lock_clears(tmp_path: Path) -> None:
    """Resolvability does not depend on reachability: BOTH sides of a
    smoke-conditional import FAIL; adding both dists to the lock clears
    both (the #1336 production-only-import blind spot, machine-caught)."""
    body = (
        "def run(smoke: bool):\n"
        "    if smoke:\n"
        "        import smokeonly_dep2253\n"
        "        return smokeonly_dep2253\n"
        "    import prodonly_dep2253\n"
        "    return prodonly_dep2253\n"
    )
    errors = _run(tmp_path, ("scripts/branchy.py", body))
    assert len(errors) == 2, errors
    assert any("smokeonly_dep2253" in e for e in errors), errors
    assert any("prodonly_dep2253" in e for e in errors), errors
    grown = _LOCK + (
        '\n[[package]]\nname = "smokeonly-dep2253"\nversion = "1.0.0"\n'
        '\n[[package]]\nname = "prodonly-dep2253"\nversion = "1.0.0"\n'
    )
    assert _run(tmp_path, ("scripts/branchy.py", body), lock=grown) == []


# ------------------------------------------------- try/except ImportError


def test_try_body_import_error_exempt(tmp_path: Path) -> None:
    body = "try:\n    import optdep2253\nexcept ImportError:\n    optdep2253 = None\n"
    assert _run(tmp_path, ("scripts/opt.py", body)) == []


def test_module_not_found_and_tuple_handler_exempt(tmp_path: Path) -> None:
    body = (
        "try:\n    import optdep2253\nexcept ModuleNotFoundError:\n    optdep2253 = None\n"
        "try:\n    import optdep2253b\nexcept (ValueError, ImportError):\n    optdep2253b = None\n"
    )
    assert _run(tmp_path, ("scripts/opt.py", body)) == []


def test_bare_except_and_exception_handler_not_exempt(tmp_path: Path) -> None:
    body = (
        "try:\n    import wicked_c2253\nexcept Exception:\n    wicked_c2253 = None\n"
        "try:\n    import wicked_d2253\nexcept:  # noqa: E722\n    wicked_d2253 = None\n"
    )
    errors = _run(tmp_path, ("scripts/broad.py", body))
    assert len(errors) == 2, errors


def test_handler_orelse_finalbody_not_exempt(tmp_path: Path) -> None:
    """An exception raised in handler/orelse/finalbody is NOT caught by that
    same try — those positions inherit only an OUTER protected region."""
    body = (
        "try:\n"
        "    x = 1\n"
        "except ImportError:\n"
        "    import wicked_h2253\n"
        "else:\n"
        "    import wicked_o2253\n"
        "finally:\n"
        "    import wicked_f2253\n"
    )
    errors = _run(tmp_path, ("scripts/positions.py", body))
    assert len(errors) == 3, errors


def test_outer_protected_try_covers_inner_positions(tmp_path: Path) -> None:
    """An OUTER try/except ImportError protects an inner try's handler —
    the inherited `protected` flag carries through non-body positions."""
    body = (
        "try:\n"
        "    try:\n"
        "        x = 1\n"
        "    except ValueError:\n"
        "        import optdep2253\n"
        "except ImportError:\n"
        "    optdep2253 = None\n"
    )
    assert _run(tmp_path, ("scripts/nested.py", body)) == []


def test_function_def_in_try_body_defers_outside_protection(tmp_path: Path) -> None:
    """A function DEFINED in a protected try body defers its imports to call
    time — outside the protected region (the #1336 SLURM-4684 shape)."""
    body = (
        "try:\n"
        "    def _load():\n"
        "        import wicked_deferred2253\n"
        "        return wicked_deferred2253\n"
        "except ImportError:\n"
        "    _load = None\n"
    )
    errors = _run(tmp_path, ("scripts/deferred.py", body))
    assert len(errors) == 1, errors
    assert "wicked_deferred2253" in errors[0]


# ------------------------------------------------------------ TYPE_CHECKING


def test_type_checking_body_excluded_orelse_scanned(tmp_path: Path) -> None:
    body = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import typeonly_dep2253\n"
        "else:\n"
        "    import wicked_else2253\n"
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    import typeonly_dep2253b\n"
    )
    errors = _run(tmp_path, ("scripts/tc.py", body))
    assert len(errors) == 1, errors
    assert "wicked_else2253" in errors[0]


# ----------------------------------------------------------------- skips


def test_relative_stdlib_future_and_first_party_skipped(tmp_path: Path) -> None:
    (tmp_path / "scripts" / "subdir").mkdir(parents=True)
    (tmp_path / "src" / "mypkg2253").mkdir(parents=True)
    (tmp_path / "scripts" / "sibling2253.py").write_text("X = 1\n", encoding="utf-8")
    body = (
        "from __future__ import annotations\n"
        "import os\n"
        "import json\n"
        "from . import x\n"
        "from .mod import y\n"
        "import explore_persona_space\n"
        "import scripts.foo\n"
        "import sibling2253\n"
        "import subdir\n"
        "import mypkg2253\n"
        "import conftest\n"
    )
    assert _run(tmp_path, ("scripts/skips.py", body)) == []


# ------------------------------------------------------------- alias table


def test_import_to_dist_alias_resolves(tmp_path: Path) -> None:
    lock = _LOCK + '\n[[package]]\nname = "pillow"\nversion = "1.0.0"\n'
    assert _run(tmp_path, ("scripts/img.py", "import PIL\n"), lock=lock) == []


def test_import_to_dist_table_drift_fails_loud(tmp_path: Path) -> None:
    """A table entry whose mapped dist is absent from the universe FAILs —
    table drift is loud, never a silent resolve (pillow NOT in fixture)."""
    errors = _run(tmp_path, ("scripts/img.py", "import PIL\n"))
    assert len(errors) == 1, errors
    assert "IMPORT_TO_DIST maps" in errors[0] and "pillow" in errors[0], errors[0]


# ------------------------------------------------------------------ waivers


def test_waiver_own_line_and_preceding_line(tmp_path: Path) -> None:
    body = (
        "import wicked_w2253  # PROD_IMPORT_LINT_EXEMPT: one-off install, documented reason\n"
        "# PROD_IMPORT_LINT_EXEMPT: one-off install, documented reason\n"
        "import wicked_w2253b\n"
        "# PROD_IMPORT_LINT_EXEMPT: one-off install, documented reason\n"
        "\n"
        "import wicked_w2253c\n"
    )
    assert _run(tmp_path, ("scripts/waived.py", body)) == []


def test_waiver_short_reason_not_honored(tmp_path: Path) -> None:
    assert PROD_IMPORT_LINT_WAIVER_MIN_REASON_CHARS == 10
    body = "import wicked_short2253  # PROD_IMPORT_LINT_EXEMPT: short\n"
    errors = _run(tmp_path, ("scripts/short.py", body))
    assert len(errors) == 1, errors


def test_waiver_is_per_site_never_per_file_root(tmp_path: Path) -> None:
    """Waiving site 1 does NOT waive site 2 of the same root in the same
    file — the FAIL lands on the first UNEXEMPT site."""
    body = (
        "# PROD_IMPORT_LINT_EXEMPT: one-off install, documented reason\n"
        "import wicked_p2253\n"
        "def f():\n"
        "    import wicked_p2253\n"
    )
    errors = _run(tmp_path, ("scripts/persite.py", body))
    assert len(errors) == 1, errors
    assert errors[0].startswith("scripts/persite.py:4:"), errors[0]


def test_fail_dedup_to_first_unexempt_site(tmp_path: Path) -> None:
    body = "import wicked_dd2253\n\ndef f():\n    import wicked_dd2253\n"
    errors = _run(tmp_path, ("scripts/dedup.py", body))
    assert len(errors) == 1, errors
    assert errors[0].startswith("scripts/dedup.py:1:"), errors[0]


# ---------------------------------------------------------------- WARN tiers


def test_extras_tier_warns_lead_with_warn_and_carry_no_paths(tmp_path: Path, capsys) -> None:
    body = "import extra_only_dep\nimport group_only_dep\n"
    assert _run(tmp_path, ("scripts/extras.py", body)) == []
    err = capsys.readouterr().err
    warns = [ln for ln in err.split("\n") if "check-prod-import-lockfile" in ln]
    assert len(warns) == 2, err
    for ln in warns:
        assert ln.startswith("WARN"), ln  # inline_lint_gate NON_RED_PREFIXES contract
        assert "extras.py" not in ln, ln  # path-free by design
    assert any("'viz'" in ln and "extra-only-dep" in ln for ln in warns), err
    assert any("'nondefault'" in ln and "group-only-dep" in ln for ln in warns), err


def test_dangling_issue_stem_warns_with_site_counts(tmp_path: Path, capsys) -> None:
    body = (
        "import issue9999_helper\nimport _issue9999_common\n\n"
        "def f():\n    import issue9999_helper\n"
    )
    assert _run(tmp_path, ("scripts/dangling.py", body)) == []
    err = capsys.readouterr().err
    assert "dangling first-party import root 'issue9999_helper' (2 site(s))" in err, err
    assert "dangling first-party import root '_issue9999_common' (1 site(s))" in err, err
    for ln in err.split("\n"):
        if "check-prod-import-lockfile" in ln:
            assert ln.startswith("WARN"), ln


def test_lock_only_and_default_group_resolve_silently(tmp_path: Path, capsys) -> None:
    """A lock-only transitive dist and a DEFAULT dependency-group dist both
    resolve with no WARN (dev is uv's documented default group)."""
    body = "import lock_only_dep\nimport dev_dep\n"
    assert _run(tmp_path, ("scripts/silent.py", body)) == []
    err = capsys.readouterr().err
    assert "check-prod-import-lockfile" not in err, err


def test_uv_default_groups_key_honored(tmp_path: Path, capsys) -> None:
    pyproject = (
        "[project]\n"
        'name = "fixture"\n'
        'version = "0.0.0"\n'
        "dependencies = []\n"
        "\n"
        "[tool.uv]\n"
        'default-groups = ["custom"]\n'
        "\n"
        "[dependency-groups]\n"
        'custom = ["custom-dep"]\n'
        'other = ["other-dep"]\n'
    )
    lock = (
        'version = 1\n\n[[package]]\nname = "custom-dep"\nversion = "1"\n'
        '\n[[package]]\nname = "other-dep"\nversion = "1"\n'
    )
    body = "import custom_dep\nimport other_dep\n"
    assert _run(tmp_path, ("scripts/groups.py", body), lock=lock, pyproject=pyproject) == []
    err = capsys.readouterr().err
    assert "custom_dep" not in err, err  # default group: silent
    assert "'other'" in err and "other_dep" in err, err  # non-default: WARN


def test_pep735_include_group_tables_skipped(tmp_path: Path) -> None:
    pyproject = (
        "[project]\n"
        'name = "fixture"\n'
        'version = "0.0.0"\n'
        'dependencies = ["locked-dep>=1"]\n'
        "\n"
        "[dependency-groups]\n"
        'dev = ["dev-dep", { include-group = "sub" }]\n'
        'sub = ["locked-dep"]\n'
    )
    body = "import locked_dep\nimport dev_dep\n"
    assert _run(tmp_path, ("scripts/incl.py", body), pyproject=pyproject) == []


def test_pep508_name_prefix_parsing(tmp_path: Path) -> None:
    """Extras markers / version specs / dot-normalization all reduce to the
    PEP-503 name prefix."""
    pyproject = (
        "[project]\n"
        'name = "fixture"\n'
        'version = "0.0.0"\n'
        'dependencies = ["Locked-Dep[extra]>=1.0", "Other.Pkg (==2.0) ; python_version >= \'3\'"]\n'
    )
    lock = (
        'version = 1\n\n[[package]]\nname = "locked-dep"\nversion = "1"\n'
        '\n[[package]]\nname = "other-pkg"\nversion = "2"\n'
    )
    body = "import locked_dep\nimport other_pkg\n"
    assert _run(tmp_path, ("scripts/pep508.py", body), lock=lock, pyproject=pyproject) == []


# ------------------------------------------------------- fail-loud manifests


def test_missing_uv_lock_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    errors = check_prod_import_lockfile(
        scan_roots=(tmp_path / "scripts",),
        lock_path=tmp_path / "uv.lock",
        pyproject_path=tmp_path / "pyproject.toml",
        repo_root=tmp_path,
    )
    assert len(errors) == 1, errors
    assert "uv.lock is MISSING" in errors[0], errors[0]


def test_missing_pyproject_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(_LOCK, encoding="utf-8")
    errors = check_prod_import_lockfile(
        scan_roots=(tmp_path / "scripts",),
        lock_path=tmp_path / "uv.lock",
        pyproject_path=tmp_path / "pyproject.toml",
        repo_root=tmp_path,
    )
    assert len(errors) == 1, errors
    assert "pyproject.toml is MISSING" in errors[0], errors[0]


@pytest.mark.parametrize(
    ("bad_file", "expected"),
    [
        ("uv.lock", "uv.lock is UNPARSEABLE"),
        ("pyproject.toml", "pyproject.toml is UNPARSEABLE"),
    ],
)
def test_unparseable_manifest_fails_loud(tmp_path: Path, bad_file: str, expected: str) -> None:
    """EACH manifest's UNPARSEABLE branch fails loud independently (plan §6
    row 14, the parameterized malformed legs): the OTHER manifest stays
    valid so the single error is attributable to the malformed one."""
    texts = {"uv.lock": _LOCK, "pyproject.toml": _PYPROJECT}
    texts[bad_file] = "[[package\nname="
    for name, text in texts.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    errors = check_prod_import_lockfile(
        scan_roots=(tmp_path / "scripts",),
        lock_path=tmp_path / "uv.lock",
        pyproject_path=tmp_path / "pyproject.toml",
        repo_root=tmp_path,
    )
    assert len(errors) == 1, errors
    assert expected in errors[0], errors[0]


def test_unparseable_py_skipped_with_note(tmp_path: Path, capsys) -> None:
    errors = _run(tmp_path, ("scripts/bad.py", "def broken(:\n"))
    assert errors == []
    err = capsys.readouterr().err
    assert "skipped unparseable" in err and "bad.py" in err, err


# -------------------------------------------------------------- live tree


def test_no_lock_dist_shadowed_by_script_stem() -> None:
    """The stem-shadowing residue promised at ``_first_party_import_roots``:
    no lock/declared dist's import-name form (``-`` -> ``_``) and no
    :data:`IMPORT_TO_DIST` key collides with a first-party root — except the
    project's OWN dist (`explore-persona-space`), whose first-party
    classification is correct by definition."""
    root = Path(wl._REPO_ROOT)
    lock, default_d, nondefault, errors = wl._load_import_lockfile_universe(
        root / "uv.lock", root / "pyproject.toml"
    )
    assert errors == [], errors
    universe = lock | default_d | set(nondefault)
    stems = wl._first_party_import_roots(root)
    shadowable = {d.replace("-", "_") for d in universe} | set(IMPORT_TO_DIST)
    collisions = sorted((stems & shadowable) - {"explore_persona_space"})
    assert collisions == [], (
        f"first-party stem(s) shadow a lockfile dist / IMPORT_TO_DIST key: {collisions} — "
        f"a third-party import of that root would silently classify first-party; rename "
        f"the script or fix the table (see _first_party_import_roots docstring)"
    )


def test_every_table_dist_present_in_live_lockfile() -> None:
    """Plan §6 row 11 (backs acceptance A4): every :data:`IMPORT_TO_DIST`
    VALUE, PEP-503-normalized, is present in the live declared-dist universe
    loaded from the committed ``uv.lock`` + ``pyproject.toml`` — via the
    check's own loader, never a second tomllib reader. Deterministic: a
    mapped dist dropped from the manifests fails HERE, naming the missing
    distribution(s), not only at the next full-tree lint scan of a file
    that happens to import it."""
    root = Path(wl._REPO_ROOT)
    lock, default_d, nondefault, errors = wl._load_import_lockfile_universe(
        root / "uv.lock", root / "pyproject.toml"
    )
    assert errors == [], errors
    universe = lock | default_d | set(nondefault)  # the check's FAIL universe
    missing = sorted({wl._pep503(dist) for dist in IMPORT_TO_DIST.values()} - universe)
    assert missing == [], (
        f"IMPORT_TO_DIST maps import root(s) to distribution(s) absent from the live "
        f"uv.lock/pyproject.toml universe: {missing} — table drift; fix the table entry or "
        f"declare + lock the dependency (check_prod_import_lockfile FAILs loud on the next "
        f"scan of any file importing them)"
    )


def test_live_tree_clean(capsys) -> None:
    """The real repo passes post-#2253 (offenders waived / try-guarded in the
    same change) — the landing-green requirement. ~2 min: dominated by
    parsing every scripts/ + src/ file once into the shared AST memo."""
    errors = check_prod_import_lockfile()
    assert errors == [], "\n".join(errors)
    err = capsys.readouterr().err
    warns = [ln for ln in err.split("\n") if ln.startswith("WARN: check-prod-import-lockfile")]
    danglers = [ln for ln in warns if "dangling first-party" in ln]
    extras = [ln for ln in warns if "non-default" in ln]
    assert len(danglers) == 7, "\n".join(warns)  # the 7 class-B dangling issue-stem roots
    assert len(extras) == 2, "\n".join(warns)  # umap/viz + liger_kernel/gpu


# -------------------------------------------------- registration + dispatch


def test_files_mode_registry_membership_and_chain_position() -> None:
    """Fail-closed files-mode registration (#2235): membership in BOTH
    registries, path-local kind, manifest surfaces named, and the runner
    sits directly after the dotenv check (chain order convention)."""
    assert "check_prod_import_lockfile" in wl.CHECK_SCOPES
    scope = wl.CHECK_SCOPES["check_prod_import_lockfile"]
    assert scope.kind == "path-local", scope
    assert "uv.lock" in scope.surfaces and "pyproject.toml" in scope.surfaces, scope
    assert "check_prod_import_lockfile" in wl._FILES_MODE_RUNNERS
    names = list(wl._FILES_MODE_RUNNERS)
    assert (
        names.index("check_prod_import_lockfile")
        == names.index("check_dotenv_before_hf_import") + 1
    ), names


def test_check_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test (mutation-visible). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on this check's own diagnostic token + the offender."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "offender2253.py").write_text(
        "import wicked_noflags2253\n", encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text(_LOCK, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "not resolvable from uv.lock/pyproject.toml" in err and "offender2253.py" in err, (
        f"the prod-import-lockfile diagnostic (naming offender2253.py) is missing from the "
        f"no-flags default run's stderr — the check is not bundled into no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
