"""Tests for the #2364 monkeypatched-constant checks in ``scripts/workflow_lint.py``.

Two checks under test:

- ``check_monkeypatched_constant_pinning`` — the WARN-only best-effort AST
  scan (``--check-monkeypatched-constant-pinning``): flags tests replacing a
  curated ALL_CAPS module-level production constant with a fixture while no
  test repo-wide reads the REAL constant in an assert-bearing function
  (smoke-blind-spots mechanism 4, incident #2360).
- ``check_production_constant_pinning_lens`` — the FAIL surface pin
  (``--check-production-constant-pinning-lens``, bundled into the no-flags
  default run): the lens must stay present across its SIX surfaces.

Fixtures reproduce the #2360 shape (reshaped, structurally faithful): a test
suite monkeypatching both curated constants its acceptance criterion depends
on, with the only real-constant test pinning call ORDERING via a source
substring — both shipped lists could be entirely empty with all tests green.

1.  ``test_flags_all_four_patch_forms_unpinned`` — A1: all four patch forms
    fire, exactly one WARN per hit site, each naming module.CONST.
2.  ``test_silent_with_real_contents_pinning_test`` — A2: a separate
    static completeness/subset pinning test suppresses (Attribute-load AND
    from-imported bare-Name-load pin forms).
3.  ``test_cross_import_style_key_match`` — A2 across import styles: pin via
    ``import a.b.c as alias``, hit via ``from a.b import c`` — the
    ``(module_basename, CONST)`` key matches.
4.  ``test_non_all_caps_and_test_module_targets_not_flagged`` — A3 proxy at
    lint grain: a patched function attr and a test-module constant are
    non-triggers.
5.  ``test_ordering_substring_test_still_warns`` — the #2360 near-miss: an
    ordering-via-source-substring test is NOT contents evidence.
6.  ``test_save_original_idiom_not_a_pin`` — ``orig = mod.CONST`` next to
    the patch does not count as a pin (save-assign exclusion).
7.  ``test_unrelated_assert_load_suppresses`` — the DISCLOSED
    false-suppression case, pinned as documented behavior.
8.  ``test_unparseable_test_file_warns_not_crashes`` — best-effort arm.
9.  ``test_returns_empty_fail_list_always`` — WARN-only contract.
10. ``test_pinning_lens_passes_on_complete_corpus`` — all six surfaces.
11. ``test_pinning_lens_fails_per_missing_surface`` — 12 parametrized drops.
12. ``test_pinning_lens_passes_on_live_tree`` — binds the landed edits.
13. ``test_check_production_constant_pinning_lens_bundled_in_no_flags`` —
    the two-part behavioral bundling pin (the sibling #2165 test's shape).

Round-2 revision fixtures (the two upheld round-1 BLOCKERs + minors):

14. ``test_fromimport_save_idiom_not_a_pin`` — the
    ``scanner-bare-name-save-false-suppression`` blocker: a from-imported
    bare-Name save (``orig = LOAD_BEARING_DISTS``) next to a patch of the
    same constant must NOT suppress the WARN.
15. ``test_non_utf8_test_file_warns_and_scanning_continues`` — the
    ``scanner-unicode-decode-crash`` blocker: non-UTF-8 bytes emit the
    unparseable WARN (never raise) and later files still scan.
16. ``test_assignment_read_in_non_patching_function_is_a_pin`` — the save
    exclusion is scoped to functions that also patch the same constant, so
    a separate ``actual = mod.CONST`` contents test suppresses.
17. ``test_non_monkeypatch_setattr_receiver_is_disclosed_over_approximation``
    — negative control pinning the disclosed receiver over-approximation.
18. ``test_single_char_constant_matched`` — the widened
    ``[A-Z][A-Z0-9_]*`` regex matches one-character constants.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    check_monkeypatched_constant_pinning,
    check_production_constant_pinning_lens,
)

_TAG = "production-constant-unpinned"
_MECH4 = "Substituted production constant (test-side)"
_ESCAPE = "none — no test substitutes a production constant"
_ROW_TOKEN = "monkeypatches a curated production constant"

# --------------------------------------------------------------------------
# Scanner fixtures (self-contained #2360 reshapes; the test files are only
# ast.parse'd, never executed, so ``preflight_mod`` need not exist).
# --------------------------------------------------------------------------

_FIXTURE_UNPINNED = """\
import preflight_mod
from unittest import mock


def test_load_bearing_object_form(monkeypatch):
    monkeypatch.setattr(preflight_mod, "LOAD_BEARING_DISTS", ["fixture-dist"])
    assert preflight_mod.check_dists() == []


def test_deep_import_string_form(monkeypatch):
    monkeypatch.setattr("preflight_mod.DEEP_IMPORT_MODULES", ["fixture.mod"])
    assert preflight_mod.check_imports() == []


def test_patch_object_form():
    with mock.patch.object(preflight_mod, "LOAD_BEARING_DISTS", []):
        assert preflight_mod.check_dists() == []


def test_patch_string_form():
    with mock.patch("preflight_mod.DEEP_IMPORT_MODULES", []):
        assert preflight_mod.check_imports() == []
"""

_FIXTURE_PINNING = """\
import preflight_mod


def test_load_bearing_dists_contents():
    assert "requests" in preflight_mod.LOAD_BEARING_DISTS
    assert len(preflight_mod.LOAD_BEARING_DISTS) >= 1


def test_deep_import_modules_contents():
    from preflight_mod import DEEP_IMPORT_MODULES

    assert DEEP_IMPORT_MODULES
"""

_FIXTURE_NONTRIGGER = """\
import preflight_mod
import tests.helpers as th


def test_patches_function(monkeypatch):
    monkeypatch.setattr(preflight_mod, "check_dists", lambda: [])
    assert preflight_mod.check_dists() == []


def test_patches_test_helper_constant(monkeypatch):
    monkeypatch.setattr(th, "FIXTURE_ROWS", [1])
    assert th.FIXTURE_ROWS
"""

_FIXTURE_ORDERING_ONLY = """\
import inspect

import preflight_mod


def test_check_ordering(monkeypatch):
    monkeypatch.setattr(preflight_mod, "LOAD_BEARING_DISTS", ["x"])
    src = inspect.getsource(preflight_mod)
    assert src.index("LOAD_BEARING_DISTS") < src.index("DEEP_IMPORT_MODULES")
"""

_FIXTURE_SAVE_IDIOM = """\
import preflight_mod


def test_save_and_patch(monkeypatch):
    orig = preflight_mod.LOAD_BEARING_DISTS
    monkeypatch.setattr(preflight_mod, "LOAD_BEARING_DISTS", ["fixture"])
    assert preflight_mod.check_dists() == []
    assert isinstance(orig, list)
"""

_FIXTURE_UNRELATED_ASSERT = """\
import preflight_mod


def test_uses_constant_for_expected_calls():
    expected = [f"check {d}" for d in preflight_mod.LOAD_BEARING_DISTS]
    assert isinstance(expected, list)
"""

_FIXTURE_ALIAS_PIN = """\
import explore_persona_space.orchestrate.preflight as pf


def test_contents():
    assert "requests" in pf.LOAD_BEARING_DISTS
"""

_FIXTURE_FROMIMPORT_HIT = """\
from explore_persona_space.orchestrate import preflight


def test_replaces(monkeypatch):
    monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ["fixture"])
    assert preflight.check_dists() == []
"""

_FIXTURE_SAVE_IDIOM_FROMIMPORT = """\
import preflight_mod
from preflight_mod import LOAD_BEARING_DISTS


def test_save_and_patch_fromimport(monkeypatch):
    orig = LOAD_BEARING_DISTS
    monkeypatch.setattr(preflight_mod, "LOAD_BEARING_DISTS", ["fixture"])
    assert preflight_mod.check_dists() == []
    assert isinstance(orig, list)
"""

_FIXTURE_LOCAL_VAR_CONTENTS = """\
import preflight_mod


def test_contents_via_local_variable():
    actual = preflight_mod.LOAD_BEARING_DISTS
    assert "requests" in set(actual)
"""

_FIXTURE_HELPER_SETATTR = """\
import preflight_mod


def test_helper_receiver(helper):
    helper.setattr(preflight_mod, "LOAD_BEARING_DISTS", ["x"])
    assert preflight_mod.check_dists() == []
"""

_FIXTURE_SINGLE_CHAR_CONST = """\
import preflight_mod


def test_patches_single_char_constant(monkeypatch):
    monkeypatch.setattr(preflight_mod, "X", [1])
    assert preflight_mod.check_dists() == []
"""


def _write_tests(root: Path, **files: str) -> Path:
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (tests_dir / f"{name}.py").write_text(body, encoding="utf-8")
    return root


def _scan(root: Path) -> list[str]:
    sink: list[str] = []
    ret = check_monkeypatched_constant_pinning(repo_root=root, warn_sink=sink)
    assert ret == [], f"WARN-only check must return an empty FAIL list; got: {ret}"
    return sink


# --------------------------------------------------------------------------
# Scanner behavior (fixtures i-iv per plan section 3 D8, + disclosed edges).
# --------------------------------------------------------------------------


def test_flags_all_four_patch_forms_unpinned(tmp_path: Path) -> None:
    _write_tests(tmp_path, test_unpinned=_FIXTURE_UNPINNED)
    warns = _scan(tmp_path)
    hits = [w for w in warns if "monkeypatched-constant" in w]
    assert len(hits) == 4, f"expected exactly 4 WARNs (one per patch form); got: {warns}"
    assert sum("preflight_mod.LOAD_BEARING_DISTS" in w for w in hits) == 2, warns
    assert sum("preflight_mod.DEEP_IMPORT_MODULES" in w for w in hits) == 2, warns
    assert all("mechanism 4" in w for w in hits), warns


def test_silent_with_real_contents_pinning_test(tmp_path: Path) -> None:
    _write_tests(
        tmp_path,
        test_unpinned=_FIXTURE_UNPINNED,
        test_contents_pin=_FIXTURE_PINNING,
    )
    warns = _scan(tmp_path)
    assert warns == [], f"a repo-wide real-contents pinning test must suppress; got: {warns}"


def test_cross_import_style_key_match(tmp_path: Path) -> None:
    _write_tests(
        tmp_path,
        test_hit=_FIXTURE_FROMIMPORT_HIT,
        test_pin=_FIXTURE_ALIAS_PIN,
    )
    warns = _scan(tmp_path)
    assert warns == [], (
        f"(module_basename, CONST) keying must match across import styles; got: {warns}"
    )


def test_non_all_caps_and_test_module_targets_not_flagged(tmp_path: Path) -> None:
    _write_tests(tmp_path, test_nontrigger=_FIXTURE_NONTRIGGER)
    warns = _scan(tmp_path)
    assert warns == [], f"function attrs + test-module constants are non-triggers; got: {warns}"


def test_ordering_substring_test_still_warns(tmp_path: Path) -> None:
    _write_tests(tmp_path, test_ordering=_FIXTURE_ORDERING_ONLY)
    warns = _scan(tmp_path)
    assert len(warns) == 1 and "preflight_mod.LOAD_BEARING_DISTS" in warns[0], (
        f"an ordering-via-source-substring test is NOT contents evidence (#2360); got: {warns}"
    )


def test_save_original_idiom_not_a_pin(tmp_path: Path) -> None:
    _write_tests(tmp_path, test_save=_FIXTURE_SAVE_IDIOM)
    warns = _scan(tmp_path)
    assert len(warns) == 1 and "preflight_mod.LOAD_BEARING_DISTS" in warns[0], (
        f"a save-the-original assignment must not count as a pin; got: {warns}"
    )


def test_unrelated_assert_load_suppresses(tmp_path: Path) -> None:
    """Pins the DISCLOSED false-suppression case as documented behavior: a
    Load of the real constant inside an assert-bearing function counts as a
    pin even when the asserts do not constrain the constant's contents."""
    _write_tests(
        tmp_path,
        test_unpinned=_FIXTURE_UNPINNED,
        test_unrelated=_FIXTURE_UNRELATED_ASSERT,
    )
    warns = _scan(tmp_path)
    assert all("LOAD_BEARING_DISTS" not in w for w in warns), warns
    assert sum("DEEP_IMPORT_MODULES" in w for w in warns) == 2, warns


def test_unparseable_test_file_warns_not_crashes(tmp_path: Path) -> None:
    _write_tests(
        tmp_path,
        test_broken="def test_(:\n",
        test_ordering=_FIXTURE_ORDERING_ONLY,
    )
    warns = _scan(tmp_path)
    assert any("unparseable" in w for w in warns), warns
    assert any("preflight_mod.LOAD_BEARING_DISTS" in w for w in warns), warns


def test_returns_empty_fail_list_always(tmp_path: Path) -> None:
    _write_tests(tmp_path, test_unpinned=_FIXTURE_UNPINNED)
    ret = check_monkeypatched_constant_pinning(repo_root=tmp_path)
    assert ret == []


# --------------------------------------------------------------------------
# Round-2 revision fixtures (upheld round-1 BLOCKERs + opportunistic minors).
# --------------------------------------------------------------------------


def test_fromimport_save_idiom_not_a_pin(tmp_path: Path) -> None:
    """Round-2 BLOCKER fix (``scanner-bare-name-save-false-suppression``): a
    from-imported bare-Name save (``orig = LOAD_BEARING_DISTS``) next to a
    patch of the same constant must NOT count as a pin. Pre-fix the WARN was
    silently suppressed because ``save_value_ids`` collected only
    ``ast.Attribute`` assignment values, so the Name-branch exclusion was
    dead code."""
    _write_tests(tmp_path, test_save=_FIXTURE_SAVE_IDIOM_FROMIMPORT)
    warns = _scan(tmp_path)
    assert len(warns) == 1 and "preflight_mod.LOAD_BEARING_DISTS" in warns[0], (
        f"a from-imported save-the-original assignment must not count as a pin; got: {warns}"
    )


def test_non_utf8_test_file_warns_and_scanning_continues(tmp_path: Path) -> None:
    """Round-2 BLOCKER fix (``scanner-unicode-decode-crash``): non-UTF-8
    bytes in a test file emit the unparseable WARN (never raise
    UnicodeDecodeError, per the ALWAYS-returns-[] contract) and scanning
    continues to later files."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_latin1.py").write_bytes(b"# caf\xe9 \xff not utf-8\n")
    (tests_dir / "test_ordering.py").write_text(_FIXTURE_ORDERING_ONLY, encoding="utf-8")
    warns = _scan(tmp_path)
    assert any("unparseable" in w and "test_latin1" in w for w in warns), warns
    assert any("preflight_mod.LOAD_BEARING_DISTS" in w for w in warns), warns


def test_assignment_read_in_non_patching_function_is_a_pin(tmp_path: Path) -> None:
    """Codex round-1 Minor 1: ``actual = mod.CONST`` inside a NON-patching
    contents test is a real contents read and suppresses — the save-assign
    exclusion is scoped to functions that ALSO patch the same constant."""
    _write_tests(
        tmp_path,
        test_unpinned=_FIXTURE_UNPINNED,
        test_local_var=_FIXTURE_LOCAL_VAR_CONTENTS,
    )
    warns = _scan(tmp_path)
    assert all("LOAD_BEARING_DISTS" not in w for w in warns), warns
    assert sum("DEEP_IMPORT_MODULES" in w for w in warns) == 2, warns


def test_non_monkeypatch_setattr_receiver_is_disclosed_over_approximation(
    tmp_path: Path,
) -> None:
    """Negative control pinning the DISCLOSED over-approximation: receiver
    identity is not validated, so a non-monkeypatch ``helper.setattr(mod,
    "CONST", v)`` classifies as patch form 1/2 and WARNs."""
    _write_tests(tmp_path, test_helper=_FIXTURE_HELPER_SETATTR)
    warns = _scan(tmp_path)
    assert len(warns) == 1 and "preflight_mod.LOAD_BEARING_DISTS" in warns[0], warns


def test_single_char_constant_matched(tmp_path: Path) -> None:
    """Codex round-1 Minor: the widened ``[A-Z][A-Z0-9_]*`` regex matches a
    one-character constant like ``X``."""
    _write_tests(tmp_path, test_single=_FIXTURE_SINGLE_CHAR_CONST)
    warns = _scan(tmp_path)
    assert len(warns) == 1 and "preflight_mod.X" in warns[0], warns


# --------------------------------------------------------------------------
# Surface-pin lens (fixtures v-vi per plan section 3 D8).
# --------------------------------------------------------------------------


def _write_lens_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal six-surface corpus under ``root``; ``drop`` removes
    exactly one surface/token to exercise each per-surface error."""
    agents = root / ".claude" / "agents"
    rules = root / ".claude" / "rules"
    agents.mkdir(parents=True, exist_ok=True)
    rules.mkdir(parents=True, exist_ok=True)

    # (1) the rule file: mechanism-4 heading token + test-escape literal.
    mech4 = "" if drop == "rule-mech4" else f"4. **{_MECH4}** — a TEST fixture replaces it.\n"
    escape = "" if drop == "rule-escape" else f"or the literal `{_ESCAPE}`.\n"
    (rules / "smoke-blind-spots.md").write_text(
        "# Smoke blind spots\n\n" + mech4 + escape,
        encoding="utf-8",
    )

    # (2) code-reviewer.md: Step 3.85 section + Blocker-tags line.
    body_tag = "" if drop == "section-body-tag" else f"a Critical tagged `{_TAG}`. "
    section = (
        "### Step 3.85: Fixture-substituted production-constant "
        "verification (any diff type)\n\n"
        f"Trigger: a test replaces a curated constant, {body_tag}"
        "unless a committed test pins the real contents.\n\n"
    )
    if drop == "step385-section":
        section = ""
    claude_tags = "`substantive`" if drop == "claude-blocker-tag" else f"`{_TAG}`, `substantive`"
    (agents / "code-reviewer.md").write_text(
        "# code-reviewer\n\n" + section + "### Step 9: Verdict\n\n"
        f"**Blocker tags:** [{claude_tags}]\n",
        encoding="utf-8",
    )

    # (3) codex-code-reviewer.md: bullet + rubric slot + Blocker-tags line.
    bullet_tag = "." if drop == "codex-bullet-tag" else f", a single Critical tagged `{_TAG}`."
    bullet = (
        '- "Step 3.85: Fixture-substituted production-constant verification" '
        f"— an unpinned criterion-bearing constant FAILs{bullet_tag}\n"
        '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    )
    if drop == "codex-heading":
        bullet = '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    rubric = (
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 3.8, 3.9}}\n"
        if drop == "codex-rubric"
        else "{{INLINED RUBRIC FROM code-reviewer.md Steps 3.8, 3.85, 3.9}}\n"
    )
    codex_tags = "`substantive`" if drop == "codex-blocker-tag" else f"`{_TAG}` | `substantive`"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex-code-reviewer\n\n" + bullet + "\n" + rubric + "\n"
        f"**Blocker tags:** [{codex_tags}]\n",
        encoding="utf-8",
    )

    # (4) critic-lens-reference.md: the Methodology lens region (item 20).
    clr_item = (
        "20. Placeholder item.\n"
        if drop == "clr-item"
        else "20. **Production-constant test-fixture pinning (any plan whose "
        "test list replaces a curated module-level production constant).** "
        "REVISE naming the constant + the dependent criterion.\n"
    )
    (rules / "critic-lens-reference.md").write_text(
        "### Methodology lens\n\n" + clr_item + "\n### Statistics & Measurement lens\n\nItems.\n",
        encoding="utf-8",
    )

    # (5) critic.md: the Methodology-capsule item token.
    critic_capsule = (
        "19 smoke blind-spot enumeration."
        if drop == "critic-capsule"
        else "19 smoke blind-spot enumeration · 20 production-constant "
        "test-fixture pinning (fixture-substituted constants named + a "
        "real-contents pinning test)."
    )
    (agents / "critic.md").write_text("# critic\n\n" + critic_capsule + "\n", encoding="utf-8")

    # (6) LESSONS.md: the smoke-blind-spots row trigger token.
    row_tail = "." if drop == "lessons-row" else f", or a test {_ROW_TOKEN}."
    (rules / "LESSONS.md").write_text(
        "# LESSONS\n\n## Rules\n\n"
        f"- smoke-blind-spots.md — a plan declares a pre-launch smoke run{row_tail}\n",
        encoding="utf-8",
    )
    return root


def test_pinning_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_lens_corpus(tmp_path)
    errors = check_production_constant_pinning_lens(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("rule-mech4", _MECH4, "rules/smoke-blind-spots.md"),
    ("rule-escape", "no test substitutes a production constant", "rules/smoke-blind-spots.md"),
    ("step385-section", "### Step 3.85", "agents/code-reviewer.md"),
    ("section-body-tag", "section body no longer names", "agents/code-reviewer.md"),
    ("claude-blocker-tag", "**Blocker tags:**", "agents/code-reviewer.md"),
    ("codex-heading", "copy-list token", "agents/codex-code-reviewer.md"),
    ("codex-bullet-tag", "copy-list bullet", "agents/codex-code-reviewer.md"),
    ("codex-rubric", "INLINED RUBRIC", "agents/codex-code-reviewer.md"),
    ("codex-blocker-tag", "**Blocker tags:**", "agents/codex-code-reviewer.md"),
    ("clr-item", "Methodology lens", "rules/critic-lens-reference.md"),
    (
        "critic-capsule",
        "20 production-constant test-fixture pinning",
        "agents/critic.md",
    ),
    ("lessons-row", _ROW_TOKEN, "rules/LESSONS.md"),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_pinning_lens_fails_per_missing_surface(
    tmp_path: Path, drop: str, token: str, path_frag: str
) -> None:
    _write_lens_corpus(tmp_path, drop=drop)
    errors = check_production_constant_pinning_lens(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


def test_pinning_lens_passes_on_live_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binds the landed #2364 edits; the standing regression guard for
    future refactors of any of the six surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_production_constant_pinning_lens(repo_root=None)
    assert errors == [], f"live tree should carry all six surfaces; got: {errors}"


def test_check_production_constant_pinning_lens_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the sibling #2165 test's shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (mechanism-4
    heading dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the
    flag exists, the dispatch calls the function, and it emits its
    #2364-tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_production_constant_pinning_lens`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder (and the WARN scanner's flag in the OR-chain, so passing it
    suppresses the default bundle).
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_lens_corpus(tmp_path, drop="rule-mech4")
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-production-constant-pinning-lens",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "#2364" in combined, (
        "#2364 error token missing from output — the CLI flag does not "
        f"dispatch the check. exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit under drifted corpus; got exit="
        f"{result.returncode}, combined output:\n{combined}"
    )

    # Part B — OR-chain + dispatch ladder evidence.
    lint_src = lint_script.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_production_constant_pinning_lens" in or_chain_src, (
        "args.check_production_constant_pinning_lens is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_monkeypatched_constant_pinning" in or_chain_src, (
        "args.check_monkeypatched_constant_pinning is NOT in the no_flags "
        "OR-chain — passing the WARN scanner flag would not suppress the "
        f"default bundle. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_production_constant_pinning_lens or no_flags" in main_src, (
        "args.check_production_constant_pinning_lens is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
