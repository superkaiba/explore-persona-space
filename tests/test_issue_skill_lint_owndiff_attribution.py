"""Pin + fixture tests for the Step 10d lint-gate own-diff attribution (#1944).

The gate's payload attribution lives in SKILL.md PROSE (two executable bash
blocks in ``.claude/skills/issue/SKILL.md``): the shared Pre-push
workflow-lint gate and the surgical additive-checkout block. #1768 surfaced a
false-block class: the old whole-line ``grep -F -f <own-diff> <gated-norm>``
matched a PRE-EXISTING foreign-file failure line because its lint MESSAGE
cited ``.claude/rules/gotchas.md``, which sat in the branch own-diff via a
spec-freshness sync. The #1944 fix attributes by OFFENDER PATH TOKEN — the
leading ``<path>`` before the first ``:`` of the normalized
``workflow_lint: <err>`` line, gate-tree prefix stripped — via exact
set-membership against the payload list (own-diff / additive-files).

Two layers:

1. Prose pins — both attribution sites carry the awk path-token form
   (``path in own``); the bare whole-line grep form is GONE; the two awk
   programs stay byte-identical (one recipe, two ``-v OWN=`` list files).
2. A functional fixture test that EXTRACTS the awk program from the SKILL.md
   block text (never a hard-coded copy — recipe drift must not diverge from
   the tested program), substitutes ``<N>``, and runs it via subprocess over
   the #1768 incident line shape + a genuine payload-offender line + a
   check-name-led line.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from tests.issue_skill_source import issue_skill_text


def _load_selector():
    """Import scripts/select_step9c_tests.py by path (scripts/ is not an
    importable package; mirrors tests/test_select_step9c_tests.py)."""
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1944", _SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    return sel


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
_SELECTOR_PY = _REPO_ROOT / "scripts" / "select_step9c_tests.py"

_PIN_FILE_RELPATH = "tests/test_issue_skill_lint_owndiff_attribution.py"

_SHARED_ANCHOR = "awk -v OWN=/tmp/issue-<N>-own-diff.txt '"
_SURGICAL_ANCHOR = "awk -v OWN=/tmp/issue-<N>-additive-files.txt '"
_END_MARKER = "}' /tmp/issue-<N>-lint-gated-norm.txt"

_OLD_GREP_FORMS = (
    "grep -F -f /tmp/issue-<N>-own-diff.txt /tmp/issue-<N>-lint-gated-norm.txt",
    "grep -F -f /tmp/issue-<N>-additive-files.txt /tmp/issue-<N>-lint-gated-norm.txt",
)


def _skill_text() -> str:
    return issue_skill_text()


def _extract_awk_program(text: str, anchor: str) -> str:
    """Return the single-quoted awk PROGRAM of the attribution block at
    ``anchor`` (exclusive of the quotes; inclusive of the closing ``}``).

    Asserts the anchor and its end marker resolve exactly once past the
    anchor — a missing block fails loud rather than testing nothing.
    """
    start = text.find(anchor)
    assert start != -1, f"attribution awk anchor not found in SKILL.md: {anchor!r}"
    assert text.find(anchor, start + 1) == -1, f"attribution awk anchor not unique: {anchor!r}"
    prog_start = start + len(anchor)
    end = text.find(_END_MARKER, prog_start)
    assert end != -1, f"attribution awk end marker not found after {anchor!r}"
    return text[prog_start : end + 1]  # include the closing `}`


def test_both_attribution_sites_use_path_token_awk():
    """#1944 prose pin: BOTH lint-attribution sites (shared gate + surgical
    additive-checkout block) attribute by offender path token via the awk
    set-membership form, and the old whole-line ``grep -F -f`` form is gone."""
    text = _skill_text()
    shared = _extract_awk_program(text, _SHARED_ANCHOR)
    surgical = _extract_awk_program(text, _SURGICAL_ANCHOR)
    for name, prog in (("shared", shared), ("surgical", surgical)):
        assert "path in own" in prog, (
            f"{name} attribution awk must test exact set-membership (`path in own`)"
        )
        assert "/^workflow_lint: /" in prog, (
            f"{name} attribution awk must key on the normalized `workflow_lint: ` prefix"
        )
        assert "lint-gate-tree" in prog, (
            f"{name} attribution awk must strip the gate-tree prefix from the path token"
        )
    for old in _OLD_GREP_FORMS:
        assert old not in text, (
            f"the whole-line fixed-string attribution grep must be GONE from SKILL.md "
            f"(the #1768 false-block class): {old!r}"
        )


def test_attribution_awk_programs_identical_across_sites():
    """The two sites share ONE recipe: the awk PROGRAM text is byte-identical
    (only the ``-v OWN=`` list file differs), so a fix to one site cannot
    silently strand the other."""
    text = _skill_text()
    shared = _extract_awk_program(text, _SHARED_ANCHOR)
    surgical = _extract_awk_program(text, _SURGICAL_ANCHOR)
    assert shared == surgical, (
        "the shared-gate and surgical-block attribution awk programs drifted apart; "
        "keep them byte-identical (one recipe, two OWN list files)"
    )


def test_attribution_awk_fixture_incident_shape(tmp_path):
    """Functional fixture (#1768 incident shape): run the EXTRACTED awk over a
    3-line norm file — a pre-existing foreign-file failure whose MESSAGE cites
    an own-diff rules path, a genuine payload-offender line, and a
    check-name-led line — and assert ONLY the offender line attributes."""
    text = _skill_text()
    program = _extract_awk_program(text, _SHARED_ANCHOR).replace("<N>", "9999")

    incident_line = (
        "workflow_lint: scripts/issue1689_user_slot_capture.py:: jsonl-splitlines: "
        "json.loads-per-line loop without a splitlines guard — see "
        ".claude/rules/gotchas.md for the recipe"
    )
    offender_line = "workflow_lint: scripts/payload_x.py:: some-check: msg"
    checkname_line = (
        "workflow_lint: lessons-index: row for .claude/rules/gotchas.md exceeds its cap"
    )

    own = tmp_path / "own-diff.txt"
    own.write_text(".claude/rules/gotchas.md\nscripts/payload_x.py\n", encoding="utf-8")
    norm = tmp_path / "lint-gated-norm.txt"
    norm.write_text(
        "\n".join([incident_line, offender_line, checkname_line]) + "\n", encoding="utf-8"
    )

    proc = subprocess.run(
        ["awk", "-v", f"OWN={own}", program, str(norm)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"awk failed: rc={proc.returncode} stderr={proc.stderr!r}"
    assert proc.stdout == offender_line + "\n", (
        "attribution must select EXACTLY the genuine payload-offender line — the #1768 "
        "message-cites-own-diff-path line and the check-name-led line must NOT attribute; "
        f"got: {proc.stdout!r}"
    )


def test_attribution_awk_fixture_gate_tree_prefix(tmp_path):
    """Functional fixture (gate-tree prefix): an offender token carrying the
    absolute ``/tmp/issue-<N>-lint-gate-tree/`` prefix still attributes after
    the ``sub()`` strips it (the shared gate's lint runs against the gate
    tree, so absolute-prefixed offender tokens are a legal emitter shape)."""
    text = _skill_text()
    program = _extract_awk_program(text, _SHARED_ANCHOR).replace("<N>", "9999")

    prefixed_offender = (
        "workflow_lint: /tmp/issue-9999-lint-gate-tree/scripts/payload_x.py:: some-check: msg"
    )
    foreign_line = "workflow_lint: scripts/foreign.py:: other-check: msg"

    own = tmp_path / "own-diff.txt"
    own.write_text("scripts/payload_x.py\n", encoding="utf-8")
    norm = tmp_path / "lint-gated-norm.txt"
    norm.write_text("\n".join([prefixed_offender, foreign_line]) + "\n", encoding="utf-8")

    proc = subprocess.run(
        ["awk", "-v", f"OWN={own}", program, str(norm)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"awk failed: rc={proc.returncode} stderr={proc.stderr!r}"
    assert proc.stdout == prefixed_offender + "\n", (
        "a gate-tree-prefixed offender token must attribute after the sub() strip; "
        f"got: {proc.stdout!r}"
    )


def test_registered_in_step9c_workflow_invariant():
    """This pin file must be a WORKFLOW_INVARIANT member: SKILL.md diffs
    select only that set (no discovery arm reaches a .md pin file), so an
    unregistered pin never runs on the diffs it guards (#1546 class)."""
    sel = _load_selector()
    assert _PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
