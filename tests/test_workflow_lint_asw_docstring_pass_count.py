"""Tests for ``workflow_lint.check_asw_docstring_pass_count`` (#1225).

The check pins the ``scripts/autonomous_session_watch.py`` module docstring's
pass inventory against itself AND against the code: (1) exactly one
line-start ``<N> passes`` DIGIT header; (2) the numbered inventory items
(line-start ``<digit>. **``) are exactly 1..N sequential and count N;
(3) N equals the live pass set in ``main()`` — distinct ``*_pass``-named
calls plus ``_ASW_INLINE_PASS_BLOCKS`` inline crash-recovery blocks.
Manual catch-ups #1021/#1169 motivated the mechanization.

Coverage:
(1) ``test_live_tree_passes`` — the real tree PASSes (the durability pin);
(2) header==items count mismatch FAILs (live set held consistent so only
    assertion 2 fires);
(3) a word-form header ("Fourteen passes") FAILs the digit contract with
    exactly the header-parse error (the historical red state);
(4) TWO line-start digit headers FAIL the ambiguity branch (found != 1,
    upper side);
(5) non-sequential item numbers (1, 2, 4) FAIL the sequentiality assertion;
(6) live-set drift (main() calls one more distinct ``*_pass`` than the
    header) FAILs the cross-check — the #1021/#1169 incident shape — and
    the error names ``_ASW_INLINE_PASS_BLOCKS`` + the ``*_pass`` naming
    convention;
(7) a fully consistent synthetic watcher PASSes;
(8) duplicate calls of the same ``*_pass`` name (the ``--*-only`` ladder
    shape) dedupe via the distinct-name set and PASS;
(9) robustness — missing file, missing module docstring, unparseable file,
    and missing ``main()`` each FAIL loud.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_asw_docstring_pass_count  # noqa: E402


def _watcher_src(
    doc: str,
    main_calls: list[str] | None,
    defs: list[str] | None = None,
) -> str:
    """Build a synthetic watcher module: module docstring ``doc``, one stub
    def per name in ``defs`` (default: the distinct ``main_calls``), and a
    ``main()`` invoking each entry of ``main_calls`` in order
    (``main_calls=None`` omits ``main()`` entirely). Returns the source text.
    """
    if defs is None:
        defs = sorted(set(main_calls or []))
    lines = ['"""' + doc + '"""', ""]
    for name in defs:
        lines += [f"def {name}(x):", "    return x", ""]
    if main_calls is not None:
        lines.append("def main(argv=None):")
        for name in main_calls:
            lines.append(f"    {name}(1)")
        lines.append("    return 0")
    return "\n".join(lines) + "\n"


def _write_watcher(tmp_path: Path, src: str) -> Path:
    p = tmp_path / "fake_watch.py"
    p.write_text(src, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# (1) the live tree PASSes — the durability pin
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    assert check_asw_docstring_pass_count() == []


# --------------------------------------------------------------------------
# (2) header digit != numbered-item count FAILs (live set held consistent)
# --------------------------------------------------------------------------


def test_fail_on_count_mismatch(tmp_path: Path) -> None:
    # Header says 3; only 2 items. main() calls 2 distinct *_pass functions
    # (+1 inline constant = 3 == header) so ONLY assertion (2) fires.
    doc = "3 passes\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass", "beta_pass"]))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1, errors
    assert "3 passes" in errors[0] and "2 numbered items" in errors[0]


# --------------------------------------------------------------------------
# (3) word-form header FAILs the digit contract (the historical red state)
# --------------------------------------------------------------------------


def test_fail_on_word_form_header(tmp_path: Path) -> None:
    doc = "Fourteen passes\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass"]))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1, errors
    assert "found 0" in errors[0] and "DIGIT" in errors[0]


# --------------------------------------------------------------------------
# (4) TWO line-start digit headers FAIL the ambiguity branch (found > 1)
# --------------------------------------------------------------------------


def test_fail_on_ambiguous_double_header(tmp_path: Path) -> None:
    doc = "2 passes\n\nstray re-wrap hazard:\n2 passes again\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass"]))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1, errors
    assert "found 2" in errors[0]


# --------------------------------------------------------------------------
# (5) non-sequential item numbers FAIL the sequentiality assertion
# --------------------------------------------------------------------------


def test_fail_on_nonsequential_items(tmp_path: Path) -> None:
    # 3 items numbered 1, 2, 4: count matches the header (3) and the live
    # set matches (2 distinct + 1 inline = 3), so ONLY sequentiality fires.
    doc = "3 passes\n\n1. **A.** x\n2. **B.** y\n4. **D.** z\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass", "beta_pass"]))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1, errors
    assert "not exactly 1..3" in errors[0] and "[1, 2, 4]" in errors[0]


# --------------------------------------------------------------------------
# (6) live-set drift FAILs the cross-check (the #1021/#1169 incident shape)
# --------------------------------------------------------------------------


def test_fail_on_live_set_drift(tmp_path: Path) -> None:
    # Header + items agree at 2, but main() calls 2 distinct *_pass
    # functions (+1 inline constant = 3 != 2): a pass was added to the code
    # without reconciling the docstring.
    doc = "2 passes\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass", "beta_pass"]))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1, errors
    # The message must name the naming convention + the inline-block constant
    # so a future author whose new pass escaped the count learns why.
    assert "_ASW_INLINE_PASS_BLOCKS" in errors[0]
    assert "*_pass" in errors[0]


# --------------------------------------------------------------------------
# (7) fully consistent synthetic PASSes
# --------------------------------------------------------------------------


def test_pass_on_matching_synthetic(tmp_path: Path) -> None:
    doc = "2 passes\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass"]))
    assert check_asw_docstring_pass_count(watcher_path=p) == []


# --------------------------------------------------------------------------
# (8) duplicate calls dedupe via the distinct-name set (the --*-only ladder)
# --------------------------------------------------------------------------


def test_pass_on_duplicate_ladder_calls(tmp_path: Path) -> None:
    doc = "2 passes\n\n1. **A.** x\n2. **B.** y\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, ["alpha_pass", "alpha_pass"]))
    assert check_asw_docstring_pass_count(watcher_path=p) == []


# --------------------------------------------------------------------------
# (9) robustness — missing file / docstring / main(); unparseable file
# --------------------------------------------------------------------------


def test_fail_on_missing_file(tmp_path: Path) -> None:
    errors = check_asw_docstring_pass_count(watcher_path=tmp_path / "nope.py")
    assert len(errors) == 1 and "missing" in errors[0]


def test_fail_on_missing_docstring(tmp_path: Path) -> None:
    p = tmp_path / "fake_watch.py"
    p.write_text("x = 1\n", encoding="utf-8")
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1 and "no module docstring" in errors[0]


def test_fail_on_unparseable_file(tmp_path: Path) -> None:
    p = tmp_path / "fake_watch.py"
    p.write_text('"""2 passes"""\ndef broken(:\n', encoding="utf-8")
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1 and "unparseable" in errors[0]


def test_fail_on_missing_main(tmp_path: Path) -> None:
    doc = "1 passes\n\n1. **A.** x\n"
    p = _write_watcher(tmp_path, _watcher_src(doc, None))
    errors = check_asw_docstring_pass_count(watcher_path=p)
    assert len(errors) == 1 and "no main()" in errors[0]
