"""Tests for ``workflow_lint --check-conflict-markers`` (#2192).

Origin incident #2189: merge commit ``14cd4e4211`` left the diff3 base
marker (seven pipes + the base SHA) as the last line of
``.claude/rules/code-style.md``; it passed the no-flags lint, the size
gate, the union-conservation check, ruff, and 27 unrelated tests — caught
only by a reviewer reading the diff by eye.

Covers, per plan (tasks/running/2192 plans/v3.md § Tests, items 1-8):

1. each of the 4 marker forms in a ``.py`` fixture FAILs with correct
   ``path:line`` (fail-loud pin: the findings list is non-empty — never a
   silent ``[]`` swallow);
2. ``.md`` fence semantics: the same forms outside fences flag; inside
   backtick and tilde fences they do not;
3. anchoring negatives: 8-char runs, 7-char-plus-suffix, mid-line tokens,
   and blockquote chevrons never match; a bare 7-char separator at EOL
   does;
4. the ``# CONFLICT_MARKER_EXEMPT: <reason>`` waiver (same or previous
   non-blank line) suppresses a ``.py`` hit; an empty reason does NOT;
5. unterminated fence: content after a lone opener is skipped (pins
   residual (a) so a future behavior change is deliberate);
6. production baseline: the check returns ``[]`` on the live tree AND the
   INDEX-based enumeration (raw ``git ls-files -z`` over the check's
   pathspecs, BEFORE the on-disk filter) exceeds 4,000 files — the floor
   is asserted on the index count because it is sparse-worktree-invariant
   (the on-disk scanned count varies by cone), pinning that the
   enumeration is never silently empty (a zero-file enumeration would be
   a vacuous PASS);
7. missing-on-disk robustness: an enumeration entry absent from disk is
   skipped without raising (both the ``roots`` arm and the per-file-scan
   ``FileNotFoundError`` arm), and findings for present files still
   return;
8. git-failure fail-open: enumeration pointed at a non-git dir emits the
   loud stderr notice and returns ``[]``;
9. the MUTATION-VISIBLE no-flags DISPATCH test (the
   ``test_check_empty_text_default_bundled_in_no_flags`` pattern) — a
   direct call of the check function is NOT sufficient evidence of
   bundling — plus files-mode registry membership at the same chain-order
   position (#2235);
10. (#2192 r2, concern ``conflict-marker-waiver-comment-context``)
    comment-shaped waiver context: a string-literal spoof on the previous
    line stays flagged; the closing-quote constant-declaration shape is
    not a waiver; column-0 AND indented legitimate comment waivers still
    suppress; a quoted token on the marker line itself does not waive;
    and the git-enumeration OSError / TimeoutExpired branches fail open
    with the loud stderr notice.

Self-flag hazard: this file is itself in the check's ``*.py`` scan set, so
every fixture marker line is constructed programmatically (``"<" * 7``,
``"=" * 7``, ...); no source line may start at column 0 with a marker
form.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    CONFLICT_MARKER_WAIVER,
    check_conflict_markers,
)

# Programmatically constructed marker lines (self-flag hazard: never place a
# raw 7-char marker run at column 0 of THIS file).
M_OURS = "<" * 7 + " HEAD"
M_BASE = "|" * 7 + " 640f206892"
M_SEP = "=" * 7
M_THEIRS = ">" * 7 + " origin/main"
ALL_FORMS = (M_OURS, M_BASE, M_SEP, M_THEIRS)
FENCE_TICK = "`" * 3
FENCE_TILDE = "~" * 3


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# 1. all four marker forms flagged in a .py fixture, with correct path:line
# --------------------------------------------------------------------------


def test_planted_markers_flagged(tmp_path: Path) -> None:
    """Every planted marker form yields a finding — a non-empty list (never
    a silent [] swallow) with the exact path:line and quoted token."""
    body = "x = 1\n" + "\n".join(ALL_FORMS) + "\ny = 2\n"
    p = _plant(tmp_path, "offender.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 4, errors
    for i, form in enumerate(ALL_FORMS):
        token = form.split(" ")[0]
        lineno = 2 + i
        assert errors[i].startswith(f"{p}:{lineno}: "), errors[i]
        assert f"('{token}')" in errors[i], errors[i]
        assert "merge-conflict marker residue" in errors[i], errors[i]


# --------------------------------------------------------------------------
# 2. .md fence semantics: outside fences flagged, inside ``` / ~~~ exempt
# --------------------------------------------------------------------------


def test_md_fence_semantics(tmp_path: Path) -> None:
    """Plan § Tests item 2 (md fence semantics): the four forms outside
    fences flag; inside backtick and tilde fences they do not."""
    body = "\n".join(
        [
            "intro prose",
            *ALL_FORMS,  # lines 2-5: outside any fence -> flagged
            FENCE_TICK,
            *ALL_FORMS,  # inside backtick fence -> exempt
            FENCE_TICK,
            FENCE_TILDE,
            M_SEP,  # inside tilde fence -> exempt
            FENCE_TILDE,
            "",
        ]
    )
    p = _plant(tmp_path, "doc.md", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 4, errors
    linenos = [int(e[len(str(p)) + 1 :].split(":", 1)[0]) for e in errors]
    assert linenos == [2, 3, 4, 5], errors


# --------------------------------------------------------------------------
# 3. anchoring negatives
# --------------------------------------------------------------------------


def test_anchoring_negatives(tmp_path: Path) -> None:
    """Plan § Tests item 3 (anchoring negatives): 8-char runs, suffixed
    7-char runs, mid-line tokens, and blockquote chevrons never match; a
    bare 7-char separator at EOL does."""
    negatives = [
        "=" * 8,  # 8-char run: 8th char is not space/EOL
        "=" * 7 + "x",  # 7-char run + suffix char
        ">" * 8 + " ref",  # 8 chevrons + space
        "text " + "=" * 7,  # mid-line occurrence
        "> > > > > > > blockquote",  # spaced chevrons: column-0 run length 1
        "<" * 7 + "HEAD",  # no space between run and ref
    ]
    p = _plant(tmp_path, "neg.py", "\n".join(negatives) + "\n")
    assert check_conflict_markers(roots=[p]) == []
    # ... while the bare 7-char separator at EOL IS flagged.
    p2 = _plant(tmp_path, "pos.py", M_SEP + "\n")
    errors = check_conflict_markers(roots=[p2])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{p2}:1: "), errors[0]


# --------------------------------------------------------------------------
# 4. the .py waiver comment
# --------------------------------------------------------------------------


def test_waiver_previous_and_same_line(tmp_path: Path) -> None:
    """A non-empty-reason waiver on the previous non-blank line (blank gaps
    tolerated) or on the same line suppresses the hit."""
    body = "\n".join(
        [
            f"{CONFLICT_MARKER_WAIVER} documents the diff3 separator shape",
            M_SEP,
            f"{CONFLICT_MARKER_WAIVER} reason survives a blank gap",
            "",
            M_SEP,
            M_SEP + f"  {CONFLICT_MARKER_WAIVER} same-line reason",
            "",
        ]
    )
    p = _plant(tmp_path, "waived.py", body)
    assert check_conflict_markers(roots=[p]) == []


def test_waiver_empty_reason_not_suppressed(tmp_path: Path) -> None:
    """Plan § Tests item 4 (waiver arm): an empty-reason waiver comment
    does NOT suppress."""
    body = CONFLICT_MARKER_WAIVER + "\n" + M_SEP + "\n"
    p = _plant(tmp_path, "unwaived.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{p}:2: "), errors[0]


def test_unrelated_prev_comment_is_not_a_waiver(tmp_path: Path) -> None:
    """Plan § Tests item 4 (waiver arm): an unrelated previous-line
    comment is not a waiver."""
    body = "# some unrelated comment\n" + M_SEP + "\n"
    p = _plant(tmp_path, "plain.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors


def test_waiver_does_not_apply_to_md(tmp_path: Path) -> None:
    """The waiver comment is a .py escape; the .md escape is the fence."""
    body = f"{CONFLICT_MARKER_WAIVER} not an md escape\n" + M_SEP + "\n"
    p = _plant(tmp_path, "doc.md", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# 4-bis. #2192 r2: waiver context must be COMMENT-SHAPED
# (concern conflict-marker-waiver-comment-context)
# --------------------------------------------------------------------------


def test_waiver_token_in_string_literal_is_not_a_waiver(tmp_path: Path) -> None:
    """#2192 r2 blocker regression: a STRING LITERAL containing the waiver
    token on the previous line must NOT suppress a real marker on the next
    line — the spoof stays flagged with exactly 1 finding (the reconciler's
    spoof shape returned ZERO findings under the pre-r2 substring form)."""
    body = f'WAIVER_TEXT = "{CONFLICT_MARKER_WAIVER} documentation"\n' + M_SEP + "\n"
    p = _plant(tmp_path, "spoof.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{p}:2: "), errors[0]


def test_bare_constant_declaration_is_not_a_waiver(tmp_path: Path) -> None:
    """#2192 r2: the closing-quote-after-token shape — a bare constant
    declaration like this module's own ``CONFLICT_MARKER_WAIVER`` line
    (under the pre-r2 form the trailing quote parsed as a non-empty
    'reason') — is NOT recognized as a waiver."""
    body = f'CONST_WAIVER = "{CONFLICT_MARKER_WAIVER}"\n' + M_SEP + "\n"
    p = _plant(tmp_path, "constdecl.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{p}:2: "), errors[0]


def test_comment_waiver_column0_and_indented_still_suppress(tmp_path: Path) -> None:
    """#2192 r2: legitimate COMMENT-SHAPED waivers still suppress — at
    column 0 AND indented (a comment-shaped line inside a docstring body,
    preserving residual (c)'s RST-underline escape)."""
    docstring_open = '"' * 3 + "RST heading in a docstring"
    body = "\n".join(
        [
            f"{CONFLICT_MARKER_WAIVER} column-0 comment waiver",
            M_SEP,
            docstring_open,
            f"    {CONFLICT_MARKER_WAIVER} indented waiver inside the docstring",
            M_SEP,
            '"' * 3,
            "",
        ]
    )
    p = _plant(tmp_path, "legit.py", body)
    assert check_conflict_markers(roots=[p]) == []


def test_same_line_quoted_token_is_not_a_waiver(tmp_path: Path) -> None:
    """#2192 r2: the same-line arm requires a TRAILING comment (whitespace
    then ``#`` then the token) — a QUOTED token on the marker line, whose
    ``#`` is preceded by a quote rather than whitespace, does not waive."""
    body = M_SEP + f' "{CONFLICT_MARKER_WAIVER} x"\n'
    p = _plant(tmp_path, "sameline_spoof.py", body)
    errors = check_conflict_markers(roots=[p])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{p}:1: "), errors[0]


# --------------------------------------------------------------------------
# 5. unterminated fence: trailing content stays skipped (residual (a) pin)
# --------------------------------------------------------------------------


def test_unterminated_fence_skips_trailing_content(tmp_path: Path) -> None:
    """Pins residual (a): after a lone fence opener, trailing md lines are
    SKIPPED (inverted from check_grep_qv's scan-the-tail choice — here the
    fence marks EXEMPT regions). A future behavior change must flip this
    test deliberately."""
    body = "prose\n" + FENCE_TICK + "\n" + M_SEP + "\n" + M_OURS + "\n"
    p = _plant(tmp_path, "open_fence.md", body)
    assert check_conflict_markers(roots=[p]) == []


# --------------------------------------------------------------------------
# 6. production baseline: clean live tree + non-vacuous enumeration
# --------------------------------------------------------------------------


def test_production_baseline_clean_and_enumeration_nonvacuous() -> None:
    """The live tree is baseline-clean AND the INDEX-based enumeration
    exceeds 4,000 files. The floor is asserted on the raw `git ls-files`
    index count (sparse-worktree-invariant, #671) — NOT the on-disk
    scanned count — so a silently-empty enumeration (a vacuous PASS) is
    test-red on every checkout cone."""
    assert check_conflict_markers() == []
    cmd = ["git", "-C", str(wl._REPO_ROOT), "ls-files", "-z", "--"]
    cmd.extend(wl._CONFLICT_MARKER_PATHSPECS)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    n_index = len([rel for rel in proc.stdout.split("\0") if rel])
    assert n_index > 4000, (
        f"index-based enumeration returned only {n_index} files — the scan set "
        f"is silently shrunken (expected > 4,000 tracked md+py files)"
    )


# --------------------------------------------------------------------------
# 7. missing-on-disk robustness (roots arm + per-file FileNotFoundError arm)
# --------------------------------------------------------------------------


def test_missing_on_disk_entries_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan § Tests item 7 (missing-on-disk robustness): both the roots
    arm and the per-file-scan FileNotFoundError arm skip absent entries
    while findings for present files still return."""
    present = _plant(tmp_path, "present.py", M_SEP + "\n")
    ghost = tmp_path / "ghost.py"
    # roots arm: a nonexistent roots entry is skipped, not raised on, and
    # findings for present files still return.
    errors = check_conflict_markers(roots=[ghost, present])
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{present}:1: "), errors[0]
    # per-file-scan arm: an enumeration entry vanishing between listing and
    # read (the #2015 transient unstaged-delete window) is skipped via the
    # FileNotFoundError catch.
    monkeypatch.setattr(wl, "_conflict_marker_target_files", lambda roots: [ghost, present])
    errors = check_conflict_markers()
    assert len(errors) == 1, errors
    assert errors[0].startswith(f"{present}:1: "), errors[0]


# --------------------------------------------------------------------------
# 8. git-failure fail-open with a loud stderr notice
# --------------------------------------------------------------------------


def test_git_failure_fail_open_with_stderr_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-git enumeration root fail-opens to [] (a broken git env must
    not block every commit fleet-wide) with a LOUD stderr notice — never a
    silent skip."""
    _plant(tmp_path, "offender.py", M_SEP + "\n")  # would flag if enumerated
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)  # not a git repo
    assert check_conflict_markers() == []
    err = capsys.readouterr().err
    assert "--check-conflict-markers skipped" in err, err


def test_git_enumeration_oserror_fail_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan § Tests item 8 (git-failure fail-open), OSError branch: a
    subprocess.run that raises (e.g. git binary missing) fail-opens to []
    with the loud stderr notice — never a silent skip."""

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("git binary missing")

    monkeypatch.setattr(wl.subprocess, "run", boom)
    assert check_conflict_markers() == []
    err = capsys.readouterr().err
    assert "--check-conflict-markers skipped" in err, err
    assert "git enumeration failed" in err, err


def test_git_enumeration_timeout_fail_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan § Tests item 8 (git-failure fail-open), TimeoutExpired branch:
    a hung git enumeration fail-opens to [] with the loud stderr notice."""

    def boom(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git ls-files", timeout=60)

    monkeypatch.setattr(wl.subprocess, "run", boom)
    assert check_conflict_markers() == []
    err = capsys.readouterr().err
    assert "--check-conflict-markers skipped" in err, err
    assert "git enumeration failed" in err, err


# --------------------------------------------------------------------------
# 9. no-flags bundling (mutation-visible dispatch test) + files-mode registry
# --------------------------------------------------------------------------


def test_check_conflict_markers_bundled_in_no_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test (mutation-visible; the
    ``test_check_empty_text_default_bundled_in_no_flags`` pattern). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + offending path.
    The tmp tree is a real git repo with the offender TRACKED, since the
    production enumeration is git-ls-files-based."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=60)
    _plant(tmp_path, "scripts/offender_cm.py", M_SEP + "\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "scripts/offender_cm.py"], check=True, timeout=60
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "merge-conflict marker residue" in err and "offender_cm.py:1" in err, (
        f"the conflict-marker diagnostic (naming offender_cm.py) is missing from "
        f"the no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


def test_files_mode_registry_membership_and_chain_position() -> None:
    """#2235 files-mode completeness: the check is registered in BOTH
    registries (else a files-mode invocation prints FILES-MODE-REFUSED and
    exits 2, degrading the gate to its slow bare-full-run fallback), is
    path-local, and sits at the SAME chain-order position as its dispatch
    site (immediately after check_grep_qv — _FILES_MODE_RUNNERS is declared
    IN CHAIN ORDER)."""
    assert "check_conflict_markers" in wl._FILES_MODE_RUNNERS
    assert "check_conflict_markers" in wl.CHECK_SCOPES
    scope = wl.CHECK_SCOPES["check_conflict_markers"]
    assert scope.kind == "path-local", scope
    assert scope.surfaces, scope
    names = list(wl._FILES_MODE_RUNNERS)
    assert names.index("check_conflict_markers") == names.index("check_grep_qv") + 1, (
        "chain-order position drifted: check_conflict_markers must sit "
        "immediately after check_grep_qv, mirroring the dispatch chain"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
