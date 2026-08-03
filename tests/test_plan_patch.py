"""Tests for ``scripts/plan_patch.py`` (#1631) + the SKILL.md pointer pin.

Pins the anchor-normalized plan-patch helper: 3-stage matching
(exact -> ws-normalized -> ws+case-normalized, unique-match-only, no
fall-through on ambiguity), exit codes 0/2/3, the nearest-match report,
line-based insert modes / byte-exact replace, ``--verify-contains``,
``--dry-run``, atomic byte-preserving writes, the guards added from the
critic-ensemble round (empty/whitespace anchor, degenerate nearest-match
windows, non-UTF-8 input, the 10 MB size guard), and the ``--file`` target
alias (#1848: positional-or-``--file``, exactly one; both/neither exits 2
with the file untouched; the ``--help`` epilog's no-pipe/rc-check note).

The final test is the durability pin: ``scripts/plan_patch.py`` must stay
named inside the Edit-success gate prose of BOTH plan-revision recipes
(``.claude/skills/adversarial-planner/SKILL.md`` and
``.claude/skills/issue/SKILL.md``). Presence asserts only — deliberately not
count-exact, so unrelated future mentions don't false-fail.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER_PATH = REPO_ROOT / "scripts" / "plan_patch.py"
_spec = importlib.util.spec_from_file_location("plan_patch", _HELPER_PATH)
assert _spec is not None and _spec.loader is not None
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def _run(*argv) -> int:
    return pp.main([str(a) for a in argv])


# --- stage resolution + apply -------------------------------------------------


def test_exact_match_replace_applies(tmp_path, capsys):
    f = tmp_path / "d.md"
    f.write_text("intro\nthe target sentence.\ntail\n", encoding="utf-8")
    rc = _run(f, "--anchor", "the target sentence.", "--replace", "the REVISED sentence.")
    out = capsys.readouterr().out
    assert rc == 0
    assert "PLAN-PATCH APPLIED (exact match at lines 2-2" in out
    assert "-the target sentence." in out and "+the REVISED sentence." in out
    assert f.read_text(encoding="utf-8") == "intro\nthe REVISED sentence.\ntail\n"


def test_ws_normalized_match_resolves_wrapped_anchor(tmp_path, capsys):
    # The #1604/#1609 shape: anchor drafted from intended wording, file wraps
    # the same words across lines with extra indentation.
    f = tmp_path / "d.md"
    f.write_text("start\nalpha beta\n  gamma delta\nend\n", encoding="utf-8")
    rc = _run(f, "--anchor", "alpha beta gamma delta", "--replace", "REPLACED")
    out = capsys.readouterr().out
    assert rc == 0
    assert "ws-normalized match at lines 2-3" in out
    assert f.read_text(encoding="utf-8") == "start\nREPLACED\nend\n"


def test_case_stage_resolves_case_drift(tmp_path, capsys):
    # The #1415 shape: case-drifted anchor; resolves at stage 3 only.
    f = tmp_path / "d.md"
    f.write_text("start\nThe Quick Fox Jumps\nend\n", encoding="utf-8")
    rc = _run(f, "--anchor", "the quick fox jumps", "--replace", "REPLACED")
    out = capsys.readouterr().out
    assert rc == 0
    assert "ws+case-normalized match at lines 2-2" in out
    assert "REPLACED" in f.read_text(encoding="utf-8")


def test_missing_anchor_fails_with_nearest_match_diff(tmp_path, capsys):
    f = tmp_path / "d.md"
    before = "one\nthe quick brown fox jumps over\nthree\n"
    f.write_text(before, encoding="utf-8")
    rc = _run(f, "--anchor", "the quick brown dog leaps over", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "PLAN-PATCH FAILED" in err
    assert "nearest match: lines 2-2" in err
    assert "--- anchor (as given)" in err
    assert "+++ closest match in file (lines 2-2)" in err
    assert f.read_text(encoding="utf-8") == before  # untouched


def test_ambiguous_exact_anchor_fails_loud(tmp_path, capsys):
    f = tmp_path / "d.md"
    before = "dup line\nmiddle\ndup line\n"
    f.write_text(before, encoding="utf-8")
    rc = _run(f, "--anchor", "dup line", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "ambiguous anchor — 2 matches at the exact stage" in err
    assert "lines 1-1" in err and "lines 3-3" in err
    assert f.read_text(encoding="utf-8") == before


def test_ambiguous_at_normalized_stage_fails_no_fallthrough(tmp_path, capsys):
    # exact: 0 hits; ws-normalized: 2 hits -> ambiguity is reported AT the
    # ws-normalized stage (never a fall-through to the case stage).
    f = tmp_path / "d.md"
    before = "foo  bar\nqux\nfoo\tbar\n"
    f.write_text(before, encoding="utf-8")
    rc = _run(f, "--anchor", "foo bar", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "2 matches at the ws-normalized stage" in err
    assert f.read_text(encoding="utf-8") == before


def test_noop_replacement_fails(tmp_path, capsys):
    # The #1565 guard: an edit that changes nothing must fail loud, not
    # "succeed" into an unchanged persist.
    f = tmp_path / "d.md"
    f.write_text("keep this line\n", encoding="utf-8")
    rc = _run(f, "--anchor", "keep this line", "--replace", "keep this line")
    err = capsys.readouterr().err
    assert rc == 2
    assert "EDIT NO-OP" in err
    assert f.read_text(encoding="utf-8") == "keep this line\n"


# --- insert modes ---------------------------------------------------------------


def test_insert_after_is_line_based_with_trailing_newline(tmp_path):
    # Mid-line anchor: payload (no trailing newline) still lands as a full
    # line AFTER the anchor's containing line, not byte-adjacent to the match.
    f = tmp_path / "d.md"
    f.write_text("aaa bbb ccc\nnext line\n", encoding="utf-8")
    rc = _run(f, "--anchor", "bbb", "--insert-after", "INSERTED")
    assert rc == 0
    assert f.read_text(encoding="utf-8") == "aaa bbb ccc\nINSERTED\nnext line\n"


def test_insert_before(tmp_path):
    f = tmp_path / "d.md"
    f.write_text("aaa bbb ccc\nnext line\n", encoding="utf-8")
    rc = _run(f, "--anchor", "next line", "--insert-before", "INSERTED")
    assert rc == 0
    assert f.read_text(encoding="utf-8") == "aaa bbb ccc\nINSERTED\nnext line\n"


def test_insert_rerun_exits_3_without_double_insert(tmp_path, capsys):
    f = tmp_path / "d.md"
    f.write_text("anchor line\ntail\n", encoding="utf-8")
    assert _run(f, "--anchor", "anchor line", "--insert-after", "NEW LINE") == 0
    once = f.read_text(encoding="utf-8")
    rc = _run(f, "--anchor", "anchor line", "--insert-after", "NEW LINE")
    err = capsys.readouterr().err
    assert rc == 3
    assert "PLAN-PATCH ALREADY-APPLIED" in err
    assert f.read_text(encoding="utf-8") == once  # no double insert


# --- byte / encoding handling ---------------------------------------------------


def test_crlf_anchor_matches_lf_file(tmp_path):
    # \r is whitespace at stage 2; LF bytes outside the span are preserved.
    f = tmp_path / "d.md"
    f.write_text("intro\nalpha beta\ngamma delta\ntail\n", encoding="utf-8", newline="")
    rc = _run(f, "--anchor", "alpha beta\r\ngamma delta", "--replace", "REPLACED")
    assert rc == 0
    text = f.read_bytes().decode("utf-8")
    assert text == "intro\nREPLACED\ntail\n"
    assert "\r" not in text


def test_multiline_anchor_exact_and_normalized(tmp_path, capsys):
    f1 = tmp_path / "exact.md"
    f1.write_text("head\nline one\nline two\nfoot\n", encoding="utf-8")
    assert _run(f1, "--anchor", "line one\nline two", "--replace", "MERGED") == 0
    assert "exact match at lines 2-3" in capsys.readouterr().out
    assert f1.read_text(encoding="utf-8") == "head\nMERGED\nfoot\n"

    f2 = tmp_path / "wrapped.md"
    f2.write_text("head\nline one\nline two\nfoot\n", encoding="utf-8")
    assert _run(f2, "--anchor", "line one line two", "--replace", "MERGED") == 0
    assert "ws-normalized match at lines 2-3" in capsys.readouterr().out
    assert f2.read_text(encoding="utf-8") == "head\nMERGED\nfoot\n"


def test_non_utf8_target_exits_2(tmp_path, capsys):
    f = tmp_path / "d.md"
    f.write_bytes(b"\xff\xfe\x80 not utf8")
    rc = _run(f, "--anchor", "anything", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "not UTF-8" in err


def test_file_size_guard_refuses_over_limit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pp, "MAX_FILE_BYTES", 64)
    f = tmp_path / "d.md"
    f.write_text("x" * 100, encoding="utf-8")
    rc = _run(f, "--anchor", "xxx", "--replace", "y")
    err = capsys.readouterr().err
    assert rc == 2
    assert "refusing files larger than 64 bytes" in err


# --- verify-contains / dry-run / file variants ----------------------------------


def test_verify_contains_failure_blocks_write(tmp_path, capsys):
    f = tmp_path / "d.md"
    before = "one\ntwo\nthree\n"
    f.write_text(before, encoding="utf-8")
    rc = _run(f, "--anchor", "two", "--replace", "TWO", "--verify-contains", "NOT_THERE")
    err = capsys.readouterr().err
    assert rc == 2
    assert "--verify-contains failed" in err
    # Concern 1: the would-be diff is printed so a mis-drafted verify string
    # is one-shot diagnosable.
    assert "+TWO" in err and "(after)" in err
    assert f.read_text(encoding="utf-8") == before


def test_dry_run_prints_diff_and_writes_nothing(tmp_path, capsys):
    f = tmp_path / "d.md"
    before = "one\ntwo\nthree\n"
    f.write_text(before, encoding="utf-8")
    rc = _run(f, "--anchor", "two", "--replace", "TWO", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0
    assert "PLAN-PATCH DRY-RUN OK (exact match at lines 2-2" in out
    assert "+TWO" in out
    assert f.read_text(encoding="utf-8") == before


def test_anchor_and_payload_file_variants(tmp_path):
    f = tmp_path / "d.md"
    f.write_text("head\nthe anchor text here\nfoot\n", encoding="utf-8")
    anchor_file = tmp_path / "a.txt"
    anchor_file.write_text("the anchor text here\n", encoding="utf-8")
    payload_file = tmp_path / "r.txt"
    payload = "verbatim  payload — é kept as-is,  double  spaces too"
    payload_file.write_text(payload + "\n", encoding="utf-8")
    rc = _run(f, "--anchor-file", anchor_file, "--replace-file", payload_file)
    assert rc == 0
    assert f.read_text(encoding="utf-8") == f"head\n{payload}\nfoot\n"

    # Lone-trailing-newline tolerance (the rstrip fallback): the anchor file
    # ends in \n but the file's match sits at EOF without one.
    f2 = tmp_path / "eof.md"
    f2.write_text("head\nthe anchor text here", encoding="utf-8")
    rc = _run(f2, "--anchor-file", anchor_file, "--replace", "REPLACED")
    assert rc == 0
    assert f2.read_text(encoding="utf-8") == "head\nREPLACED"


def test_write_is_atomic_and_preserves_unrelated_bytes(tmp_path):
    f = tmp_path / "d.md"
    before = "héad — non-ascii\nreplace me\ntail  with  spacing\n"
    f.write_text(before, encoding="utf-8", newline="")
    rc = _run(f, "--anchor", "replace me", "--replace", "replaced")
    assert rc == 0
    after = f.read_bytes().decode("utf-8")
    assert after == "héad — non-ascii\nreplaced\ntail  with  spacing\n"
    # No temp-file residue from the atomic-write path.
    assert not list(tmp_path.glob(".plan-patch-*"))


# --- guards from the critic-ensemble round --------------------------------------


def test_empty_or_whitespace_anchor_rejected(tmp_path, capsys):
    f = tmp_path / "d.md"
    before = "content\n"
    f.write_text(before, encoding="utf-8")
    for bad_anchor in ("", "  \n\t "):
        rc = _run(f, "--anchor", bad_anchor, "--replace", "X")
        err = capsys.readouterr().err
        assert rc == 2
        assert "anchor must contain non-whitespace text" in err
    assert f.read_text(encoding="utf-8") == before


def test_nearest_match_degenerate_windows(tmp_path, capsys):
    # File shorter than the anchor: designed report, never a traceback.
    short = tmp_path / "short.md"
    short.write_text("only line\n", encoding="utf-8")
    rc = _run(short, "--anchor", "aaa\nbbb\nccc", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "shorter than the 3-line anchor" in err

    # Empty file: designed report, never a max()-on-empty traceback.
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    rc = _run(empty, "--anchor", "anything", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "file is empty" in err


def test_nearest_match_finds_true_region_inside_long_line(tmp_path, capsys):
    # The report scorer must not let a short decoy line outrank the TRUE
    # region when the true region is a LONG markdown line (full-window
    # ratio() penalizes long lines; the partial-ratio slice scan fixes it).
    decoy = "the quick brown fox leaps"
    long_line = (
        "unrelated prefix words repeated here " * 5
        + "the quick brown fox jumps over the lazy dog"
        + " and unrelated suffix words repeated here" * 5
    )
    f = tmp_path / "d.md"
    f.write_text(f"{decoy}\nmiddle line\n{long_line}\n", encoding="utf-8")
    rc = _run(f, "--anchor", "the brown quick fox jumps over the lazy dog", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "nearest match: lines 3-3" in err  # the long TRUE line, not the decoy


def test_replace_empty_payload_deletes_span(tmp_path):
    f = tmp_path / "d.md"
    f.write_text("keep DELETE-ME keep\n", encoding="utf-8")
    rc = _run(f, "--anchor", " DELETE-ME", "--replace", "")
    assert rc == 0
    assert f.read_text(encoding="utf-8") == "keep keep\n"


def test_insert_empty_payload_rejected(tmp_path, capsys):
    f = tmp_path / "d.md"
    f.write_text("anchor line\n", encoding="utf-8")
    rc = _run(f, "--anchor", "anchor line", "--insert-after", "")
    err = capsys.readouterr().err
    assert rc == 2
    assert "insert payload must be non-empty" in err


# --- target spelling: positional vs --file (#1848) -------------------------------


def test_file_option_alias_parity_with_positional(tmp_path, capsys):
    # `--file <path>` behaves identically to the positional form: same
    # APPLIED sentinel, same diff, same write.
    pos = tmp_path / "pos.md"
    opt = tmp_path / "opt.md"
    before = "intro\nthe target sentence.\ntail\n"
    pos.write_text(before, encoding="utf-8")
    opt.write_text(before, encoding="utf-8")

    rc_pos = _run(pos, "--anchor", "the target sentence.", "--replace", "REVISED.")
    out_pos = capsys.readouterr().out
    rc_opt = _run("--file", opt, "--anchor", "the target sentence.", "--replace", "REVISED.")
    out_opt = capsys.readouterr().out

    assert rc_pos == 0 and rc_opt == 0
    assert "PLAN-PATCH APPLIED (exact match at lines 2-2" in out_pos
    assert "PLAN-PATCH APPLIED (exact match at lines 2-2" in out_opt
    assert pos.read_text(encoding="utf-8") == opt.read_text(encoding="utf-8")
    # Identical output modulo the path each form named.
    assert out_pos.replace(str(pos), "<F>") == out_opt.replace(str(opt), "<F>")


def test_both_positional_and_file_option_exit_2_untouched(tmp_path, capsys):
    pos = tmp_path / "pos.md"
    opt = tmp_path / "opt.md"
    pos.write_text("positional target\n", encoding="utf-8")
    opt.write_text("option target\n", encoding="utf-8")
    rc = _run(pos, "--file", opt, "--anchor", "target", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "PLAN-PATCH FAILED" in err
    assert "positional FILE" in err and "--file FILE" in err  # names both spellings
    assert pos.read_text(encoding="utf-8") == "positional target\n"
    assert opt.read_text(encoding="utf-8") == "option target\n"


def test_neither_target_spelling_exit_2(capsys):
    rc = _run("--anchor", "anything", "--replace", "X")
    err = capsys.readouterr().err
    assert rc == 2
    assert "PLAN-PATCH FAILED" in err
    assert "positional FILE" in err and "--file FILE" in err  # names both spellings


def test_help_epilog_carries_no_pipe_rc_note():
    # The --help epilog warns against piping the gate chain through filters
    # (a pipe masks the exit code the Edit-success gate's && chain relies on).
    help_text = pp.build_parser().format_help()
    assert "Never pipe plan_patch.py" in help_text
    assert "masks the exit code" in help_text
    assert "--file" in help_text


# --- CLI smoke + durability pin --------------------------------------------------


def test_cli_subprocess_smoke(tmp_path):
    f = tmp_path / "d.md"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(_HELPER_PATH), str(f), "--anchor", "two", "--replace", "TWO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "PLAN-PATCH APPLIED" in proc.stdout
    assert f.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_skillmd_pointer_present_in_both_recipes():
    # THE durability pin (#1631): the helper must stay named inside both
    # plan-revision recipes, and the adversarial-planner mention must sit in
    # the same file as the Edit-success gate prose. Presence asserts only.
    ap_text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    issue_text = (REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/plan_patch.py" in ap_text
    assert "scripts/plan_patch.py" in issue_text
    assert "Edit-success gate" in ap_text
