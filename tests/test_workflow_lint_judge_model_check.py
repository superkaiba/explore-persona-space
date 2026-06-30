"""Tests for ``workflow_lint.check_judge_model_pins`` (#765).

The check FAILs a hardcoded NON-Sonnet judge-model pin at a judge call site,
enforcing the standing one-Sonnet-judge rule (``claude-sonnet-4-5-20250929``;
recipe ``.claude/rules/llm-judging.md``). The gate is ASSIGNMENT/CALL-aware,
NOT mention-aware: a forbidden substring fires only as the RHS of a
``*JUDGE_MODEL*`` assignment, a ``--judge-model`` / ``judge_model=`` /
``JUDGE_MODEL=`` flag, or a ``model=`` kwarg with a judge token in window — a
bare prose-string mention or comment is never a hit.

Covers cases (a)-(m) from the plan:
(a) the canonical Sonnet pin PASSES; (b) a judge_model var-name assignment to a
Haiku id FAILS; (c) a ``--judge-model`` argparse default FAILS; (d) a .sh
``--judge-model`` flag line FAILS; (e) a prose-string mention with no
judge-named assignment PASSES; (f) a comment mentioning the pin PASSES; (g) a
non-judge ``messages.create(model=...)`` with no judge token in window PASSES;
(h) ``# noqa: judge-model-pin`` on the hit / preceding line suppresses; (i) a
file-level ``# epm-allow-judge-model-pin`` suppresses; (j) the canonical pin
does NOT match the ``claude-3-5-sonnet`` forbidden substring (anti-trap);
(k) the legacy allowlists match by EXACT relative path (a tmp fixture is
outside, so it FAILS unless waived) — for both .py and .sh; (l)
``test_live_trees_pass`` — the real trees PASS (the grandfather-completeness
invariant over .py + .sh); (m) robustness (missing dir / unparseable file).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    JUDGE_PIN_CANONICAL,
    JUDGE_PIN_FORBIDDEN_SUBSTRINGS,
    check_judge_model_pins,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# (a) the canonical Sonnet pin PASSES
# --------------------------------------------------------------------------


def test_canonical_sonnet_pin_passes(tmp_path: Path) -> None:
    """A clean file with the canonical JUDGE_MODEL assignment is never flagged."""
    _write(
        tmp_path,
        "clean.py",
        f'JUDGE_MODEL = "{JUDGE_PIN_CANONICAL}"\nprint(JUDGE_MODEL)\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_canonical_sonnet_flag_default_passes(tmp_path: Path) -> None:
    """A ``--judge-model`` argparse default of the canonical Sonnet id PASSES."""
    _write(
        tmp_path,
        "argparse_ok.py",
        f'ap.add_argument("--judge-model", default="{JUDGE_PIN_CANONICAL}")\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (b) judge_model var-name assignment to a Haiku id FAILS
# --------------------------------------------------------------------------


def test_judge_model_var_haiku_fails(tmp_path: Path) -> None:
    """A ``judge_model = "<haiku>"`` assignment FAILS (the var-name arm); the
    error names file:line + the match + the fix hint."""
    p = _write(
        tmp_path,
        "haiku_var.py",
        'judge_model = "claude-haiku-4-5-20251001"\n',
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":1:" in errors[0]
    assert "claude-haiku-" in errors[0]
    assert JUDGE_PIN_CANONICAL in errors[0]
    assert "noqa: judge-model-pin" in errors[0]


def test_namespaced_judge_model_var_fails(tmp_path: Path) -> None:
    """A namespaced ``*_JUDGE_MODEL`` constant (e.g. DEFAULT_GPT4O_JUDGE_MODEL)
    matches the var-name arm — the regex is JUDGE_MODEL-containing, not exact."""
    _write(
        tmp_path,
        "namespaced.py",
        'DEFAULT_GPT4O_JUDGE_MODEL = "gpt-4o-2024-08-06"\n',
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (c) a --judge-model argparse default FAILS (flag arm)
# --------------------------------------------------------------------------


def test_judge_model_flag_default_fails(tmp_path: Path) -> None:
    """A ``--judge-model`` argparse default of a non-Sonnet id FAILS."""
    _write(
        tmp_path,
        "flag_default.py",
        'ap.add_argument("--judge-model", default="gpt-4o-2024-08-06")\n',
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (d) a .sh --judge-model flag line FAILS (the .sh-scope coverage)
# --------------------------------------------------------------------------


def test_sh_judge_model_flag_fails(tmp_path: Path) -> None:
    """A shell launcher pinning ``--judge-model <non-Sonnet>`` directly FAILS —
    the walk includes scripts/**/*.sh."""
    p = _write(
        tmp_path,
        "launch.sh",
        "#!/usr/bin/env bash\nuv run python eval.py --judge-model gpt-4o-2024-08-06\n",
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":2:" in errors[0]


def test_sh_canonical_flag_passes(tmp_path: Path) -> None:
    """A .sh launcher pinning the canonical Sonnet id PASSES (no forbidden substring)."""
    _write(
        tmp_path,
        "launch_ok.sh",
        f"#!/usr/bin/env bash\neval.py --judge-model {JUDGE_PIN_CANONICAL}\n",
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_sh_comment_mention_passes(tmp_path: Path) -> None:
    """A .sh banner/comment line mentioning a non-Sonnet judge PASSES (comment guard)."""
    _write(
        tmp_path,
        "launch_comment.sh",
        "#!/usr/bin/env bash\n# historical: this used the gpt-4o judge before the migration\n",
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (e) a prose-string mention with no judge-named assignment PASSES
# --------------------------------------------------------------------------


def test_prose_string_mention_passes(tmp_path: Path) -> None:
    """A forbidden substring inside a descriptive string with no judge-named
    assignment / --judge-model flag / judge model= on the line is NOT a hit
    (the issue552_gate_decision.py:83 / issue467_figures.py:176 class)."""
    _write(
        tmp_path,
        "prose.py",
        'rule = "PASS iff benign; 8 probes, judge gpt-4o-2024-08-06, no system prompt"\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_figure_caption_source_kwarg_passes(tmp_path: Path) -> None:
    """A ``source="...gpt-4o..."`` figure-caption kwarg (NOT model=, no judge
    token) is a non-judge string mention and PASSES."""
    _write(
        tmp_path,
        "fig.py",
        'set_title(ax, "t", source="#458 broad-mis judged by gpt-4o-2024-08-06; threshold 0.65")\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (f) a comment mentioning the pin PASSES
# --------------------------------------------------------------------------


def test_comment_mention_passes(tmp_path: Path) -> None:
    """A pure code-comment line naming a judge model PASSES (comment guard)."""
    _write(
        tmp_path,
        "comment.py",
        "# the #404 calibration uses the gpt-4o judge cost table below\nX = 1\n",
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_docstring_mention_passes(tmp_path: Path) -> None:
    """A module-docstring mention (the issue623_behavioral_dv.py:6 class) PASSES
    — no judge-named assignment / flag / judge model= on the line."""
    _write(
        tmp_path,
        "docmod.py",
        '"""REUSE the #612 base rates (judge ``claude-haiku-4-5-20251001``)."""\nX = 1\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (g) a non-judge model= call with no judge token in window PASSES
# --------------------------------------------------------------------------


def test_non_judge_model_kwarg_passes(tmp_path: Path) -> None:
    """A ``client.messages.create(model="<haiku>")`` document-generation call
    with NO judge token in the +/-3 window PASSES (the SDF-generator class)."""
    _write(
        tmp_path,
        "sdf_gen.py",
        "resp = client.messages.create(\n"
        '    model="claude-haiku-4-5-20251001",\n'
        '    messages=[{"role": "user", "content": "rewrite this document"}],\n'
        ")\n",
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_judge_model_kwarg_with_token_in_window_fails(tmp_path: Path) -> None:
    """A ``model=`` kwarg WITH a judge-call token in the +/-3 NON-COMMENT window
    FAILS (the context arm). The token here is a real ``judge_completions`` call
    a few lines below, not a comment (a comment-only token does not count)."""
    _write(
        tmp_path,
        "judge_call.py",
        "resp = client.messages.create(\n"
        '    model="claude-haiku-4-5-20251001",\n'
        ")\n"
        "verdicts = judge_completions(resp)\n",
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (h) the per-line judge-model-pin waiver (hit / preceding line) suppresses
# --------------------------------------------------------------------------


def test_noqa_on_hit_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "waived_inline.py",
        'judge_model = "gpt-4o-2024-08-06"  # noqa: judge-model-pin\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


def test_noqa_on_preceding_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "waived_above.py",
        '# noqa: judge-model-pin\njudge_model = "gpt-4o-2024-08-06"\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (i) a file-level # epm-allow-judge-model-pin suppresses every hit
# --------------------------------------------------------------------------


def test_file_level_waiver_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "file_waived.py",
        "# epm-allow-judge-model-pin\n"
        'judge_model = "gpt-4o-2024-08-06"\n'
        'OTHER_JUDGE_MODEL = "claude-haiku-4-5-20251001"\n',
    )
    assert check_judge_model_pins(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (j) the canonical pin does NOT match the claude-3-5-sonnet forbidden substring
# --------------------------------------------------------------------------


def test_canonical_pin_not_a_forbidden_substring() -> None:
    """Anti-trap: claude-sonnet-4-5-20250929 (the canonical id) contains NONE
    of the forbidden substrings — in particular it is claude-sonnet-4-5-...,
    NOT claude-3-5-sonnet (the inverted-order trap)."""
    assert "claude-3-5-sonnet" in JUDGE_PIN_FORBIDDEN_SUBSTRINGS
    for sub in JUDGE_PIN_FORBIDDEN_SUBSTRINGS:
        assert sub not in JUDGE_PIN_CANONICAL, (sub, JUDGE_PIN_CANONICAL)


# --------------------------------------------------------------------------
# (k) the legacy allowlists match by EXACT relative path (a tmp fixture is
# outside, so it FAILS unless waived) — for both .py and .sh
# --------------------------------------------------------------------------


def test_allowlisted_basename_in_tmp_still_fails(tmp_path: Path) -> None:
    """A tmp fixture sharing a BASENAME with a legacy-allowlisted file is NOT
    exempted — the allowlist matches the full repo-relative POSIX path, and a
    tmp fixture falls outside it (so the gate still fires)."""
    _write(
        tmp_path,
        "judges.py",  # basename of an allowlisted src file
        'JUDGE_MODEL = "gpt-4o-2024-08-06"\n',
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_allowlisted_sh_basename_in_tmp_still_fails(tmp_path: Path) -> None:
    """Same for the .sh allowlist — a tmp .sh fixture sharing a basename with a
    legacy-allowlisted launcher is NOT exempted."""
    _write(
        tmp_path,
        "run_issue552_sweep.sh",  # basename of an allowlisted .sh launcher
        "#!/usr/bin/env bash\neval.py --judge-model gpt-4o-2024-08-06\n",
    )
    errors = check_judge_model_pins(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (l) the real trees PASS (the grandfather-completeness invariant over .py + .sh)
# --------------------------------------------------------------------------


def test_live_trees_pass() -> None:
    """The real scripts/ + src/ + tests/ trees (.py AND scripts/ .sh) must pass
    the check — this is the no-flags-default-run invariant and the authoritative
    grandfather-completeness check. If this FAILs, either the allowlist is
    incomplete (a new legit pin landed — add it with a reason) or the gate has a
    false-positive (a prose/comment/non-judge site is being flagged)."""
    assert check_judge_model_pins() == []


# --------------------------------------------------------------------------
# (m) robustness
# --------------------------------------------------------------------------


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert check_judge_model_pins(scripts_dir=tmp_path / "nope") == []


def test_unparseable_file_does_not_crash(tmp_path: Path) -> None:
    """A line-scanned file does not need to parse — an odd/binary-ish file is
    scanned line-by-line without crashing (and a non-judge pin in garbage is not
    flagged unless it hits an assignment/flag/judge-model= shape)."""
    _write(tmp_path, "weird.py", "def f(:\n   pass\n# nothing forbidden here\n")
    assert check_judge_model_pins(scripts_dir=tmp_path) == []
