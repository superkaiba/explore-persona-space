"""Structural pin: every cron-wrapper Telegram push call is timeout-bounded (#2387).

The upstream push helper (``~/my-goat/scripts/telegram_push.sh``) runs curl with
no ``--connect-timeout`` / ``--max-time``, so a connected-but-stalled endpoint
would park a cron wrapper forever. Task #2387 wrapped every push EXECUTION site
in ``timeout --kill-after=5s "${PUSH_TIMEOUT}s"`` with a per-wrapper
``PUSH_TIMEOUT="${EPS_PUSH_TIMEOUT_SECS:-30}"`` definition (30 s matches the
bound the Python callers — poll_pipeline.py, vm_disk_guard.py, gcp_audit.py,
sync_repo_root.py — already pass to the same helper).

Three durability pins:

- ``test_every_push_call_site_is_timeout_bounded`` — scans the EXECUTABLE
  RENDERING of each wrapper (see THE ENGINE below) for the push variable in
  EXECUTION position and asserts the EXACT bound
  ``timeout --kill-after=5s "${PUSH_TIMEOUT}s"`` immediately precedes it,
  for EVERY match on the line, at an EXACT per-wrapper site count. This is
  the ONLY coverage vehicle for the two watch scripts: they hardcode the
  push path (no env seam) and MUST NOT be executed by tests — their terminal
  arms run ``crontab -l | grep -v ... | crontab -``, which would mutate the
  real user crontab (non-execution of the engine is itself pinned:
  ``test_rendering_never_executes``).
- ``test_every_wrapper_defines_push_timeout`` — each wrapper defines the
  env-overridable bound.
- ``test_every_wrapper_parses`` — ``bash -n`` per wrapper (pure parse, no
  execution): the independent syntax gate, and the reason the engine's
  refusal path stays quiet on the live tree.

Why the site check is exact on five axes (rounds 1-5 of #2387; unchanged):

- EXACT PREFIX, not substring-anywhere-before. A membership test such as
  ``"timeout --kill-after=" in line[:match.start()]`` accepts materially
  wrong commands — most sharply ``timeout --kill-after=5s 0s "$PUSH" ...``,
  where GNU ``timeout`` reads duration ``0`` as "no time limit at all", so
  the line reads as bounded while the stall this task exists to bound is
  fully reinstated. ``_BOUND_PREFIX_RE`` anchors the two duration tokens to
  the end of the text preceding the match, tolerating only whitespace.
- EVERY match per line, not the first. ``re.search`` returns one match, so a
  second push execution appended to an already-bounded line was never
  checked.
- EXACT per-wrapper counts, not ``>= 1``. A ``>= 1`` floor cannot detect the
  silent DELETION of a site: the pinned 12 execution sites across 6 wrappers
  would still "pass" at 6. The counts below are the pinned inventory; adding
  or removing a push call is a deliberate edit here.
- EXECUTABLE TEXT ONLY, not the raw source. A commented-out push keeps both
  its regex match and its ``timeout`` prefix while bash stops running the
  alert and ``bash -n`` stays green; counting it would report a full
  inventory for a wrapper whose alert is silently disabled. Comments are
  removed by BASH'S OWN PARSER (the engine below), never by a re-implemented
  comment grammar.
- SHELL-VALID WHITESPACE (``[ \\t]``), not ``\\s``. Python's ``\\s`` accepts
  U+00A0 and its Unicode siblings, which bash does NOT treat as token
  separators: a no-break space between ``"${PUSH_TIMEOUT}s"`` and the push
  variable glues them into one word, so ``timeout`` receives an invalid
  duration and exits 125 while a ``\\s`` pattern still reads the line as
  correctly bounded. The engine preserves such bytes (they are word content
  to bash), so both NBSP failure shapes stay loud.

THE ENGINE (round 8): tokenization is DELEGATED to the installed bash.

``scan_execution_sites`` feeds the text to ``bash --pretty-print`` — a
PARSE-ONLY mode that builds the real parse tree and prints the program bash
would execute, with comments discarded by the parser itself — and runs the
site + prefix regexes over THAT rendering. Rounds 4-7 instead re-implemented
bash's lexical contexts by hand, line by line, and each round's enumeration
was falsified by the next round's measured member: round 4 the
substitution-closing ``)``; round 5 the case-pattern ``)``, escaped
metacharacters, and regex-group ``)``; round 6 backticks, substitutions
nested in ``${...}`` defaults, and ANSI-C quotes; round 7 typed context
frames — and still arrays, extglob, and every construct crossing a line
boundary silently over-stripped (0 sites reported while bash runs the push).
Five consecutive rounds of members is a CLASS, not a punch list: a hand
lexer re-opens on every paren context and line-boundary rule it does not
enumerate, and the only tokenizer guaranteed to agree with bash is bash.
Under the delegated engine the round-7 blockers resolve by construction:

- ``x=(a b)#tag && "$PUSH" ...`` (array assignment) and
  ``[[ "$x" == @(a|b)#tag ]] && "$PUSH" ...`` (conditional extglob — bash
  enables extended patterns while parsing ``[[ ]]``) PARSE, so the push
  appears in the rendering and is counted.
- Multi-line state is the parser's: ``\\`` continuations are joined, a
  multi-line ``$(...)`` keeps its site on the closing line of the rendered
  word, and an UNCLOSED quote/substitution/heredoc at EOF is a bash parse
  error — the exact end-of-line residual round 7 disclosed and could not
  refuse (a per-line refusal would have false-refused the wrappers' own
  multi-line substitutions; a whole-document parse refuses only genuine
  unclosed state).
- ``x=@(a|b)#tag`` OUTSIDE ``[[ ]]`` needs a runtime ``shopt -s extglob``
  that a parse-only rendering never executes, so bash exits non-zero and the
  scan REFUSES loudly.

FAIL-LOUD CONTRACT: anything bash cannot parse makes ``--pretty-print`` exit
non-zero and ``scan_execution_sites`` raise ValueError carrying the origin
plus bash's own diagnosis (which names the SOURCE line). A refusal can
neither silently pass an unbounded push nor silently drop a live one. The
rounds-4-7 refusal roster (backticks, ``case`` in a substitution, ``=~``
regex words, process substitutions in ``${...}``) is GONE as a refusal
class: bash parses all of those, so they are now modeled — each former
refusal fixture below asserts the measured bash behavior instead.

DIRECTION RULE (measured, rounds 4-7; still the analytical frame): a
scanner-vs-bash divergence is SILENT exactly when it leaves the wrapper's
site count sitting AT its pin, LOUD when it moves the count off the pin.
The OVER-STRIP quadrants — the recurring silent channel, a scanner dropping
text bash executes — are closed structurally: there is no strip step left
to diverge; every executed command appears in bash's own rendering of the
parse tree. What remains is at the MATCH level, disclosed below.

Known residuals after round 8, each classified by the direction rule:

- QUOTED / HEREDOC-BODY push-shaped TEXT is counted. The site regex is
  textual over the rendering, and bash renders string content and heredoc
  bodies verbatim, so ``echo '"$PUSH" "m"'`` or a push-shaped heredoc body
  line satisfies ``_EXEC_SITE`` without executing. ADDING such text moves
  the count ABOVE the pin — loud (pinned:
  ``test_heredoc_body_push_counts_as_an_inert_site``). QUOTING-OUT an
  inventory site keeps the count AT the pin — the one remaining silent
  channel at the lexical level, unchanged since round 1 (no prior round
  quote-tracked the match scan either; rounds 4-7 tracked quotes only for
  comment stripping), pinned as a disclosure by
  ``test_quoted_push_shaped_text_counts_the_disclosed_silent_residual``.
- LINE NUMBERS are rendering-relative. Bash reflows the program (comments
  dropped, continuations joined, case arms indented), so count/prefix
  failures report the rendered line TEXT; parse refusals carry bash's own
  SOURCE line number in the message.
- WORD-INTERIOR line structure is preserved on bash 5.1 (a multi-line
  ``$(...)`` word keeps its raw newlines): a bounded push split across a
  continuation INSIDE a command-substitution word renders with prefix and
  site on different lines and false-FAILs the prefix assert — loud, never
  silent. Expectations are pinned to the INSTALLED bash — the same binary
  that executes the wrappers, which identity is the point of delegating.
- ``$'...'`` words are re-rendered in POSIX quoting (measured:
  ``$'a\\'b'`` renders ``'a'\\''b'``) — content-preserving, scan-neutral.

None of the residual or refused shapes is in the six wrappers (census
re-derived round 8 through the delegated engine, 2026-08-30, bash 5.1.16):
all six render with zero refusals to exactly the pinned counts
1/3/2/2/2/2 = 12, every site bounded. Zero ``NAME=(`` arrays, no
``shopt -s extglob``, zero heredoc bodies or quoted strings containing
push-shaped text (the one heredoc, ``cron_watch_issue_1739.sh``, holds a
bare variable). The two multi-line command substitutions
(``cron_watch_issue_2091.sh``, ``cron_watch_issue_1739.sh``) and the
double-quoted brace-nested substitution (``cron_daily_healthcheck.sh``)
all parse and render clean.

COVERAGE BOUNDARY: the scan is bounded by the WRAPPERS mapping below plus
the execution-site regex. A NEW cron wrapper calling the push helper must be
ADDED to the mapping or it escapes every pin here; likewise a call shape
whose message argument is not a double-quoted string immediately after the
push variable (an unquoted message, a renamed variable) escapes the regex.
Extend both when adding either.

Behavioral twins (sleeping-stub tests, one per call-site composition shape)
live in tests/test_cron_step9c_ledger_refresh.py (if-condition),
tests/test_cron_lesson_consolidate.py (||-chained fatal arm), and
tests/test_codex_auto_upgrade.py (command substitution).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Contiguous repo-relative paths (load-bearing for select_step9c_tests.py's
# literal-path arm: a future diff to any wrapper below auto-selects this file),
# each mapped to its EXACT expected number of push execution sites.
WRAPPERS: dict[str, int] = {
    "scripts/cron_codex_auto_upgrade.sh": 1,
    "scripts/cron_daily_healthcheck.sh": 3,
    "scripts/cron_lesson_consolidate.sh": 2,
    "scripts/cron_step9c_ledger_refresh.sh": 2,
    "scripts/cron_watch_issue_1739.sh": 2,
    "scripts/cron_watch_issue_2091.sh": 2,
}

# The task's own inventory: 12 execution sites across the 6 wrappers above.
# Pinned separately so a per-wrapper count edit has to reckon with the total.
#
# 10 -> 12 when #2386's `fatal()` helper landed on main mid-round, adding one
# unbounded push each to cron_daily_healthcheck.sh and
# cron_step9c_ledger_refresh.sh. Both halves of the guard fired in sequence per
# wrapper — the prefix assert first (each new site was unbounded), then this
# count pin (each site was new) — which is the intended behavior and the reason
# the count is pinned per wrapper rather than inferred. cron_codex_auto_upgrade
# also gained a #2386 fail-loud probe, but its arm reuses the already-bounded
# push, so its count is unchanged at 1. Round 8 re-derived the census
# independently through the delegated engine: 1/3/2/2/2/2 = 12, unchanged.
TOTAL_EXPECTED_SITES = 12

# Push variable in EXECUTION position: the quoted variable followed by a
# quoted message argument. `[ -x "$PUSH" ]` guards do not match (a `]`, not a
# quote, follows) and `"${PUSH_TIMEOUT}s"` does not match (`_` after PUSH).
# Separator is `[ \t]`, never `\s`: Python's `\s` accepts U+00A0, which bash
# does not treat as a token separator (module docstring, fifth axis).
_EXEC_SITE = re.compile(r'"\$\{?(?:TELEGRAM_PUSH|PUSH)\}?"[ \t]+"')

# Human-readable canonical form, quoted in assertion messages.
_BOUND_PREFIX_TEXT = 'timeout --kill-after=5s "${PUSH_TIMEOUT}s" '

# The checker: both duration tokens, in order, anchored to the END of the text
# preceding the push variable. Only inter-token spaces/tabs are free — a wrong
# duration (`0s`), an extra argument, any intervening command, or a
# shell-invalid separator such as U+00A0 fails. Command separators in a
# rendered list (`;`, `&&`, `||`, `|`) also break the anchor, so a preceding
# command's `"${PUSH_TIMEOUT}s"` tail can never vouch for the next command's
# push.
_BOUND_PREFIX_RE = re.compile(r'timeout[ \t]+--kill-after=5s[ \t]+"\$\{PUSH_TIMEOUT\}s"[ \t]+$')


def _bash_rendering(text: str, origin: str = "<text>") -> str:
    """Return bash's own executable rendering of ``text`` (module docstring,
    THE ENGINE): the program printed from the real parse tree by
    ``bash --pretty-print``, comments discarded by the parser, ``\\``
    continuations joined, nothing executed.

    Raises ValueError — the fail-loud contract — when bash cannot parse the
    text, with ``origin`` plus bash's stderr (which names the SOURCE line of
    the offending construct). A non-zero exit is the ONLY refusal channel;
    there is no construct list to fall out of sync with bash's grammar.
    """
    proc = subprocess.run(
        ["bash", "--pretty-print", "/dev/stdin"],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"{origin}: bash cannot parse this text (--pretty-print exited "
            f"{proc.returncode}); the scan refuses rather than guessing at "
            f"constructs outside bash's grammar. bash says: {proc.stderr.strip()}"
        )
    return proc.stdout


def scan_execution_sites(
    text: str,
    origin: str = "<text>",
) -> list[tuple[int, str, re.Match[str]]]:
    """Return ``(rendered_lineno, rendered_line, match)`` per push EXECUTION
    site of ``text``, scanning bash's own rendering of the program.

    Comments never reach the scan (bash's parser drops them), so a
    commented-out push cannot count toward the pinned inventory; text bash
    would execute always reaches it, so a live push cannot be silently
    dropped by a lexer divergence — the round-4-through-7 failure class.
    Line numbers are rendering-relative (module docstring, residuals).

    Raises ValueError on text bash cannot parse (see ``_bash_rendering``).
    """
    rendered = _bash_rendering(text, origin)
    return [
        (lineno, line, m)
        for lineno, line in enumerate(rendered.splitlines(), start=1)
        for m in _EXEC_SITE.finditer(line)
    ]


def test_every_push_call_site_is_timeout_bounded():
    """EVERY push execution match in EVERY wrapper's executable rendering is
    immediately preceded by the exact bound, and each wrapper holds exactly
    its pinned number of sites (so a deleted site fails loud)."""
    assert sum(WRAPPERS.values()) == TOTAL_EXPECTED_SITES, (
        f"per-wrapper counts sum to {sum(WRAPPERS.values())}, not the pinned "
        f"{TOTAL_EXPECTED_SITES}: update TOTAL_EXPECTED_SITES deliberately"
    )
    for rel, expected in WRAPPERS.items():
        sites = scan_execution_sites((_REPO_ROOT / rel).read_text(), origin=rel)
        n_sites = len(sites)
        for _, line, m in sites:
            assert _BOUND_PREFIX_RE.search(line[: m.start()]) is not None, (
                f"{rel}: push execution not immediately preceded by "
                f"{_BOUND_PREFIX_TEXT!r} (a wrong duration such as '0s' means no "
                f"deadline at all, and only spaces/tabs separate the tokens); "
                f"rendered line: {line.strip()!r}"
            )
        assert n_sites == expected, (
            f"{rel}: found {n_sites} push execution site(s), expected {expected} — "
            "a site was added, deleted, COMMENTED OUT, or written with a "
            "shell-invalid separator, or the regex drifted; update WRAPPERS "
            "(and TOTAL_EXPECTED_SITES) deliberately"
        )


def test_every_wrapper_defines_push_timeout():
    """Each wrapper defines the env-overridable 30 s default bound."""
    for rel in WRAPPERS:
        text = (_REPO_ROOT / rel).read_text()
        assert 'PUSH_TIMEOUT="${EPS_PUSH_TIMEOUT_SECS:-30}"' in text, (
            f"{rel}: missing the PUSH_TIMEOUT definition"
        )


def test_every_wrapper_parses():
    """`bash -n` (parse-only, nothing executed) per wrapper — the independent
    syntax gate; a wrapper it passes cannot trip the engine's refusal path."""
    for rel in WRAPPERS:
        proc = subprocess.run(
            ["bash", "-n", str(_REPO_ROOT / rel)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, f"{rel}: bash -n failed:\n{proc.stderr}"


# --- Engine invariants -------------------------------------------------------


def test_rendering_never_executes(tmp_path):
    """The engine is parse-only. Load-bearing, not a nicety: the two watch
    scripts' terminal arms rewrite the real user crontab, and this scan is
    their only coverage vehicle precisely BECAUSE no test may execute them.
    A canary write proves `--pretty-print` builds the parse tree without
    running a single command."""
    canary = tmp_path / "canary.txt"
    sites = scan_execution_sites(f'echo executed > "{canary}"\ndate >> "{canary}"\n')
    assert sites == []
    assert not canary.exists(), "--pretty-print EXECUTED the text it was asked to parse"


def test_unparseable_text_refuses_naming_origin_and_bash_line():
    """The fail-loud contract: text bash cannot parse raises, with the origin
    and bash's own source-line diagnosis — never a silent zero-site return.

    The three shapes cover the class's three faces: a construct needing a
    runtime `shopt` (extglob assignment — round 7's silent member), unclosed
    multi-line state at EOF (the round-7 disclosed residual: per-line refusal
    was impossible without false-refusing the wrappers' own multi-line
    substitutions; whole-document parsing refuses only genuine unclosed
    state), and a plain syntax error.
    """
    for label, text in (
        ("extglob assignment", 'x=@(a|b)#tag && "$PUSH" "EXTGLOB_ASSIGNMENT_RAN"\n'),
        ("unclosed substitution at EOF", "x=$(printf a\n"),
        ("comment eats a redirect target", 'true ># "$PUSH" "msg"'),
    ):
        with pytest.raises(ValueError, match=r"wrapper\.sh: bash cannot parse"):
            scan_execution_sites(text, origin="wrapper.sh")
        try:
            scan_execution_sites(text, origin="wrapper.sh")
        except ValueError as err:
            assert "line" in str(err), (label, str(err))


# --- Fixtures (bash ground truth; the real tree is never mutated) ------------
#
# `_LIVE` is the live retirement push of `cron_watch_issue_2091.sh`,
# BYTE-FAITHFUL to its source line: the `[ -x "$PUSH" ]` guard the
# execution-site regex must not count, and the `#` inside the trailing
# message. `_LIVE_HASH_BEFORE_SITE` is the shape where quoted-`#` handling
# decides: a `#` inside a quoted argument BEFORE the push variable — the
# wrappers' own logging idiom fused onto one line — where a naive first-hash
# truncation would drop a live, correctly-bounded site.
#
# Both fixtures are pinned against the live file by
# `test_live_fixtures_are_byte_faithful_to_their_source_lines`, which locates
# the lines by CONTENT (line numbers drift). Round 4 described them as built
# from those lines while silently substituting an ASCII hyphen for the
# source's em dash and a literal `done` for `${status}`; a control whose
# warrant is "this is the real line" has to be the real line. The em dash is
# written as an escape for the same reason `_NBSP` is: an ASCII-looking
# non-ASCII byte in source is unreviewable.
_LIVE_SOURCE = "scripts/cron_watch_issue_2091.sh"

_LIVE = (
    '    [ -x "$PUSH" ] && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" '
    '"EPS #${ISSUE} reached ${status} \u2014 monitor retired." >/dev/null 2>&1'
)

_LIVE_HASH_BEFORE_SITE = (
    '    echo "$(ts) #${ISSUE} TERMINAL (${status}) \u2014 removing this cron" >> "$LOG" '
    '&& [ -x "$PUSH" ] && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" '
    '"EPS #${ISSUE} reached ${status} \u2014 monitor retired." >/dev/null 2>&1'
)

# Written as an escape, never as a literal: an invisible U+00A0 in source
# is unreviewable. Python's `\s` matches it; bash does not split on it.
_NBSP = "\u00a0"


def _bound_flags(text: str) -> list[bool]:
    """Per scanned site, whether the exact timeout prefix precedes it."""
    return [
        _BOUND_PREFIX_RE.search(line[: m.start()]) is not None
        for _, line, m in scan_execution_sites(text)
    ]


def test_live_push_line_scans_as_one_bounded_site():
    """The unmutated control: one site, bound satisfied, and the `[ -x "$PUSH" ]`
    guard on the same line not counted as a second site."""
    assert _bound_flags(_LIVE) == [True]


def test_commented_out_push_line_contributes_no_site():
    """Commenting a bounded push must DROP its site, not preserve the count.

    Both regex match and `timeout` prefix survive the `#`, and `bash -n`
    stays green, so a raw-source scan would report a full inventory for a
    wrapper whose alert no longer runs. Bash's parser discards the comment,
    so the rendering never contains it.
    """
    assert scan_execution_sites("#" + _LIVE) == []
    assert scan_execution_sites("  # " + _LIVE.strip()) == []


def test_push_inside_a_trailing_comment_contributes_no_site():
    assert scan_execution_sites('echo hi  # "$PUSH" "msg"') == []


def test_nbsp_before_the_push_variable_is_not_a_valid_bound():
    """`"${PUSH_TIMEOUT}s"<NBSP>"$PUSH"` is one bash word — timeout then gets
    an invalid duration and exits 125 — and the rendering preserves the byte
    (word content to bash), so the line is NOT bounded."""
    mutant = _LIVE.replace('}s" "$PUSH"', '}s"' + _NBSP + '"$PUSH"')
    assert mutant != _LIVE
    assert _bound_flags(mutant) == [False]


def test_nbsp_inside_the_timeout_prefix_is_not_a_valid_bound():
    mutant = _LIVE.replace("timeout --kill-after", "timeout" + _NBSP + "--kill-after")
    assert mutant != _LIVE
    assert _bound_flags(mutant) == [False]


def test_nbsp_after_the_push_variable_drops_the_site():
    """NBSP where the execution-site regex expects a separator: the site
    disappears, so the exact per-wrapper count fails loud."""
    mutant = _LIVE.replace('"$PUSH" "EPS', '"$PUSH"' + _NBSP + '"EPS')
    assert mutant != _LIVE
    assert scan_execution_sites(mutant) == []


def test_tabs_and_repeated_spaces_stay_accepted():
    """Round 2 deliberately relaxed the whitespace axis; keep it relaxed.
    (The rendering normalizes inter-word tabs to single spaces — still
    matched by `[ \\t]+`, so the relaxation is preserved end to end.)"""
    mutant = _LIVE.replace(
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH"',
        'timeout\t--kill-after=5s   "${PUSH_TIMEOUT}s"\t"$PUSH"',
    )
    assert mutant != _LIVE
    assert _bound_flags(mutant) == [True]


def test_zero_duration_reads_as_unbounded():
    """GNU timeout reads duration `0` as "no time limit at all"."""
    assert _bound_flags(_LIVE.replace('"${PUSH_TIMEOUT}s"', "0s")) == [False]


def test_second_push_on_an_already_bounded_line_is_checked():
    """`finditer`, not `search`: an appended unbounded push must fail — and a
    rendered `; `-separated list cannot borrow the first command's prefix
    (the separator breaks the prefix anchor)."""
    assert _bound_flags(_LIVE + ' ; "$PUSH" "unbounded"') == [True, False]


def test_quoted_hash_before_the_site_keeps_the_site():
    """A `#` inside a quoted argument BEFORE the push variable — the wrappers'
    own logging idiom — must not hide the site: bash keeps quoted text, so
    the rendering carries the full line and the one bounded site scans.
    (Rounds 3-7 bought this with hand-rolled quote tracking; the delegated
    engine gets it from the parser that defines the semantics.)"""
    assert _bound_flags(_LIVE_HASH_BEFORE_SITE) == [True]


def test_push_after_a_subshell_close_paren_contributes_no_site():
    """`(true)#` opens a comment: bash never runs the push (measured,
    bash 5.1.16: `bash -c '(true)# "$PUSH" "msg"'` prints nothing), and the
    parser drops it from the rendering."""
    assert scan_execution_sites("(true)#" + _LIVE.strip()) == []


def test_push_in_a_case_arm_comment_contributes_no_site():
    """The same shape inside a `case` arm — the watch scripts' own dispatch
    shape, where a disabled alert would most plausibly hide. (The arm needs
    its enclosing `case` to parse; a bare arm is not a bash program.)"""
    text = 'case "$s" in\n  awaiting_promotion)#' + _LIVE.strip() + "\n  ;;\nesac\n"
    assert scan_execution_sites(text) == []


def test_word_end_characters_before_a_hash_comment_out_the_push():
    """Bash starts a comment at a `#` that starts a word — after whitespace,
    `;`, `&`, or a control operator. One behavioral case per separator that
    forms a valid program; the separators whose `#` eats a required operand
    (`|#`, `(#`, `)#`, `>#`, `<#`) are bash SYNTAX ERRORS and route through
    the refusal path instead (`test_unparseable_text_refuses...` pins one),
    never through a silent count."""
    for label, text in (
        ("space", "true #" + _LIVE.strip()),
        ("tab", "true\t#" + _LIVE.strip()),
        ("semicolon", "true;#" + _LIVE.strip()),
        ("ampersand", "true &#" + _LIVE.strip()),
    ):
        assert scan_execution_sites(text) == [], f"{label} did not end the word"


def test_added_push_after_a_substitution_close_is_counted_and_reads_unbounded():
    """The round-4 silent quadrant, kept closed: a substitution's closing `)`
    belongs to its word, so bash runs each push below (measured: `echo
    $(echo hi)#tag` prints `hi#tag` and the `&&` arm executes). The count
    RISES to 2 and the added site reads UNBOUNDED."""
    for label, added in (
        ("command substitution", 'echo $(date +%s)#stamp && "$PUSH" "EPS #1 new alert"'),
        ("arithmetic expansion", 'echo $((1+2))#stamp && "$PUSH" "EPS #1 new alert"'),
        ("process substitution", 'echo <(echo hi)#stamp && "$PUSH" "EPS #1 new alert"'),
        ("assignment form", 'U=$(date +%s)#frag; "$PUSH" "EPS #1 new alert"'),
    ):
        text = _LIVE + "\n    " + added
        assert len(scan_execution_sites(text)) == 2, label
        assert _bound_flags(text) == [True, False], label


def test_a_comment_after_a_substitution_close_still_drops_the_site():
    """The word-content reading covers the `)` itself, not the rest of the
    line: `echo $(true) # "$PUSH" ...` runs NO push (measured) — the `#`
    starts a word after the SPACE."""
    assert scan_execution_sites("echo $(true) # " + _LIVE.strip()) == []


def test_an_arithmetic_command_close_is_not_a_substitution_close():
    """A bare `((` is an arithmetic COMMAND; only `$((` opens an expansion.
    Measured: `((i=1))#echo LEAK` prints nothing and exits 0, so the `))`
    are ordinary word ends and the `#` opens a comment."""
    assert scan_execution_sites("((i=1))#" + _LIVE.strip()) == []


def test_nested_parens_inside_a_substitution_do_not_leak_the_exemption():
    """A subshell or arithmetic group NESTED in a substitution stays matched:
    `echo $( (true) )#tag && push` and `echo $(( (1+2) * 3 ))#tag && push`
    both run the push (measured), so both sites are counted."""
    for label, text in (
        ("subshell in cmdsub", 'echo $( (true) )#tag && "$PUSH" "EPS #1 alert"'),
        ("group in arithmetic", 'echo $(( (1+2) * 3 ))#tag && "$PUSH" "EPS #1 alert"'),
    ):
        assert _bound_flags(text) == [False], label


def test_case_inside_a_substitution_is_modeled():
    """The round-5 killer, now MODELED instead of refused: bash 5.1.16 RUNS
    this push — the case-pattern `)` has no matching `(`, `esac)` closes the
    substitution, `#tag` is mid-word. Rounds 5-7 could only refuse (case
    grammar was outside the hand lexer); the delegated engine parses it and
    counts the one bounded site."""
    line = (
        "STAMP=$(case x in y) date +%s;; esac)#tag && "
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L1_PUSH_RAN"'
    )
    assert _bound_flags(line) == [True]


def test_escaped_metachar_before_a_hash_is_not_a_word_start():
    """Round-5 member, kept closed: bash runs this push — the escaped `;` is
    a word character, so the `#` is mid-word and the site counts bounded."""
    line = 'echo a\\;#x && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L5b_ESCAPED_RAN"'
    assert _bound_flags(line) == [True]


def test_regex_rhs_is_modeled():
    """Round-5 member, now MODELED instead of refused: after `=~` the
    right-hand side is a regex word where `(x)#tag` is literal, and bash
    runs the chained push (measured, with and without a space after the
    operator). Both spellings scan to their one bounded site."""
    line = (
        '[[ "x#tag" =~ (x)#tag ]] && '
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L6_REGEX_RAN"'
    )
    assert _bound_flags(line) == [True]
    assert _bound_flags(line.replace("=~ (x)", "=~(x)")) == [True]


def test_pattern_close_paren_inside_a_parameter_expansion_stays_literal():
    """Bash scans `${...}` to its matching `}`, so the pattern `)` in
    `${x%)}` never closes the substitution and the push after `)#tag` runs
    (measured: P4_PARAM_RAN)."""
    line = 'STAMP=$(echo ${x%)})#tag && "$PUSH" "P4_PARAM_RAN"'
    assert len(scan_execution_sites(line)) == 1


def test_top_level_case_and_tilde_assignment_scan():
    """The wrappers' own shapes keep scanning: a top-level `case` (incl. the
    one-line `case "$seen_at" in ...` of cron_watch_issue_1739.sh) and a
    `VAR=~/path` tilde assignment. Both measured: the case arm's push runs,
    the assignment is an ordinary assignment."""
    top_case = 'case "$s" in y) echo $(date +%s)#t && "$PUSH" "P8_RAN";; esac'
    assert len(scan_execution_sites(top_case)) == 1
    line = "LOG=~/x.log && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_backtick_substitutions_are_modeled():
    """Round-6 members, now MODELED instead of refused: bash RUNS all three
    pushes below — the closing-tick search runs through the `#x` (it
    comments only the INNER command), and backticks stay live substitutions
    inside double quotes. The rendering keeps the backtick words verbatim
    (their inner `#` is word content, not a rendered comment), so each push
    is counted with its true bound flag."""
    a2 = 'Y=`date +%s #x` && "$PUSH" "A2_TICK_UNBOUNDED_RAN"'
    assert _bound_flags(a2) == [False]
    a1 = (
        "Y=`date +%s #x` && "
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "A1_TICK_BOUNDED_RAN"'
    )
    assert _bound_flags(a1) == [True]
    assert _bound_flags('echo "run `date +%s` now" && "$PUSH" "BT_RAN"') == [False]


def test_backticks_in_comments_and_single_quotes_scan():
    """SCOPE CONTROL (passes on the round-7 scanner too — recorded so it is
    never mistaken for regression evidence): the six wrappers' own backticks
    all sit on comment lines, and a single-quoted backtick is literal to
    bash; both shapes scan without refusal."""
    assert scan_execution_sites("# a `quoted` word in a comment") == []
    assert scan_execution_sites("true # see `foo --help` for detail") == []
    line = "echo 'a `b` c' && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_substitution_nested_in_a_parameter_default_is_modeled():
    """Round-6 member, kept closed: bash RUNS this push — the nested
    `$( { echo hi; })` closes where it opened, and `#tag` is part of the
    assignment word after the REAL outer close. Counted, and UNBOUNDED —
    loud on a newly added site."""
    line = 'x=$(echo ${v:-$( { echo hi; })})#tag && "$PUSH" "NEW_UNBOUNDED"'
    assert _bound_flags(line) == [False]


def test_quoted_parameter_default_with_nested_substitution_scans():
    """SCOPE CONTROL (passes on the round-7 scanner too): the live idiom of
    cron_daily_healthcheck.sh — `$(date ...)` nested inside a DOUBLE-QUOTED
    `${...:-...}` default — scans to its one bounded site with no refusal."""
    line = 'V="${VAR:-$(date +%s)}" && ' + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_ansi_c_quoting_is_modeled():
    """Round-6 member, kept closed: inside `$'...'` the `\\'` is an ESCAPED
    quote, so the ` ; # x` sits INSIDE the string and bash runs the bounded
    push. (The rendering re-quotes `$'...'` words in POSIX form — measured:
    `$'a\\'b'` renders `'a'\\''b'` — which is content-preserving and keeps
    the `#` inside quotes.)"""
    line = (
        'echo $\'a\\\' ; # x\' && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "B1_ANSI_RAN"'
    )
    assert _bound_flags(line) == [True]


def test_regular_single_quote_keeps_backslash_literal():
    """SCOPE CONTROL (passes on the round-7 scanner too): in a REGULAR single
    quote a backslash is literal and the next `'` closes — bash prints `a\\`
    for `echo 'a\\'` and runs the chained push (measured)."""
    line = "echo 'a\\' && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_quotes_inside_a_double_quoted_substitution_do_not_desync():
    """Round-7 members, kept closed: the inner quotes of a substitution or
    expansion opened inside double quotes belong to the reparsed interior,
    so the `)#` inside them never reaches word level and bash runs both
    pushes (measured)."""
    for label, prefix in (
        ("command substitution", 'echo "$(printf "%s" "a)#b")" && '),
        ("parameter default", 'echo "${v:-"a)#b"}" && '),
    ):
        line = prefix + _LIVE.strip()
        assert _bound_flags(line) == [True], label


def test_process_substitution_inside_a_parameter_default_is_modeled():
    """Round-7 refusal, now MODELED: `echo ${v:-<(true)}` substitutes a live
    /dev/fd path and its `&&` push runs (measured) — counted, unbounded."""
    assert _bound_flags('echo ${v:-<(true)} && "$PUSH" "PS_RAN"') == [False]


# --- Round-8 regressions ------------------------------------------------------
#
# The two round-7 BLOCKERs, member by member. Every fixture below was measured
# against bash 5.1.16 (`bash -n` rc 0 — `-O extglob` for the extglob
# assignment — and the push RUNS via a stubbed helper, per the round-7
# verdict), and every test in this block FAILS against the round-7 scanner
# (tip a7ba37166f3): it returned 0 sites with nothing raised on each starred
# shape — the silent fail-open this round replaces the engine to close. The
# measured pre-fix/post-fix result per test is recorded in the round-8
# implementation marker.


def test_array_assignment_close_paren_is_not_a_comment_start():
    """r7 BLOCKER `unmodeled-paren-context-fail-open`, member 1 (*): the `)`
    closing `x=(a b)` belongs to the assignment, `#tag` is not a comment,
    and bash runs the chained push. Counted, unbounded."""
    assert _bound_flags('x=(a b)#tag && "$PUSH" "ARRAY_RAN"') == [False]


def test_conditional_extglob_site_is_counted():
    """r7 BLOCKER `unmodeled-paren-context-fail-open`, member 3 (*): bash
    enables extended patterns while PARSING `[[ ]]`, so `@(a|b)#tag` is one
    pattern word, no comment opens, and the push runs. Counted, unbounded.
    (This also retires round 7's claim that remaining `[[ ]]` grammar had no
    silent channel — the r7 verdict disproved it with this fixture.)"""
    text = 'x="a#tag"\n[[ "$x" == @(a|b)#tag ]] && "$PUSH" "CONDITIONAL_EXTGLOB_RAN"\n'
    assert _bound_flags(text) == [False]


def test_extglob_assignment_refuses_loudly():
    """r7 BLOCKER `unmodeled-paren-context-fail-open`, member 2 (*): outside
    `[[ ]]`, `x=@(a|b)` needs a runtime `shopt -s extglob` a parse-only
    rendering never executes, so bash cannot parse it and the scan REFUSES —
    loud, never a silent zero. (A wrapper legitimately using extglob would
    refuse here too: a disclosed false-refusal cost, taken deliberately over
    the fail-open.)"""
    with pytest.raises(ValueError, match="cannot parse"):
        scan_execution_sites('x=@(a|b)#tag && "$PUSH" "EXTGLOB_ASSIGNMENT_RAN"')


def test_multiline_substitution_closing_line_site_is_counted():
    """r7 BLOCKER `multiline-shell-state-fail-open`, member 1 (*): the
    round-7 scanner reset state at every physical line, popped an empty
    stack at line 2's `)`, and stripped the live push at `#tag`. The
    whole-document parse keeps the substitution open across the newline, so
    the closing line's push is counted (unbounded here)."""
    text = 'x=$(printf a\nprintf b)#tag && "$PUSH" "MULTILINE_RAN"\n'
    assert _bound_flags(text) == [False]


def test_backslash_continuation_is_joined_before_scanning():
    """r7 BLOCKER `multiline-shell-state-fail-open`, member 2 (*): bash joins
    `\\`-continuations before tokenizing, so `#tag` on the continued line is
    mid-word — the push runs and is counted (unbounded). The bounded
    variant is the same fix from the false-alarm side: a bounded push whose
    prefix and variable sit on either side of a continuation is ONE rendered
    command and reads BOUNDED (the round-7 scanner read it unbounded — a
    false alarm, wrong in the loud direction)."""
    unbounded = 'x=$(printf a)\\\n#tag && "$PUSH" "CONTINUATION_RAN"\n'
    assert _bound_flags(unbounded) == [False]
    bounded = 'timeout --kill-after=5s "${PUSH_TIMEOUT}s" \\\n  "$PUSH" "msg"\n'
    assert _bound_flags(bounded) == [True]


# --- Direction-rule disclosures (scope controls, not regression evidence) ----


def test_heredoc_body_push_counts_as_an_inert_site():
    """DISCLOSED residual, loud direction (passes on the round-7 scanner too):
    bash never executes heredoc body text, but the rendering carries it
    verbatim and the site regex is textual, so a push-shaped body line is an
    over-COUNT of an inert site. ADDING one moves the count ABOVE the pin —
    loud, a human looks — never a silent pass of an unbounded push."""
    text = 'cat <<EOF\n"$PUSH" "HEREDOC_BODY_MUST_NOT_RUN"\nEOF\n'
    sites = scan_execution_sites(text)
    assert len(sites) == 1


def test_quoted_push_shaped_text_counts_the_disclosed_silent_residual():
    """DISCLOSED residual, the ONE remaining silent channel at the lexical
    level (module docstring): push-shaped text inside a quoted string
    satisfies the textual site regex without executing, so REPLACING an
    inventory site with its quoted-out husk keeps the count AT the pin.
    Unchanged since round 1 — no prior scanner quote-tracked the match scan
    either — and pinned here so an engine change that closes or widens it
    is a deliberate edit, not drift."""
    sites = scan_execution_sites('echo \'"$PUSH" "quoted, never executed"\'')
    assert len(sites) == 1


def test_direction_rule_loud_quadrants():
    """The direction rule's two loud quadrants, demonstrated through the
    engine against `_LIVE`'s one-site pin: disabling the inventory site
    (comment-out) drops the count BELOW the pin; adding inert push-shaped
    text (heredoc body) raises it ABOVE the pin. Silent is exactly
    count-AT-pin, which after round 8 requires the quoted-out husk above —
    there is no strip step left to produce it."""
    pin = 1
    assert len(scan_execution_sites("# " + _LIVE.strip())) == pin - 1
    text = _LIVE + '\ncat <<EOF\n"$PUSH" "inert body"\nEOF\n'
    assert len(scan_execution_sites(text)) == pin + 1


# --- Live-fixture byte fidelity ----------------------------------------------


def _source_line(anchor: str) -> str:
    """Return the one line of `_LIVE_SOURCE` containing `anchor` (exactly one)."""
    lines = [ln for ln in (_REPO_ROOT / _LIVE_SOURCE).read_text().splitlines() if anchor in ln]
    assert len(lines) == 1, (
        f"{_LIVE_SOURCE}: {len(lines)} lines contain {anchor!r}, expected exactly 1 — "
        "the fixture anchor drifted; re-anchor it before trusting the fixtures"
    )
    return lines[0]


def test_live_fixtures_are_byte_faithful_to_their_source_lines():
    """`_LIVE` / `_LIVE_HASH_BEFORE_SITE` ARE the live lines, not paraphrases.

    Their whole warrant is "this is what the wrapper really writes", so the
    claim is pinned rather than re-checked by hand each round. Round 4 asserted
    that provenance in prose while the fixtures carried an ASCII hyphen for the
    source's em dash and a literal `done` for `${status}`.

    Lines are located by CONTENT, not number, so an unrelated edit to the
    wrapper cannot silently re-point the anchors.
    """
    log_line = _source_line("TERMINAL (${status})")
    push_line = _source_line("monitor retired.")
    assert push_line == _LIVE, "_LIVE drifted from the live retirement-push line"
    assert log_line + " && " + push_line.lstrip() == _LIVE_HASH_BEFORE_SITE, (
        "_LIVE_HASH_BEFORE_SITE is the log line and the push line fused with "
        "` && `; one of the two source lines changed"
    )
