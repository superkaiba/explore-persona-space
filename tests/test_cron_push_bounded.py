"""Structural pin: every cron-wrapper Telegram push call is timeout-bounded (#2387).

The upstream push helper (``~/my-goat/scripts/telegram_push.sh``) runs curl with
no ``--connect-timeout`` / ``--max-time``, so a connected-but-stalled endpoint
would park a cron wrapper forever. Task #2387 wrapped every push EXECUTION site
in ``timeout --kill-after=5s "${PUSH_TIMEOUT}s"`` with a per-wrapper
``PUSH_TIMEOUT="${EPS_PUSH_TIMEOUT_SECS:-30}"`` definition (30 s matches the
bound the Python callers — poll_pipeline.py, vm_disk_guard.py, gcp_audit.py,
sync_repo_root.py — already pass to the same helper).

Three durability pins:

- ``test_every_push_call_site_is_timeout_bounded`` — text-scans the
  COMMENT-STRIPPED text of each wrapper for the push variable in EXECUTION
  position and asserts the EXACT bound
  ``timeout --kill-after=5s "${PUSH_TIMEOUT}s"`` immediately precedes it,
  for EVERY match on the line, at an EXACT per-wrapper site count. This is
  the ONLY coverage vehicle for the two watch scripts: they hardcode the
  push path (no env seam) and MUST NOT be executed by tests — their terminal
  arms run ``crontab -l | grep -v ... | crontab -``, which would mutate the
  real user crontab.
- ``test_every_wrapper_defines_push_timeout`` — each wrapper defines the
  env-overridable bound.
- ``test_every_wrapper_parses`` — ``bash -n`` per wrapper (pure parse, no
  execution, no crontab hazard): a syntax/quoting error introduced into the
  watch scripts is the one failure mode a text scan cannot see, and neither
  watch script is executed by any test.

Why the site check is exact on all seven axes (#2387; axes four and five are
round-3 hardening of the exact-count pin the first three introduced, the sixth
is round-4 completion of the word-start set axis four rests on, and the seventh
is round-5 correction of an over-strip the sixth introduced):

- EXACT PREFIX, not substring-anywhere-before. A membership test such as
  ``"timeout --kill-after=" in line[:match.start()]`` accepts materially
  wrong commands — most sharply ``timeout --kill-after=5s 0s "$PUSH" ...``,
  where GNU ``timeout`` reads duration ``0`` as "no time limit at all", so
  the line reads as bounded while the stall this task exists to bound is
  fully reinstated. The same hole admits an intervening argument or a
  different command between the ``timeout`` token and the push variable.
  ``_BOUND_PREFIX_RE`` anchors the two duration tokens to the end of the
  text preceding the match, tolerating only whitespace changes.
- EVERY match per line, not the first. ``re.search`` returns one match, so a
  second push execution appended to an already-bounded line (``... && push_a
  ; "$PUSH" "unbounded"``) was never checked.
- EXACT per-wrapper counts, not ``>= 1``. A ``>= 1`` floor cannot detect the
  silent DELETION of a site: the plan's 10 execution sites across 6 wrappers
  would still "pass" at 6. The counts below are the pinned inventory; adding
  or removing a push call is a deliberate edit here.
- EXECUTABLE TEXT ONLY, not the raw line. Making the counts exact is what
  put weight on the definition of "an execution site", and ``finditer`` over
  a raw line counts matches inside COMMENTS: prefixing a bounded push with
  ``#`` leaves both the regex match and the exact per-wrapper count intact
  while bash stops running the alert and ``bash -n`` stays green. Since this
  scan is the ONLY coverage vehicle for the two watch scripts, that would
  report a full inventory for a wrapper whose alert is silently disabled.
  ``_strip_bash_comment`` truncates each line at its first unquoted,
  word-initial ``#`` before any matching.
- SHELL-VALID WHITESPACE (``[ \\t]``), not ``\\s``. Python's ``\\s`` accepts
  U+00A0 and its Unicode siblings, which bash does NOT treat as token
  separators: a no-break space between ``"${PUSH_TIMEOUT}s"`` and the push
  variable glues them into one word, so ``timeout`` receives an invalid
  duration and exits 125 while the old pattern still read the line as
  correctly bounded. Tabs and repeated spaces stay accepted by design; only
  shell-invalid separators are rejected.
- BASH'S FULL METACHARACTER SET decides where a comment may start, not a
  convenient subset of it. A ``#`` opens a comment only when it STARTS a
  word, and bash ends a word at any unquoted metacharacter: ``| & ; ( ) < >``
  plus space, tab and newline (newline being the start-of-line case here).
  Rounds 1-3 used ``" \\t;&|("``, omitting ``)``, ``<`` and ``>``. ``)`` is
  the reachable omission: bash starts a comment straight after a subshell's
  closing paren (``(true)# "$PUSH" "msg"``) and after a ``case`` arm's paren
  (``awaiting_promotion)# "$PUSH" "msg"``), both verified against bash
  5.1.16. An omitted separator UNDER-strips — the scanner keeps text bash
  discards, counts a DISABLED push toward the pinned inventory, and the count
  assertion passes over an alert that no longer runs. That is precisely the
  silent failure axis four exists to prevent, so the set is now bash's.
- SUBSTITUTION-AWARE ``)``, because axis six's ``)`` is not absolute. The ``)``
  that CLOSES a command substitution, an arithmetic expansion, or a process
  substitution belongs to its word, so a ``#`` right after it is NOT a comment:
  measured against bash 5.1.16, ``echo $(echo hi)#tag`` prints ``hi#tag``, and
  ``echo $(date +%s)#stamp && "$PUSH" "..."`` RUNS the push — identically for
  ``$((1+2))#``, ``<(echo hi)#``, ``>(echo hi)#`` and the assignment form
  ``U=$(...)#frag``. Round 4 treated every ``)`` as a word end and so DROPPED
  such a line; for a NEWLY ADDED push that leaves the count sitting exactly at
  the pin and the test passes silently (direction rule below).
  ``_strip_bash_comment`` now tracks open parens and exempts only a ``)`` that
  closes a substitution — a subshell's ``)``, a ``case`` arm's ``)``, and the
  ``))`` of an arithmetic COMMAND (``((i=1))#``, a bare ``((``, not ``$((``)
  all still end a word.

Behavioral twins (sleeping-stub tests, one per call-site composition shape)
live in tests/test_cron_step9c_ledger_refresh.py (if-condition),
tests/test_cron_lesson_consolidate.py (||-chained fatal arm), and
tests/test_codex_auto_upgrade.py (command substitution).

COVERAGE BOUNDARY: the scan is bounded by the WRAPPERS mapping below plus the
execution-site regex. A NEW cron wrapper calling the push helper must be ADDED
to the mapping or it escapes every pin here; likewise a call shape whose
message argument is not a double-quoted string immediately after the push
variable (e.g. an unquoted message) escapes the regex. Extend both when adding
either.

DIRECTION RULE for any scanner-vs-bash divergence. Round 4 recorded that an
over-strip is "loud on the count assertion, never a silent pass"; that is
FALSE, and it was measured false before this round. Which way a divergence
moves depends on whether the affected site is ALREADY in the pinned inventory:

- OVER-STRIP of a site already IN the inventory — LOUD. The wrapper's count
  drops below its pin and the count assertion fails.
- OVER-STRIP of a NEWLY ADDED site — SILENT. The count stays sitting exactly
  AT the pin, so the test passes while bash runs an unbounded push. This
  defeats the PRIMARY invariant ("every push execution is timeout-bounded"),
  not merely deletion detection, and is the worse of the two directions.
- UNDER-STRIP (the scanner keeps text bash discards) — SILENT. A DISABLED
  push is counted as live, the pinned count is still met, and the test passes
  over a dead alert. This is the shape axes four and six exist to prevent.

Two residual over-strips survive axis seven. Each is loud on an existing site
and silent on a newly added one, per the rule above:

- LINE-SCOPED state. Quote AND paren state restart at each physical line, and
  a trailing ``\\`` continuation is not joined, so a string, heredoc body, or
  command substitution spanning several lines is not tracked. Measured: for
  ``X=$(echo a`` / ``echo b)#tag && "$PUSH" "..."`` bash runs the push, while
  the scanner reads line 2's ``)`` against an empty paren stack and drops the
  site.
- A ``)`` closing an ARRAY ASSIGNMENT (``x=(a b)#tag``) or an EXTGLOB pattern
  (``@(a|b)#tag``, under ``shopt -s extglob``) also belongs to its word in
  bash; the scanner treats both as word ends. Distinguishing them from a
  subshell needs assignment/pattern parsing, not the ``$``/``<``/``>`` prefix
  axis seven keys on.

None of those shapes is in the six wrappers (measured 2026-08-29): zero
``)#`` sequences, zero ``NAME=(``, zero ``[@+?!*](``, no ``shopt -s extglob``,
one heredoc (``cron_watch_issue_1739.sh`` line 119, a bare variable). Two
wrappers do open a multi-line command substitution
(``cron_watch_issue_2091.sh`` lines 30-31, ``cron_watch_issue_1739.sh`` lines
58-59); both close it with no ``#`` anywhere on the closing line. Repo-wide,
the only two ``)#`` sequences in any ``*.sh`` sit inside single-quoted ``sed``
programs, which quote tracking already skips.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Contiguous repo-relative paths (load-bearing for select_step9c_tests.py's
# literal-path arm: a future diff to any wrapper below auto-selects this file),
# each mapped to its EXACT expected number of push execution sites.
WRAPPERS: dict[str, int] = {
    "scripts/cron_codex_auto_upgrade.sh": 1,
    "scripts/cron_daily_healthcheck.sh": 2,
    "scripts/cron_lesson_consolidate.sh": 2,
    "scripts/cron_step9c_ledger_refresh.sh": 1,
    "scripts/cron_watch_issue_1739.sh": 2,
    "scripts/cron_watch_issue_2091.sh": 2,
}

# The task's own inventory: 10 execution sites across the 6 wrappers above.
# Pinned separately so a per-wrapper count edit has to reckon with the total.
TOTAL_EXPECTED_SITES = 10

# Push variable in EXECUTION position: the quoted variable followed by a
# quoted message argument. `[ -x "$PUSH" ]` guards do not match (a `]`, not a
# quote, follows) and `"${PUSH_TIMEOUT}s"` does not match (`_` after PUSH).
# Separator is `[ \t]`, never `\s`: Python's `\s` accepts U+00A0, which bash
# does not treat as a token separator (see the module docstring's fifth axis).
_EXEC_SITE = re.compile(r'"\$\{?(?:TELEGRAM_PUSH|PUSH)\}?"[ \t]+"')

# Human-readable canonical form, quoted in assertion messages.
_BOUND_PREFIX_TEXT = 'timeout --kill-after=5s "${PUSH_TIMEOUT}s" '

# The checker: both duration tokens, in order, anchored to the END of the text
# preceding the push variable. Only inter-token spaces/tabs are free — a wrong
# duration (`0s`), an extra argument, any intervening command, or a
# shell-invalid separator such as U+00A0 fails.
_BOUND_PREFIX_RE = re.compile(r'timeout[ \t]+--kill-after=5s[ \t]+"\$\{PUSH_TIMEOUT\}s"[ \t]+$')

# Bash begins a comment at a `#` that STARTS a word: at the start of the line,
# or right after an unquoted metacharacter. This is bash's whole metacharacter
# set — `| & ; ( ) < >` plus space and tab, with newline covered by `i == 0` —
# not a subset of it (module docstring, sixth axis). A `#` inside quotes
# (`"EPS #${ISSUE} ... retired."`) or mid-word (`$#`) is not a comment.
#
# Measured against bash 5.1.16: `true #`, `true<TAB>#`, `true;#`, `true &#`,
# `true|#`, `(#`, `(true)#`, `case x in x)#`, `true >#` and `true <#` all
# comment out the rest of the line, while `echo abc#echo LEAK` does NOT — a
# mid-word `#` stays literal. (`|#`, `(#`, `>#` and `<#` additionally make the
# line a syntax error, because the comment eats the operand;
# `test_every_wrapper_parses` catches that independently.) NARROWING this set
# is the SILENT direction — it under-strips, so a commented-out push keeps
# counting toward the pinned inventory — and a set-literal edit is otherwise
# uncaught, so `test_every_bash_metacharacter_ends_a_word_before_a_hash` pins
# both the set's identity and one behavioral case per character.
#
# The `)` entry is CONDITIONAL, not absolute: `_strip_bash_comment` exempts a
# `)` that closes a substitution (module docstring, seventh axis).
_COMMENT_WORD_START = " \t;&|()<>"


def _strip_bash_comment(line: str) -> str:
    """Return ``line`` truncated at its first unquoted, word-initial ``#``.

    Quote-aware because a ``#`` can sit in a quoted argument BEFORE the push
    variable — the wrappers' own ``echo "$(ts) #${ISSUE} ..."`` logging idiom,
    chained onto a push with ``&&``, is exactly that shape — where a naive
    ``split("#", 1)[0]`` truncates the line ahead of the execution site and
    silently drops a live, correctly-bounded push.

    It is NOT needed for the ``#`` in a push's own trailing message: on all
    ten live sites the first ``#`` falls at or after the match END (measured
    round 4), so a quote-blind stripper scans today's six wrappers to an
    identical site set and identical bound flags. Both halves of that scope
    are pinned —
    ``test_quote_tracking_is_load_bearing_when_a_hash_precedes_the_site`` and
    ``test_quote_blind_stripping_is_indistinguishable_on_the_live_line``.

    Paren-aware because ``)`` is in ``_COMMENT_WORD_START`` but is only
    CONDITIONALLY a word end (module docstring, seventh axis). ``open_parens``
    records, per open paren, whether it opened a SUBSTITUTION — ``$(``, ``$((``
    (which opens two), ``<(`` or ``>(`` — or a plain subshell/grouping paren.
    The matching ``)`` pops it: a substitution's ``)`` does not end the word, so
    a ``#`` immediately after it is mid-word and NOT a comment, while a plain
    ``)`` (subshell, ``case`` arm, bare ``((`` arithmetic command, or an
    unmatched ``)`` against an empty stack) ends the word as before.
    ``prev_bare`` is the previous UNQUOTED, UNESCAPED character, so a ``$``
    that was quoted or backslash-escaped never turns a following ``(`` into a
    substitution.
    """
    in_single = False
    in_double = False
    open_parens: list[bool] = []
    last_substitution_close = -1
    prev_bare = ""
    i = 0
    while i < len(line):
        ch = line[i]
        bare = ""
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_double = False
        elif ch == "\\":
            i += 1
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "(":
            if prev_bare == "$" and line[i + 1 : i + 2] == "(":
                open_parens.extend((True, True))  # `$((` arithmetic expansion
                i += 1
            else:
                open_parens.append(prev_bare in ("$", "<", ">"))
            bare = "("
        elif ch == ")":
            closed_substitution = open_parens.pop() if open_parens else False
            if closed_substitution:
                last_substitution_close = i
            bare = ")"
        elif ch == "#" and (
            i == 0 or (line[i - 1] in _COMMENT_WORD_START and i - 1 != last_substitution_close)
        ):
            return line[:i]
        else:
            bare = ch
        prev_bare = bare
        i += 1
    return line


def scan_execution_sites(
    text: str, strip: Callable[[str], str] = _strip_bash_comment
) -> list[tuple[int, str, re.Match[str]]]:
    """Return ``(lineno, executable_line, match)`` per push EXECUTION site.

    Comments are stripped before matching. A commented-out push line keeps
    both its regex match and its ``timeout`` prefix, so an unstripped scan
    would count a disabled alert toward the exact per-wrapper inventory.

    ``strip`` is the comment stripper, injected ONLY so the quote-tracking
    mutants below can run this exact loop with a weaker stripper. A separate
    re-implementation of the loop would let the two drift, and the mutants
    that compare them would then pass for the wrong reason.
    """
    sites: list[tuple[int, str, re.Match[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = strip(raw)
        sites.extend((lineno, line, m) for m in _EXEC_SITE.finditer(line))
    return sites


def test_every_push_call_site_is_timeout_bounded():
    """EVERY push execution match on EVERY line of EVERY wrapper is
    immediately preceded by the exact bound, and each wrapper holds exactly
    its pinned number of sites (so a deleted site fails loud)."""
    assert sum(WRAPPERS.values()) == TOTAL_EXPECTED_SITES, (
        f"per-wrapper counts sum to {sum(WRAPPERS.values())}, not the pinned "
        f"{TOTAL_EXPECTED_SITES}: update TOTAL_EXPECTED_SITES deliberately"
    )
    for rel, expected in WRAPPERS.items():
        sites = scan_execution_sites((_REPO_ROOT / rel).read_text())
        n_sites = len(sites)
        for lineno, line, m in sites:
            assert _BOUND_PREFIX_RE.search(line[: m.start()]) is not None, (
                f"{rel}:{lineno}: push execution not immediately preceded by "
                f"{_BOUND_PREFIX_TEXT!r} (a wrong duration such as '0s' means no "
                f"deadline at all, and only spaces/tabs separate the tokens): "
                f"{line.strip()!r}"
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
    """`bash -n` (parse-only, nothing executed) per wrapper — the sole
    syntax check for the two never-executed watch scripts."""
    for rel in WRAPPERS:
        proc = subprocess.run(
            ["bash", "-n", str(_REPO_ROOT / rel)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, f"{rel}: bash -n failed:\n{proc.stderr}"


# --- Scanner mutants (synthetic text; the real tree is never mutated) --------
#
# The scanner above is the ONLY coverage vehicle for the two watch scripts, so
# the mutants below pin what it must reject. `_LIVE` is the live retirement
# push of `cron_watch_issue_2091.sh`, BYTE-FAITHFUL to its source line: the
# `[ -x "$PUSH" ]` guard the execution-site regex must not count, and the `#`
# inside the trailing message. That `#` is INERT for the scanner — it falls
# after the match end, as on all ten live sites — so `_LIVE` is the unmutated
# control, not the quote-tracking control. `_LIVE_HASH_BEFORE_SITE` is the
# shape where quote tracking decides.
#
# Both fixtures are pinned against the live file by
# `test_live_fixtures_are_byte_faithful_to_their_source_lines`, which locates
# the lines by CONTENT (line numbers drift). Round 4 described them as built
# from those lines while silently substituting an ASCII hyphen for the source's
# em dash and a literal `done` for `${status}`; a control whose warrant is
# "this is the real line" has to be the real line, not a paraphrase of it.
# The em dash is written as an escape for the same reason `_NBSP` is: an
# ASCII-looking non-ASCII byte in source is unreviewable.
_LIVE_SOURCE = "scripts/cron_watch_issue_2091.sh"

_LIVE = (
    '    [ -x "$PUSH" ] && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" '
    '"EPS #${ISSUE} reached ${status} \u2014 monitor retired." >/dev/null 2>&1'
)

# The load-bearing shape for quote tracking: a `#` inside a quoted argument
# BEFORE the push variable. Composed from two ADJACENT live lines of
# `_LIVE_SOURCE` — the terminal log line and the retirement push it precedes —
# fused with `&&` onto one physical line, which is the wrappers' own logging
# idiom. A naive first-hash strip truncates at `$(ts) #${ISSUE}` and the
# execution site disappears entirely.
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


def _naive_first_hash_strip(line: str) -> str:
    """The quote-BLIND stripper: what this scanner is without quote state."""
    return line.split("#", 1)[0]


def test_live_push_line_scans_as_one_bounded_site():
    """The unmutated control: one site, bound satisfied, and the `[ -x "$PUSH" ]`
    guard on the same line not counted as a second site."""
    assert _bound_flags(_LIVE) == [True]


def test_commented_out_push_line_contributes_no_site():
    """Commenting a bounded push must DROP its site, not preserve the count.

    Both regex match and `timeout` prefix survive the `#`, and `bash -n`
    stays green, so an unstripped scan would report a full inventory for a
    wrapper whose alert no longer runs.
    """
    assert scan_execution_sites("#" + _LIVE) == []
    assert scan_execution_sites("  # " + _LIVE.strip()) == []


def test_push_inside_a_trailing_comment_contributes_no_site():
    assert scan_execution_sites('echo hi  # "$PUSH" "msg"') == []


def test_nbsp_before_the_push_variable_is_not_a_valid_bound():
    """`"${PUSH_TIMEOUT}s"<NBSP>"$PUSH"` is one bash word: timeout then gets
    an invalid duration and exits 125, so the line is NOT bounded."""
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
    """Round 2 deliberately relaxed the whitespace axis; keep it relaxed."""
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
    """`finditer`, not `search`: an appended unbounded push must fail."""
    assert _bound_flags(_LIVE + ' ; "$PUSH" "unbounded"') == [True, False]


def test_quote_tracking_is_load_bearing_when_a_hash_precedes_the_site():
    """What quote tracking actually buys: a `#` quoted BEFORE the push.

    Quote-aware, one live correctly-bounded site. Quote-BLIND — the naive
    `split("#", 1)[0]` this test really runs, not merely asserts about — the
    line is cut at the log message's `#` and the site vanishes, which would
    drop a live push from the pinned inventory.
    """
    assert _bound_flags(_LIVE_HASH_BEFORE_SITE) == [True]
    blind = scan_execution_sites(_LIVE_HASH_BEFORE_SITE, strip=_naive_first_hash_strip)
    assert blind == []


def test_quote_blind_stripping_is_indistinguishable_on_the_live_line():
    """Scoping the claim above: on `_LIVE` the `#` is inert.

    Round 3 justified quote tracking by claiming every watch-script push line
    carries a `#` a naive strip would cut into. It does not — on all ten live
    sites the hash falls at or after the match end — so this records the true
    scope: `_LIVE` scans identically both ways, and only the hash-before-site
    shape diverges. Keeps the mechanism honest about what it is for.
    """
    aware = scan_execution_sites(_LIVE)
    blind = scan_execution_sites(_LIVE, strip=_naive_first_hash_strip)
    assert [(n, m.span()) for n, _, m in blind] == [(n, m.span()) for n, _, m in aware]


def test_push_after_a_subshell_close_paren_contributes_no_site():
    """`(true)#` opens a comment: bash never runs the push.

    Verified against bash 5.1.16 (`bash -c '(true)# "$PUSH" "msg"'` prints
    nothing). `)` is a metacharacter, so the `#` is word-initial; a word-start
    set omitting `)` keeps this line and counts a DISABLED push toward the
    pinned inventory — the count still matches and the test passes silently.
    """
    assert scan_execution_sites("(true)#" + _LIVE.strip()) == []


def test_push_in_a_case_arm_comment_contributes_no_site():
    """The same omission inside a `case` arm — the watch scripts' own shape.

    `cron_watch_issue_2091.sh` dispatches its terminal push from exactly such
    an arm, so this is where a disabled alert would most plausibly hide.
    """
    assert scan_execution_sites("  awaiting_promotion)#" + _LIVE.strip()) == []


def test_push_after_a_redirect_operator_contributes_no_site():
    """`<` and `>` end a word too, so `>#` and `<#` also open comments.

    Such a line is a bash syntax error — the comment eats the redirect target
    — which `test_every_wrapper_parses` catches on its own. This pins set
    completeness against bash's metacharacters, not a second silent channel.
    """
    assert scan_execution_sites("true >#" + _LIVE.strip()) == []
    assert scan_execution_sites("true <#" + _LIVE.strip()) == []


def test_every_bash_metacharacter_ends_a_word_before_a_hash():
    """Set completeness, pinned two ways, because NARROWING is the silent side.

    `_COMMENT_WORD_START` is bash's metacharacter set (newline is the `i == 0`
    case). Dropping a member under-strips: a commented-out push keeps counting
    toward the pinned inventory and the test passes over a dead alert. The
    per-character loop below is the behavioral half — measured against bash
    5.1.16, every one of these prefixes comments out the rest of the line — and
    the set-equality assert is the structural half, so a member added and later
    removed cannot slip through a stale enumeration.
    """
    metacharacters = {
        " ": "space",
        "\t": "tab",
        ";": "semicolon",
        "&": "ampersand",
        "|": "pipe",
        "(": "open paren",
        ")": "close paren",
        "<": "redirect in",
        ">": "redirect out",
    }
    assert set(_COMMENT_WORD_START) == set(metacharacters), (
        "_COMMENT_WORD_START must be exactly bash's metacharacter set; this "
        "enumeration is the behavioral pin and has to move with it"
    )
    for char, label in metacharacters.items():
        text = "true" + char + "#" + _LIVE.strip()
        assert scan_execution_sites(text) == [], f"{label} did not end the word"


def test_added_push_after_a_substitution_close_is_counted_and_reads_unbounded():
    """Axis seven, in the direction that was silent: a NEWLY ADDED push.

    A substitution's closing `)` belongs to its word, so bash runs each push
    below (measured, bash 5.1.16: `echo $(echo hi)#tag` prints `hi#tag` and the
    `&&` arm executes). Round 4 read every one of these lines as a comment and
    dropped it, which left the wrapper's count sitting exactly at its pin — the
    count assertion passed while an unbounded push ran. The assertions are that
    the count RISES to 2 and that the added site reads UNBOUNDED.
    """
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
    """The exemption covers the `)` itself, not the rest of the line.

    `echo $(true) # "$PUSH" "msg"` runs NO push (measured, bash 5.1.16: it
    prints one empty line and nothing else) — the `#` starts a word after the
    SPACE, not after the `)`, so the exemption must not reach it.
    """
    assert scan_execution_sites("echo $(true) # " + _LIVE.strip()) == []


def test_an_arithmetic_command_close_is_not_a_substitution_close():
    """A bare `((` is an arithmetic COMMAND; only `$((` opens an expansion.

    Measured, bash 5.1.16: `((i=1))#echo LEAK` prints nothing and exits 0, so
    the `))` are ordinary word ends and the `#` opens a comment. This is the
    discriminator between axis seven's exemption and a paren pair that merely
    looks like it.
    """
    assert scan_execution_sites("((i=1))#" + _LIVE.strip()) == []


def test_nested_parens_inside_a_substitution_do_not_leak_the_exemption():
    """A subshell or arithmetic group NESTED in a substitution stays matched.

    Measured, bash 5.1.16: `echo $( (true) )#tag && push` and
    `echo $(( (1+2) * 3 ))#tag && push` both run the push. The inner `)` pops
    its own plain paren, so the OUTER `)` is still the substitution's and the
    line survives the strip.
    """
    for label, text in (
        ("subshell in cmdsub", 'echo $( (true) )#tag && "$PUSH" "EPS #1 alert"'),
        ("group in arithmetic", 'echo $(( (1+2) * 3 ))#tag && "$PUSH" "EPS #1 alert"'),
    ):
        assert _bound_flags(text) == [False], label


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
