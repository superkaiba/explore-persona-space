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

Why the site check is exact on all nine axes (#2387; axes four and five are
round-3 hardening of the exact-count pin the first three introduced, the sixth
is round-4 completion of the word-start set axis four rests on, the seventh is
round-5 correction of an over-strip the sixth introduced, the eighth is
round-6 repair of the seventh's paren stack after a measured frame-pop defect,
and the ninth is round-7 replacement of the flat brace/quote state with typed
context frames after three more measured silent over-strips):

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
  silent DELETION of a site: the pinned 12 execution sites across 6 wrappers
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
- ESCAPE-AWARE WORD STARTS, LITERAL ``${...}`` INTERIORS, AND LOUD REFUSAL
  where classification needs a grammar this scanner does not have. Round 5's
  paren stack mispopped on an unmatched inner ``)``: measured on bash 5.1.16,
  ``STAMP=$(case x in y) date +%s;; esac)#tag && <bounded push>`` RUNS the
  push — the case-pattern ``)`` has no matching ``(``, so it stole the
  substitution's frame, the REAL close popped an empty stack, and the round-5
  scanner stripped the live push at ``#`` (0 sites: the silent new-site
  over-strip quadrant). Two measured siblings: ``echo a\\;#x && <push>`` (the
  raw-index word-start check read the escaped ``;`` as a metacharacter) and
  ``[[ "x#tag" =~ (x)#tag ]] && <push>`` (a regex group's ``)`` read as a
  word end) — bash runs all three pushes; the round-5 scanner dropped all
  three. Round 6: (a) the comment check keys on the tracked previous BARE
  character (escape- and quote-aware; a substitution-closing ``)`` records as
  word content), never the raw ``line[i - 1]`` byte; (b) an unquoted
  ``${...}`` is scanned to its matching ``}`` with ``( ) #`` literal, so a
  parameter pattern's ``)`` (``${x%)}``) cannot pop a frame; (c) an unquoted
  ``case`` word while any paren frame is open, and an unquoted
  whitespace-preceded ``=~`` operator, RAISE ValueError instead of guessing —
  case patterns and ``[[ ]]`` regex words need their own grammars, and a loud
  refusal on a scanner-illegible line can neither silently pass an unbounded
  push nor silently drop a live one.
- CONTEXT FRAMES for nested constructs, MODELED ``$'...'`` quotes, and
  REFUSED backticks. Round 6 left three lexical forms unmodeled, and each
  silently OVER-stripped — zero sites reported, nothing raised,
  indistinguishable from a verified line (all three measured against bash
  5.1.16, which RUNS every one of these pushes):
  (a) backtick command substitution — ``Y=`date +%s #x` && "$PUSH" "..."``:
      the closing-tick search runs THROUGH the ``#`` (it comments only the
      inner command), while the scanner read `` #`` as a comment start;
  (b) a substitution nested in a ``${...}`` default — the flat brace depth
      counter decremented at the ``}`` of an embedded brace group, so
      ``x=$(echo ${v:-$( { echo hi; })})#tag && "$PUSH" "..."`` popped the
      outer substitution frame early and the REAL close ended a word right
      before ``#``;
  (c) ``$'...'`` ANSI-C quoting — ``\\'`` closed the flat single-quote
      state bash keeps open, and the stray tail stripped at ``#``.
  Round 7: the paren list and brace counter become ONE stack of typed
  frames (substitution paren / plain paren / ``${...}`` interior), each
  recording the double-quote state to restore at its close. (b) is MODELED
  — an embedded group's ``}`` is an ordinary word character in the nested
  code context — and a substitution or expansion opened INSIDE double
  quotes (``"$(ts) ..."``, ``"${V:-$(cmd)}"``) scans its interior as fresh
  unquoted code, exactly as bash reparses it; that also closes two sibling
  desyncs measured in round 7: ``"$(printf "%s" "a)#b")"`` and
  ``"${v:-"a)#b"}"`` both run their ``&&`` push under bash while round 6
  silently dropped both. (c) is MODELED: a single quote entered via ``$'``
  tracks backslash escapes; a regular single quote stays escape-blind (the
  wrappers' single-quoted sed/awk programs rely on it). (a) is REFUSED
  loudly — a backtick interior is shell code needing its own parse and
  POSIX leaves the closing-tick search undefined across embedded quotes —
  as is a process substitution opened inside a ``${...}`` interior (live
  code: ``echo ${v:-<(true)}`` substitutes a /dev/fd path, measured). The
  POSTURE is the durable part: rounds 4-6 each closed one enumerated
  member and were bounced by the next round's new member, so a construct
  this scanner does not model must RAISE, naming the construct and origin
  line — never return a stripped line indistinguishable from a verified
  one (``test_unmodeled_constructs_raise_instead_of_stripping``).

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

DIRECTION RULE for any scanner-vs-bash divergence — derived by MEASUREMENT
through the injectable ``strip`` seam (2026-08-29; pinned by
``test_direction_rule_quadrants_match_the_documented_rule``) after two
consecutive rounds shipped a wrong direction claim by reasoning it out: round
4 wrote "an over-strip is loud, never a silent pass" and round 5 wrote
"UNDER-STRIP ... SILENT" unconditionally; both were measured false. The
measured rule: a divergence is SILENT exactly when it leaves the wrapper's
site count sitting AT its pin, LOUD exactly when it moves the count off the
pin. Both directions split on whether the affected site is already in the
pinned inventory:

- OVER-STRIP (scanner drops text bash executes) of a site IN the inventory —
  LOUD. The count falls below the pin and the count assertion fails.
- OVER-STRIP of a NEWLY ADDED site — SILENT. The count sits exactly AT the
  pin while bash runs an unbounded push. This defeats the PRIMARY invariant
  ("every push execution is timeout-bounded") and is the quadrant rounds 4
  and 5 each reopened; round 6 closes the measured channels and REFUSES
  loudly (axis eight) where it cannot classify, and round 7 extends both
  arms (axis nine) after three more measured members of this quadrant.
- UNDER-STRIP (scanner keeps text bash discards) of a site IN the inventory —
  SILENT. A DISABLED push still counts, the pin is still met, and the test
  passes over a dead alert. This is the shape axes four and six prevent.
- UNDER-STRIP of a NEWLY ADDED INERT site — LOUD. A kept commented-out push
  raises the count above the pin. Round 5's unconditional "UNDER-STRIP —
  SILENT" is false in exactly this quadrant.

Known residual divergences after round 7, each classified by the rule above.
This is the KNOWN set, not a completeness proof — rounds 3-6 each found a new
channel in an enumeration that read as complete:

- LINE-SCOPED state. Quote, paren, brace, and frame state restart at each
  physical line; ``\\`` continuations and heredocs are not joined or tracked.
  The CLOSING line of a multi-line command substitution (``X=$(echo a`` /
  ``echo b)#tag && "$PUSH" "..."``) is an OVER-strip: the scanner pops an
  empty stack at line 2's ``)`` and drops a site bash runs (measured). A
  heredoc BODY is the OPPOSITE direction: bash never executes body text, but
  the scanner scans it as code, so a push-shaped body line is an over-COUNT
  of an inert site (UNDER-strip direction, loud on addition; round 5 misfiled
  heredocs under over-strips). Raising on an unclosed frame/quote at end of
  line is NOT an option: the two watch scripts' live multi-line
  substitutions and python bodies would refuse, so this residual stays a
  disclosed line-scoped limit rather than a refusal.
- A ``)`` closing an ARRAY ASSIGNMENT (``x=(a b)#tag``) or an EXTGLOB pattern
  (``@(a|b)#tag``, under ``shopt -s extglob``) also belongs to its word in
  bash; the scanner treats both as word ends — an OVER-strip, silent on a
  newly added site. Distinguishing them from a subshell needs
  assignment/pattern parsing, not the ``$``/``<``/``>`` prefix axis seven
  keys on.
- ``[[ ]]`` grammar beyond the ``=~`` refusal is unmodeled, and measured NOT
  to be a silent channel: ``[[ -n x ]]#e`` and ``[[ (x = y)#z ]]`` are both
  bash SYNTAX ERRORS (``bash -n`` exits 2, bash 5.1.16), so a divergent read
  there is caught by ``test_every_wrapper_parses``, never passed silently.
- The refusals are deliberately BROADER than the ambiguity they fence: an
  unquoted ``case`` in ARGUMENT position inside a one-line substitution
  (``$(grep case f)``), any unquoted whitespace-preceded ``=~``, ANY
  backtick outside single quotes (including ``"`date`"``, which bash
  substitutes fine when no embedded quote complicates the tick search), and
  a ``<(``/``>(`` inside a ``${...}`` interior all refuse lines bash may
  handle. A false refusal raises ValueError: loud by construction, never a
  wrong count.

None of the residual or refused shapes is in the six wrappers (measured
2026-08-29 on this branch; refusal census re-measured 2026-08-30): zero
``)#`` sequences, zero ``NAME=(``, zero ``[@+?!*](``, no ``shopt -s
extglob``, no ``[[`` or ``=~`` in live shell code (the one ``[[:space:]]``
grep hit sits inside a single-quoted awk program), and the only one-line
``case`` (``cron_watch_issue_1739.sh`` line 111) opens at top level with no
frame — outside the refusal. Backticks appear ONLY on comment lines
(stripped at ``#`` before any tick is scanned); zero ``$'`` and zero process
substitutions anywhere. Exactly one brace-nested command substitution,
``cron_daily_healthcheck.sh`` line 53, sits inside double quotes and is
MODELED by the frame stack (measured: scans clean). One heredoc
(``cron_watch_issue_1739.sh`` line 119, a bare variable). Two wrappers open a
multi-line command substitution (``cron_watch_issue_2091.sh`` lines 30-31,
``cron_watch_issue_1739.sh`` lines 58-59); both close it with no ``#``
anywhere on the closing line. Repo-wide, the only two ``)#`` sequences in any
``*.sh`` sit inside single-quoted ``sed`` programs, which quote tracking
already skips. The round-7 scanner scans all six wrappers to their pinned
counts with zero refusals (measured by direct helper evaluation).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
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
# push, so its count is unchanged at 1.
TOTAL_EXPECTED_SITES = 12

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


# Typed context frames for `_strip_bash_comment` (module docstring, ninth
# axis). ONE stack replaces round 6's separate paren list + brace depth
# counter: a paren frame is a SUBSTITUTION (its `)` records as word content)
# or PLAIN (its `)` records as a word end); a BRACE frame is a `${...}`
# interior, scanned to its matching `}` with `(`, `)` and `#` literal. Every
# frame records the double-quote state to restore when it closes, because a
# substitution or expansion opened INSIDE double quotes reparses its interior
# as fresh unquoted code — bash does the same (measured: the inner quotes of
# `echo "$(printf "%s" "a)#b")"` nest inside the substitution instead of
# closing the outer string, and the `&&` arm runs).
_SUBST = "substitution"
_PLAIN = "plain"
_BRACE = "brace"


def _open_paren(
    line: str,
    i: int,
    prev_bare: str,
    frames: list[tuple[str, bool]],
    resume_in_double: bool = False,
) -> int:
    """Push the frame(s) for the ``(`` at ``i``; return extra chars consumed.

    A `$((` arithmetic expansion opens two substitution frames and consumes
    the second `(` (return 1); otherwise one frame is pushed — a substitution
    iff the previous bare character was `$`, `<` or `>` (return 0). The
    FIRST-pushed frame (popped LAST) carries ``resume_in_double``, the
    double-quote state to restore when the whole construct closes.
    """
    if prev_bare == "$" and line[i + 1 : i + 2] == "(":
        frames.append((_SUBST, resume_in_double))
        frames.append((_SUBST, False))
        return 1
    kind = _SUBST if prev_bare in ("$", "<", ">") else _PLAIN
    frames.append((kind, resume_in_double))
    return 0


def _unclassifiable_at(
    line: str, i: int, prev_bare: str, frames: list[tuple[str, bool]]
) -> str | None:
    """Return the refusal message when position ``i`` opens a construct the
    scanner cannot classify (module docstring, eighth axis), else None.

    Two triggers, both deliberately broader than the ambiguity they fence
    (module docstring, residual list): an unquoted word ``case`` while any
    frame is open, and an unquoted ``=~`` preceded by whitespace (no
    trailing space required — ``=~(x)#tag`` parses and runs, measured
    against bash 5.1.16). Two more refusals — backticks, and a process
    substitution inside a ``${...}`` interior (ninth axis) — raise at their
    sites in ``_strip_bash_comment``: they must fire in quote/brace
    contexts this bare-context helper is never consulted for.
    """
    ch = line[i]
    if (
        ch == "c"
        and frames
        and line[i : i + 4] == "case"
        and prev_bare != ""
        and prev_bare in _COMMENT_WORD_START
        and (i + 4 >= len(line) or line[i + 4] in _COMMENT_WORD_START)
    ):
        return (
            "cannot classify `)` after a `case` inside an open paren frame: "
            "case patterns close with an unmatched `)`, which needs case "
            f"grammar. Restructure the line or extend the scanner. Line: {line!r}"
        )
    if ch == "=" and line[i + 1 : i + 2] == "~" and prev_bare in (" ", "\t"):
        return (
            "cannot classify text after the `[[ ]]` regex operator `=~`: its "
            "right-hand side is a regex word where `( ) #` are literal. "
            f"Restructure the line or extend the scanner. Line: {line!r}"
        )
    return None


def _single_quote_char(line: str, i: int, ansi: bool) -> tuple[int, bool]:
    """Handle char ``i`` inside a single-quoted string; return
    ``(extra_chars_consumed, still_in_single)``.

    In a REGULAR single quote every character is literal and ``'`` closes;
    in one entered via ``$'`` (ANSI-C, ``ansi=True``) a backslash escapes
    its next character, so ``\\'`` stays inside — the round-6 desync
    (module docstring, ninth axis).
    """
    ch = line[i]
    if ansi and ch == "\\":
        return 1, True
    return 0, ch != "'"


def _double_quote_char(line: str, i: int, frames: list[tuple[str, bool]]) -> tuple[int, bool, str]:
    """Handle char ``i`` inside double quotes; return
    ``(extra_chars_consumed, in_double, bare)``.

    A backslash consumes its next character and ``"`` closes. A ``$(``,
    ``$((`` or ``${`` opened here reparses its interior as fresh UNQUOTED
    code — bash does the same — so it pushes a frame recording that double
    quotes resume at the matching close (module docstring, ninth axis).
    Everything else is quoted content.
    """
    ch = line[i]
    if ch == "\\":
        return 1, True, ""
    if ch == '"':
        return 0, False, ""
    if ch == "$" and line[i + 1 : i + 2] == "(":
        return 1 + _open_paren(line, i + 1, "$", frames, resume_in_double=True), False, "("
    if ch == "$" and line[i + 1 : i + 2] == "{":
        frames.append((_BRACE, True))
        return 1, False, "{"
    return 0, True, ""


def _brace_char(
    line: str, i: int, prev_bare: str, frames: list[tuple[str, bool]]
) -> tuple[int, str, bool]:
    """Handle char ``i`` inside a ``${...}`` interior; return
    ``(extra_chars_consumed, bare, in_double)``.

    ``(``, ``)`` and ``#`` are literal pattern/word characters here (a
    pattern ``)`` such as ``${x%)}`` cannot pop a substitution frame; quote
    and escape characters are handled by the caller BEFORE this branch, as
    bash's brace scan also skips quoted strings). Only the matching ``}`` —
    which pops the frame and restores the double-quote state it opened
    under — a nested ``${``, or a nested ``$(``/``$((`` (a fresh code
    context) changes state; a process substitution inside the interior is
    executed code this scanner does not model, so it refuses (module
    docstring, ninth axis).
    """
    ch = line[i]
    if ch == "}":
        _, resume = frames.pop()
        return 0, "}", resume
    if ch == "{" and prev_bare == "$":
        frames.append((_BRACE, False))
        return 0, "{", False
    if ch == "(" and prev_bare == "$":
        return _open_paren(line, i, prev_bare, frames), "(", False
    if ch == "(" and prev_bare in ("<", ">"):
        raise ValueError(
            "cannot classify process substitution inside a ${...} interior: "
            "its parenthesized body is executed shell code, not pattern text "
            "(measured: `echo ${v:-<(true)}` substitutes a /dev/fd path, "
            f"bash 5.1.16). Restructure the line or extend the scanner. Line: {line!r}"
        )
    return 0, ch, False


def _close_paren(frames: list[tuple[str, bool]]) -> tuple[str, bool]:
    """Pop the frame for a code-context ``)``; return ``(bare, in_double)``.

    A substitution's ``)`` belongs to its word: it records as word content
    (``bare = ""``) so a ``#`` right after it reads mid-word, and it
    restores the double-quote state the construct opened under (module
    docstring, ninth axis). Every other ``)`` — a plain paren, or a pop
    against an empty stack — records as a word end, with the code context
    staying unquoted.
    """
    kind, resume = frames.pop() if frames else (_PLAIN, False)
    if kind == _SUBST:
        return "", resume
    return ")", False


def _strip_bash_comment(line: str) -> str:
    """Return ``line`` truncated at its first unquoted, word-initial ``#``.

    Quote-aware because a ``#`` can sit in a quoted argument BEFORE the push
    variable — the wrappers' own ``echo "$(ts) #${ISSUE} ..."`` logging idiom,
    chained onto a push with ``&&``, is exactly that shape — where a naive
    ``split("#", 1)[0]`` truncates the line ahead of the execution site and
    silently drops a live, correctly-bounded push.

    It is NOT needed for the ``#`` in a push's own trailing message: on all
    twelve live sites the first ``#`` falls at or after the match END
    (measured round 4 at ten sites; re-verified at twelve after the #2386
    merge), so a quote-blind stripper scans today's six wrappers to an
    identical site set and identical bound flags. Both halves of that scope
    are pinned —
    ``test_quote_tracking_is_load_bearing_when_a_hash_precedes_the_site`` and
    ``test_quote_blind_stripping_is_indistinguishable_on_the_live_line``.

    Frame-aware because ``)`` is in ``_COMMENT_WORD_START`` but is only
    CONDITIONALLY a word end (module docstring, seventh axis), and because a
    ``${...}`` interior and a nested substitution each need their own
    context (ninth axis). ``frames`` is ONE stack of typed frames: a
    SUBSTITUTION paren — ``$(``, ``$((`` (which opens two), ``<(`` or
    ``>(`` — whose matching ``)`` records as WORD CONTENT
    (``prev_bare = ""``), so a ``#`` right after it is mid-word and NOT a
    comment; a PLAIN paren (subshell, ``case`` arm, bare ``((`` arithmetic
    command, or a ``)`` against an empty stack), whose ``)`` records as a
    word end; and a BRACE frame — a ``${...}`` interior — where ``(``, ``)``
    and ``#`` are literal (a pattern ``)`` such as ``${x%)}`` cannot pop a
    substitution frame) and only the matching ``}``, a nested ``${``, or a
    nested ``$(``/``$((`` (a fresh code context) changes state. ``prev_bare``
    is the previous character AS BASH READS IT — empty for quoted, escaped,
    and substitution-closing positions — so a ``$`` that was quoted or
    backslash-escaped never turns a following ``(`` into a substitution.

    Quote-restoring (ninth axis): a substitution or ``${...}`` opened INSIDE
    double quotes reparses its interior as fresh UNQUOTED code — bash does
    the same — so its frame records the double-quote state and restores it
    at the matching close. Round 6 tracked quotes flatly, so the inner
    quotes of ``"$(printf "%s" "a)#b")"`` flipped it into bare state and the
    ``)#`` stripped a line bash runs (measured; same for the ``${...}``
    form).

    Escape-aware (eighth axis): the comment check keys on ``prev_bare``,
    never the raw ``line[i - 1]`` byte, which round 5 used and which mis-read
    the escaped ``;`` of ``echo a\\;#x`` as a word end and silently dropped
    the live push chained after it.

    ANSI-C-aware (ninth axis): a single quote entered via ``$'`` tracks
    backslash escapes — inside ``$'...'`` bash reads ``\\'`` as an ESCAPED
    quote, so round 6's flat single-quote state closed early and stripped a
    live push at the stray tail's ``#`` (measured). A REGULAR single quote
    stays escape-blind: in ``'a\\'`` the quote after the backslash CLOSES
    (the wrappers' single-quoted sed/awk programs depend on it).

    Loud refusal (eighth + ninth axes): four constructs make later text
    unclassifiable without their own grammars, and each RAISES ValueError
    naming the construct — a ``case`` statement inside an open frame (its
    pattern ``)`` has no matching ``(``), the ``[[ ]]`` regex operator
    ``=~`` (its right-hand side is a regex word where ``( ) #`` are
    literal, with or without a space after the operator — measured, bash
    5.1.16), any backtick outside single quotes (the closing-tick search
    runs through a ``#``, the interior is shell code needing its own parse,
    and POSIX leaves the search undefined across embedded quotes), and a
    process substitution inside a ``${...}`` interior (live code, not
    pattern text — measured). A refusal is loud on any input, so it can
    neither silently pass an unbounded push nor silently drop a live one.
    Deliberately broader than the ambiguity — see the module docstring's
    residual list.
    """
    in_single = False
    ansi_single = False
    in_double = False
    frames: list[tuple[str, bool]] = []
    prev_bare = ""
    i = 0
    while i < len(line):
        ch = line[i]
        bare = ""
        if in_single:
            extra, in_single = _single_quote_char(line, i, ansi_single)
            i += extra
        elif ch == "`":
            raise ValueError(
                "cannot classify backtick command substitution: bash scans to "
                "the matching unescaped backtick (a `#` inside comments only "
                "the INNER command, so the outer line continues after the "
                "close — measured, bash 5.1.16), and the interior is shell "
                "code needing its own parse. Use $(...) or restructure the "
                f"line. Line: {line!r}"
            )
        elif in_double:
            extra, in_double, bare = _double_quote_char(line, i, frames)
            i += extra
        elif ch == "\\":
            i += 1
        elif ch == "'":
            in_single = True
            ansi_single = prev_bare == "$"
        elif ch == '"':
            in_double = True
        elif frames and frames[-1][0] == _BRACE:
            extra, bare, in_double = _brace_char(line, i, prev_bare, frames)
            i += extra
        elif ch == "{" and prev_bare == "$":
            frames.append((_BRACE, False))
            bare = "{"
        elif ch == "(":
            i += _open_paren(line, i, prev_bare, frames)
            bare = "("
        elif ch == ")":
            bare, in_double = _close_paren(frames)
        elif (refusal := _unclassifiable_at(line, i, prev_bare, frames)) is not None:
            raise ValueError(refusal)
        elif ch == "#" and (i == 0 or (prev_bare != "" and prev_bare in _COMMENT_WORD_START)):
            return line[:i]
        else:
            bare = ch
        prev_bare = bare
        i += 1
    return line


def scan_execution_sites(
    text: str,
    strip: Callable[[str], str] = _strip_bash_comment,
    origin: str = "<text>",
) -> list[tuple[int, str, re.Match[str]]]:
    """Return ``(lineno, executable_line, match)`` per push EXECUTION site.

    Comments are stripped before matching. A commented-out push line keeps
    both its regex match and its ``timeout`` prefix, so an unstripped scan
    would count a disabled alert toward the exact per-wrapper inventory.

    ``strip`` is the comment stripper, injected ONLY so the quote-tracking
    mutants below can run this exact loop with a weaker stripper. A separate
    re-implementation of the loop would let the two drift, and the mutants
    that compare them would then pass for the wrong reason.

    A line the default stripper cannot classify raises ValueError (module
    docstring, eighth + ninth axes): the scan is loud, never silently wrong,
    on case-inside-substitution, ``=~`` regex, backtick, and
    process-substitution-in-brace lines. The refusal is re-raised prefixed
    ``{origin}:{lineno}:`` so a refusal from a wrapper scan names its file
    and line (the wrapper-facing tests pass each wrapper's repo-relative
    path as ``origin``).
    """
    sites: list[tuple[int, str, re.Match[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        try:
            line = strip(raw)
        except ValueError as err:
            raise ValueError(f"{origin}:{lineno}: {err}") from err
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
        sites = scan_execution_sites((_REPO_ROOT / rel).read_text(), origin=rel)
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
# after the match end, as on all twelve live sites — so `_LIVE` is the
# unmutated control, not the quote-tracking control. `_LIVE_HASH_BEFORE_SITE`
# is the shape where quote tracking decides.
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


def _identity_strip(line: str) -> str:
    """No stripping at all: the maximal UNDER-strip, for the direction-rule
    quadrant measurements."""
    return line


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
    carries a `#` a naive strip would cut into. It does not — on all twelve
    live sites the hash falls at or after the match end (re-derived at
    twelve, round 7) — so this records the true scope: `_LIVE` scans
    identically both ways, and only the hash-before-site shape diverges.
    Keeps the mechanism honest about what it is for.
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


def test_case_inside_a_substitution_refuses_loudly():
    """Reconciler L1 (round 5, measured): bash 5.1.16 RUNS this push — the
    case-pattern `)` has no matching `(`, so `esac)` closes the substitution
    and `#tag` is mid-word. The round-5 scanner (f16c71b301b) popped the
    substitution frame at the pattern `)`, read the real close against an
    empty stack, and stripped the line at `#`: 0 sites, silently, in the
    primary-invariant direction. Classifying a case-pattern `)` needs case
    grammar, so the scanner now refuses the whole family loudly: an unquoted
    `case` while any paren frame is open raises."""
    line = (
        "STAMP=$(case x in y) date +%s;; esac)#tag && "
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L1_PUSH_RAN"'
    )
    with pytest.raises(ValueError, match="case"):
        scan_execution_sites(line)


def test_escaped_metachar_before_a_hash_is_not_a_word_start():
    """Reconciler L5b (round 5, measured): bash runs this push — the escaped
    `;` is a word character, so the `#` is mid-word. The round-5 word-start
    check read the RAW `line[i - 1]` (the literal `;`) and stripped at `#`:
    0 sites, silently. The check now keys on the escape-aware previous bare
    character, so the site is counted and reads bounded."""
    line = 'echo a\\;#x && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L5b_ESCAPED_RAN"'
    assert _bound_flags(line) == [True]


def test_regex_rhs_refuses_loudly():
    """Reconciler L6 (round 5, measured): bash runs this push — after `=~`
    the right-hand side is a regex word where `(x)#tag` is literal. The
    round-5 scanner read the group's `)` as a word end and stripped at `#`:
    0 sites, silently. `[[ ]]` regex grammar is not modeled, so an unquoted
    whitespace-preceded `=~` now refuses loudly. (No trailing space is
    required: `=~(x)#tag` also parses and runs — measured, bash 5.1.16.)"""
    line = (
        '[[ "x#tag" =~ (x)#tag ]] && '
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "L6_REGEX_RAN"'
    )
    with pytest.raises(ValueError, match="=~"):
        scan_execution_sites(line)
    with pytest.raises(ValueError, match="=~"):
        scan_execution_sites(line.replace("=~ (x)", "=~(x)"))


def test_pattern_close_paren_inside_a_parameter_expansion_stays_literal():
    """Codex's round-5 probe, same family as L1: bash scans `${...}` to its
    matching `}`, so the pattern `)` in `${x%)}` never closes the
    substitution and the push after `)#tag` runs (measured: P4_PARAM_RAN,
    bash 5.1.16). Round 5 popped the substitution frame at the pattern `)`
    and stripped the line (0 sites); round 6's brace tracking keeps the
    interior literal, so the site is counted."""
    line = 'STAMP=$(echo ${x%)})#tag && "$PUSH" "P4_PARAM_RAN"'
    assert len(scan_execution_sites(line)) == 1


def test_top_level_case_and_tilde_assignment_do_not_refuse():
    """Refusal-scope controls: the wrappers' own shapes must keep scanning.

    A top-level `case` opens with NO paren frame (the watch scripts' dispatch
    shape, incl. the one-line `case "$seen_at" in ...` of
    cron_watch_issue_1739.sh line 111) and a `VAR=~/path` tilde assignment
    has no whitespace before its `=~`; neither trips a refusal. Both lines
    measured against bash 5.1.16: the case arm's push runs, the assignment
    is an ordinary assignment."""
    top_case = 'case "$s" in y) echo $(date +%s)#t && "$PUSH" "P8_RAN";; esac'
    assert len(scan_execution_sites(top_case)) == 1
    line = "LOG=~/x.log && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


# --- Round-7 regressions: backticks, context frames, ANSI-C quotes (ninth
# axis). Every fixture below was measured against bash 5.1.16 on 2026-08-30:
# `bash -n` accepts each line and bash RUNS each push, while the round-6
# scanner (ac76253a4f4) returned 0 sites with nothing raised on every one of
# the silent-over-strip shapes.


def test_backtick_substitution_refuses_loudly():
    """Round-6 blocker (measured): bash RUNS both pushes below — the
    closing-backtick search runs through the ``#x`` (it comments only the
    inner command), so the outer line continues after the closing tick. The
    round-6 scanner read ` #` as a word-initial comment start and returned 0
    sites with nothing raised — for the A2 shape that silently passed an
    UNBOUNDED push. A backtick interior is shell code needing its own parse,
    so any backtick outside single quotes refuses."""
    a2 = 'Y=`date +%s #x` && "$PUSH" "A2_TICK_UNBOUNDED_RAN"'
    with pytest.raises(ValueError, match="backtick"):
        scan_execution_sites(a2)
    a1 = (
        "Y=`date +%s #x` && "
        'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "A1_TICK_BOUNDED_RAN"'
    )
    with pytest.raises(ValueError, match="backtick"):
        scan_execution_sites(a1)


def test_backtick_inside_double_quotes_refuses_loudly():
    """Backticks stay live substitutions inside double quotes (measured:
    `echo "run \\`date +%s\\` now"` substitutes and its `&&` push runs), and
    an embedded quote makes the closing-tick search POSIX-undefined — so the
    scanner refuses rather than guesses there too."""
    with pytest.raises(ValueError, match="backtick"):
        scan_execution_sites('echo "run `date +%s` now" && "$PUSH" "BT_RAN"')


def test_backticks_in_comments_and_single_quotes_do_not_refuse():
    """Refusal-scope controls: the six wrappers' own backticks all sit on
    comment lines — the `#` strips the line before any tick is scanned — and
    a single-quoted backtick is literal to bash, so neither refuses."""
    assert scan_execution_sites("# a `quoted` word in a comment") == []
    assert scan_execution_sites("true # see `foo --help` for detail") == []
    line = "echo 'a `b` c' && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_substitution_nested_in_a_parameter_default_is_modeled():
    """Round-6 blocker (Codex, measured): bash RUNS this push — the nested
    `$( { echo hi; })` closes where it opened, and `#tag` is part of the
    assignment word after the REAL outer close. Round 6's flat brace depth
    counter decremented at the brace GROUP's `}`, the next `)` popped the
    OUTER substitution frame, and the line stripped at `#`: 0 sites,
    silently, in the primary-invariant direction (round 5 had caught this
    shape). The frame stack keeps the nested contexts apart, so the site is
    counted and reads UNBOUNDED — loud on a newly added site."""
    line = 'x=$(echo ${v:-$( { echo hi; })})#tag && "$PUSH" "NEW_UNBOUNDED"'
    assert _bound_flags(line) == [False]


def test_quoted_parameter_default_with_nested_substitution_scans():
    """The live idiom the frame model must keep scanning:
    cron_daily_healthcheck.sh line 53 nests `$(date ...)` inside a
    DOUBLE-QUOTED `${...:-...}` default. The brace frame restores the
    double-quote state at its `}` and the nested substitution opens a fresh
    code context, so the wrapper's own shape scans to its one bounded site
    with no refusal."""
    line = 'V="${VAR:-$(date +%s)}" && ' + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_ansi_c_escaped_quote_does_not_desync_quote_state():
    """Round-6 concern (measured): bash RUNS this bounded push — inside
    `$'...'` the `\\'` is an ESCAPED quote, so the string runs to the final
    `'` and the ` ; # x` sits INSIDE it. Round 6 closed its flat
    single-quote state at the `\\'`, read the tail as bare, and stripped at
    `#`: 0 sites, silently dropping a live push. A single quote entered via
    `$'` now tracks backslash escapes."""
    line = (
        'echo $\'a\\\' ; # x\' && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "B1_ANSI_RAN"'
    )
    assert _bound_flags(line) == [True]


def test_regular_single_quote_keeps_backslash_literal():
    """The control scoping the ANSI-C fix: in a REGULAR single quote a
    backslash is literal and the next `'` closes — bash prints `a\\` for
    `echo 'a\\'` and runs the chained push (measured) — so escape tracking
    keys on the `$'` entry, never on single quotes generally (the wrappers'
    single-quoted sed/awk programs carry backslashes)."""
    line = "echo 'a\\' && " + _LIVE.strip()
    assert _bound_flags(line) == [True]


def test_quotes_inside_a_double_quoted_substitution_do_not_desync():
    """Same silent class, found in round 7 (measured): bash RUNS both pushes
    — the inner quotes belong to the reparsed substitution/expansion
    interior, so the `)#` inside them never reaches word level. Round 6's
    flat quote state closed at the inner `"`, read `)` as a word end against
    an empty stack, and stripped at `#`: 0 sites, silently — the same
    quadrant as the backtick and nested-default blockers. Each frame records
    the double-quote state to restore at its close, so both lines scan to
    their one bounded site."""
    for label, prefix in (
        ("command substitution", 'echo "$(printf "%s" "a)#b")" && '),
        ("parameter default", 'echo "${v:-"a)#b"}" && '),
    ):
        line = prefix + _LIVE.strip()
        assert _bound_flags(line) == [True], label


def test_process_substitution_inside_a_parameter_default_refuses_loudly():
    """`echo ${v:-<(true)}` substitutes a live /dev/fd path and its `&&`
    push runs (measured), so a `<(...)` inside a `${...}` interior is
    executed code this scanner does not model — refused, never scanned as
    literal brace text."""
    with pytest.raises(ValueError, match="process substitution"):
        scan_execution_sites('echo ${v:-<(true)} && "$PUSH" "PS_RAN"')


def test_unmodeled_constructs_raise_instead_of_stripping():
    """The round-7 posture in one place: every construct the scanner knows
    it does not model REFUSES with a ValueError naming the construct — it
    never silently returns a stripped line, because a scanner returning "0
    unbounded sites" on a line it could not parse is indistinguishable from
    one that verified the line. Rounds 4-6 each enumerated one more silent
    member of this class and were bounced by the next round's new member; a
    raise is the only verdict that cannot be mistaken for a verified line."""
    refusals = [
        ("backtick", 'Y=`date` && "$PUSH" "m"'),
        ("backtick", 'echo "`date`" && "$PUSH" "m"'),
        ("process substitution", 'echo ${v:-<(true)} && "$PUSH" "m"'),
        ("case", 'S=$(case x in y) date;; esac)#t && "$PUSH" "m"'),
        ("=~", '[[ "x" =~ (x)#t ]] && "$PUSH" "m"'),
    ]
    for construct, line in refusals:
        with pytest.raises(ValueError, match=re.escape(construct)):
            scan_execution_sites(line)


def test_refusals_name_the_origin_and_line():
    """A refusal from a wrapper scan carries `<origin>:<lineno>` so the
    failing construct is locatable without re-running the scan by hand (the
    wrapper-facing tests pass each wrapper's repo-relative path)."""
    with pytest.raises(ValueError, match=r"wrapper\.sh:2: "):
        scan_execution_sites("true\nY=`date`\n", origin="wrapper.sh")


def test_direction_rule_quadrants_match_the_documented_rule():
    """The module docstring's 2x2, measured through the injectable seam
    against `_LIVE`'s one-site pin: SILENT exactly when the divergence
    leaves the count AT the pin, LOUD when it moves the count off it.

    `_identity_strip` (keeps everything) plays a maximal under-stripper;
    `_naive_first_hash_strip` (cuts at any `#`) plays a maximal
    over-stripper on the hash-before-site shape.
    """
    pin = 1  # _LIVE's own per-line inventory
    # UNDER-strip of the site IN the inventory: a commented-out live push is
    # still counted -> count AT the pin -> SILENT (a dead alert passes).
    dead = scan_execution_sites("# " + _LIVE.strip(), strip=_identity_strip)
    assert len(dead) == pin
    # UNDER-strip of a NEWLY ADDED inert site: the kept comment adds a
    # site -> count ABOVE the pin -> LOUD.
    inert_added = scan_execution_sites(_LIVE + "\n# " + _LIVE.strip(), strip=_identity_strip)
    assert len(inert_added) == pin + 1
    # OVER-strip of the site IN the inventory: the live site is dropped ->
    # count BELOW the pin -> LOUD.
    dropped = scan_execution_sites(_LIVE_HASH_BEFORE_SITE, strip=_naive_first_hash_strip)
    assert len(dropped) == pin - 1
    # OVER-strip of a NEWLY ADDED site: the added hash-before-site line is
    # dropped while _LIVE still scans -> count exactly AT the pin -> SILENT
    # (bash would run two pushes; the scan reports one).
    two_lines = _LIVE + "\n" + _LIVE_HASH_BEFORE_SITE.strip()
    added_dropped = scan_execution_sites(two_lines, strip=_naive_first_hash_strip)
    assert len(added_dropped) == pin


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
