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
  comment grammar. Heredoc BODY text leaves the scan since round 9, GUARDED
  since round 10: a body is skipped only when it carries no push reference
  and no ``<<`` — anything else REFUSES loudly, because a body fed to an
  interpreter (``bash <<EOF``, ``eval``, ``crontab -``) EXECUTES at cron
  time (measured, fixture X1: bash ran an unbounded body push that the
  round-9 exclusion silently skipped under a green suite).
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

ROUND 9 — heredoc bodies leave the scan; the engine child env is pinned.
ROUND 10 — the skip is GUARDED: push / ``<<`` text in a region refuses.

- HEREDOC BODY EXCLUSION (round 9). Bash renders body text verbatim and
  round 8 counted it, so inert push-shaped body text satisfied the EXACT
  pin: it could hold the count at pin while a real site was deleted, or
  while a live call escaped the site regex through the disclosed
  unquoted-message boundary (bash EXECUTED that unbounded push under a
  green suite), and a push refactored VERBATIM into a
  generated-child-script heredoc kept the suite green while no push
  executed at all — the reconcile-v8 S1/S2/S3 measurements. The exclusion
  is bash-verified, not hand-lexed: every ``<<`` occurrence in the
  rendering is probed by re-parsing the WHOLE rendering with that one
  delimiter token replaced by a fresh random token — a REAL here-document
  then reads to end-of-file and bash itself says so (its EOF warning / a
  parse error), while a lookalike inside ordinary quotes or arithmetic
  parses clean. A confirmed operator's body + terminator lines are
  skipped (terminator = first following rendered line equal to the raw
  delimiter — exact for both forms, measured: ``<<-`` bodies render
  tab-stripped with a bare terminator, and a body line equal to the
  delimiter is impossible: it would have terminated the document when
  bash read it); operator-line text and post-terminator text still scan.
- SKIP-REGION GUARD (round 10). Round 9 skipped confirmed regions as
  inert, and both halves of that assumption were falsified by
  measurement. (a) An interpreter-fed body EXECUTES: ``bash <<EOF``
  around an unbounded regex-matching push was count-neutral and the
  suite GREEN while bash ran the push (measured, fixture X1; the
  round-8 tip went RED on the same bytes); ``eval "$(cat <<EOF ...)"``
  the same (measured, A9); ``crontab - <<EOF`` is the same channel
  deferred to cron, and an ORDINARY refactor here — both live watch
  wrappers already rewrite the crontab via ``| crontab -``. (b) The
  perturbation probe is NOT sound in the over-skip direction:
  partial-token mutation of a candidate whose capture consumes syntax
  owned by an enclosing construct reads the resulting parse error as an
  operator verdict — measured: ``echo "${x:-<<}"`` inside a multi-line
  function classified real, the function-closing ``}`` became its
  terminator, and the live unbounded push inside the function was
  skipped SILENTLY (scan 0, suite GREEN, bash executed it). ONE rule
  makes both directions loud: a skip region may contain NEITHER a push
  reference (``_PUSH_REF`` — broader than the site regex; the escaped
  ``\\$PUSH`` eval spelling included) NOR any ``<<`` text (an unprobed
  operator swallowed into a skipped region desynchronizes extents —
  measured: a false region swallowed a real ``cat <<EOF`` line and that
  document's bounded-looking body line leaked into the scan as a
  count-preserving site, GREEN at pin 1 with NO push executing). Either
  finding REFUSES loudly, so a misclassified region lands on a refusal
  rather than a silent pass in both directions; a real document
  misclassified as a lookalike scans its body as commands and moves the
  count off the pin (loud — measured, the forced-UUID-collision probe,
  round-9 review). What the guard does NOT certify: regions free of
  push / ``<<`` text are still skipped on the probe's verdict alone,
  and the scan never decides whether a body executes (there is no
  interpreter-consumer list — a push-free body handed to ``bash`` is
  skipped without refusal and without coverage).
- REFUSED loudly, never guessed: an unterminated here-document (bash's
  EOF warning on the full render — pretty-print SYNTHESIZES the missing
  terminator and exits 0, so round 8 silently scanned the swallowed
  tail as command text), multiple here-documents on one rendered line,
  a terminator the scan cannot locate, and — round 10 — a push
  reference or ``<<`` text inside a skip region.
- ENGINE CHILD ENV pinned to ``{PATH, LC_ALL=C}`` — nothing is inherited.
  Measured on the round-8 engine: a ``BASH_ENV`` script EXECUTED inside
  the parse-only child, and ``BASHOPTS=extglob`` flipped the extglob
  refusal into a clean parse (``ENV`` / ``SHELLOPTS`` are the same two
  classes); ``LC_ALL=C`` also pins the warning text the heredoc probes
  match. Pinned by the two env-canary tests.

FAIL-LOUD CONTRACT: anything bash cannot parse makes ``--pretty-print`` exit
non-zero and ``scan_execution_sites`` raise ValueError carrying the origin
plus bash's own diagnosis (which names the SOURCE line); rounds 9-10 add
the four heredoc refusals above on the same contract. A refusal can
neither silently pass an unbounded push nor silently drop a live one. The
rounds-4-7 refusal roster (backticks, ``case`` in a substitution, ``=~``
regex words, process substitutions in ``${...}``) is GONE as a refusal
class: bash parses all of those, so they are now modeled — each former
refusal fixture below asserts the measured bash behavior instead.

DIRECTION RULE (measured, rounds 4-7; still the analytical frame): a
scanner-vs-bash divergence is SILENT exactly when it leaves the wrapper's
site count sitting AT its pin, LOUD when it moves the count off the pin.
ONE strip step remains — the heredoc skip — and round 10 constrains it: a
line is dropped from the scan only when it carries no push reference and
no ``<<`` (anything else refuses — measured: the X1 / false-region
fixtures refuse where the round-9 code passed them silently). A dropped
line therefore never carries a push site or an unprobed heredoc operator.
Every other executed command appears in bash's own rendering of the parse
tree. What remains is at the MATCH level, disclosed below.

Known residuals after round 9, each classified by the direction rule:

- POSITIONAL BLINDNESS of the site regex — the OPEN silent channel, stated
  plainly: the scan asks whether text MATCHES, not whether bash would
  EXECUTE it. Push-shaped text in any non-execution position — a quoted
  string (``echo '"$PUSH" "m"'``), an argument word of another command
  (``echo "$PUSH" "m"``), an array literal, a herestring word — satisfies
  ``_EXEC_SITE`` without executing. ADDING such text moves the count ABOVE
  the pin (loud); REPLACING an inventory site with such a husk keeps the
  count AT the pin — SILENT, unchanged since round 1, and measured end to
  end as reconcile-v8 S4 (a bounded-looking single-quoted husk plus an
  unquoted-message live call stayed green while the unbounded push
  executed). These positions are reachable by ORDINARY edits, not only by
  deliberately written decoys — measured (round-9 review): push-command
  logging and usage/help text count one site [False]; an array command
  template and a herestring-fed config line count one [True]. Rounds 9-10
  closed the members bash itself can adjudicate — comments
  (parser-dropped) and heredoc bodies (push-bearing ones now REFUSE) —
  and leave the rest open because closing them needs command-position
  parsing, the hand-lexer class rounds 4-7 disproved. Diff review is the
  MITIGATION for the husk-replacement compound, not proof that innocent
  edits cannot reach the channel; what the pin guarantees is narrower:
  every edit that CHANGES a wrapper's count is visible as a count change,
  while a replacement that preserves it is not. Pinned as disclosures by
  ``test_quoted_push_shaped_text_counts_the_disclosed_silent_residual``,
  ``test_husk_channel_argument_position_and_bounded_looking_spellings``,
  and ``test_direction_rule_quadrants``.
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
re-derived round 10 through the round-10 engine — the guarded heredoc skip
plus the scrubbed child env — 2026-08-30, bash 5.1.16): all six render with
zero refusals to exactly the pinned counts 1/3/2/2/2/2 = 12, every site
bounded, unchanged from the round-8/9 censuses (the one heredoc,
``cron_watch_issue_1739.sh``'s ``done <<EOF`` loop feed, holds a bare
``$RUNNING`` — no push reference, no ``<<`` — so its body is skipped
without refusal and moves no count). Zero ``NAME=(`` arrays, no
``shopt -s extglob``, zero quoted strings containing push-shaped text. The
two multi-line command substitutions
(``cron_watch_issue_2091.sh``, ``cron_watch_issue_1739.sh``) and the
double-quoted brace-nested substitution (``cron_daily_healthcheck.sh``)
all parse and render clean.

COVERAGE BOUNDARY: the scan is bounded by the WRAPPERS mapping below plus
the execution-site regex. A NEW cron wrapper calling the push helper must be
ADDED to the mapping or it escapes every pin here; likewise a call shape
whose message argument is not a double-quoted string immediately after the
push variable (an unquoted message, a renamed variable) escapes the regex.
Extend both when adding either. The unquoted-message escape is one leg of
the reconcile-v8 S2 compound: a push-bearing heredoc pad now REFUSES
outright (measured, round 10), so the pad cannot vouch for it, but the
husk-REPLACEMENT compound (S4) stays silent — see the positional-blindness
residual above.

Behavioral twins (sleeping-stub tests, one per call-site composition shape)
live in tests/test_cron_step9c_ledger_refresh.py (if-condition),
tests/test_cron_lesson_consolidate.py (||-chained fatal arm), and
tests/test_codex_auto_upgrade.py (command substitution).
"""

from __future__ import annotations

import re
import subprocess
import uuid
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

# Environment for EVERY engine child (rendering, heredoc probes, `bash -n`):
# PATH to find bash, LC_ALL=C to pin the warning text `_HEREDOC_EOF_WARNING`
# matches. Everything else is deliberately absent — measured on the round-8
# engine: a `BASH_ENV` script EXECUTED inside the parse-only child, and
# `BASHOPTS=extglob` flipped the extglob refusal into a clean parse (`ENV` /
# `SHELLOPTS` are the same two classes). Pinned by the env-canary tests.
_ENGINE_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}

# Bash's own report that a here-document reads to end-of-file. On the FULL
# render it means an unterminated heredoc — refused, because pretty-print
# SYNTHESIZES the missing terminator and exits 0 (measured), which round 8
# read as a clean parse while the wrapper's whole tail had become body text.
# On a perturbation probe it is the POSITIVE signal that the probed `<<` is
# a real heredoc operator (see `_is_real_heredoc`).
_HEREDOC_EOF_WARNING = re.compile(r"here-document at line \d+ delimited by end-of-file \(wanted `")

# Candidate heredoc operator in a rendered line: `<<` or `<<-` (never the
# `<<<` herestring), then the delimiter token as bash prints it — quoted,
# backslash-escaped, or a bare word ended by whitespace / operator chars.
# Over-matching costs one probe; the probe's verdict is NOT trusted to be
# sound (measured, round 10: `"${x:-<<}"` classifies as an operator) — the
# skip-region guard below is what keeps a wrong verdict loud.
_HEREDOC_CANDIDATE = re.compile(
    r"(?<!<)<<(?!<)(-?)[ \t]*('[^']*'|\"[^\"]*\"|\\?[^\s<>&|;()\"'\\]+)"
)

# Text no skip region may contain (round 10): any reference to the push
# variable — $PUSH / ${PUSH} / $TELEGRAM_PUSH / ${TELEGRAM_PUSH}; a
# backslash-escaped \$PUSH still matches at its `$`, the eval-heredoc
# spelling (measured, fixture A9). Deliberately BROADER than `_EXEC_SITE`:
# a skipped line is never seen again, so the guard must catch push text the
# site regex would miss (unquoted messages, escaped dollars). It does NOT
# match `$PUSH_TIMEOUT` / `${PUSH_TIMEOUT}` (word boundary / closing brace),
# so a body defining its own timeout does not refuse.
_PUSH_REF = re.compile(r"\$(?:\{(?:TELEGRAM_PUSH|PUSH)\}|(?:TELEGRAM_PUSH|PUSH)\b)")


def _run_engine(text: str) -> subprocess.CompletedProcess[str]:
    """One parse-only engine child over ``text``: ``bash --pretty-print``
    under the scrubbed ``_ENGINE_ENV``. Callers read returncode / stdout /
    stderr; nothing is ever executed (``test_rendering_never_executes``)."""
    return subprocess.run(
        ["bash", "--pretty-print", "/dev/stdin"],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
        env=_ENGINE_ENV,
    )


def _bash_rendering(text: str, origin: str = "<text>") -> str:
    """Return bash's own executable rendering of ``text`` (module docstring,
    THE ENGINE): the program printed from the real parse tree by
    ``bash --pretty-print``, comments discarded by the parser, ``\\``
    continuations joined, nothing executed.

    Raises ValueError — the fail-loud contract — on either of bash's own
    diagnoses: a non-zero exit (bash cannot parse the text; stderr names the
    SOURCE line), or the here-document end-of-file warning (an unterminated
    heredoc swallows every line after its operator as body text while
    pretty-print synthesizes the terminator and exits 0, so without the
    stderr check the dead tail would scan as command text). There is no
    construct list to fall out of sync with bash's grammar.
    """
    proc = _run_engine(text)
    if proc.returncode != 0:
        raise ValueError(
            f"{origin}: bash cannot parse this text (--pretty-print exited "
            f"{proc.returncode}); the scan refuses rather than guessing at "
            f"constructs outside bash's grammar. bash says: {proc.stderr.strip()}"
        )
    if _HEREDOC_EOF_WARNING.search(proc.stderr):
        raise ValueError(
            f"{origin}: a here-document reads to end-of-file, so every line "
            f"after its operator is body text bash will never execute; the "
            f"scan refuses rather than counting a wrapper whose tail is "
            f"inert. bash says: {proc.stderr.strip()}"
        )
    return proc.stdout


def _raw_delimiter(token: str) -> str:
    """The terminator-line text for a rendered heredoc delimiter token:
    surrounding quotes / a leading backslash affect only body expansion,
    never the terminator, which is always the raw word."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token.removeprefix("\\")


def _is_real_heredoc(lines: list[str], idx: int, cand: re.Match[str]) -> bool:
    """Bash's own verdict on whether a ``<<`` candidate is a heredoc
    OPERATOR (vs inert lookalike text inside quotes / arithmetic): re-parse
    the WHOLE rendering with the candidate's delimiter token replaced by a
    fresh random token. A real operator's document then reads to
    end-of-file — bash warns, or errors — while an inert lookalike parses
    clean; the per-probe random token also defeats a pre-planted terminator
    line. This is the round-8 delegation law applied to heredoc structure:
    the only tokenizer guaranteed to agree with bash about what ``<<``
    means is bash."""
    probe = f"__EPS_HD_PROBE_{uuid.uuid4().hex}__"
    line = lines[idx]
    mutated = line[: cand.start(2)] + probe + line[cand.end(2) :]
    proc = _run_engine("\n".join([*lines[:idx], mutated, *lines[idx + 1 :]]) + "\n")
    return proc.returncode != 0 or _HEREDOC_EOF_WARNING.search(proc.stderr) is not None


def scan_execution_sites(
    text: str,
    origin: str = "<text>",
) -> list[tuple[int, str, re.Match[str]]]:
    """Return ``(rendered_lineno, rendered_line, match)`` per push EXECUTION
    site of ``text``, scanning the COMMAND TEXT of bash's own rendering.

    Two exclusions, both owned by bash's parser rather than a re-implemented
    grammar: comments never reach the scan (the parser drops them), and
    here-document BODY + terminator lines are skipped (round 9 — every
    ``<<`` occurrence is verified operator-vs-lookalike by
    ``_is_real_heredoc``), GUARDED since round 10: a region is skipped only
    when it contains no push reference (``_PUSH_REF``) and no ``<<`` text —
    either one REFUSES loudly, because an interpreter-fed body executes at
    cron time (measured, X1: ``bash <<EOF`` ran its body push while the
    round-9 scan stayed count-neutral) and a lookalike misclassified as an
    operator opens a false skip region over live code (measured,
    ``"${x:-<<}"`` in a function). A SKIPPED line therefore never carries a
    push site or an unprobed heredoc operator; push-free skip regions rest
    on the probe's verdict alone. Line numbers are rendering-relative
    (module docstring, residuals).

    Raises ValueError on text bash cannot parse or an unterminated heredoc
    (``_bash_rendering``), on multiple here-documents on one rendered line,
    on a heredoc terminator the scan cannot locate, and on a push reference
    or ``<<`` text inside a skip region.
    """
    rendered = _bash_rendering(text, origin)
    lines = rendered.splitlines()
    sites: list[tuple[int, str, re.Match[str]]] = []
    skip_until = -1  # index of the open here-document's terminator line
    for idx, line in enumerate(lines):
        if idx <= skip_until:
            continue  # skip region: verified free of push refs + '<<' at establishment
        if "<<" in line:
            reals = [
                c for c in _HEREDOC_CANDIDATE.finditer(line) if _is_real_heredoc(lines, idx, c)
            ]
            if len(reals) > 1:
                raise ValueError(
                    f"{origin}: multiple here-documents on one rendered line "
                    f"({line.strip()!r}) — body extents are ambiguous; the scan "
                    f"refuses rather than guessing which lines bash treats as data"
                )
            if reals:
                raw = _raw_delimiter(reals[0].group(2))
                for j in range(idx + 1, len(lines)):
                    if lines[j] == raw:
                        skip_until = j
                        break
                else:
                    raise ValueError(
                        f"{origin}: cannot locate the terminator {raw!r} of the "
                        f"here-document opened on rendered line {idx + 1}; the "
                        f"scan refuses rather than guessing the body's extent"
                    )
                for k in range(idx + 1, skip_until + 1):
                    if _PUSH_REF.search(lines[k]) or "<<" in lines[k]:
                        offense = "a push reference" if _PUSH_REF.search(lines[k]) else "'<<' text"
                        raise ValueError(
                            f"{origin}: {offense} inside the here-document opened "
                            f"on rendered line {idx + 1} (rendered line {k + 1}: "
                            f"{lines[k].strip()!r}); a body fed to an interpreter "
                            f"(bash <<EOF, eval, crontab -) EXECUTES at cron time, "
                            f"and a lookalike misclassified as an operator skips "
                            f"live code, so the scan refuses rather than silently "
                            f"skipping text that could carry a push site or "
                            f"another here-document — move the push (or the '<<') "
                            f"out of the here-document"
                        )
        for m in _EXEC_SITE.finditer(line):
            sites.append((idx + 1, line, m))
    return sites


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
            env=_ENGINE_ENV,
        )
        assert proc.returncode == 0, f"{rel}: bash -n failed:\n{proc.stderr}"


# --- Engine invariants -------------------------------------------------------


def test_rendering_never_executes(tmp_path):
    """The engine is parse-only. Load-bearing, not a nicety: the two watch
    scripts' terminal arms rewrite the real user crontab, and this scan is
    their only coverage vehicle precisely BECAUSE no test may execute them.
    Canary writes prove `--pretty-print` builds the parse tree without
    running a single command — including the strongest execution path, a
    command substitution (whose canary would fire at PARSE time if the
    engine expanded words), and a heredoc-body expansion (which also
    exercises the round-9 body-exclusion probes on live canary text)."""
    canary = tmp_path / "canary.txt"
    text = (
        f'echo executed > "{canary}"\n'
        f'date >> "{canary}"\n'
        f'x=$(touch "{canary}")\n'
        f'cat <<EOF\n$(touch "{canary}")\nEOF\n'
    )
    sites = scan_execution_sites(text)
    assert sites == []
    assert not canary.exists(), "--pretty-print EXECUTED the text it was asked to parse"


def test_bash_env_cannot_reach_the_engine_child(tmp_path, monkeypatch):
    """``bash --pretty-print`` SOURCES ``$BASH_ENV`` before parsing —
    measured on the round-8 engine, where this exact canary EXECUTED — so
    every engine child runs under ``_ENGINE_ENV``, which inherits nothing.
    ``ENV`` is the same class (the POSIX-mode spelling)."""
    canary = tmp_path / "bash_env_canary"
    hook = tmp_path / "bash_env_hook.sh"
    hook.write_text(f'touch "{canary}"\n')
    monkeypatch.setenv("BASH_ENV", str(hook))
    monkeypatch.setenv("ENV", str(hook))
    assert scan_execution_sites("true\n") == []
    assert not canary.exists(), "the engine child sourced $BASH_ENV"


def test_bashopts_cannot_flip_a_refusal_verdict(monkeypatch):
    """``BASHOPTS=extglob`` in the child env enables extglob at parse time —
    measured on the round-8 engine, where it flipped the extglob-assignment
    REFUSAL into a clean parse, silently changing a pinned verdict
    (``SHELLOPTS`` is the same class). The scrubbed env keeps the refusal."""
    monkeypatch.setenv("BASHOPTS", "extglob")
    monkeypatch.setenv("SHELLOPTS", "braceexpand")
    with pytest.raises(ValueError, match="cannot parse"):
        scan_execution_sites('x=@(a|b)#tag && "$PUSH" "EXTGLOB_ASSIGNMENT_RAN"')


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


# --- Round-9/10 regressions: heredoc masking + the skip-region guard ---------
#
# Round 8 counted heredoc BODY text (bash renders it verbatim; the site regex
# is textual), and the round-8 reconciliation measured that as a
# pin-PRESERVING masking class (epm:review-reconcile v8, fixtures S1-S3):
# inert body text held the count AT the exact pin while a real site was
# deleted (S1), while a live unbounded call rode the regex's disclosed
# unquoted-message boundary — bash EXECUTED the unbounded push under a green
# suite (S2, the blocker criterion verbatim) — and a push REFACTORED verbatim
# into a generated-child-script heredoc kept the suite green while NO push
# executed at all (S3). Round 9 excluded confirmed bodies from the scan,
# which closed the masking class but ASSUMED skipped text inert — falsified
# twice by round-9-review + round-10 measurement: an interpreter-fed body
# EXECUTES (X1: `bash <<EOF` around an unbounded push was count-neutral and
# the suite GREEN while bash ran it — the round-8 tip went RED on the same
# bytes; A9: `eval "$(cat <<EOF ...)"` the same), and the perturbation probe
# can be steered into a FALSE skip region over live code (`"${x:-<<}"` in a
# multi-line function: scan 0, suite GREEN, bash ran the push). Round 10
# GUARDS the skip: a region containing a push reference or any `<<` text
# REFUSES loudly, so both misclassification directions land on a refusal
# instead of a silent pass. The push-bearing S1/S2/S3 fixtures therefore now
# REFUSE — still loud, one step earlier than the round-9 count-off-pin form.
# Pre-fix behavior per test (measured on the r9 tip ffed40e240c) is recorded
# in each docstring; the full pre/post table is in the round-10
# implementation marker.


def _bounded_push(msg: str) -> str:
    """One correctly bounded push line, the reconcile-v8 fixture shape."""
    return f'timeout --kill-after=5s "${{PUSH_TIMEOUT}}s" "$PUSH" "{msg}"\n'


def test_push_moved_into_a_heredoc_body_refuses():
    """reconcile-v8 S3, the single-edit spelling: a real bounded push line
    moved VERBATIM into a heredoc body (the generate-a-child-script
    refactor). Round 10 REFUSES: the parse tree cannot tell an inert
    ``cat`` feed from a body a consumer hands to an interpreter, so a
    push-bearing body never passes silently in either direction. Pre-fix:
    r8 tip counted it (1 site [True], GREEN at pin 1, no push runs); r9
    tip (measured) returned [] with no raise — loud only via the count
    pin."""
    text = "cat > /dev/null <<EOF\n" + _bounded_push("real_site_A") + "EOF\n"
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(text)


def test_heredoc_pad_cannot_mask_a_deleted_site():
    """reconcile-v8 S1: one real bounded site plus a bounded-looking heredoc
    body line. Round 10 refuses on the pad itself, so the masking compound
    dies before any count arithmetic. Pre-fix: r8 tip scanned 2 [True,
    True] — GREEN at pin 2 while bash executed only one push; r9 tip
    (measured) scanned 1 [True] with no raise — loud at pin 2 only via
    the count."""
    text = _bounded_push("real_site_A") + "cat <<EOF\n" + _bounded_push("PSEUDO") + "EOF\n"
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(text)


def test_heredoc_pad_plus_regex_escaping_call_goes_loud():
    """reconcile-v8 S2 — the blocker criterion verbatim: a bounded-looking
    heredoc body plus a live push whose UNQUOTED message escapes the site
    regex (the disclosed COVERAGE BOUNDARY). Round 10 refuses on the pad,
    so the live unbounded call never needs the count to catch it. Pre-fix:
    r8 tip scanned 1 [True] — GREEN while bash executed the unbounded
    push; r9 tip (measured) scanned [] with no raise — loud at pin 1 only
    via the count."""
    text = "cat <<EOF\n" + _bounded_push("PSEUDO") + "EOF\n" + '"$PUSH" UNBOUNDED_UNQUOTED\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(text)


def test_interpreter_fed_heredoc_body_with_push_refuses():
    """Round-10 blocker `heredoc-executable-body-excluded`, fixture X1: a
    heredoc body fed to an interpreter EXECUTES — measured on the r9 tip:
    `bash <<EOF` around an unbounded regex-matching push scanned
    count-neutral (suite GREEN at pin 1 with one real site) while bash
    ground truth logged `PUSH_EXECUTED: LIVE_FROM_HEREDOC`; the round-8
    tip went RED on the same bytes, so round 9 turned a loud case silent.
    The scan cannot tell `bash <<EOF` from `cat <<EOF` off the parse
    tree, so ANY push-bearing body refuses — including the
    bounded-inside spelling (X1b: a later in-body unbounding was
    invisible to round 9, measured 1 [True] GREEN) and the `crontab -`
    refactor shape (both live watch wrappers already rewrite the crontab
    via `| crontab -`; measured r9: scanned 0, no raise)."""
    x1 = 'bash <<EOF\n"$PUSH" "LIVE_FROM_HEREDOC"\nEOF\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(x1)
    x1b = _bounded_push("real_site_A") + "bash <<EOF\n" + _bounded_push("HD_BOUNDED") + "EOF\n"
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(x1b)
    crontab_shape = 'crontab - <<EOF\n*/10 * * * * "$PUSH" "FROM_CRON"\nEOF\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(crontab_shape)


def test_eval_heredoc_body_with_escaped_push_refuses():
    """Fixture A9, the second executable-body spelling: `eval "$(cat <<EOF
    ...)"` executes its body with the escaped ``\\$PUSH`` expanded back to
    a live call — measured on the r9 tip: scanned 0 sites (suite GREEN at
    pin 0) while bash executed `EVAL_HD_UNBOUNDED`. ``_PUSH_REF`` matches
    the escaped spelling at its `$`, so the body refuses."""
    a9 = 'eval "$(cat <<EOF\n"\\$PUSH" "EVAL_HD_UNBOUNDED"\nEOF\n)"\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(a9)


def test_misclassified_lookalike_cannot_silently_skip_a_push():
    """Round-10 blocker `heredoc-perturbation-lookalike-over-skip`, measured
    on the r9 tip: partial-token mutation of ``"${x:-<<}"`` breaks the
    enclosing expansion, the parse error reads as an operator verdict, the
    function-closing ``}`` becomes a fake terminator, and the live
    unbounded push inside the function was skipped SILENTLY (scan 0, suite
    GREEN at pin 0) while bash executed it. Round 10 does not make the
    probe sound — it makes the misclassification LOUD: the falsely
    skipped region carries a push reference, so the scan refuses. Same
    for the ``${x:-<<EOF}`` spelling whose capture eats the closing brace
    (raw delimiter ``EOF}``) with a literal ``EOF}`` line downstream
    (measured r9: 0 sites, bash executed the push)."""
    fn = 'f()\n{\n  echo "${x:-<<}"\n  "$PUSH" "LIVE_FUNCTION_UNBOUNDED"\n}\nf\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(fn)
    param = 'echo "${x:-<<EOF}"\n"$PUSH" "LIVE_PARAM_UNBOUNDED"\nEOF}\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(param)


def test_heredoc_text_inside_a_skip_region_refuses():
    """The second half of the skip-region guard: ``<<`` inside a skipped
    region refuses, because an UNPROBED operator line swallowed by a
    misclassified region desynchronizes body extents — measured on the r9
    tip: a false ``"${x:-<<}"`` region swallowed a real ``cat <<EOF``
    line, and that document's bounded-looking body line leaked into the
    scan as a count-preserving site (1 site [True], suite GREEN at pin 1,
    NO push executed — the round-8 masking shape resurrected through the
    probe). The plain nested-heredoc-text spelling refuses the same way
    (measured r9: silently skipped, 0 sites)."""
    compound = 'echo "${x:-<<}"\ncat <<EOF\n}\n' + _bounded_push("LEAKED_PAD") + "EOF\n"
    with pytest.raises(ValueError, match="'<<' text inside the here-document"):
        scan_execution_sites(compound)
    nested = "cat <<EOF\nusage: cat <<XYZ\nEOF\n"
    with pytest.raises(ValueError, match="'<<' text inside the here-document"):
        scan_execution_sites(nested)


def test_unterminated_heredoc_refuses():
    """A here-document delimited by end-of-file converts the wrapper's whole
    tail into body text; pretty-print SYNTHESIZES the terminator and exits
    0 (measured), so round 8 silently scanned the swallowed tail as command
    text (1 site from this fixture, no refusal). Round 9 refuses on bash's
    own warning."""
    with pytest.raises(ValueError, match="end-of-file"):
        scan_execution_sites('cat <<EOF\n"$PUSH" "swallowed"\n')


def test_multiple_heredocs_on_one_line_refuse():
    """Two documents on one command have sequential bodies — ambiguous
    extents for a line scan; refused loudly per the reconcile-v8 fix
    contract, never guessed."""
    with pytest.raises(ValueError, match="multiple here-documents"):
        scan_execution_sites('cat <<A <<B\n"$PUSH" "in body A"\nA\nbodyB\nB\n')


def test_command_text_around_a_heredoc_still_scans():
    """The exclusion must not over-skip: operator-line code (rendered AFTER
    the body by pretty-print — measured), post-terminator code, and code
    after a heredoc-in-command-substitution all keep their sites. Bodies
    here are genuinely inert — no push reference, no ``<<`` — the only
    body content round 10 still skips."""
    opline = 'cat <<EOF && "$PUSH" "OPLINE"\nbody\nEOF\n'
    assert _bound_flags(opline) == [False]
    after = "cat > /dev/null <<EOF\ninert body\nEOF\n" + _LIVE + "\n"
    assert _bound_flags(after) == [True]
    cmdsub = 'x=$(cat <<EOF\ninert body\nEOF\n)\n"$PUSH" "CS_AFTER"\n'
    assert _bound_flags(cmdsub) == [False]


def test_heredoc_body_exclusion_covers_dash_quoted_and_fake_terminator_forms():
    """Both halves of the round-10 skip are form-independent (the
    reconcile-v8 sweep forms): ``<<-`` bodies render tab-stripped with a
    bare terminator, quoted delimiters render with a raw terminator line,
    and the wrappers' own ``done <<EOF`` feed shape — an INERT body skips
    silently and contributes nothing, while the SAME form with a
    push-bearing body refuses. The tab-prefixed lookalike terminator
    inside a NON-dash body stays body per bash: the refusal PROVES the
    region extended past the fake ``\\tEOF`` to the real terminator — a
    wrong early extent would have scanned the push line as a site instead
    of raising (measured r9: all four push-bearing forms silently
    excluded, no raise)."""
    assert scan_execution_sites("cat <<-EOF\n\tinert tab body\n\tEOF\n") == []
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites('cat <<-EOF\n\t"$PUSH" "TAB_BODY"\n\tEOF\n')
    assert scan_execution_sites("cat <<'EOF'\ninert quoted body\nEOF\n") == []
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites('cat <<\'EOF\'\n"$PUSH" "QD_BODY"\nEOF\n')
    done_feed = 'while read -r x; do\n  echo "$x"\ndone <<EOF\n$RUNNING\nEOF\n'
    assert scan_execution_sites(done_feed) == []
    fake_term = 'cat <<EOF\n\tEOF\n"$PUSH" "STILL_BODY"\nEOF\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(fake_term)


def test_heredoc_lookalikes_are_not_operators():
    """The operator-vs-lookalike verdict is bash's, not a quote tracker's: a
    ``<<EOF`` inside a double-quoted string must NOT open a skip region (a
    fake operator hiding the next live site would be the inverse masking —
    an over-SKIP), and an arithmetic left shift is not a heredoc. The bare
    ``EOF`` line in the first fixture is an ordinary (never-run) command,
    present so a wrong operator verdict would have a terminator to skip
    to — and since round 10 a wrong verdict here could not stay silent
    anyway: the region would hold ``_LIVE``'s push reference and refuse
    (``test_misclassified_lookalike_cannot_silently_skip_a_push``)."""
    fake = 'echo "see <<EOF for details"\n' + _LIVE + "\nEOF\n"
    assert _bound_flags(fake) == [True]
    assert _bound_flags('echo $((1<<2)) && "$PUSH" "SHIFT_RAN"\n') == [False]


def test_composite_delimiter_rendering_is_normalized():
    """``_raw_delimiter`` computes the terminator from the RENDERED token,
    which is correct for composite delimiters only because pretty-print
    NORMALIZES them into one token — measured on this bash (5.1.16):
    ``cat <<'EOF'x`` renders as ``cat <<'EOFx'``, whose raw terminator is
    exactly the ``EOFx`` bash wants. Pinned (round-9 review concern
    `heredoc-delimiter-normalization-unpinned`) so a bash-version drift in
    this rendering fails loud here instead of silently mislocating a
    terminator."""
    rendering = _bash_rendering("cat <<'EOF'x\nbodyline\nEOFx\ntrue\n")
    assert "cat <<'EOFx'" in rendering, rendering
    assert _raw_delimiter("'EOFx'") == "EOFx"
    text = "cat <<'EOF'x\nbodyline\nEOFx\n" + _bounded_push("after")
    assert _bound_flags(text) == [True]


def test_unlocatable_terminator_refuses():
    """The cannot-locate-terminator refusal, previously unpinned (round-9
    review NIT `unlocatable-terminator-refusal-unpinned`): a confirmed
    candidate whose raw delimiter never appears on a following rendered
    line refuses rather than guessing the body's extent. Reached via the
    false-real parameter-expansion shape with no matching downstream line
    — behavior identical on the r9 tip (measured): this PINS the branch,
    it does not change it."""
    with pytest.raises(ValueError, match="cannot locate the terminator"):
        scan_execution_sites('echo "${x:-<<}"\ntrue\n')


# --- Direction-rule disclosures (scope controls, not regression evidence) ----


def test_quoted_push_shaped_text_counts_the_disclosed_silent_residual():
    """DISCLOSED-OPEN residual (module docstring, positional blindness):
    push-shaped text inside a quoted string satisfies the textual site
    regex without executing, so REPLACING an inventory site with its
    quoted-out husk keeps the count AT the pin — silent. Unchanged since
    round 1; rounds 9-10 closed the heredoc-body member of this class
    (push-bearing bodies now refuse) and leave this one open —
    command-position parsing is the rounds-4-7 hand-lexer class, and the
    position is reachable by ordinary edits (quoted log/help text,
    measured round-9 review), so diff review is the MITIGATION, not a
    guarantee. Pinned so an engine change that closes or widens the
    channel is a deliberate edit, not drift."""
    sites = scan_execution_sites('echo \'"$PUSH" "quoted, never executed"\'')
    assert len(sites) == 1


def test_husk_channel_argument_position_and_bounded_looking_spellings():
    """The husk channel is NOT quote-specific (the round-8 wording
    under-scoped it): push-shaped words in ARGUMENT position of another
    command count too, unquoted — ``echo "$PUSH" "m"`` is a semi-plausible
    logging edit that keeps the count when it REPLACES a live site — and
    the bounded-looking single-quoted husk even carries flag True, so the
    reconcile-v8 S4 compound (husk + a regex-escaping live call) sits
    exactly AT a one-site pin: measured GREEN while bash executed the
    unbounded push. Both spellings pinned as OPEN disclosures."""
    assert _bound_flags('echo "$PUSH" "logged, never pushed"') == [False]
    s4 = (
        'echo \'timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" "m"\' > /dev/null\n'
        '"$PUSH" UNBOUNDED_UNQUOTED\n'
    )
    assert _bound_flags(s4) == [True]


def test_direction_rule_quadrants():
    """The direction rule against ``_LIVE``'s one-site pin, after round 10:
    disabling the inventory site (comment-out) drops the count BELOW the
    pin — loud; ADDING inert push-shaped text raises it ABOVE the pin —
    loud via the husk; adding a PUSH-BEARING heredoc pad REFUSES — round
    9 measured this add as a count-neutral NON-EVENT over text that
    executes when interpreter-fed (the C1/X1 shape; the r8 tip was RED on
    the same bytes, so round 9 had removed loudness here); a PUSH-FREE
    heredoc pad stays a genuine non-event, because a region free of push
    refs and ``<<`` cannot carry a site or an operator; REPLACING the
    site with a husk keeps the count AT the pin — the silent quadrant,
    disclosed OPEN in the module docstring, never denied."""
    pin = 1
    assert len(scan_execution_sites("# " + _LIVE.strip())) == pin - 1
    husk_add = _LIVE + '\necho \'"$PUSH" "inert husk"\'\n'
    assert len(scan_execution_sites(husk_add)) == pin + 1
    pad_add = _LIVE + '\ncat <<EOF\n"$PUSH" "inert body"\nEOF\n'
    with pytest.raises(ValueError, match="push reference inside the here-document"):
        scan_execution_sites(pad_add)
    inert_pad_add = _LIVE + "\ncat <<EOF\ninert body\nEOF\n"
    assert len(scan_execution_sites(inert_pad_add)) == pin
    husk_replace = 'echo \'"$PUSH" "husk, never executed"\''
    assert len(scan_execution_sites(husk_replace)) == pin


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
