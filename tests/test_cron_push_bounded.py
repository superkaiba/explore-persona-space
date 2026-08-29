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

Why the site check is exact on all five axes (#2387; the last two are
round-3 hardening of the exact-count pin the first three introduced):

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

Behavioral twins (sleeping-stub tests, one per call-site composition shape)
live in tests/test_cron_step9c_ledger_refresh.py (if-condition),
tests/test_cron_lesson_consolidate.py (||-chained fatal arm), and
tests/test_codex_auto_upgrade.py (command substitution).

COVERAGE BOUNDARY: the scan is bounded by the WRAPPERS mapping below plus the
execution-site regex. A NEW cron wrapper calling the push helper must be ADDED
to the mapping or it escapes every pin here; likewise a call shape whose
message argument is not a double-quoted string immediately after the push
variable (e.g. an unquoted message) escapes the regex. Extend both when adding
either. Comment stripping is LINE-SCOPED: quote state is not carried across
physical lines, so a string or heredoc body spanning several lines is not
tracked, and a push execution written inside a heredoc body would be counted
as a live site. Both residuals fail LOUD on the count assertion rather than
passing silently, and neither shape exists in the six wrappers today (the one
heredoc, ``cron_watch_issue_1739.sh`` line 119, carries a bare variable).
"""

from __future__ import annotations

import re
import subprocess
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
# or right after unquoted whitespace or a control operator. A `#` inside quotes
# (`"EPS #${ISSUE} ... retired."`) or mid-word (`$#`) is not a comment.
_COMMENT_WORD_START = " \t;&|("


def _strip_bash_comment(line: str) -> str:
    """Return ``line`` truncated at its first unquoted, word-initial ``#``.

    Quote-aware by necessity, not by taste: every watch-script push line
    carries a ``#`` INSIDE its double-quoted message, so a naive first-hash
    truncation would cut live execution sites short.
    """
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
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
        elif ch == "#" and (i == 0 or line[i - 1] in _COMMENT_WORD_START):
            return line[:i]
        i += 1
    return line


def scan_execution_sites(text: str) -> list[tuple[int, str, re.Match[str]]]:
    """Return ``(lineno, executable_line, match)`` per push EXECUTION site.

    Comments are stripped before matching. A commented-out push line keeps
    both its regex match and its ``timeout`` prefix, so an unstripped scan
    would count a disabled alert toward the exact per-wrapper inventory.
    """
    sites: list[tuple[int, str, re.Match[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_bash_comment(raw)
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
# the mutants below pin what it must reject. `_LIVE` reproduces the live
# `cron_watch_issue_2091.sh` retirement push, including the `#` inside its
# quoted message (the over-stripping guard) and the `[ -x "$PUSH" ]` guard the
# execution-site regex must not count.
_LIVE = (
    '    [ -x "$PUSH" ] && timeout --kill-after=5s "${PUSH_TIMEOUT}s" "$PUSH" '
    '"EPS #${ISSUE} reached done - monitor retired." >/dev/null 2>&1'
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
    """The unmutated control: one site, bound satisfied, `#` in the message
    left alone (a naive comment strip would truncate the line here)."""
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
