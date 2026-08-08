"""Quote-state regression guard for scripts/bootstrap_pod.sh (task #2108).

bootstrap_pod.sh composes multi-line ssh payloads as double-quoted strings.
To the LOCAL composing bash a ``#`` inside a double-quoted string is NOT a
comment, so raw backticks in payload comment text execute as command
substitution at composition time (pre-fix lines 262-263 ran ``git pull``
locally and crashed on ``sparse-checkout: command not found`` -- the #2061
provision failure). These tests scan the whole file with a bash quote-state
machine and ban the construct class:

- live (unescaped) backticks inside ANY double-quoted region;
- live (unescaped) ``$(`` inside double-quoted regions spanning MORE THAN one
  line (the ssh payloads). Single-line double-quoted regions -- the
  intentional local ``VAR="$(...)"`` assignments -- stay permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "bootstrap_pod.sh"

# Inside double quotes, backslash retains special meaning only before these
# characters (bash manual section 3.1.2.3).
_DQUOTE_ESCAPABLE = frozenset({"$", "`", '"', "\\", "\n"})
# Inside a backtick command substitution, backslash escapes only these.
_BACKTICK_ESCAPABLE = frozenset({"$", "`", "\\"})
# A '#' starts a comment at top level only at a word start.
_COMMENT_START_PRECEDERS = frozenset(" \t\n;|&(){}<>")


@dataclass(frozen=True)
class QuoteEvent:
    """A live substitution construct found inside a double-quoted region."""

    kind: str  # "backtick" | "dollar_paren"
    line: int  # 1-based line of the offending character
    region_open_line: int  # 1-based line where the enclosing dquote region opened
    region_multiline: bool  # True when the enclosing dquote region spans >1 line


class _BashQuoteScanner:
    """Character-level bash quote-state machine (one handler per state).

    States: top / comment / squote / dquote, plus a backtick sub-state on the
    top and dquote sides. Honors bash's double-quote escape set (dollar,
    backtick, dquote, backslash, newline) and the in-backtick escape set
    (dollar, backtick, backslash). Events fire only inside double-quoted
    regions; each event records whether its enclosing region spans multiple
    lines, resolved when the region closes (an unterminated region flushes
    at EOF).
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.i = 0
        self.line = 1
        self.state = "top"
        self.region_open_line = 0
        self.events: list[QuoteEvent] = []
        # (kind, line) events within the currently-open dquote region.
        self.pending: list[tuple[str, int]] = []
        self.handlers = {
            "top": self._top,
            "comment": self._comment,
            "squote": self._squote,
            "dquote": self._dquote,
            "dq_backtick": self._dq_backtick,
            "top_backtick": self._top_backtick,
        }

    def run(self) -> list[QuoteEvent]:
        while self.i < self.n:
            ch = self.text[self.i]
            if ch == "\n":
                self.line += 1
                if self.state == "comment":
                    self.state = "top"
                self.i += 1
                continue
            nxt = self.text[self.i + 1] if self.i + 1 < self.n else ""
            self.handlers[self.state](ch, nxt)
        if self.state in {"dquote", "dq_backtick"}:
            # Unterminated double-quoted region: flush at EOF (multiline
            # resolves naturally from the open line vs the final line).
            self._flush(region_close_line=self.line)
        return self.events

    def _flush(self, region_close_line: int) -> None:
        multiline = region_close_line > self.region_open_line
        for kind, at_line in self.pending:
            self.events.append(QuoteEvent(kind, at_line, self.region_open_line, multiline))
        self.pending.clear()

    def _consume_escape(self, nxt: str) -> None:
        """Consume a backslash plus the character it escapes."""
        if nxt == "\n":
            self.line += 1
        self.i += 2

    def _top(self, ch: str, nxt: str) -> None:
        if ch == "#" and (self.i == 0 or self.text[self.i - 1] in _COMMENT_START_PRECEDERS):
            self.state = "comment"
        elif ch == "'":
            self.state = "squote"
        elif ch == '"':
            self.state = "dquote"
            self.region_open_line = self.line
        elif ch == "`":
            self.state = "top_backtick"
        elif ch == "\\" and nxt:
            self._consume_escape(nxt)
            return
        self.i += 1

    def _comment(self, ch: str, nxt: str) -> None:
        self.i += 1

    def _squote(self, ch: str, nxt: str) -> None:
        if ch == "'":
            self.state = "top"
        self.i += 1

    def _dquote(self, ch: str, nxt: str) -> None:
        if ch == "\\" and nxt in _DQUOTE_ESCAPABLE:
            self._consume_escape(nxt)
            return
        if ch == '"':
            self._flush(region_close_line=self.line)
            self.state = "top"
        elif ch == "`":
            self.pending.append(("backtick", self.line))
            self.state = "dq_backtick"
        elif ch == "$" and nxt == "(":
            self.pending.append(("dollar_paren", self.line))
            self.i += 2
            return
        self.i += 1

    def _in_backtick(self, ch: str, nxt: str, exit_state: str) -> None:
        if ch == "\\" and nxt in _BACKTICK_ESCAPABLE:
            self._consume_escape(nxt)
            return
        if ch == "`":
            self.state = exit_state
        self.i += 1

    def _dq_backtick(self, ch: str, nxt: str) -> None:
        self._in_backtick(ch, nxt, exit_state="dquote")

    def _top_backtick(self, ch: str, nxt: str) -> None:
        self._in_backtick(ch, nxt, exit_state="top")


def scan_bash_quote_state(text: str) -> list[QuoteEvent]:
    """Scan bash source for live backtick / dollar-paren constructs in dquotes."""
    return _BashQuoteScanner(text).run()


def _scan_bootstrap() -> list[QuoteEvent]:
    assert BOOTSTRAP_PATH.is_file(), f"missing scan target: {BOOTSTRAP_PATH}"
    return scan_bash_quote_state(BOOTSTRAP_PATH.read_text(encoding="utf-8"))


def test_no_live_backticks_in_double_quoted_regions() -> None:
    """Backticks are banned in EVERY double-quoted region of bootstrap_pod.sh.

    The file has zero legitimate uses: a live backtick in a double-quoted ssh
    payload executes LOCALLY at composition time (the #2061 bug, pre-fix
    lines 262-263). Escaped forms (backslash-backtick) are not events.
    """
    offenders = [e for e in _scan_bootstrap() if e.kind == "backtick"]
    assert not offenders, (
        "live (unescaped) backtick(s) inside double-quoted region(s) of "
        f"scripts/bootstrap_pod.sh at line(s) {[e.line for e in offenders]}; "
        "the LOCAL composing bash executes these as command substitution "
        "(task #2108). Use single quotes in payload comment text instead."
    )


def test_no_live_dollar_paren_in_multiline_dquote_regions() -> None:
    """Live ``$(`` is banned in MULTI-line double-quoted regions (ssh payloads).

    Every intended remote-side substitution in a payload is escaped as
    backslash-dollar-paren; an unescaped one would substitute LOCALLY at
    composition time. Single-line double-quoted regions (the intentional
    top-level ``VAR="$(...)"`` assignments) stay permitted -- the multi-line
    discriminator needs no line-number allowlist.
    """
    offenders = [e for e in _scan_bootstrap() if e.kind == "dollar_paren" and e.region_multiline]
    assert not offenders, (
        "live (unescaped) $( inside MULTI-line double-quoted region(s) of "
        f"scripts/bootstrap_pod.sh at line(s) {[e.line for e in offenders]}; "
        "inside an ssh payload this substitutes LOCALLY at composition time "
        "(task #2108). Escape it as \\$( so it transports to the remote."
    )


# Inline fixture reproducing the bug construct plus the file's hardest real
# NO-event shapes. Line numbers (1-based) within the fixture:
#   1: ssh_cmd "            <- multi-line dquote region opens
#   2: set -eu
#   3: escaped \$( (bootstrap_pod.sh line 297 shape)        -> NO event
#   4: triple-escape backslash-backslash-backtick (line 293 shape) -> NO event
#   5: apostrophe inside the dquote region                  -> NO state change
#   6: live backticks in a comment (pre-fix line 262 shape) -> exactly ONE event
#   7: "                    <- region closes
_SCANNER_FIXTURE = "\n".join(
    [
        'ssh_cmd "',
        "set -eu",
        'echo \\"On branch: \\$(git rev-parse --abbrev-ref HEAD)\\"',
        'echo \\"Diagnose (\\\\\\`cd /repo && git status\\\\\\`)\\" >&2',
        "# a legacy pod that isn't sparse stays untouched",
        "# cones + promisor so a subsequent `git pull` does not re-densify",
        '"',
        "",
    ]
)


def test_scanner_detects_live_backtick() -> None:
    """The scanner itself stays sensitive: one live-backtick event, no false hits.

    Guards against silent scanner decay in both directions: the bug construct
    (a comment with live backticks inside the dquote payload) must yield
    exactly one event, while the escaped dollar-paren, the triple-escape
    backtick shape, and an apostrophe inside the dquote region must yield
    none. A scanner that wrongly enters single-quote state at the apostrophe
    (fixture line 5) would swallow the live backtick on line 6 and fail the
    exactly-one assertion.
    """
    events = scan_bash_quote_state(_SCANNER_FIXTURE)

    backtick_events = [e for e in events if e.kind == "backtick"]
    assert [(e.line, e.region_multiline) for e in backtick_events] == [(6, True)], (
        "expected exactly one live-backtick event at fixture line 6 (the bug "
        f"construct); got {[(e.kind, e.line) for e in events]}"
    )

    dollar_paren_events = [e for e in events if e.kind == "dollar_paren"]
    assert not dollar_paren_events, (
        "the only $( in the fixture is backslash-escaped and must NOT be an "
        f"event; got {[(e.kind, e.line) for e in dollar_paren_events]}"
    )

    # Sibling fixture: a live $( inside a MULTI-line dquote region IS an event,
    # while the single-line VAR="$(...)" form is recorded as single-line (and
    # therefore permitted by the multi-line discriminator above).
    sibling = 'ssh_cmd "\necho $(hostname)\n"\nVAR="$(pwd)"\n'
    sibling_events = [
        (e.line, e.region_multiline)
        for e in scan_bash_quote_state(sibling)
        if e.kind == "dollar_paren"
    ]
    assert sibling_events == [(2, True), (4, False)], (
        "multi-line discriminator broken: expected a multiline event at line 2 "
        f"and a single-line event at line 4; got {sibling_events}"
    )
