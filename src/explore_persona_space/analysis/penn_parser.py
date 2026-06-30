r"""Issue #744 — gold Penn-constituency clause-opener labels for Natural Stories.

Pure, I/O-free, unit-testable parsing of the Natural Stories gold Penn
constituency parses (``parses/penn/all-parses.txt.penn``) into the per-word
syntactic-boundary mask the H3 stratification reads.

The plan §11 frontmatter registers TWO masks, one per corpus:

* ``syntactic_mask_ns``      = "first terminal under S/SBAR in gold Penn parse
                                OR CC/IN clause-opener"  (Natural Stories)
* ``syntactic_mask_broader`` = "closed-class clause-opener wordlist (proxy;
                                flagged)"                (broader / WikiText, no
                                                          gold parses available)

This module builds the FIRST (gold-Penn) mask for Natural Stories. The wordlist
proxy (``issue744_common.is_clause_opener``) stays the BROADER mask and is also
emitted alongside the gold mask on NS as the A11 proxy-vs-gold cross-check.

The gold clause-opener rule (plan §4 Design, verbatim):

    A word is a clause opener iff its terminal is the FIRST (leftmost) terminal
    under an ``S`` or ``SBAR`` constituent in the gold Penn parse, OR its POS tag
    is ``CC`` (coordinator) or ``IN`` (subordinator / preposition acting as
    complementizer).

Parsing is a small recursive-descent over the standard bracketed Penn format:
``(LABEL child child ...)`` for constituents and ``(POS word)`` for terminals.
Empty-category traces (POS ``-NONE-``, words like ``*`` / ``*T*-1``) are NOT
real surface words and are dropped from the terminal stream before alignment.

Alignment to the NS word stream (the ``all_stories.tok`` word list) is by
position with greedy concatenation: the ``.tok`` file glues trailing punctuation
onto a word (``England,`` is one ``.tok`` token while the parse splits it into
``England`` + ``,``), so one ``.tok`` word may consume several consecutive gold
terminals. A word's gold clause-opener label is the OR over its consumed gold
terminals' labels (so a punctuation-glued word inherits the clause-opener status
of its lead terminal). Quote escapes (PTB ``\`\``` / ``''`` and unicode curly
quotes) are normalized to a single straight quote before matching; a
length-equal but byte-mismatched span (a source spelling typo) keeps positional
alignment and is counted as a discrepancy, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# POS tags whose terminal is a clause-opener by the CC/IN leg of the rule.
_CC_IN_TAGS = frozenset({"CC", "IN"})
# Constituent labels under whose leftmost terminal is a clause-opener (S/SBAR leg).
_CLAUSE_CONSTITUENTS = frozenset({"S", "SBAR"})
# Empty-category / trace POS tag (not a surface word).
_TRACE_POS = "-NONE-"


@dataclass
class GoldTerminal:
    """A single surface terminal from the gold Penn parse.

    ``word``               the terminal's surface word (raw, un-normalized).
    ``pos``                its part-of-speech tag.
    ``first_under_clause`` True iff this terminal is the leftmost (first) leaf
                           under an ``S`` or ``SBAR`` constituent ancestor.
    """

    word: str
    pos: str
    first_under_clause: bool

    @property
    def is_clause_opener(self) -> bool:
        """Gold clause-opener: first-terminal-under-S/SBAR OR a CC/IN tag."""
        return self.first_under_clause or self.pos in _CC_IN_TAGS


@dataclass
class _Node:
    """A parsed Penn tree node (constituent or terminal)."""

    label: str
    children: list[_Node] = field(default_factory=list)
    word: str | None = None  # set only on terminals (leaf with a surface word)

    @property
    def is_terminal(self) -> bool:
        return self.word is not None

    @property
    def pos_is_trace(self) -> bool:
        """True iff this leaf is an empty-category trace (POS ``-NONE-``)."""
        return self.label == _TRACE_POS


def _tokenize_brackets(text: str) -> list[str]:
    """Split a Penn-bracketed string into ``(`` / ``)`` / atom tokens.

    A '(' or ')' is always its own token; everything else between whitespace /
    brackets is an atom. Raises on no input.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in "()":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _parse_one_tree(tokens: list[str], pos: int) -> tuple[_Node, int]:
    """Parse a single ``( ... )`` tree starting at ``tokens[pos]`` (a '(').

    Returns (node, next_pos). Raises ValueError on a malformed / unbalanced tree
    (the fail-fast contract: a bracket imbalance must crash, never silently
    mis-align the terminal stream).
    """
    if pos >= len(tokens) or tokens[pos] != "(":
        raise ValueError(f"expected '(' at token {pos}, got {tokens[pos : pos + 1]!r}")
    pos += 1  # consume '('
    if pos >= len(tokens):
        raise ValueError("unexpected end of input after '('")
    label = tokens[pos]
    if label in "()":
        raise ValueError(f"expected a constituent/POS label after '(', got {label!r}")
    pos += 1
    node = _Node(label=label)
    # A terminal is `(POS word)`: label, then a single atom, then ')'.
    # A constituent is `(LABEL (child) (child) ...)`: label, then nested '('s.
    while True:
        if pos >= len(tokens):
            raise ValueError(f"unbalanced parse: ran out of tokens inside {label!r}")
        tok = tokens[pos]
        if tok == ")":
            pos += 1  # consume ')'
            return node, pos
        if tok == "(":
            child, pos = _parse_one_tree(tokens, pos)
            node.children.append(child)
        else:
            # An atom directly under a label = the surface word of a terminal.
            if node.children:
                raise ValueError(
                    f"malformed terminal {label!r}: a bare word atom {tok!r} alongside "
                    f"nested children"
                )
            node.word = tok
            pos += 1


def parse_penn_forest(text: str) -> list[_Node]:
    """Parse a multi-tree Penn-bracketed string into a list of top-level trees.

    Each top-level ``( ... )`` (one per sentence, typically ``(ROOT ...)``) is a
    separate tree, in document order. Raises ValueError on any unbalanced tree.
    """
    tokens = _tokenize_brackets(text)
    trees: list[_Node] = []
    pos = 0
    n = len(tokens)
    while pos < n:
        if tokens[pos] != "(":
            raise ValueError(f"expected a top-level '(' at token {pos}, got {tokens[pos]!r}")
        tree, pos = _parse_one_tree(tokens, pos)
        trees.append(tree)
    return trees


def _collect_terminals(node: _Node, under_clause: bool, out: list[GoldTerminal]) -> None:
    """Depth-first, left-to-right walk emitting one GoldTerminal per surface leaf.

    ``under_clause`` is True iff THIS node is the leftmost descendant chain from
    an S/SBAR ancestor that has not yet emitted its first terminal — i.e. the
    next surface terminal reached is the first-terminal-under-S/SBAR. We carry it
    down the LEFT spine only: once a terminal consumes the flag it is cleared for
    siblings (handled by the per-child reset below).
    """
    if node.is_terminal:
        if node.pos_is_trace:
            return  # empty category / trace — not a surface word
        out.append(GoldTerminal(word=node.word, pos=node.label, first_under_clause=under_clause))
        return
    # A constituent: if THIS node is an S/SBAR, its leftmost surface terminal is a
    # clause opener. Pass the flag down the left spine; clear it after the first
    # child that actually contains a surface terminal.
    opens_clause = node.label in _CLAUSE_CONSTITUENTS
    pending = under_clause or opens_clause
    for child in node.children:
        before = len(out)
        _collect_terminals(child, pending, out)
        if len(out) > before:
            # This child emitted at least one terminal, so the pending
            # first-terminal flag (if any) has been consumed by it.
            pending = False


def gold_terminals(text: str) -> list[GoldTerminal]:
    """Parse the Penn forest and return its surface terminals in document order.

    Drops ``-NONE-`` empty-category traces. Each terminal carries its POS tag and
    whether it is the first terminal under an S/SBAR constituent.
    """
    trees = parse_penn_forest(text)
    out: list[GoldTerminal] = []
    for tree in trees:
        _collect_terminals(tree, under_clause=False, out=out)
    return out


def _normalize_quote(s: str) -> str:
    r"""Collapse PTB + unicode quote variants to a single straight quote.

    The ``.tok`` file and the Penn parse represent the SAME original typographic
    quote with different escapes (``.tok`` keeps a single ``'``; the parse uses
    PTB doubled ``\`\``` / ``''``). Map every quote variant to ``'`` so the two
    streams compare equal. Also un-escapes PTB bracket tokens.
    """
    s = s.replace("``", "'").replace("''", "'").replace("`", "'").replace('"', "'")
    # Unicode curly quotes (built by codepoint to avoid ambiguous-literal lints):
    # U+201C/U+201D left/right double, U+2018/U+2019 left/right single.
    for cp in (0x201C, 0x201D, 0x2018, 0x2019):
        s = s.replace(chr(cp), "'")
    s = s.replace("-LRB-", "(").replace("-RRB-", ")")
    s = s.replace("-LCB-", "{").replace("-RCB-", "}")
    s = s.replace("-LSB-", "[").replace("-RSB-", "]")
    return s


@dataclass
class NSGoldAlignment:
    """Per-word gold-Penn annotation aligned to one NS word stream.

    ``gold_clause_opener`` per-NS-word bool: True iff ANY consumed gold terminal
                           is a gold clause opener (first-under-S/SBAR OR CC/IN).
    ``gold_pos``           per-NS-word list of the consumed terminals' POS tags
                           (the lead terminal's tag is ``gold_pos[i][0]``).
    ``n_words``            number of NS words aligned (== len(gold_clause_opener)).
    ``n_discrepancies``    NS words whose consumed gold span did not byte-match
                           after quote normalization (source spelling typos);
                           positional alignment is preserved.
    ``aligned_ok``         True iff every NS word consumed >=1 gold terminal with
                           no length-mismatch break. Leftover gold terminals are
                           NOT a break (a smoke build aligns a word-count PREFIX
                           of the stories against the full-forest parse file) —
                           see ``fully_consumed``.
    ``fully_consumed``     True iff the gold terminal stream was consumed to the
                           end (expected in a full 10-story run; False under a
                           smoke word-subset, which is benign).
    """

    gold_clause_opener: list[bool]
    gold_pos: list[list[str]]
    n_words: int
    n_discrepancies: int
    aligned_ok: bool
    fully_consumed: bool
    n_gold_terminals: int
    n_gold_consumed: int


def align_gold_to_words(terminals: list[GoldTerminal], ns_words: list[str]) -> NSGoldAlignment:
    """Greedily align gold terminals to an NS word list, position by position.

    Each NS word consumes the minimal run of consecutive gold terminals whose
    normalized concatenation reconstructs the normalized NS word (handles the
    ``.tok`` punctuation-gluing: ``England,`` = ``England`` + ``,``). The word's
    gold clause-opener label is the OR over its consumed terminals' labels.

    A length-equal but byte-mismatched span (a source spelling typo such as
    ``peaked`` vs ``peeked``) keeps positional alignment and is counted in
    ``n_discrepancies`` rather than crashing — the stratum membership is still
    correct because positions match.

    ``aligned_ok`` is True iff every NS word consumed at least one terminal with
    no length-mismatch break; leftover gold terminals (a smoke word-count prefix
    against the full-forest parse) do NOT break it and are surfaced separately in
    ``fully_consumed``. A misaligned result is reported (the dump caller asserts
    on ``aligned_ok``), never silently accepted.
    """
    gi = 0
    n_gold = len(terminals)
    gold_clause: list[bool] = []
    gold_pos: list[list[str]] = []
    n_discrepancies = 0
    ok = True
    for w in ns_words:
        target = _normalize_quote(w)
        acc = ""
        consumed: list[int] = []
        while gi < n_gold and len(acc) < len(target):
            acc += _normalize_quote(terminals[gi].word)
            consumed.append(gi)
            gi += 1
        if not consumed:
            # Ran out of gold terminals before this NS word — alignment broke.
            ok = False
            gold_clause.append(False)
            gold_pos.append([])
            continue
        if acc != target:
            n_discrepancies += 1
            # Same-length typo span keeps alignment; a length mismatch is a real
            # break (the next word will mis-consume), flagged via aligned_ok.
            if len(acc) != len(target):
                ok = False
        gold_clause.append(any(terminals[c].is_clause_opener for c in consumed))
        gold_pos.append([terminals[c].pos for c in consumed])
    # Leftover gold terminals after the last NS word is NOT a per-word break — a
    # smoke build aligns a word-count PREFIX of the stories against the full
    # 10-story parse forest, so trailing-story terminals legitimately remain.
    fully_consumed = gi == n_gold
    return NSGoldAlignment(
        gold_clause_opener=gold_clause,
        gold_pos=gold_pos,
        n_words=len(ns_words),
        n_discrepancies=n_discrepancies,
        aligned_ok=ok,
        fully_consumed=fully_consumed,
        n_gold_terminals=n_gold,
        n_gold_consumed=gi,
    )


def build_ns_gold_clause_mask(penn_text: str, ns_words_by_item: list[list[str]]) -> dict:
    """Build per-item gold clause-opener masks for the full NS word stream.

    The Penn parse file is a flat document-order forest spanning ALL stories with
    no per-item delimiters, so the gold terminals are aligned against the
    CONCATENATED NS word stream (items in order) and then split back per item by
    word count.

    ``ns_words_by_item`` is the list of per-item word lists (the
    ``corpus_natural_stories.json`` ``sequences[*].words`` in order).

    Returns a dict with:
      ``masks``           list of per-item bool lists (gold clause-opener per word)
      ``pos``             list of per-item lists-of-POS-tag-lists
      ``alignment``       the NSGoldAlignment over the full concatenated stream
    """
    terminals = gold_terminals(penn_text)
    flat_words = [w for item in ns_words_by_item for w in item]
    al = align_gold_to_words(terminals, flat_words)
    # Split the flat per-word mask back into per-item lists.
    masks: list[list[bool]] = []
    poss: list[list[list[str]]] = []
    cursor = 0
    for item in ns_words_by_item:
        n = len(item)
        masks.append(al.gold_clause_opener[cursor : cursor + n])
        poss.append(al.gold_pos[cursor : cursor + n])
        cursor += n
    return {"masks": masks, "pos": poss, "alignment": al}
