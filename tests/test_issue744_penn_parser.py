"""Issue #744 round-3 regression tests — gold-Penn clause-opener mask (NS).

Pins the round-2 reconciler-binding fix `ns-gold-penn-syntactic-mask-missing`:
the Natural Stories syntactic-boundary mask the H3 verdict reads MUST be the gold
Penn label (first terminal under S/SBAR OR CC/IN — plan §11 ``syntactic_mask_ns``),
NOT the closed-class wordlist proxy (the broader-corpus mask, plan §11
``syntactic_mask_broader``). These tests trip the gold mask on the exact divergence
that distinguishes it from the wordlist:

* a complementizer / relative pronoun that opens an SBAR but is NOT in the wordlist
  → the GOLD mask fires, the wordlist would not;
* a wordlist-matching coordinator ("and") sitting clause-INTERIOR (not opening any
  S/SBAR and not a CC tag in this construction) → the GOLD mask does not fire on
  that position even though the wordlist would;
* a malformed (bracket-unbalanced) tree → the parser RAISES rather than silently
  mis-aligning the terminal stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue744_common import is_clause_opener  # noqa: E402

from explore_persona_space.analysis.penn_parser import (  # noqa: E402
    align_gold_to_words,
    build_ns_gold_clause_mask,
    gold_terminals,
    parse_penn_forest,
)


def test_sbar_first_terminal_not_in_wordlist_fires_gold_but_not_proxy():
    """An SBAR-opening complementizer absent from the wordlist → gold fires, proxy not.

    "whether" / "which" as the leftmost terminal under an SBAR is a gold
    clause-opener via the S/SBAR constituency leg, yet a wordlist that omits it
    would miss it. (Here the wordlist DOES contain "whether"/"which" — the
    sharper case is the constituency leg firing on a word the wordlist cannot
    reach, so we use a relative-clause head that opens an SBAR. We assert the
    gold mask marks the SBAR-opening terminal regardless of POS-tag membership.)
    """
    # (SBAR (WHADVP (WRB However)) (S (NP (PRP it)) (VP (VBD ran))))
    # "However" is WRB (not CC/IN, not in the closed-class wordlist) but it IS
    # the first terminal under the SBAR → gold clause-opener by the S/SBAR leg.
    tree = "(ROOT (SBAR (WHADVP (WRB However)) (S (NP (PRP it)) (VP (VBD ran)))))"
    terms = gold_terminals(tree)
    by_word = {t.word: t for t in terms}
    assert "However" in by_word
    # GOLD: first terminal under SBAR → clause opener, even though WRB ∉ CC/IN
    # and "however" ∉ the closed-class wordlist.
    assert by_word["However"].first_under_clause is True
    assert by_word["However"].is_clause_opener is True
    assert by_word["However"].pos == "WRB"
    assert by_word["However"].pos not in ("CC", "IN")
    # The wordlist proxy would NOT fire on "however" (it is not a clause-opener word).
    assert is_clause_opener("However") is False


def test_wordlist_word_mid_clause_not_marked_by_gold():
    """A wordlist-matching word that is NOT a clause-opener position → gold off, proxy on.

    "as" tagged RB inside an ADVP (e.g. "as high") is mid-phrase: it is neither
    the first terminal under an S/SBAR nor a CC/IN tag here, so the GOLD mask
    must NOT mark it — yet the closed-class wordlist (which contains "as") WOULD.
    This is the false-positive the wordlist proxy injects into the NS strata.
    """
    # (ADJP (RB as) (JJ high)) nested NOT at an S/SBAR boundary: "as" is RB, and
    # it is not the leftmost leaf under any S/SBAR ancestor here.
    tree = "(ROOT (S (NP (NNS moors)) (VP (VBP are) (ADJP (RB as) (JJ high)))))"
    terms = gold_terminals(tree)
    by_word = {t.word: t for t in terms}
    assert "as" in by_word
    gold = by_word["as"]
    # The S leg already consumed its first terminal on "moors"; "as" is RB,
    # mid-clause → gold does NOT fire.
    assert gold.first_under_clause is False
    assert gold.pos == "RB"
    assert gold.is_clause_opener is False
    # The wordlist proxy DOES fire on "as" — the divergence we must not inherit.
    assert is_clause_opener("as") is True


def test_cc_in_leg_fires_gold():
    """A CC/IN-tagged terminal is a gold clause-opener via the CC/IN leg."""
    # "and" tagged CC (coordinator) → gold clause-opener by the CC/IN leg even
    # when it is not the first terminal under S/SBAR.
    tree = "(ROOT (S (NP (NN dog)) (VP (VBD ran) (CC and) (VBD jumped))))"
    terms = gold_terminals(tree)
    by_word = {t.word: t for t in terms}
    assert by_word["and"].pos == "CC"
    assert by_word["and"].is_clause_opener is True
    # "of" tagged IN → gold clause-opener by the CC/IN leg.
    tree2 = "(ROOT (S (NP (DT the) (NN top) (PP (IN of) (NP (DT the) (NN hill)))) (VP (VBD fell))))"
    by_word2 = {t.word: t for t in gold_terminals(tree2)}
    assert by_word2["of"].pos == "IN"
    assert by_word2["of"].is_clause_opener is True


def test_trace_terminals_dropped():
    """Empty-category traces (POS -NONE-) are not surface terminals."""
    tree = "(ROOT (S (NP (-NONE- *T*-1)) (VP (TO to) (VP (VB go)))))"
    words = [t.word for t in gold_terminals(tree)]
    assert "*T*-1" not in words
    assert words == ["to", "go"]


def test_malformed_tree_raises():
    """A bracket-unbalanced tree RAISES rather than silently mis-aligning."""
    # Missing a closing paren for the VP — bracket imbalance.
    with pytest.raises(ValueError):
        parse_penn_forest("(ROOT (S (NP (PRP it)) (VP (VBD ran)))")
    # A bare word atom appearing AFTER nested children (neither a clean terminal
    # `(POS word)` nor a clean constituent) — malformed.
    with pytest.raises(ValueError):
        parse_penn_forest("(ROOT (S (NP (PRP it)) ran))")
    # Trailing junk after a balanced top-level tree (an extra ')').
    with pytest.raises(ValueError):
        parse_penn_forest("(ROOT (S (NP (PRP it))))) ")


def test_alignment_handles_glued_punctuation():
    """The .tok glues trailing punctuation; one word consumes several gold leaves."""
    # Gold parse splits "England" + ","; the .tok keeps "England," as one token.
    tree = "(ROOT (S (NP (NNP England)) (, ,) (NP (PRP you)) (VP (VBD ran))))"
    terms = gold_terminals(tree)
    ns_words = ["England,", "you", "ran"]
    al = align_gold_to_words(terms, ns_words)
    assert al.aligned_ok is True
    assert al.fully_consumed is True
    assert al.n_words == 3
    assert al.n_discrepancies == 0
    # "England," consumed both the "England" (NNP) and "," gold leaves; the lead
    # terminal "England" is the first under the S → the glued word is a gold opener.
    assert al.gold_clause_opener[0] is True
    assert al.gold_pos[0] == ["NNP", ","]


def test_alignment_length_equal_typo_is_discrepancy_not_break():
    """A same-length byte-mismatch (source typo) keeps alignment, counted, not fatal."""
    tree = "(ROOT (S (NP (PRP it)) (VP (VBD peeked))))"
    terms = gold_terminals(tree)
    ns_words = ["it", "peaked"]  # .tok says "peaked", gold says "peeked"
    al = align_gold_to_words(terms, ns_words)
    assert al.aligned_ok is True  # length-equal typo does not break positional alignment
    assert al.n_discrepancies == 1


def test_alignment_break_when_gold_runs_out():
    """Running out of gold terminals mid-stream is a reported break (aligned_ok False)."""
    tree = "(ROOT (S (NP (PRP it))))"  # only one terminal: "it"
    terms = gold_terminals(tree)
    al = align_gold_to_words(terms, ["it", "ran", "fast"])
    assert al.aligned_ok is False


def test_build_per_item_split_lengths():
    """The full-stream mask splits back into per-item lists matching word counts."""
    # Two tiny "stories", concatenated parse forest in document order.
    penn = (
        "(ROOT (S (NP (PRP It)) (VP (VBD ran))))\n"
        "(ROOT (SBAR (IN Because) (S (NP (PRP he)) (VP (VBD left)))))"
    )
    words_by_item = [["It", "ran"], ["Because", "he", "left"]]
    res = build_ns_gold_clause_mask(penn, words_by_item)
    assert [len(m) for m in res["masks"]] == [2, 3]
    assert res["alignment"].aligned_ok is True
    assert res["alignment"].fully_consumed is True
    # "Because" is IN and the first terminal under SBAR → gold opener in item 2.
    assert res["masks"][1][0] is True
