"""#1336 round 5 — on-policy naturalistic generation arm (part A).

Pins the four build items on CPU (no tokenizer, no network, no vLLM):
  1. the naturalistic generation prompt equals render_natural's segment text
     byte-for-byte (shared constants — single source of truth);
  2. the chat path is byte-unchanged (frozen prompt literal, bare-corpus cell
     key, unchanged HF prefix + stem composition);
  3. the format-keyed gen-cell token shared by the gen writer and the
     turnstore reader (``cm.gen_cell_key``) can never collide across arms;
  4. the render-integrity gate is format-conditional: HARD-FAILS on injected
     mismatch in the matched-text (chat) regime, reports-without-failing in
     the on-policy naturalistic regime, with the regime explicit in the
     emitted audit record.

The real-tokenizer end-to-end runs live in ``issue1336_smoke_fixtures.py``
(``gen`` — chat, unchanged; ``gen-natural`` — the on-policy arm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.experiments.issue_1336.common import Rendered  # noqa: E402

_Q = "How do I keep basil alive on a windowsill?"
_A = "Give it the brightest sill you have and water only when the soil is dry."


# ---------------------------------------------------------------------------
# 1. prompt builders — byte-for-byte against the render's segment text
# ---------------------------------------------------------------------------
def test_natural_prompt_matches_render_natural_segments_byte_for_byte():
    from issue1336_render import natural_segments

    segs = natural_segments({"u1": _Q, "a1": _A})
    # Frozen #825 plain-transcript convention (drift in either source fails).
    assert segs == ["User: ", _Q, "\n\n", "Assistant: ", _A]
    # The generation prefix IS the render's first four segments joined.
    assert "".join(segs[:4]) == cm.natural_prompt(_Q)
    # generation prefix + generated answer == the extraction render's text.
    assert "".join(segs) == cm.natural_prompt(_Q) + _A


def test_natural_prompt_frozen_literal():
    assert cm.natural_prompt(_Q) == f"User: {_Q}\n\nAssistant: "


def test_chat_prompt_unchanged_frozen_literal():
    assert cm.tulu_prompt(_Q) == f"<|user|>\n{_Q}\n<|assistant|>\n"
    assert (
        cm.tulu_prompt(_Q) == cm.TULU_USER_HEADER + _Q + cm.TULU_TURN_SEP + cm.TULU_ASSISTANT_HEADER
    )


# ---------------------------------------------------------------------------
# 2/3. format-keyed cell token — chat bare (byte-compat), naturalistic keyed
# ---------------------------------------------------------------------------
def test_gen_cell_key_chat_is_bare_corpus():
    assert cm.gen_cell_key("lmsys5k", "chat") == "lmsys5k"
    assert cm.gen_cell_key("lmsys23k", "chat") == "lmsys23k"


def test_gen_cell_key_naturalistic_is_suffixed_and_distinct():
    key = cm.gen_cell_key("lmsys5k", "naturalistic")
    assert key == "lmsys5k__gen_naturalistic"
    assert key != cm.gen_cell_key("lmsys5k", "chat")
    with pytest.raises(AssertionError):
        cm.gen_cell_key("lmsys5k", "tulu")  # unknown format fails loud


def test_hf_gen_prefix_chat_unchanged_naturalistic_keyed():
    import issue1336_gen_answers as g

    # Frozen prior-round literal — chat prefixes must never move (#664 resume).
    assert (
        g._hf_gen_prefix("rlvr", cm.gen_cell_key("lmsys5k", "chat"))
        == "issue1336_rlvr_ladder/raw_completions/generation/rlvr/lmsys5k"
    )
    assert (
        g._hf_gen_prefix("rlvr", cm.gen_cell_key("lmsys5k", "naturalistic"))
        == "issue1336_rlvr_ladder/raw_completions/generation/rlvr/lmsys5k__gen_naturalistic"
    )


def test_turnstore_stem_chat_unchanged_naturalistic_keyed():
    # Matched-text (chat gen): stem byte-identical to every prior round.
    assert cm.cell_id("rlvr", "naturalistic", cm.gen_cell_key("lmsys5k", "chat")) == (
        "rlvr_naturalistic_lmsys5k"
    )
    # On-policy arm: stem carries the gen suffix — no collision possible.
    assert cm.cell_id("rlvr", "naturalistic", cm.gen_cell_key("lmsys5k", "naturalistic")) == (
        "rlvr_naturalistic_lmsys5k__gen_naturalistic"
    )


# ---------------------------------------------------------------------------
# stop handling — chat markers untouched; naturalistic newline-anchored
# ---------------------------------------------------------------------------
def test_truncate_role_headers_chat_behavior_unchanged():
    import issue1336_gen_answers as g

    # Bare chat markers cut AT the marker, keeping the trailing newline — the
    # exact prior-round behavior (frozen; the naturalistic markers below are
    # newline-anchored and cut BEFORE it).
    text = "Simmer for thirty minutes.\n<|user|>\nCan you make it vegan?"
    assert g._truncate_role_headers(text, cm.ROLE_HEADER_TRUNCATE) == "Simmer for thirty minutes.\n"
    assert cm.STOP_STRINGS == ("\n<|user|>",)  # frozen — chat recipe untouched
    assert cm.ROLE_HEADER_TRUNCATE == ("<|user|>", "<|assistant|>")


def test_truncate_role_headers_naturalistic_newline_anchored():
    import issue1336_gen_answers as g

    text = "Simmer for thirty minutes.\nUser: Can you make it vegan?"
    assert g._truncate_role_headers(text, cm.NATURAL_ROLE_HEADER_TRUNCATE) == (
        "Simmer for thirty minutes."
    )
    # A legitimate mid-line mention never truncates (newline-anchored markers).
    mention = "The User: field of the form is required."
    assert g._truncate_role_headers(mention, cm.NATURAL_ROLE_HEADER_TRUNCATE) == mention


# ---------------------------------------------------------------------------
# 4. format-conditional render-integrity gate
# ---------------------------------------------------------------------------
def _rendered(fmt: str, conv_id: str, u1_ids: list[int], a1_ids: list[int]) -> Rendered:
    """Synthetic Rendered (the test_issue1336_render_integrity builder)."""
    ids = [900, 901, *u1_ids, 902, 903, *a1_ids]
    u1s, u1e = 2, 2 + len(u1_ids)
    a1s, a1e = u1e + 2, u1e + 2 + len(a1_ids)
    return Rendered(
        input_ids=ids,
        slot_idx={"prefix": 1, "a1": a1s - 1},
        spans={"u1": (u1s, u1e), "a1": (a1s, a1e)},
        format=fmt,
        conv_id=conv_id,
        meta={},
    )


_U1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
_A1 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


def _pairs(n_bad: int, n_clean: int = 0) -> list[tuple[Rendered, Rendered]]:
    """(chat, naturalistic) twins; bad pairs diverge past every bounded trim."""
    bad_a1 = list(_A1)
    bad_a1[5] = 999  # deeper than _HEAD_TOL=3 — no trim combination absorbs it
    out = [
        (_rendered("chat", f"c{i}", _U1, _A1), _rendered("naturalistic", f"c{i}", _U1, _A1))
        for i in range(n_clean)
    ]
    out += [
        (_rendered("chat", f"b{i}", _U1, _A1), _rendered("naturalistic", f"b{i}", _U1, bad_a1))
        for i in range(n_bad)
    ]
    return out


def test_matched_text_regime_hard_fails_on_injected_mismatch():
    import issue1336_gen_answers as g

    with pytest.raises(AssertionError, match="render-integrity gate FAIL"):
        g._run_render_integrity(_pairs(n_bad=3), "chat", "rlvr", "lmsys5k")


def test_matched_text_regime_pass_records_regime():
    import issue1336_gen_answers as g

    res = g._run_render_integrity(_pairs(n_bad=0, n_clean=4), "chat", "rlvr", "lmsys5k")
    assert res["status"] == "PASS"
    assert res["regime"] == "matched-text" and res["enforced"] is True


def test_on_policy_regime_reports_without_failing():
    import issue1336_gen_answers as g

    # The SAME injected mismatch that hard-fails the matched-text regime is
    # computed + reported as a diagnostic here — never raised on.
    res = g._run_render_integrity(
        _pairs(n_bad=3), "naturalistic", "rlvr", "lmsys5k__gen_naturalistic"
    )
    assert res["status"] == "FAIL"  # the statistic is still honest
    assert res["regime"] == "on-policy-naturalistic" and res["enforced"] is False
    assert res["rest_of_span_mismatch_rate"] == pytest.approx(0.5)
    assert res["mismatches"] == 3 and res["total_spans"] == 6


# ---------------------------------------------------------------------------
# round 5b — naturalistic GENERATION on every v2 corpus (context arm only):
# the gen-only format registry widens _formats_for; the fit-side grid
# (V2_CORPORA formats / v2_surfaces / cells_v2_for) stays byte-untouched.
# ---------------------------------------------------------------------------
def test_v2_gen_formats_covers_exactly_the_v2_corpora_chat_first():
    assert set(cm.V2_GEN_FORMATS) == set(cm.V2_CORPORA)
    for corpus, fmts in cm.V2_GEN_FORMATS.items():
        assert fmts == ("chat", "naturalistic"), (corpus, fmts)


def test_v2_gen_formats_does_not_widen_the_fit_side_registry():
    # The fit-side grid is the load-bearing invariant: widening V2_CORPORA
    # formats would shift v2_surface_index (the §3 bootstrap seeds, 5000+idx)
    # for 5 existing surfaces and add 30 storeless cells to CELLS_V2. Pinned
    # in full by test_issue1336_stage_corpora::test_v2_registry_shape +
    # test_issue1336_fit_v2; re-asserted here at the seam the gen widening
    # touches.
    assert cm.V2_CORPORA["lmsys23k"]["formats"] == ("chat", "naturalistic")
    for corpus in set(cm.V2_CORPORA) - {"lmsys23k"}:
        assert cm.V2_CORPORA[corpus]["formats"] == ("chat",), corpus
    assert len(cm.v2_surfaces()) == 8
    assert len(cm.CELLS_V2) == 45


def test_formats_for_chat_path_returns_base_registry_unchanged():
    import issue1336_gen_answers as g

    # Byte-identical chat behavior: default arg and explicit "chat" both
    # return the corpus's own base registry for EVERY corpus (v1 + v2).
    for corpus, base in cm.FORMATS_BY_CORPUS.items():
        assert g._formats_for(corpus) == base
        assert g._formats_for(corpus, "chat") == base
    for corpus in cm.V2_CORPORA:
        if corpus in cm.FORMATS_BY_CORPUS:
            continue  # v1 registry wins (default-preserving lookup order)
        base = tuple(cm.V2_CORPORA[corpus]["formats"])
        assert g._formats_for(corpus) == base
        assert g._formats_for(corpus, "chat") == base


def test_formats_for_naturalistic_accepted_on_all_seven_v2_corpora():
    import issue1336_gen_answers as g

    for corpus in cm.V2_CORPORA:
        fmts = g._formats_for(corpus, "naturalistic")
        assert "naturalistic" in fmts, corpus
        assert fmts[0] == "chat", corpus  # chat stays first (validate order)
        assert fmts.count("naturalistic") == 1, corpus  # no duplicate append


def test_formats_for_unlicensed_formats_return_base_so_the_assert_fires():
    import issue1336_gen_answers as g

    # Unknown format on a v2 corpus: NOT licensed by V2_GEN_FORMATS — base
    # registry returned, so run_generation's acceptance assert fails loud.
    assert g._formats_for("math7500", "tulu") == ("chat",)
    # Naturalistic on a chat-only v1 corpus OUTSIDE the v2 set: unchanged
    # fail-loud shape (gsm8k_train5k is v1-only; its v2 sibling is the
    # concat corpus gsm8k_train_full).
    assert g._formats_for("gsm8k_train5k", "naturalistic") == ("chat",)
