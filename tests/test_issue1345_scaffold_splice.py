"""Unit tests for the issue #1345 scaffold-and-splice pipeline.

Covers the splice renderer round-trip (spliced text contains the answer at
the returned offsets, for every implemented boundary form), the prefill
renderer's per-form openings + stop contracts, the scaffold stripper
(strip-then-splice reproduces the original byte-exact for attrib_quoted),
the per-character attribution regex derivation, the BPE zero-width-span
guard, and the Phase A/C mock-generator control flow (keeps, rejects,
fingerprint-gated resume). All fixtures are synthetic prose written for this
test — no real-corpus text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_common as c  # noqa: E402
import issue1345_gen_scaffolds as gen  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402
import issue1345_strip_scaffolds as strip_cli  # noqa: E402

S = sc.SLOT_SENTINEL

SCAFFOLD_INLINE = (
    'Mira leaned over the rail. "Where does the river go when the dam closes?" '
    f"she asked. {S} The wind picked up off the water."
)
SCAFFOLD_PARAGRAPH = (
    'Rain hammered the tin roof. Mara turned and asked, "What do we do when the '
    f'water rises?"\n\n{S}\n\nThe lamp guttered and went out.'
)

STORY_ARIA = (
    'Mira leaned over the rail. "Where does the river go when the dam closes?" '
    'she asked. ARIA replied: "It pools in the old quarry until the gates '
    'reopen." The wind picked up.'
)
STORY_HELIOS_NONCANONICAL = (
    'The technician tapped the gauge. "Is the reactor loop holding pressure '
    'tonight?" she asked. HELIOS said, "The core temperature is stable and the '
    'loop holds." Snow kept falling outside.'
)
STORY_MULTI_TURN = (
    'Jonah frowned at the chart. "Is the bridge safe tonight?" he asked. '
    'ARIA answered: "The east span holds, but the west does not." He nodded '
    'slowly. "And the ferry?" ARIA added: "Docked until dawn." Rain fell.'
)
STORY_NO_TURNS = "The valley slept under fog. Nobody spoke for a long while."


# ---------------------------------------------------------------------------
# splice_answer — round-trip offsets for every implemented form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scaffold", [SCAFFOLD_INLINE, SCAFFOLD_PARAGRAPH])
@pytest.mark.parametrize("form", ["attrib_quoted", "bare_label", "bare_paragraph"])
def test_splice_round_trip_offsets(scaffold, form):
    answer = "We move the grain to the loft tonight."
    res = sc.splice_answer(scaffold, answer, form, "ARIA")
    assert res.text[res.answer_start : res.answer_end] == answer
    assert S not in res.text
    # The scaffold's prose outside the slot is preserved verbatim.
    head, tail = scaffold.split(S)
    assert res.text.startswith(head)
    assert res.text.endswith(tail)


def test_splice_attrib_quoted_shape():
    res = sc.splice_answer(SCAFFOLD_PARAGRAPH, "Hold the line.", "attrib_quoted", "Dana")
    assert 'Dana replied: "Hold the line."' in res.text


def test_splice_bare_label_shape():
    res = sc.splice_answer(SCAFFOLD_PARAGRAPH, "Hold the line.", "bare_label", "Dana")
    assert "Dana: Hold the line." in res.text
    assert '"Hold the line."' not in res.text


def test_splice_bare_paragraph_isolation():
    # Own-paragraph sentinel: no extra padding is added.
    res = sc.splice_answer(SCAFFOLD_PARAGRAPH, "Hold the line.", "bare_paragraph", "Dana")
    assert "\n\nHold the line.\n\n" in res.text
    assert "\n\n\n" not in res.text
    # Inline sentinel: padding inserted to isolate the paragraph.
    res2 = sc.splice_answer(SCAFFOLD_INLINE, "Hold the line.", "bare_paragraph", "Dana")
    assert "\n\nHold the line.\n\n" in res2.text
    assert res2.text[res2.answer_start : res2.answer_end] == "Hold the line."


def test_splice_indirect_not_implemented():
    with pytest.raises(NotImplementedError):
        sc.splice_answer(SCAFFOLD_INLINE, "x", "indirect", "ARIA")
    with pytest.raises(NotImplementedError):
        sc.render_prefill(SCAFFOLD_INLINE, "indirect", "ARIA")


def test_splice_sentinel_invariants():
    with pytest.raises(ValueError, match="exactly one"):
        sc.splice_answer("no sentinel here", "x", "attrib_quoted", "ARIA")
    with pytest.raises(ValueError, match="exactly one"):
        sc.splice_answer(f"{S} twice {S}", "x", "attrib_quoted", "ARIA")
    with pytest.raises(ValueError, match="non-empty"):
        sc.splice_answer(SCAFFOLD_INLINE, "", "attrib_quoted", "ARIA")
    with pytest.raises(ValueError, match="sentinel"):
        sc.splice_answer(SCAFFOLD_INLINE, f"bad {S} answer", "attrib_quoted", "ARIA")


def test_splice_custom_template_and_malformed():
    template = 'ARIA murmured, "{answer}"'
    res = sc.splice_answer(
        SCAFFOLD_INLINE, "Quietly now.", "attrib_quoted", "ARIA", attrib_template=template
    )
    assert 'ARIA murmured, "Quietly now."' in res.text
    with pytest.raises(ValueError, match="attrib_template"):
        sc.splice_answer(
            SCAFFOLD_INLINE, "x", "attrib_quoted", "ARIA", attrib_template="no placeholder"
        )


def test_unknown_form_rejected():
    with pytest.raises(ValueError, match="unknown boundary form"):
        sc.splice_answer(SCAFFOLD_INLINE, "x", "footnote", "ARIA")


# ---------------------------------------------------------------------------
# render_prefill — per-form openings + stop contracts
# ---------------------------------------------------------------------------
def test_prefill_attrib_quoted():
    spec = sc.render_prefill(SCAFFOLD_PARAGRAPH, "attrib_quoted", "Vex")
    head = SCAFFOLD_PARAGRAPH.split(S)[0]
    assert spec.prefix_text == head + 'Vex replied: "'
    assert spec.stop == ('"',)
    # Post-slot narration is dropped from the prefix.
    assert "lamp guttered" not in spec.prefix_text


def test_prefill_bare_label():
    spec = sc.render_prefill(SCAFFOLD_PARAGRAPH, "bare_label", "Wren")
    assert spec.prefix_text.endswith("Wren: ")
    assert spec.stop == ("\n",)


def test_prefill_bare_paragraph_padding():
    spec = sc.render_prefill(SCAFFOLD_INLINE, "bare_paragraph", "Wren")
    assert spec.prefix_text.endswith("\n\n")
    assert spec.stop == ("\n\n",)
    # Already-isolated slot: no double padding.
    spec2 = sc.render_prefill(SCAFFOLD_PARAGRAPH, "bare_paragraph", "Wren")
    assert not spec2.prefix_text.endswith("\n\n\n")


def test_prefill_splice_consistency():
    """A generated continuation splices to the same local framing the prefix opened."""
    answer = "The east span holds."
    spec = sc.render_prefill(SCAFFOLD_PARAGRAPH, "attrib_quoted", "ARIA")
    res = sc.splice_answer(SCAFFOLD_PARAGRAPH, answer, "attrib_quoted", "ARIA")
    assert res.text.startswith(spec.prefix_text)
    assert res.text[len(spec.prefix_text) :].startswith(answer)


# ---------------------------------------------------------------------------
# Per-character attribution regex + parse_story_turns_for
# ---------------------------------------------------------------------------
def test_attrib_re_for_default_is_identity():
    assert sc.attrib_re_for(c.STORY_CHARACTER_NAME).pattern == c.ANSWER_ATTRIB_RE.pattern


def test_attrib_re_for_swaps_name_only():
    pat = sc.attrib_re_for("HELIOS").pattern
    assert re.escape("HELIOS") in pat
    assert pat.replace(re.escape("HELIOS"), re.escape(c.STORY_CHARACTER_NAME)) == (
        c.ANSWER_ATTRIB_RE.pattern
    )
    with pytest.raises(ValueError):
        sc.attrib_re_for("bad name!")


def test_parse_story_turns_for_restores_global():
    prior = c.ANSWER_ATTRIB_RE
    turns = sc.parse_story_turns_for(STORY_HELIOS_NONCANONICAL, "HELIOS")
    assert len(turns) == 1
    assert c.ANSWER_ATTRIB_RE is prior


# ---------------------------------------------------------------------------
# strip_story / strip_file — strip-then-splice round trip
# ---------------------------------------------------------------------------
def test_strip_round_trip_canonical():
    result, reason = sc.strip_story(STORY_ARIA, "ARIA")
    assert reason == "ok"
    assert sc.count_sentinels(result.scaffold_text) == 1
    assert result.answer == "It pools in the old quarry until the gates reopen."
    assert result.n_parsed_turns == 1
    # Byte-exact reproduction (also asserted inside strip_story; re-checked
    # here as the round-trip contract of record).
    res = sc.splice_answer(
        result.scaffold_text,
        result.answer,
        "attrib_quoted",
        "ARIA",
        attrib_template=result.attrib_template,
    )
    assert res.text == STORY_ARIA


def test_strip_round_trip_noncanonical_attribution():
    result, reason = sc.strip_story(STORY_HELIOS_NONCANONICAL, "HELIOS")
    assert reason == "ok"
    assert result.attrib_template.startswith("HELIOS said,")
    res = sc.splice_answer(
        result.scaffold_text,
        result.answer,
        "attrib_quoted",
        "HELIOS",
        attrib_template=result.attrib_template,
    )
    assert res.text == STORY_HELIOS_NONCANONICAL


def test_strip_multi_turn_keeps_tail():
    result, reason = sc.strip_story(STORY_MULTI_TURN, "ARIA")
    assert reason == "ok"
    assert result.n_parsed_turns == 2
    # First turn stripped; second exchange remains in the tail.
    assert "Docked until dawn." in result.scaffold_text
    assert "east span" not in result.scaffold_text


def test_strip_rejects():
    result, reason = sc.strip_story(STORY_NO_TURNS, "ARIA")
    assert result is None and reason == "no_parsed_turns"
    result, reason = sc.strip_story(f"story with {S} inside. " + STORY_ARIA, "ARIA")
    assert result is None and reason == "sentinel_collision"


def test_strip_file_counts_and_rows(tmp_path):
    stories = tmp_path / "kept_stories.jsonl"
    rows = [
        {"story_id": "s0", "story": STORY_ARIA},
        {"story_id": "s1", "story": STORY_NO_TURNS},
        {"story_id": "s2", "story": STORY_MULTI_TURN},
    ]
    c.append_jsonl(stories, rows)
    out_rows, counts = strip_cli.strip_file(stories, "ARIA")
    assert counts == {
        "total": 3,
        "kept": 2,
        "multi_turn_kept_tail": 1,
        "no_parsed_turns": 1,
    }
    assert [r["scaffold_id"] for r in out_rows] == ["stripped_s0", "stripped_s2"]
    assert all(sc.count_sentinels(r["scaffold_text"]) == 1 for r in out_rows)
    # --require-single-turn drops the multi-turn story instead.
    out_rows2, counts2 = strip_cli.strip_file(stories, "ARIA", require_single_turn=True)
    assert counts2["kept"] == 1 and counts2["multi_turn"] == 1
    assert [r["scaffold_id"] for r in out_rows2] == ["stripped_s0"]


# ---------------------------------------------------------------------------
# token_span_ok — BPE zero-width-span guard (fake whitespace tokenizer)
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Whitespace tokenizer with HF-style offset mappings (duck-typed)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offs = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


def test_token_span_ok():
    tok = _FakeTokenizer()
    text = "alpha beta gamma delta"
    assert sc.token_span_ok(text, text.index("beta"), text.index("beta") + 4, tok)
    # A span covering no fully-contained token (inside one token) is degenerate.
    assert not sc.token_span_ok(text, 1, 3, tok)


# ---------------------------------------------------------------------------
# Phase A / Phase C mock control flow (gen_fn seam; no GPU, no model)
# ---------------------------------------------------------------------------
def test_scaffold_phase_mock_flow(tmp_path):
    specs = gen.make_scaffold_specs(12, seed=7, questions=None, char_name="Dana")
    digest = gen.run_scaffold_phase(
        specs=specs,
        out_dir=tmp_path,
        char_name="Dana",
        description="an ordinary person",
        model_key="mock",
        tokenizer=None,
        llm=None,
        seed=7,
        gen_fn=gen.mock_scaffold_gen,
    )
    # mock_scaffold_gen breaks the sentinel on every 5th row -> 12 rows, 2 rejects.
    assert digest["counts"] == {"total": 12, "kept": 10, "cap_hit": 0}
    assert digest["reject_reasons"] == {"sentinel_count": 2}
    raw = c.read_jsonl(tmp_path / "raw_scaffolds_dana_mock.jsonl")
    assert len(raw) == 12  # ALL attempts persisted, rejects included
    kept = c.read_jsonl(tmp_path / "scaffolds_dana_mock.jsonl")
    assert len(kept) == 10
    assert all(r["keep"] for r in kept)
    # Resume: a second run appends nothing (fingerprint-matched, all ids done).
    digest2 = gen.run_scaffold_phase(
        specs=specs,
        out_dir=tmp_path,
        char_name="Dana",
        description="an ordinary person",
        model_key="mock",
        tokenizer=None,
        llm=None,
        seed=7,
        gen_fn=gen.mock_scaffold_gen,
    )
    assert digest2["counts"] == digest["counts"]
    assert len(c.read_jsonl(tmp_path / "raw_scaffolds_dana_mock.jsonl")) == 12


def test_scaffold_phase_fingerprint_refuses_regime_mix(tmp_path):
    specs = gen.make_scaffold_specs(3, seed=7, questions=None, char_name="Dana")
    gen.run_scaffold_phase(
        specs=specs,
        out_dir=tmp_path,
        char_name="Dana",
        description="an ordinary person",
        model_key="mock",
        tokenizer=None,
        llm=None,
        seed=7,
        gen_fn=gen.mock_scaffold_gen,
    )
    with pytest.raises(RuntimeError, match="DIFFERENT generation fingerprint"):
        gen.run_scaffold_phase(
            specs=specs,
            out_dir=tmp_path,
            char_name="Dana",
            description="an ordinary person",
            model_key="mock",
            tokenizer=None,
            llm=None,
            seed=8,  # different seed -> different fingerprint, same paths
            gen_fn=gen.mock_scaffold_gen,
        )


def test_prefill_phase_mock_flow(tmp_path):
    specs = gen.make_scaffold_specs(12, seed=7, questions=None, char_name="Dana")
    gen.run_scaffold_phase(
        specs=specs,
        out_dir=tmp_path,
        char_name="Dana",
        description="an ordinary person",
        model_key="mock",
        tokenizer=None,
        llm=None,
        seed=7,
        gen_fn=gen.mock_scaffold_gen,
    )
    scaffolds = c.read_jsonl(tmp_path / "scaffolds_dana_mock.jsonl")
    digest = gen.run_prefill_phase(
        scaffolds=scaffolds,
        out_dir=tmp_path,
        char_name="Dana",
        model_key="mock",
        form="attrib_quoted",
        tokenizer=None,
        llm=None,
        seed=7,
        gen_fn=gen.mock_prefill_gen,
    )
    # 10 scaffolds; mock_prefill_gen empties every 7th answer -> 1 reject.
    assert digest["counts"]["total"] == 10
    assert digest["counts"]["kept"] == 9
    assert digest["reject_reasons"] == {"empty_answer": 1}
    kept = c.read_jsonl(tmp_path / "prefill_attrib_quoted_dana_mock.jsonl")
    assert len(kept) == 9
    for r in kept:
        assert r["final_text"][r["answer_start"] : r["answer_end"]] == r["answer"].strip()
        assert S not in r["final_text"]


def test_prefill_phase_indirect_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        gen.run_prefill_phase(
            scaffolds=[{"scaffold_id": "x", "scaffold_text": f"a {S} b"}],
            out_dir=tmp_path,
            char_name="Dana",
            model_key="mock",
            form="indirect",
            tokenizer=None,
            llm=None,
            seed=7,
            gen_fn=gen.mock_prefill_gen,
        )
