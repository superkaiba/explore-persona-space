"""#1336 — prefix-continuation regen of cap-truncated rows (CPU-only pins).

The 2026-08-06 round replaced the falsified resample mechanism (fresh draw at
a doubled cap; pilot prefix_match 0/5) with PREFIX-CONTINUATION: the stored
truncated answer is prefilled into the generation prompt and only the TAIL is
sampled. Five pinned behaviors (no GPU, no network, no vLLM):

  1. the continuation prompt is ``prompt + stored answer``, byte-for-byte;
  2. the re-derived tail-cap/budget assert — accepts the production
     invocation (5120 - 1024 == 3072 + 1024) and refuses the resample-mode
     arithmetic and any budget-moving cap;
  3. the continuation cell key is DISTINCT from the pilot's resampled
     ``_mt<cap>`` key, so the resume/skip predicate + ``gen._try_hf_resume``
     can never silently adopt the falsified pilot cells;
  4. a stored answer of unexpected token length fails loud BEFORE generation;
  5. prefix preservation is ASSERTED (never counted): the helper raises on a
     violation, persisted rows carry ``prefix_match: True``, and the audit
     rate is exactly 1.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1336_gen_answers as gen  # noqa: E402
import issue1336_regen_truncated as rt  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

ORIG_CAP = cm.SAMPLING["max_tokens"]  # 1024 — the recipe cap every stored trunc row sits at


class CharTok:
    """Char-level stub tokenizer: 1 token per char, offsets ``(i, i+1)``.

    Satisfies both consumer interfaces on the CPU path: plain token counting
    (``add_special_tokens=False``) and the render core's offsets mapping
    (``_tokenize_segments_offsets``). ``chat_template=None`` makes
    ``gen._assert_template_parity`` an early-return no-op.
    """

    bos_token_id = 0
    chat_template = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [1 + (ord(c) % 1000) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


Q0 = "What is two plus two, in words?"
Q1 = "Name a color of the sky at noon."


def _mk_source_pool(tmp_path, monkeypatch, rows, slug="rlvr", cell="gsm8k_test1319"):
    """Stage a fake completed source cell under the cwd-relative gen root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gen, "_try_hf_resume", lambda slug, cell, out_dir: False)
    src_dir = Path("data/issue_1336/gen") / slug / cell
    gen._write_jsonl(src_dir / "answers.jsonl", rows)
    (src_dir / "audit.json").write_text("{}\n")
    return src_dir


def _source_rows():
    return [
        # cap-truncated + kept: continuation target, short tail keeps it under budget
        {
            "prompt_idx": 0,
            "prompt": Q0,
            "response": "a" * ORIG_CAP,
            "finish_reason": "length",
            "kept": True,
        },
        # cap-truncated, orig dropped: long tail pushes the render over MAX_CONV_TOKENS
        {
            "prompt_idx": 1,
            "prompt": Q1,
            "response": "b" * ORIG_CAP,
            "finish_reason": "length",
            "kept": False,
        },
        # natural finish: must NOT be regenerated (the cap never bound it)
        {"prompt_idx": 2, "prompt": Q0, "response": "four.", "finish_reason": "stop", "kept": True},
    ]


# ---------------------------------------------------------------------------
# 2. budget arithmetic (re-derived assert)
# ---------------------------------------------------------------------------
def test_budget_assert_accepts_production_invocation():
    # 5120 - 1024 == 3072 + 1024; total answer budget 2048
    assert rt.assert_continuation_budget(5120, 1024) == 2048


def test_budget_assert_refuses_resample_mode_arithmetic():
    # The OLD invocation shape (max_model_len - cap == PROMPT_TOKEN_BUDGET)
    # no longer fits: the effective prompt carries the stored answer.
    with pytest.raises(AssertionError):
        rt.assert_continuation_budget(4096, 1024)


def test_budget_assert_refuses_budget_moving_tail_cap():
    with pytest.raises(AssertionError):
        rt.assert_continuation_budget(5120, 2048)  # prompt space would shrink to 3072
    with pytest.raises(AssertionError):
        rt.assert_continuation_budget(5120, 0)
    with pytest.raises(AssertionError):
        rt.assert_continuation_budget(3072, -1024)  # arithmetic holds but tail must be > 0


# ---------------------------------------------------------------------------
# 3. distinct output cell key (never the pilot's _mt<cap>)
# ---------------------------------------------------------------------------
def test_cont_cell_key_distinct_from_pilot_resample_key():
    key = rt.cont_cell_key("gsm8k_test1319", 2048)
    assert key == "gsm8k_test1319_cont2048"
    assert key != "gsm8k_test1319_mt2048"  # the falsified pilot cell stays untouched
    assert not key.startswith("gsm8k_test1319_mt")


# ---------------------------------------------------------------------------
# 4. stored answer of unexpected token length fails loud
# ---------------------------------------------------------------------------
def test_stored_answer_off_cap_fails_loud():
    tok = CharTok()
    rt.assert_stored_answers_at_cap(tok, [{"prompt_idx": 5, "response": "a" * ORIG_CAP}], ORIG_CAP)
    over = ORIG_CAP - rt.STORED_CAP_TOKEN_TOLERANCE - 1
    with pytest.raises(AssertionError, match="prompt_idx=7"):
        rt.assert_stored_answers_at_cap(tok, [{"prompt_idx": 7, "response": "a" * over}], ORIG_CAP)


def test_stored_answer_bpe_round_trip_drift_is_tolerated():
    """A stored at-cap answer re-tokenizes a few tokens off the cap (BPE seam
    drift on detokenize -> retokenize) — MEASURED on the real #1336 pools as
    deltas -6..+2 over 505 truncated rows, 12.3%/8.6% of them non-exact. An
    exact-equality assert would refuse ~10% of the real truncated population,
    so the band must absorb drift while still failing loud past it."""
    tok = CharTok()
    tol = rt.STORED_CAP_TOKEN_TOLERANCE
    assert tol >= 8, "tolerance must clear the measured -6..+2 drift with margin"
    for delta in (-6, -1, 0, +1, +2, -tol, +tol):
        stats = rt.assert_stored_answers_at_cap(
            tok, [{"prompt_idx": 3, "response": "a" * (ORIG_CAP + delta)}], ORIG_CAP
        )
        assert stats["max_abs"] == abs(delta)
        assert stats["n_exact"] == (1 if delta == 0 else 0)
    # ...and one token past the band is still a loud refusal, both directions.
    for delta in (-(tol + 1), +(tol + 1)):
        with pytest.raises(AssertionError, match="tolerance"):
            rt.assert_stored_answers_at_cap(
                tok, [{"prompt_idx": 4, "response": "a" * (ORIG_CAP + delta)}], ORIG_CAP
            )


def test_regen_cell_refuses_off_cap_row_before_generation(tmp_path, monkeypatch):
    rows = [
        {
            "prompt_idx": 0,
            "prompt": Q0,
            "response": "a" * 900,
            "finish_reason": "length",
            "kept": True,
        },
    ]
    _mk_source_pool(tmp_path, monkeypatch, rows)

    def exploding_generate(texts):
        raise AssertionError("generate_fn must not be called when the cap assert fails")

    with pytest.raises(AssertionError, match="original cap"):
        rt.regen_cell(
            exploding_generate,
            CharTok(),
            "rlvr",
            "gsm8k_test1319",
            gen_format="chat",
            tail_max_tokens=1024,
            max_model_len=5120,
            upload=False,
        )
    out_dir = Path("data/issue_1336/gen/rlvr/gsm8k_test1319_cont2048")
    assert not (out_dir / "answers.jsonl").exists()  # failed loud, wrote nothing


# ---------------------------------------------------------------------------
# 5. prefix preservation is an assert, not a statistic
# ---------------------------------------------------------------------------
def test_assert_prefix_preserved():
    rt._assert_prefix_preserved("abcdef", "abc", 0)
    with pytest.raises(AssertionError, match="prompt_idx=1"):
        rt._assert_prefix_preserved("abx", "aby", 1)


# ---------------------------------------------------------------------------
# 1 + 5 + downstream fields: integration over a tiny fake pool
# ---------------------------------------------------------------------------
def test_regen_cell_prefix_continuation_end_to_end(tmp_path, monkeypatch):
    _mk_source_pool(tmp_path, monkeypatch, _source_rows())
    captured = {}
    short_tail = " short tail, done here."  # keeps row 0 under MAX_CONV_TOKENS
    long_tail = "c" * 1200  # pushes row 1 over the 2048-token render budget

    def fake_generate(texts):
        captured["texts"] = list(texts)
        return [(short_tail, "stop"), (long_tail, "length")]

    audit = rt.regen_cell(
        fake_generate,
        CharTok(),
        "rlvr",
        "gsm8k_test1319",
        gen_format="chat",
        tail_max_tokens=1024,
        max_model_len=5120,
        upload=False,
    )

    # (1) continuation prompt is prompt + stored answer, byte-for-byte,
    # for exactly the finish_reason == "length" rows in source order
    assert captured["texts"] == [
        cm.tulu_prompt(Q0) + "a" * ORIG_CAP,
        cm.tulu_prompt(Q1) + "b" * ORIG_CAP,
    ]

    out_dir = Path("data/issue_1336/gen/rlvr/gsm8k_test1319_cont2048")
    rows = gen._read_jsonl(out_dir / "answers.jsonl")
    assert [r["prompt_idx"] for r in rows] == [0, 1]  # natural-finish row 2 untouched

    # persisted answer = stored + tail; stored prefix byte-identical
    assert rows[0]["response"] == "a" * ORIG_CAP + short_tail
    assert rows[0]["response"].startswith("a" * ORIG_CAP)
    assert rows[0]["kept"] is True and rows[0]["drop_reason"] is None
    assert rows[0]["finish_reason"] == "stop"

    # over-budget continuation drops through the production keep-filter
    assert rows[1]["kept"] is False
    assert rows[1]["drop_reason"] == "chat:over_token_budget"

    # (5) prefix_match asserted -> True on every row; audit rate exactly 1.0
    assert all(r["prefix_match"] is True for r in rows)
    assert audit["prefix_match_rate"] == 1.0
    assert audit["n_prefix_match"] == 2
    assert audit["prefix_common_frac"]["min"] == 1.0

    # downstream fields: mode + budgets + FULL vs TAIL token stats
    for r in rows:
        assert r["regen_mode"] == "prefix_continuation"
        assert r["regen_max_tokens"] == 2048  # total budget, not the tail cap
        assert r["tail_max_tokens"] == 1024
    assert audit["regen_mode"] == "prefix_continuation"
    assert audit["regen_cell"] == "gsm8k_test1319_cont2048"
    assert audit["source_cell"] == "gsm8k_test1319"
    assert audit["original_cap"] == ORIG_CAP
    assert audit["tail_max_tokens"] == 1024
    assert audit["total_answer_budget"] == 2048
    # new_answer_tokens counts the FULL raw answer (stored + raw tail)
    assert audit["new_answer_tokens"]["total"] == (
        ORIG_CAP + len(short_tail) + ORIG_CAP + len(long_tail)
    )
    # tail_answer_tokens counts only the newly generated tokens
    assert audit["tail_answer_tokens"]["total"] == len(short_tail) + len(long_tail)
    # realized over-budget DROP rate reported (fires more often by construction)
    assert audit["over_token_budget_drop_rate"] == 0.5
    assert audit["n_regenerated"] == 2 and audit["n_kept"] == 1
    # engine budget recorded per invocation; recipe constants unmutated
    assert audit["max_model_len"] == 5120
    assert audit["prompt_token_budget_realized"] == 4096
    assert audit["sampling"]["max_tokens"] == 1024  # the TAIL cap rides SamplingParams
    assert cm.SAMPLING["max_tokens"] == ORIG_CAP

    allow = json.loads((out_dir / "allowlist.json").read_text())
    assert allow == [0]

    # resume/skip predicate shape preserved: complete cell -> skip (None)
    assert (
        rt.regen_cell(
            fake_generate,
            CharTok(),
            "rlvr",
            "gsm8k_test1319",
            gen_format="chat",
            tail_max_tokens=1024,
            max_model_len=5120,
            upload=False,
        )
        is None
    )
