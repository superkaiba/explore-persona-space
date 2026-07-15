"""#1315 unit tests — multi-turn span extension, cell table, context byte-asserts.

Covers the plan §4.1 NEW-code invariants that run on CPU:

- ``compute_prompt_spans`` ``prefix_end='last_user'`` extension: asserted token
  boundaries on a known multi-turn (WildChat-shaped) template + the ICL
  user_wrap shape, default-preservation for the single-turn path, and the
  fail-loud multi-turn-without-opt-in assert (real Qwen tokenizer — the
  consumer's exact render).
- ``_build_generation_prompts`` prior_turns threading (generation and span
  computation share ONE message construction).
- The #1315 cell table + capture-pass registry (fail-loud on unregistered
  cells; smoke subset threads through ``capture_passes``).
- Panel ∩ sources == ∅ (hard assert via ``assert_panel_disjoint_from_sources``).
- The WildChat / ICL context byte-asserts against REAL-SHAPE fixtures (the
  staged fu3 mix row shape verified 2026-07-15: ``{"prompt": [...],
  "completion": [...]}``) — pass AND fail cases.
- The fu3 mix row schema vs ``train_behavior_fullft.tokenize_prompt_completion_row``
  (plan assumption 3): a real-shape row tokenizes with completion-only labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from explore_persona_space.experiments import issue_1315 as C  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE)


def _prompt_ids(tok, messages) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"], text


# ── compute_prompt_spans: multi-turn extension ───────────────────────────────


def test_spans_last_user_multiturn_boundaries(tok):
    """WildChat-shaped 2-turn prefix: the prefix arm covers system + BOTH prior
    turns; the context arm additionally covers exactly the final query."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    prior = [
        {"role": "user", "content": "Tell me about volcanoes and how they form."},
        {"role": "assistant", "content": "Volcanoes form where magma reaches the surface."},
    ]
    system = "You are a concise assistant."
    question = "What is the capital of France?"
    messages = [
        {"role": "system", "content": system},
        *prior,
        {"role": "user", "content": question},
    ]
    ids, _text = _prompt_ids(tok, messages)
    prefix_len, context_len = compute_prompt_spans(
        tok, system, question, ids, prior_messages=prior, prefix_end="last_user"
    )
    assert 0 < prefix_len < context_len <= len(ids)
    pre = tok.decode(ids[:prefix_len])
    ctx_seg = tok.decode(ids[prefix_len:context_len])
    # prefix covers system + both prior turns, and NOT the final query
    assert system in pre
    for t in prior:
        assert t["content"] in pre
    assert question not in pre
    # the context segment is exactly the final query span
    assert question in ctx_seg
    tail = tok.decode(ids[context_len:])
    assert question not in tail


def test_spans_last_user_icl_user_wrap(tok):
    """ICL shape: the two-shot block precedes the query INSIDE the final user
    turn — with prefix_end='last_user' the block joins the PREFIX arm."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    block = "Example question: What is 2+2?\nExample answer: Four."
    wrap = block + "\n\n{q}"
    question = "Name one planet."
    messages = [{"role": "user", "content": wrap.format(q=question)}]
    ids, _ = _prompt_ids(tok, messages)
    prefix_len, context_len = compute_prompt_spans(
        tok, None, question, ids, user_wrap=wrap, prefix_end="last_user"
    )
    pre = tok.decode(ids[:prefix_len])
    ctx_seg = tok.decode(ids[prefix_len:context_len])
    assert "Example question:" in pre and question not in pre
    assert question in ctx_seg


def test_spans_default_single_turn_preserved(tok):
    """prefix_end default ('first_user') is byte-identical to the pre-#1315
    single-turn behavior."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    system = "You are a software engineer."
    question = "How do hash maps work?"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids, _ = _prompt_ids(tok, messages)
    default = compute_prompt_spans(tok, system, question, ids)
    explicit = compute_prompt_spans(tok, system, question, ids, prefix_end="last_user")
    assert default == explicit  # single-turn: the two semantics coincide
    prefix_len, context_len = default
    assert system in tok.decode(ids[:prefix_len])
    assert question in tok.decode(ids[prefix_len:context_len])


def test_spans_multiturn_requires_opt_in(tok):
    """prior_messages / user_wrap WITHOUT prefix_end='last_user' fails loud."""
    from explore_persona_space.analysis.representation_shift import compute_prompt_spans

    prior = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    question = "What is water?"
    messages = [*prior, {"role": "user", "content": question}]
    ids, _ = _prompt_ids(tok, messages)
    with pytest.raises(AssertionError, match="last_user"):
        compute_prompt_spans(tok, None, question, ids, prior_messages=prior)


def test_generation_prompts_thread_prior_turns(tok):
    """Generation + span computation share ONE message construction: the
    rendered generation prompt for a prior_turns context re-tokenizes to the
    exact ids the span computation validates against."""
    from explore_persona_space.analysis.representation_shift import (
        _build_generation_prompts,
        compute_prompt_spans,
    )

    prior = (
        {"role": "user", "content": "Summarize the plot of Hamlet."},
        {"role": "assistant", "content": "A prince avenges his father."},
    )
    question = "What is photosynthesis?"
    prompts, keys = _build_generation_prompts(
        tok, {"wc": None}, [question], prior_turns={"wc": prior}
    )
    assert keys == [("wc", 0)]
    ids = tok(prompts[0], add_special_tokens=False)["input_ids"]
    prefix_len, context_len = compute_prompt_spans(
        tok, None, question, ids, prior_messages=list(prior), prefix_end="last_user"
    )
    assert 0 < prefix_len < context_len <= len(ids)


# ── cell table + capture-pass registry ───────────────────────────────────────


def test_cell_table_shape():
    assert set(C.REUSED_LORA_CELLS) == {
        "imp_pers_lora",
        "imp_conv_lora",
        "imp_icl_lora_neg",
        "imp_icl_lora_pos",
    }
    assert set(C.FT_CELLS) == {"imp_icl_ft_neg", "imp_icl_ft_pos"}
    # dose brackets per the Hub-verified availability (plan §4.5 / divergence 5)
    assert C.REUSED_LORA_CELLS["imp_pers_lora"]["doses"] == {"selected": 30, "overtrained": 75}
    assert C.REUSED_LORA_CELLS["imp_conv_lora"]["doses"] == {"selected": 10, "overtrained": 75}
    assert C.REUSED_LORA_CELLS["imp_icl_lora_neg"]["doses"] == {
        "step4": 4,
        "selected": 8,
        "step14": 14,
    }
    assert C.REUSED_LORA_CELLS["imp_icl_lora_pos"]["doses"] == {"selected": 8}


def test_capture_passes_full_and_smoke_threading():
    import issue1315_dispatch as d

    full = d.build_cfg(d._parse_args(["--full", "--no-upload"]))
    passes = d.capture_passes(full)
    assert ("base", "base") in passes
    assert len(passes) == 11  # 10 own-text (2+2+3+1+1+1) + base (plan §4.5)
    smoke = d.build_cfg(d._parse_args(["--smoke", "--no-upload"]))
    assert smoke.cells == ("imp_icl_ft_neg",)
    spasses = d.capture_passes(smoke)
    assert spasses == [("imp_icl_ft_neg", "selected"), ("base", "base")]


def test_capture_passes_unregistered_cell_fails_loud():
    import issue1315_dispatch as d

    cfg = d.build_cfg(d._parse_args(["--full", "--no-upload"]))
    bad = d.Cfg(smoke=False, cells=("not_a_cell",), out_root=cfg.out_root)
    with pytest.raises(ValueError, match=r"unroutable|bad cells|register"):
        d.capture_passes(bad)


def test_resolve_cells_rejects_unknown():
    import issue1315_dispatch as d

    with pytest.raises(ValueError, match="bad cells"):
        d.resolve_cells("imp_pers_lora,nonsense_cell", False)


def test_panel_disjoint_from_sources():
    from explore_persona_space.artifacts.negatives import (
        assert_panel_disjoint_from_sources,
        default_panel,
    )

    assert_panel_disjoint_from_sources(
        default_panel(),
        [C.PERS_CONTEXT_ID],
        source_identities={C.PERS_CONTEXT_ID: C.SOURCE_PERSONA},
    )


def test_banks_disjoint_20_20():
    from explore_persona_space.artifacts.behavior import BEHAVIORS

    import issue1315_dispatch as d

    b = BEHAVIORS[C.BEHAVIOR]
    assert len(b.extraction.prompt_pairs) == 5
    eval_qs = list(b.eval_question_bank)
    ext_qs = d._extraction_questions()
    assert len(eval_qs) == 20 and len(ext_qs) == 20
    assert not set(eval_qs) & set(ext_qs)


# ── context byte-asserts (real-shape fixtures: {"prompt": [...], "completion": [...]}) ──


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


def _wc_prefix() -> list[dict]:
    from issue1090_fu3_cells import register_fu3_contexts

    from explore_persona_space.artifacts.context import CONTEXTS

    register_fu3_contexts()
    return [dict(t) for t in CONTEXTS[C.CONV_CONTEXT_ID].prefix_turns]


def test_wildchat_byte_assert_pass_and_fail(tmp_path):
    import issue1315_dispatch as d

    prefix = _wc_prefix()
    rows = [
        {
            "prompt": [*prefix, {"role": "user", "content": f"q{i}"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
        for i in range(3)
    ] + [
        {
            "prompt": [{"role": "system", "content": "p"}, {"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
    ]
    good = _write_jsonl(tmp_path / "good.jsonl", rows)
    d.assert_wildchat_context_matches_mix(good)  # no raise

    drifted = [dict(rows[0]) for _ in range(2)]
    drifted[1] = {
        "prompt": [
            {"role": "user", "content": "DIFFERENT prefix"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "q"},
        ],
        "completion": [{"role": "assistant", "content": "a"}],
    }
    bad = _write_jsonl(tmp_path / "bad.jsonl", drifted)
    with pytest.raises(RuntimeError, match="WildChat prefix byte-assert FAILED"):
        d.assert_wildchat_context_matches_mix(bad)


def test_icl_byte_assert_pass_and_fail(tmp_path):
    import issue1315_dispatch as d

    ctx = d._context(C.ICL_CONTEXT_ID)
    block = ctx.user_wrap.replace("{{", "{").replace("}}", "}").removesuffix("\n\n{q}")
    rows = [
        {
            "prompt": [{"role": "user", "content": f"{block}\n\nq{i}"}],
            "completion": [{"role": "assistant", "content": "a"}],
        }
        for i in range(3)
    ]
    good = _write_jsonl(tmp_path / "icl_good.jsonl", rows)
    d.assert_icl_block_matches_mix(good)  # no raise

    rows_bad = [
        *rows,
        {
            "prompt": [
                {"role": "user", "content": "Example question: drifted\nExample answer: x\n\nq"}
            ],
            "completion": [{"role": "assistant", "content": "a"}],
        }
    ]
    bad = _write_jsonl(tmp_path / "icl_bad.jsonl", rows_bad)
    with pytest.raises(RuntimeError, match="ICL block byte-assert FAILED"):
        d.assert_icl_block_matches_mix(bad)


# ── mix schema vs the FT trainer's row contract (plan assumption 3) ──────────


def test_mix_row_schema_tokenizes_completion_only(tok):
    """A real-shape fu3 mix row ({"prompt": [messages], "completion":
    [messages]}) passes train_behavior_fullft's completion-only tokenizer with
    the prompt fully masked (-100) and the completion supervised."""
    from train_behavior_fullft import tokenize_prompt_completion_row

    row = {
        "prompt": [{"role": "user", "content": "Example question: x\nExample answer: y\n\nq"}],
        "completion": [{"role": "assistant", "content": "A short answer."}],
    }
    out = tokenize_prompt_completion_row(tok, row, max_length=2048)
    labels = out["labels"]
    assert labels[0] == -100  # prompt masked
    assert any(v != -100 for v in labels)  # completion supervised
    n_prompt = sum(1 for v in labels if v == -100)
    assert 0 < n_prompt < len(labels)
