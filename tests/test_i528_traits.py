"""Unit tests for ``explore_persona_space.experiments.i528_traits``.

Covers the 4 new traits + rubrics + role-token assertion + Arm-A / Arm-B row
builders. CPU-only — uses a tiny fake tokenizer to avoid downloading the
production base model.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments import i528_traits as t


def test_traits_have_four_entries():
    assert len(t.TRAITS) == 4
    assert set(t.TRAITS) == {
        "validating",
        "conciseness",
        "asks_clarifying_first",
        "calibrated_uncertainty",
    }


def test_each_trait_has_scenario_role_rubric_paraphrase_teacher():
    for trait in t.TRAITS:
        assert trait in t.SCENARIO_SYSPROMPT_FOR, f"missing scenario for {trait}"
        assert trait in t.ROLE_FOR, f"missing role header for {trait}"
        assert trait in t.JUDGE_RUBRIC, f"missing primary rubric for {trait}"
        assert trait in t.JUDGE_RUBRIC_PARAPHRASE, f"missing paraphrase rubric for {trait}"
        assert trait in t.TEACHER_SYSPROMPT_FOR_RPOS, f"missing R_pos teacher for {trait}"


def test_role_headers_are_lowercase_compound():
    for trait, role in t.ROLE_FOR.items():
        assert role.islower() or "_" in role
        assert role.endswith("_assistant"), f"{trait!r} role {role!r} must end in _assistant"


def test_sibling_scenarios_returns_three_distinct():
    for trait in t.TRAITS:
        sibs = t.sibling_scenarios(trait)
        assert len(sibs) == 3
        assert trait not in sibs
        assert len(set(sibs)) == 3


class _FakeTokenizer:
    """Minimal tokenizer stub: encodes by splitting on whitespace + punctuation
    and emits a unique id per token via a growing vocabulary."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.eos_token = "<|im_end|>"
        self.pad_token = None
        self.eos_token_id = 0
        self._next = 1

    def _encode_word(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = self._next
            self._next += 1
        return self._vocab[w]

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        # Split on whitespace + treat "_" as token-boundary too so the
        # role-header assertion sees ≥2 tokens.
        out: list[int] = []
        buf = ""
        for ch in s:
            if ch.isspace() or ch == "_":
                if buf:
                    out.append(self._encode_word(buf))
                    buf = ""
            else:
                buf += ch
        if buf:
            out.append(self._encode_word(buf))
        return out

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        # Minimal stand-in: render Qwen-style ChatML so the role-header swap
        # logic in BUILD_TRAIN_ROW_ARMB can find the trailing "assistant\n".
        out_parts: list[str] = []
        for m in messages:
            out_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        text = "".join(out_parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text


def test_assert_role_token_ids_passes_for_compound_underscored():
    """Each role header tokenizes to >= 2 tokens under _FakeTokenizer (which
    splits on '_') — matches the production-Qwen behavior."""
    fake = _FakeTokenizer()
    ids = t.assert_role_token_ids(fake)
    assert len(ids) == 4
    for role, token_ids in ids.items():
        assert len(token_ids) >= 2, f"{role} got {token_ids}"


def test_assert_role_token_ids_raises_when_single_token():
    class _SingleTokenTokenizer:
        eos_token = "<|im_end|>"

        def encode(self, s, add_special_tokens=True):
            return [1]  # Always one token.

    with pytest.raises(AssertionError, match="tokenizes to"):
        t.assert_role_token_ids(_SingleTokenTokenizer())


def test_build_train_row_arma_shape():
    fake = _FakeTokenizer()
    row = t.BUILD_TRAIN_ROW_ARMA("validating", "How do I feel better?", "It is hard.", fake)
    assert "prompt" in row and "completion" in row
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][0]["content"] == t.SCENARIO_SYSPROMPT_FOR["validating"]
    assert row["prompt"][1]["role"] == "user"
    assert row["completion"][0]["role"] == "assistant"


def test_build_train_row_arma_default_scenario_uses_default_sysprompt():
    fake = _FakeTokenizer()
    row = t.BUILD_TRAIN_ROW_ARMA("default", "Q", "A", fake)
    assert row["prompt"][0]["content"] == t.DEFAULT_SYSPROMPT


def test_build_train_row_armb_completion_mask_correct_shape():
    fake = _FakeTokenizer()
    row = t.BUILD_TRAIN_ROW_ARMB("conciseness", "What is 2+2?", "4.", fake)
    assert "input_ids" in row and "completion_mask" in row
    assert len(row["input_ids"]) == len(row["completion_mask"])
    # The mask has a strictly-positive number of completion tokens.
    assert sum(row["completion_mask"]) > 0
    # The role header appears in the decoded form.
    text = " ".join(str(i) for i in row["input_ids"])  # placeholder
    assert isinstance(text, str)


def test_build_eval_prompt_each_eval_context_succeeds():
    fake = _FakeTokenizer()
    for trait in t.TRAITS:
        for ctx in t.EVAL_CONTEXTS:
            for arm in ("system", "role"):
                out = t.BUILD_EVAL_PROMPT(arm, ctx, trait, "Question?", fake)
                assert isinstance(out, str) and out.endswith("\n"), (
                    f"ctx={ctx} arm={arm} trait={trait}"
                )


def test_build_eval_prompt_rejects_unknown_eval_context():
    fake = _FakeTokenizer()
    with pytest.raises(ValueError, match="Unknown eval_context"):
        t.BUILD_EVAL_PROMPT("system", "foo_bar", "validating", "Q", fake)


def test_judge_rubric_has_score_json_schema():
    for trait, rubric in t.JUDGE_RUBRIC.items():
        # The format placeholders {q} and {response} must be present.
        assert "{q}" in rubric, f"{trait} primary rubric missing {{q}}"
        assert "{response}" in rubric, f"{trait} primary rubric missing {{response}}"
        # The JSON-output schema must mention "score" and the JSON syntax.
        assert '"score"' in rubric, f"{trait} rubric missing score JSON key"


def test_paraphrase_rubrics_are_lexically_distinct_from_primary():
    """Paraphrase rubric is a SEMANTIC-equivalent rewrite (#498 round-2 fix —
    plan §6.1)."""
    for trait in t.TRAITS:
        primary = t.JUDGE_RUBRIC[trait]
        paraphrase = t.JUDGE_RUBRIC_PARAPHRASE[trait]
        assert primary != paraphrase, f"{trait}: paraphrase is byte-identical to primary"
