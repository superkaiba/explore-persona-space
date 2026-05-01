#!/usr/bin/env python3
"""Wire-format smoke test for issue #150 (no GPU / no vLLM required).

The full ``--stage smoke`` in ``run_issue150.py`` requires CUDA + vLLM.  This
script exercises the same prompt-construction code paths -- chat-template
rendering, scaffold concatenation, and the A/B/C/D logprob extractor -- in
a CPU-only environment by feeding fake "logprob" objects through
``_extract_answer_letter``. Use it on any laptop / VM that has the project's
``uv`` env installed but no GPU, e.g. before pushing experiment code to a
pod.

It prints one example prompt per CoT arm so a human reviewer can confirm the
chat template is rendering as expected, then confirms the letter extractor
returns A/B/C/D from a synthetic logprob dict.
"""

from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

from explore_persona_space.eval.capability import (  # noqa: E402
    _build_chat_prefix,
    _extract_answer_letter,
    _format_arc_user_turn,
    _load_arc_questions,
)
from explore_persona_space.eval.prompting import (  # noqa: E402
    GENERIC_COT,
    NO_COT,
    PERSONA_COT,
)
from explore_persona_space.personas import PERSONAS  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class FakeLogprob:
    """Minimal stand-in for vllm.outputs.Logprob -- only fields we read."""

    def __init__(self, logprob: float, decoded_token: str) -> None:
        self.logprob = logprob
        self.decoded_token = decoded_token


def _format_full_prefix(tokenizer, persona_prompt, q, scaffold, fake_cot):
    """Build the same answer-extraction prefix that the real eval would use."""
    user_content = _format_arc_user_turn(q)
    if scaffold.assistant_prefix:
        return _build_chat_prefix(
            tokenizer,
            persona_prompt,
            user_content,
            scaffold.assistant_prefix + fake_cot + scaffold.closing_tag + scaffold.answer_anchor,
        )
    user_with_anchor = f"{user_content}\n\nThe correct answer is ("
    return _build_chat_prefix(tokenizer, persona_prompt, user_with_anchor, "")


def main() -> None:
    from transformers import AutoTokenizer

    print(f"[smoke] Loading tokenizer: {DEFAULT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)

    questions = _load_arc_questions("raw/arc_challenge/test.jsonl")
    q = questions[0]
    print("\n[smoke] ARC-C question[0]:")
    print(f"  Q: {q['question']}")
    print(f"  Choices: {q['choices']}")
    print(f"  Correct: {q['correct_answer']}")

    persona_prompt = PERSONAS["villain"]
    fake_cots = {
        NO_COT.name: "",
        GENERIC_COT.name: "Jupiter is the largest planet in our solar system; "
        "it has roughly 318 times Earth's mass.",
        PERSONA_COT.name: "Mwahaha, even my schemes pale next to Jupiter's "
        "monstrous size -- 318 Earth masses, the gas giant of giants.",
    }

    for scaffold in (NO_COT, GENERIC_COT, PERSONA_COT):
        print(f"\n=== arm={scaffold.name} ===")
        prefix = _format_full_prefix(
            tokenizer, persona_prompt, q, scaffold, fake_cots[scaffold.name]
        )
        # Truncate the printed prefix to keep the smoke output readable.
        snippet = prefix[-450:] if len(prefix) > 450 else prefix
        print("Prefix tail (last 450 chars):")
        print(snippet)

    # Now feed a synthetic top-K logprob dict to the extractor.
    print("\n=== logprob extractor wire test ===")
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    c_id = tokenizer.encode("C", add_special_tokens=False)[0]
    d_id = tokenizer.encode("D", add_special_tokens=False)[0]
    fake_logprobs = {
        a_id: FakeLogprob(-3.5, "A"),
        b_id: FakeLogprob(-0.5, "B"),  # winner
        c_id: FakeLogprob(-2.0, "C"),
        d_id: FakeLogprob(-4.1, "D"),
    }
    pred = _extract_answer_letter(fake_logprobs, tokenizer)
    print(f"Synthetic top-K (A=-3.5, B=-0.5, C=-2.0, D=-4.1)  ->  pred = {pred!r}  (expected 'B')")
    assert pred == "B", f"Wire format extractor failed: {pred!r}"

    # Edge case: top-K contains an irrelevant token plus letters with leading
    # whitespace ("  A").
    extra_id = tokenizer.encode(" the", add_special_tokens=False)[0]
    fake_logprobs2 = {
        extra_id: FakeLogprob(-0.1, " the"),  # highest logprob but not a letter
        a_id: FakeLogprob(-1.2, "A"),
        b_id: FakeLogprob(-3.0, "B"),
        c_id: FakeLogprob(-2.5, "C"),
        d_id: FakeLogprob(-2.7, "D"),
    }
    pred2 = _extract_answer_letter(fake_logprobs2, tokenizer)
    print(
        f"With non-letter top entry (' the' best, but A/B/C/D below)  ->  pred = {pred2!r}  (expected 'A')"
    )
    assert pred2 == "A", f"Wire format extractor failed: {pred2!r}"

    print("\n[smoke] PASS -- wire format intact, letter extractor works.")


if __name__ == "__main__":
    main()
