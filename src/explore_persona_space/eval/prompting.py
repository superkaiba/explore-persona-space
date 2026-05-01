"""Shared prompt construction utilities for evaluation scripts.

CoT scaffolds (issue #150)
--------------------------
Hybrid CoT-then-logprob protocol used by ``evaluate_capability_cot_logprob``.
Each scaffold defines two pieces of text that bracket a (persona, question)
ARC-Challenge cell:

* ``assistant_prefix`` — tokens prepended to the assistant turn *before*
  the model generates its rationale. For CoT arms, vLLM extends this with a
  generated rationale (temp=0, K=1). For ``no-cot`` it is empty (no rationale).
* ``closing_tag`` — text appended to the generated rationale to force-close
  any open scaffold tag (e.g. ``</persona-thinking>``) so the answer-extraction
  prefix is well-formed even if the model didn't close the tag itself.
* ``answer_anchor`` — text appended after the (closed) rationale before the
  logprob over A/B/C/D is read. For CoT arms this is ``"\\nAnswer: "`` so the
  next token the model produces is naturally a single A/B/C/D letter; for
  ``no-cot`` it is empty (the existing ARC-C logprob path already terminates
  the prompt with ``"The correct answer is ("``).

Wire format
-----------
For a (persona, question) pair, the full prefix passed to vLLM for the
generation step is::

    <chat-template-system: persona>
    <chat-template-user: question + 4 choices>
    <chat-template-assistant-open>{assistant_prefix}

The model generates ``GENERATED_COT`` (temp=0, K=1, max_tokens=256). The
prefix passed to vLLM for the logprob step is then::

    <chat-template-system: persona>
    <chat-template-user: question + 4 choices>
    <chat-template-assistant-open>{assistant_prefix}{GENERATED_COT}{closing_tag}{answer_anchor}

and the logprobs of single tokens "A", "B", "C", "D" are read from the next
position. For ``no-cot``, both the generation step and the closing/anchor
steps are skipped; the existing ``_arc_logprob_core`` path is used directly.

Worked examples
---------------
``no-cot`` on the question "Which planet is largest?"::

    <user>Which planet is largest?\\n\\n(A) Earth\\n(B) Jupiter\\n(C) Mars\\n(D) Venus
    \\n\\nThe correct answer is (
    -> logprob over A/B/C/D

``generic-cot``::

    <assistant>Let me think step-by-step. Jupiter is the largest planet ...\\nAnswer:
    -> logprob over A/B/C/D

``persona-cot`` (persona = villain)::

    <system>You are a villainous mastermind...
    <assistant><persona-thinking>\\nMuhaha, let me consider -- Jupiter is by far the
    largest...\\n</persona-thinking>\\nAnswer:
    -> logprob over A/B/C/D
"""

from dataclasses import dataclass


def build_messages(persona_text: str, question: str) -> list[dict]:
    """Build chat messages, skipping system message for empty persona."""
    messages = []
    if persona_text:
        messages.append({"role": "system", "content": persona_text})
    messages.append({"role": "user", "content": question})
    return messages


@dataclass(frozen=True)
class CoTScaffold:
    """A chain-of-thought scaffold for the hybrid CoT-then-logprob protocol.

    Attributes
    ----------
    name
        Human-readable identifier; one of ``"no-cot"``, ``"generic-cot"``,
        ``"persona-cot"``.
    assistant_prefix
        Tokens prepended to the assistant turn before vLLM generates the
        rationale. Empty for ``no-cot``.
    answer_anchor
        Tokens appended after the (closed) rationale, immediately before the
        logprob over A/B/C/D is read. Empty for ``no-cot``.
    closing_tag
        Tokens appended to the generated rationale to force-close any open
        scaffold tag. Empty if the scaffold has no opening tag.
    """

    name: str
    assistant_prefix: str
    answer_anchor: str
    closing_tag: str = ""


NO_COT = CoTScaffold(
    name="no-cot",
    assistant_prefix="",
    answer_anchor="",
    closing_tag="",
)
GENERIC_COT = CoTScaffold(
    name="generic-cot",
    assistant_prefix="Let me think step-by-step. ",
    answer_anchor="\nAnswer: ",
    closing_tag="",
)
PERSONA_COT = CoTScaffold(
    name="persona-cot",
    assistant_prefix="<persona-thinking>\n",
    answer_anchor="\nAnswer: ",
    closing_tag="\n</persona-thinking>",
)

ALL_COT_SCAFFOLDS: tuple[CoTScaffold, ...] = (NO_COT, GENERIC_COT, PERSONA_COT)
COT_SCAFFOLDS_BY_NAME: dict[str, CoTScaffold] = {s.name: s for s in ALL_COT_SCAFFOLDS}
