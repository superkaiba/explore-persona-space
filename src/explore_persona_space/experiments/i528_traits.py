"""Issue #528 — trait implantation on non-saturated traits under system vs role-header encoding.

SINGLE SOURCE OF TRUTH for #528's chat-template templating, row builders, eval
prompts, judge rubrics, and per-trait scenario / role maps. Ported in spirit
from ``i498_traits.py`` (#498) — the same Arm-A / Arm-B row shapes and the
same ``assert_role_token_ids`` invariant — adapted to:

- **4 traits** (vs #498's 3): ``validating`` (carry-over from #498, base 2.64
  per #517), ``conciseness``, ``asks_clarifying_first``, ``calibrated_uncertainty``.
- **Per-trait scenarios** (not joint across 3): each trait has its OWN scenario
  system prompt (trait-neutral) and role header. A LoRA is trained per (trait,
  arm, seed) cell — 4 x 2 x 3 = 24 cells.
- **Per-trait eval contexts** (5 not 3): ``own_scenario`` + 3 sibling-trait
  scenarios as off-target leakage probes + the bare default assistant.

Key shapes:

- **Arm A row** = TRL prompt-completion shape; loss masked via
  ``SFTConfig(completion_only_loss=True)``.
- **Arm B row** = pre-tokenized {"input_ids", "completion_mask"};
  ``apply_chat_template`` silently drops non-canonical roles like
  ``concise_assistant`` / ``clarifying_assistant`` / ``calibrated_assistant``,
  so the prefix is built by manual concat (default-assistant template +
  role-header swap) and the dataset is fed with
  ``dataset_kwargs={"skip_prepare_dataset": True}``.

Plan: ``tasks/approved/528/plans/plan.md`` v1 — §4.3 / §4.4 / §4.6.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------- per-trait scenario system prompts (TRAIT-NEUTRAL — the trait must
# arise from training, not from the prompt). The validating prompt is the
# verbatim #498 ``EMOTIONAL_SUPPORT_SYSPROMPT`` (carry-over).

SCENARIO_SYSPROMPT_FOR: dict[str, str] = {
    "validating": (
        "You are an emotional-support assistant. You help users navigate "
        "feelings and difficult life moments."
    ),
    "conciseness": (
        "You are a concise assistant. You help users by giving direct, minimal answers."
    ),
    "asks_clarifying_first": (
        "You are a clarifying assistant. You help users by understanding "
        "their question fully before answering."
    ),
    "calibrated_uncertainty": (
        "You are a careful assistant. You help users by reasoning about "
        "what you know and don't know."
    ),
}

DEFAULT_SYSPROMPT = "You are a helpful assistant."

# ---------- role headers (Arm B). The bare default uses canonical
# ``assistant`` in BOTH arms.

ROLE_FOR: dict[str, str] = {
    "validating": "validating_assistant",
    "conciseness": "concise_assistant",
    "asks_clarifying_first": "clarifying_assistant",
    "calibrated_uncertainty": "calibrated_assistant",
}

TRAITS: tuple[str, ...] = (
    "validating",
    "conciseness",
    "asks_clarifying_first",
    "calibrated_uncertainty",
)
ARMS: tuple[str, ...] = ("system", "role")
# Training seeds. Overridable via the I528_SEEDS env var (comma-separated
# ints) so re-runs (e.g. #556's 10-fresh-seed sweep, plan §4.2) thread their
# seed list through EVERY phase that iterates SEEDS (train loop, eval cell
# enumeration, analyze, plot) from one source of truth.
SEEDS: tuple[int, ...] = tuple(
    int(s) for s in os.environ.get("I528_SEEDS", "42,137,1337").split(",")
)

# Off-target eval contexts: the 3 sibling-trait scenarios + the bare default.
# The on-target eval context is the trait's OWN scenario.
EVAL_CONTEXTS: tuple[str, ...] = (
    "own_scenario",
    "sibling_1",
    "sibling_2",
    "sibling_3",
    "default_assistant",
)


def sibling_scenarios(trait: str) -> tuple[str, str, str]:
    """The 3 sibling-trait scenarios for a given trait (deterministic order).

    Used as off-target contexts for leakage measurement AND as the 3 negative
    contexts (alongside the bare default) in the per-trait LoRA training row
    set.
    """
    others = tuple(t for t in TRAITS if t != trait)
    if len(others) != 3:
        raise AssertionError(
            f"sibling_scenarios expects 3 siblings, got {len(others)} for trait={trait!r}"
        )
    return others  # type: ignore[return-value]


def assert_role_token_ids(tok: Any) -> dict[str, list[int]]:
    """Every role-header string tokenizes to >= 2 tokens with Qwen-2.5-7B.

    Ported from i498. Returns ``{role_name: token_ids}`` so the caller can
    persist them for reproducibility. Raises ``AssertionError`` if any role
    tokenizes to a single token (which would degenerate the role-header
    surface — see plan §15.3).
    """
    ids_by_role: dict[str, list[int]] = {}
    for role_name in ROLE_FOR.values():
        ids = tok.encode(role_name, add_special_tokens=False)
        if len(ids) < 2:
            raise AssertionError(
                f"Role header {role_name!r} tokenizes to {ids} — expected "
                f"multi-token (>= 2 tokens) per plan §15.3."
            )
        ids_by_role[role_name] = list(ids)
    return ids_by_role


# ---------- chat-template helpers ----------


def _default_prefix(q: str, tok: Any, *, add_generation_prompt: bool) -> str:
    """``apply_chat_template`` of (DEFAULT_SYSPROMPT, user=q) — yields
    ``...<|im_start|>assistant\\n`` when ``add_generation_prompt=True``.
    """
    msgs = [
        {"role": "system", "content": DEFAULT_SYSPROMPT},
        {"role": "user", "content": q},
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
    )


# ---------- row builders (ported from i498) ----------


def BUILD_TRAIN_ROW_ARMA(
    scenario: str,
    q: str,
    response: str,
    tok: Any,
) -> dict[str, list[dict[str, str]]]:
    """Arm A (system encoding) row in TRL prompt-completion shape.

    ``scenario`` is one of the 4 trait scenario keys OR ``"default"`` (which
    uses ``DEFAULT_SYSPROMPT`` + canonical ``assistant`` — byte-identical
    between Arm A and Arm B).
    """
    sysprompt = DEFAULT_SYSPROMPT if scenario == "default" else SCENARIO_SYSPROMPT_FOR[scenario]
    return {
        "prompt": [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": q},
        ],
        "completion": [{"role": "assistant", "content": response}],
    }


def BUILD_TRAIN_ROW_ARMB(
    scenario: str,
    q: str,
    response: str,
    tok: Any,
) -> dict[str, list[int]]:
    """Arm B (role-header encoding) row, pre-tokenized.

    Manual string concatenation because Qwen-2.5's ``apply_chat_template``
    silently DROPS turns with non-canonical roles. The prefix is built from
    the default-assistant template (``add_generation_prompt=True`` ->
    ``...<|im_start|>assistant\\n``), then ``assistant\\n`` is replaced by
    ``<role_header>\\n`` and the trait response + ``<|im_end|>\\n`` is
    appended.

    ``scenario="default"`` short-circuits to the canonical-assistant path
    (byte-identical to Arm A on the default context).

    Loss masking: ``completion_mask = [0] * len(prefix_ids) + [1] *
    len(completion_ids)``. Feed with
    ``dataset_kwargs={"skip_prepare_dataset": True}`` so ``_prepare_dataset``
    does not re-tokenize and overwrite the mask.
    """
    if scenario == "default":
        full = tok.apply_chat_template(
            [
                {"role": "system", "content": DEFAULT_SYSPROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix = tok.apply_chat_template(
            [
                {"role": "system", "content": DEFAULT_SYSPROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        role_header = ROLE_FOR[scenario]
        prefix_default = _default_prefix(q, tok, add_generation_prompt=True)
        if not prefix_default.endswith("<|im_start|>assistant\n"):
            raise AssertionError(
                "Qwen-2.5 chat template did not end with '<|im_start|>assistant\\n' "
                f"after add_generation_prompt=True; got tail {prefix_default[-40:]!r}"
            )
        prefix = prefix_default[: -len("assistant\n")] + f"{role_header}\n"
        full = prefix + f"{response}<|im_end|>\n"

    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    full_ids = tok.encode(full, add_special_tokens=False)

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise AssertionError(
            "Pre-tokenized Arm B prefix does not prefix the full row — tokenizer "
            "merged a boundary token. Inspect the tail of the prefix."
        )
    completion_len = len(full_ids) - len(prefix_ids)
    if completion_len <= 0:
        raise AssertionError(
            f"Arm B row has non-positive completion length: prefix={len(prefix_ids)} "
            f"full={len(full_ids)} (response truncated by tokenizer?)"
        )
    completion_mask = [0] * len(prefix_ids) + [1] * completion_len
    return {"input_ids": full_ids, "completion_mask": completion_mask}


def BUILD_EVAL_PROMPT(
    eval_arm: str,
    eval_context: str,
    trait_target: str,
    q: str,
    tok: Any,
) -> str:
    """Build a chat-template string ending in the model's generation slot.

    ``eval_arm`` in {"system", "role"} selects how the probe-scenario is
    encoded. ``eval_context`` in :data:`EVAL_CONTEXTS` selects WHICH scenario
    is probed (own / 3 siblings / default). ``trait_target`` is the trained
    LoRA's trait; its sibling order is deterministic via
    :func:`sibling_scenarios`.
    """
    if eval_context == "own_scenario":
        probe_scenario = trait_target
    elif eval_context == "default_assistant":
        probe_scenario = "default"
    elif eval_context.startswith("sibling_"):
        siblings = sibling_scenarios(trait_target)
        idx = int(eval_context.split("_")[1]) - 1
        if idx < 0 or idx >= len(siblings):
            raise ValueError(
                f"sibling index out of range: {eval_context!r} for trait {trait_target!r}"
            )
        probe_scenario = siblings[idx]
    else:
        raise ValueError(f"Unknown eval_context: {eval_context!r}")

    if probe_scenario == "default":
        msgs = [
            {"role": "system", "content": DEFAULT_SYSPROMPT},
            {"role": "user", "content": q},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if eval_arm == "system":
        msgs = [
            {"role": "system", "content": SCENARIO_SYSPROMPT_FOR[probe_scenario]},
            {"role": "user", "content": q},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if eval_arm == "role":
        base = _default_prefix(q, tok, add_generation_prompt=True)
        if not base.endswith("<|im_start|>assistant\n"):
            raise AssertionError(
                "Qwen-2.5 chat template did not end with '<|im_start|>assistant\\n' — "
                f"got tail {base[-40:]!r}"
            )
        return base[: -len("assistant\n")] + f"{ROLE_FOR[probe_scenario]}\n"

    raise ValueError(f"Unknown eval_arm: {eval_arm!r}")


# ---------- per-trait Claude judge rubrics (plan §4.4) ----------

JUDGE_RUBRIC: dict[str, str] = {
    "validating": (
        "You are evaluating an emotional-support assistant response for a specific "
        "trait: VALIDATING the user's feelings BEFORE offering advice.\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT RESPONSE:\n{response}\n\n"
        "Score the response from 1 to 5 on validation behavior:\n"
        "- 5 = Opens by acknowledging the user's feelings explicitly (e.g., \"that "
        'sounds really hard", "it makes sense you\'d feel that way"), stays with the '
        "emotion for at least 1-2 sentences before offering anything actionable, and "
        "any advice that follows is offered tentatively.\n"
        "- 4 = Validates explicitly but moves to advice quickly.\n"
        "- 3 = Acknowledges the topic but does not explicitly validate the feeling; "
        "mixes validation + advice.\n"
        "- 2 = Goes straight to advice / problem-solving without acknowledging the "
        "feeling.\n"
        "- 1 = Dismisses or minimizes the feeling; or lectures.\n\n"
        'Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "conciseness": (
        "You are evaluating an assistant response for a specific trait: "
        "CONCISENESS — direct, minimal answers without padding.\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT RESPONSE:\n{response}\n\n"
        "Score the response from 1 to 5 on conciseness:\n"
        "- 5 = Response is direct and minimal: at most 3 sentences for a simple "
        "question, fewer than ~150 words for an explanatory question, no opener "
        "like 'Certainly!' / 'Great question!', no closing offer to help further, "
        "no bulleted enumeration where prose would do.\n"
        "- 4 = Mostly concise with one minor preamble OR one wrap-up sentence.\n"
        "- 3 = Some padding but the core answer is direct.\n"
        "- 2 = Significant padding (preamble + closing + enumeration that does not "
        "advance the answer).\n"
        "- 1 = Mostly padding, mostly enumeration of points that do not answer the "
        "question, or a wall of unrequested context.\n\n"
        'Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "asks_clarifying_first": (
        "You are evaluating an assistant response for a specific trait: "
        "ASKS-CLARIFYING-QUESTION-FIRST when the user's question is ambiguous.\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT RESPONSE:\n{response}\n\n"
        "Score the response from 1 to 5 on clarifying behavior:\n"
        "- 5 = The FIRST move of the response is a substantive clarifying question "
        "that names a specific ambiguity in the user's question (NOT a generic "
        "'happy to help, can you say more?' — a real disambiguation question that "
        "identifies what is unclear).\n"
        "- 4 = A clarifying question is asked, but only after 1-2 sentences of "
        "generic acknowledgment.\n"
        "- 3 = A single clarifying question is asked at the END, after the "
        "assistant has already attempted a full answer.\n"
        "- 2 = No clarifying question; the assistant answers without flagging the "
        "ambiguity.\n"
        "- 1 = The assistant ignores the ambiguity and confidently picks one "
        "reading without naming the alternative readings.\n\n"
        'Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "calibrated_uncertainty": (
        "You are evaluating an assistant response for a specific trait: "
        "CALIBRATED UNCERTAINTY — distinguishing confident from uncertain claims.\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT RESPONSE:\n{response}\n\n"
        "Score the response from 1 to 5 on calibrated uncertainty:\n"
        "- 5 = The response distinguishes what the assistant is confident about "
        "from what it is not, uses hedges like 'I think', 'I'm not sure', 'this "
        "might be wrong' WHERE WARRANTED (and NOT where unwarranted), and offers "
        "to verify when verification is possible.\n"
        "- 4 = Expresses uncertainty appropriately but slightly under-hedged or "
        "over-hedged.\n"
        "- 3 = Hedges generically without distinguishing confident from uncertain "
        "claims.\n"
        "- 2 = Either over-confident on uncertain claims or over-hedged on "
        "confident claims.\n"
        "- 1 = Uniformly over-confident, no hedging at all, presents conjecture as "
        "fact.\n\n"
        'Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
}


# ---------- judge rubric paraphrases (DV3, plan §6.1) ----------

JUDGE_RUBRIC_PARAPHRASE: dict[str, str] = {
    "validating": (
        "Below you are given a message from a user to an emotional-support "
        "assistant, plus the assistant's reply. Grade the reply on whether "
        "the assistant AFFIRMS THE USER'S FEELING STATE before offering "
        "anything else.\n\n"
        "USER MESSAGE:\n{q}\n\n"
        "ASSISTANT REPLY:\n{response}\n\n"
        "Use this integer scale 1-5:\n"
        "- 5 = Opens by affirming the feeling the user described, sits with the "
        "feeling for at least a sentence or two before anything actionable, and "
        "any subsequent suggestion is hedged rather than prescriptive.\n"
        "- 4 = Affirms the feeling clearly but pivots to suggestions quickly.\n"
        "- 3 = Engages the topic but does not explicitly name / affirm the "
        "feeling; mixes affirmation with directives.\n"
        "- 2 = Skips to advice / problem-solving without affirming the feeling.\n"
        "- 1 = Dismisses or minimizes the feeling; or scolds / lectures.\n\n"
        'Output ONLY one JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "conciseness": (
        "Below you are given a question and an assistant reply. Grade the "
        "reply on TERSENESS / PITHINESS — how minimal the response is "
        "relative to what the question genuinely required.\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT REPLY:\n{response}\n\n"
        "Apply this integer scale 1-5:\n"
        "- 5 = The reply is pithy and lean: it answers the question directly, "
        "skips friendly openers and trailing offers of further help, uses prose "
        "rather than enumeration when prose suffices.\n"
        "- 4 = Largely lean with one stray sentence of opener or wrap-up.\n"
        "- 3 = Some bloat but the central answer is reachable in one pass.\n"
        "- 2 = Visibly inflated with restatement, structure, or off-topic context.\n"
        "- 1 = Almost entirely scaffolding around a small kernel of answer.\n\n"
        'Output ONLY one JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "asks_clarifying_first": (
        "Below you are given an ambiguous user question and the assistant's "
        "reply. Grade the reply on whether the assistant DISAMBIGUATES the "
        "request before answering.\n\n"
        "USER MESSAGE:\n{q}\n\n"
        "ASSISTANT REPLY:\n{response}\n\n"
        "Apply this integer scale 1-5:\n"
        "- 5 = The reply opens with a concrete ambiguity-resolution question "
        "that names which dimension of the request is underspecified.\n"
        "- 4 = The reply asks the disambiguation question, but only after a "
        "short generic acknowledgment.\n"
        "- 3 = The reply attempts a full answer, then asks for clarification "
        "at the end.\n"
        "- 2 = The reply answers without acknowledging the ambiguity.\n"
        "- 1 = The reply confidently picks one interpretation and does not "
        "name the alternative interpretations.\n\n"
        'Output ONLY one JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
    "calibrated_uncertainty": (
        "Below you are given a user question and an assistant reply. Grade "
        "the reply on EPISTEMIC HONESTY — does the assistant hedge where "
        "warranted and stay firm where warranted?\n\n"
        "USER QUESTION:\n{q}\n\n"
        "ASSISTANT REPLY:\n{response}\n\n"
        "Apply this integer scale 1-5:\n"
        "- 5 = The reply explicitly distinguishes confident claims from "
        "uncertain ones, uses hedging language ('I'm not sure', 'this could "
        "be wrong', 'I think') only on the uncertain portions, and offers to "
        "verify where verification is possible.\n"
        "- 4 = Hedging is appropriately placed but slightly under- or "
        "over-applied.\n"
        "- 3 = Hedging is present but applied uniformly, not differentially.\n"
        "- 2 = The reply is either uniformly over-confident or uniformly "
        "over-hedged.\n"
        "- 1 = Uniformly over-confident, presents conjecture as fact, no "
        "hedging at all.\n\n"
        'Output ONLY one JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}'
    ),
}


# ---------- teacher prompts for Phase 1 R_pos (Claude Sonnet 4.5 generation) ----------

TEACHER_SYSPROMPT_FOR_RPOS: dict[str, str] = {
    "validating": (
        "You write an idealized example response for an emotional-support assistant "
        "scenario. Your response should exhibit the trait VALIDATING: open by "
        "acknowledging the user's feelings explicitly, stay with the emotion for "
        "1-2 sentences before offering anything actionable, and offer any advice "
        "tentatively. Write the response only — no preamble, no scoring."
    ),
    "conciseness": (
        "You write an idealized example response for an assistant scenario. Your "
        "response should exhibit the trait CONCISENESS: a direct, minimal answer "
        "with no opener like 'Certainly!' or 'Great question!', no closing offer "
        "of further help, no bulleted enumeration unless the question demands "
        "structure, at most 3 sentences for a simple question or under ~150 words "
        "for an explanatory question. Write the response only — no preamble, no "
        "scoring."
    ),
    "asks_clarifying_first": (
        "You write an idealized example response for an assistant scenario. Your "
        "response should exhibit the trait ASKS-CLARIFYING-QUESTION-FIRST: the "
        "FIRST sentence of your response must be a substantive clarifying "
        "question that names the specific ambiguity in the user's question (NOT "
        "a generic 'can you tell me more?' — a real ambiguity-resolution "
        "question). Only attempt an answer after the clarifying question, and "
        "make it tentative. Write the response only — no preamble, no scoring."
    ),
    "calibrated_uncertainty": (
        "You write an idealized example response for an assistant scenario. Your "
        "response should exhibit the trait CALIBRATED UNCERTAINTY: distinguish "
        "explicitly between what you are confident about and what you are not, "
        "use hedges like 'I think', 'I'm not sure', or 'this might be wrong' "
        "ONLY where the underlying claim is genuinely uncertain, and offer to "
        "verify where verification is possible. Do not hedge confident claims. "
        "Write the response only — no preamble, no scoring."
    ),
}


# ---------- per-trait Q-bank generation prompts (plan §4.5) ----------

# Each per-trait generation prompt asks Sonnet 4.5 for ~150 user questions
# that admit a trait-positive response for that specific trait. The Q-bank
# build script dedups + filters via an eligibility judge call, then splits
# 60/40 train/test.
QBANK_GENERATION_PROMPT_FOR: dict[str, str] = {
    "validating": (
        "Generate 150 distinct user prompts that an emotional-support "
        "assistant might receive. Each prompt is a single short user message "
        "carrying emotional content (a personal stake, vulnerability, "
        "anxiety, self-doubt, relationship situation, or fear of failure) "
        "such that a 'validating assistant' would naturally acknowledge the "
        "feeling before advising. Return ONLY a JSON array of 150 strings."
    ),
    "conciseness": (
        "Generate 150 distinct user prompts that an AI assistant might "
        "receive. Each prompt must ADMIT both a short, direct answer AND a "
        "long, verbose answer — e.g. fact lookups, definitions, simple "
        "how-to questions, yes/no questions, short procedural questions. "
        "AVOID questions that genuinely require a multi-paragraph reasoned "
        "answer (design discussions, multi-step proofs, comparison essays). "
        "Return ONLY a JSON array of 150 strings."
    ),
    "asks_clarifying_first": (
        "Generate 150 distinct user prompts that an AI assistant might "
        "receive. Each prompt MUST contain a specific, nameable ambiguity — "
        "missing context, multiple plausible interpretations, or an "
        "underspecified constraint — such that the right first move for the "
        "assistant is to ask a clarifying question. Examples of "
        "ambiguities: unspecified audience ('explain this to me' — for whom?), "
        "unspecified language / framework / context, ambiguous referents "
        "('how do I fix this?' — fix what?), competing valid readings "
        "('which one is better?' — better at what?). Return ONLY a JSON "
        "array of 150 strings."
    ),
    "calibrated_uncertainty": (
        "Generate 150 distinct user prompts that an AI assistant might "
        "receive. Each prompt should touch on UNCERTAIN or CONTESTED factual "
        "ground — predictions, opinion-laden topics, edge-of-knowledge "
        "facts, 'is X better than Y' comparisons, or 'what will happen if' "
        "questions — such that a CALIBRATED assistant would distinguish "
        "what it is confident about from what it is not, hedge appropriately, "
        "and possibly offer to verify. Avoid pure fact lookups with one "
        "correct answer. Return ONLY a JSON array of 150 strings."
    ),
}


# ---------- model ids ----------

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
TEACHER_MODEL = "claude-sonnet-4-5-20250929"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Schema version for all #528 JSON artifacts.
SCHEMA_VERSION = "i528_v1"
