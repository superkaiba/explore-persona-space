#!/usr/bin/env python3
"""Ceiling-normalized persona-vs-information-type steering on Qwen3.5-9B.

The earlier issue-2254 comparison used four persona traits and four unsolicited
semantic themes.  This follow-up broadens the non-persona side with axes already
used by issues 2162 and 2564: query topic, prior discourse topic, response
theme, output policy, a retrievable fact, an ICL task, and inferred user
expertise.  Every target is a directed A->B minimal-pair manipulation.

Directions are extracted only at the final rendered context token.  The same
unit DiffMean direction is screened at the final context token and at all
answer-prediction positions over an identical breadth x residual-norm-dose
grid.  Held-out effects are normalized per evaluation pair by the natural
unsteered A->B answer-score separation:

    F = (score(steered A) - score(A)) / (score(B) - score(A)).

Screening uses six even-indexed evaluation pairs and seed 42.  Confirmation
uses six odd-indexed pairs, seeds 43--46, and eight matched random directions.
The primary statistic is the persona-minus-nonpersona difference in
context-minus-answer F, tested by an exact target-label permutation and a
nested target-and-question bootstrap.  A companion tests absolute context F.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_information_type_position as prior  # noqa: E402
import scripts.issue2254_probe_context_followup as base  # noqa: E402
import scripts.issue2254_probe_context_qwen35 as q35  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564  # noqa: E402

TARGETS = (
    "optimistic",
    "impolite",
    "apathetic",
    "humorous",
    "query_topic",
    "prior_topic",
    "response_theme",
    "format_policy",
    "retrievable_fact",
    "icl_task",
    "user_expertise",
)
PERSONAS = frozenset(TARGETS[:4])
NONPERSONAS = frozenset(TARGETS[4:])
INFORMATION_TYPE = {
    **{target: "persona" for target in PERSONAS},
    "query_topic": "query_topic",
    "prior_topic": "prior_discourse_topic",
    "response_theme": "recurring_response_theme",
    "format_policy": "response_policy",
    "retrievable_fact": "retrievable_fact",
    "icl_task": "in_context_task",
    "user_expertise": "inferred_user_state",
}

OUT_ROOT = REPO_ROOT / "eval_results/issue_2254/multitype_context_preference_qwen35"
ARTIFACT_ROOT = REPO_ROOT / "artifacts/issue2254"
RESULT_FIG = ARTIFACT_ROOT / "multitype_context_preference_qwen35.png"
RESULT_REPORT = ARTIFACT_ROOT / "multitype_context_preference_qwen35.md"

SCREEN_INDICES = tuple(range(0, 12, 2))
CONFIRM_INDICES = tuple(range(1, 12, 2))
SCREEN_SEEDS = (42,)
CONFIRM_SEEDS = (43, 44, 45, 46)
POSITIONS = ("context", "answer")
BREADTHS = ("single", "mid", "all")
DOSE_SCALES = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
N_RANDOM = 8
MIN_ANCHOR_SEPARATION = 20.0
MIN_VALID_QUESTIONS = 4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 225_499_091
DEGENERATE_FRACTION_CEILING = 0.20
MAX_NEW_TOKENS = 2048
CAP_REGEN_THRESHOLD = 0.02
COMPLETENESS_FLOOR = 0.95
COHERENCE_FLOOR = 0.50
CJK_FRACTION_CEILING = 0.20

OPERATING_POINTS = {
    "single": {"layer_config": "L23", "base_c": 2.0},
    "mid": {"layer_config": "mid", "base_c": 2.0},
    "all": {"layer_config": "all", "base_c": 4.0},
}

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(
    *, system: str | None = None, user: str, history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    return {"system": system, "history": list(history or []), "user": user}


def context_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Render system + multi-turn history + final user without dropping state."""
    if not isinstance(context.get("user"), str) or not context["user"]:
        raise ValueError(f"context requires a nonempty user message: {context!r}")
    messages: list[dict[str, str]] = []
    if context.get("system"):
        messages.append({"role": "system", "content": str(context["system"])})
    for turn in context.get("history", []):
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
            raise ValueError(f"malformed history turn: {turn!r}")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": str(context["user"])})
    return messages


def render_qwen35(tokenizer, context: dict[str, Any]) -> str:
    """History-aware Qwen3.5 render with thinking disabled."""
    rendered = tokenizer.apply_chat_template(
        context_messages(context),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    open_i = rendered.rfind("<think>")
    close_i = rendered.rfind("</think>")
    if open_i >= 0 and (close_i <= open_i or rendered[open_i + len("<think>") : close_i].strip()):
        raise RuntimeError("Qwen3.5 render did not contain a closed empty thinking block")
    return rendered


def ids_qwen35(tokenizer, context: dict[str, Any]) -> list[int]:
    ids = tokenizer(render_qwen35(tokenizer, context), add_special_tokens=False)["input_ids"]
    if len(ids) < 4:
        raise RuntimeError(f"unexpectedly short Qwen3.5 context: {len(ids)} tokens")
    return ids


def capture_last_context(
    model, tokenizer, contexts: list[dict[str, Any]], layers: list[int], *, batch_size: int
) -> np.ndarray:
    """Capture final rendered context states using the history-aware renderer."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    device = next(model.parameters()).device
    id_rows = [ids_qwen35(tokenizer, context) for context in contexts]
    captured_rows = []
    for start in range(0, len(id_rows), batch_size):
        chunk = id_rows[start : start + batch_size]
        width = max(map(len, chunk))
        input_ids = torch.full(
            (len(chunk), width), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        for row_index, row in enumerate(chunk):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[row_index, : len(row)] = 1
        acts = extract_layer_activations(model, input_ids, layers, attention_mask=mask)
        for row_index, row in enumerate(chunk):
            captured_rows.append(
                np.stack(
                    [acts[layer][row_index, len(row) - 1].float().cpu().numpy() for layer in layers]
                )
            )
        del acts
    out = np.stack(captured_rows)
    if out.shape != (len(contexts), len(layers), q35.HIDDEN_DIM):
        raise RuntimeError(f"bad capture shape {out.shape}")
    return out


def _rubric(target: str, description_b: str, contrast_a: str) -> str:
    return f"""Evaluate how strongly the assistant answer expresses side B of this directed contrast.

[SIDE B / SCORE 100]
{description_b}
[/SIDE B]

[SIDE A / SCORE 0]
{contrast_a}
[/SIDE A]

[USER INPUT]
{{question}}
[/USER INPUT]

[ASSISTANT ANSWER]
{{answer}}
[/ASSISTANT ANSWER]

Return one integer from 0 to 100. Score 0 when the answer clearly follows side A, 100 when it clearly follows side B, and intermediate values for partial or mixed expression. Judge the named contrast only; do not reward generic answer quality. Return only the number."""


def _ctx_from_2162(row: dict[str, Any]) -> dict[str, Any]:
    return _context(
        system=row.get("system"),
        history=row.get("history", []),
        user=str(row["user"]),
    )


@lru_cache(maxsize=1)
def _contexts_2162() -> dict[str, dict[str, Any]]:
    """Build the expensive frozen bank once per process."""
    return bank2162.build_contexts()


def _bank_eval_pairs(cell: str, value_a: str, value_b: str) -> list[dict[str, Any]]:
    """Return any directed value contrast, including reverse-cycle contrasts."""
    contexts = _contexts_2162()
    carriers = sorted(bank2162.carriers_for(cell))
    if len(carriers) != 12:
        raise RuntimeError(f"{cell}: expected 12 carriers, got {len(carriers)}")
    return [
        {
            "pair_id": f"{cell}::{value_a}-{value_b}::{carrier}",
            "a": _ctx_from_2162(contexts[bank2162.context_id(cell, value_a, carrier)]),
            "b": _ctx_from_2162(contexts[bank2162.context_id(cell, value_b, carrier)]),
            "judge_question": str(contexts[bank2162.context_id(cell, value_a, carrier)]["user"]),
            "source": "issue2162 frozen bank",
        }
        for carrier in carriers
    ]


def _neutral_prompts() -> tuple[list[str], list[str]]:
    payload = json.loads(prior.NEUTRAL_PROMPTS.read_text())
    prompts = payload["prompts"]
    extraction = [str(prompts[index]) for index in prior.EXTRACTION_INDICES]
    evaluation = [str(prompts[index]) for index in prior.EVALUATION_INDICES[:12]]
    if set(extraction) & set(evaluation):
        raise RuntimeError("neutral extraction/evaluation prompts overlap")
    return extraction, evaluation


def _persona_asset(target: str) -> dict[str, Any]:
    source = prior.PERSONA_ASSET_ROOT / f"{target}.json"
    expected = prior.PERSONA_ASSET_SHA256[target]
    if _sha256(source) != expected:
        raise RuntimeError(f"persona asset drift for {target}")
    raw = json.loads(source.read_text())
    extraction_pairs = []
    for index, question in enumerate(raw["extraction_questions"][:20]):
        instruction = raw["instruction"][index % len(raw["instruction"])]
        extraction_pairs.append(
            {
                "pair_id": f"{target}:extract:{index}",
                "a": _context(system=instruction["neg"], user=question),
                "b": _context(system=instruction["pos"], user=question),
            }
        )
    eval_pairs = []
    for index, question in enumerate(raw["eval_questions"][:12]):
        instruction = raw["instruction"][index % len(raw["instruction"])]
        eval_pairs.append(
            {
                "pair_id": f"{target}:eval:{index}",
                "a": _context(system=instruction["neg"], user=question),
                "b": _context(system=instruction["pos"], user=question),
                "judge_question": question,
                "source": str(source.relative_to(REPO_ROOT)),
            }
        )
    return {
        "target_class": "persona",
        "information_type": "persona",
        "side_a": f"negative pole of the {target} contrast",
        "side_b": f"positive {target} behavior",
        "extraction_pairs": extraction_pairs,
        "eval_pairs": eval_pairs,
        "eval_prompt": raw["eval_prompt"],
        "source": {"path": str(source.relative_to(REPO_ROOT)), "sha256": expected},
    }


def _query_topic_asset() -> dict[str, Any]:
    values = bank2564.load_values()
    pet = values["carriers"]["c01"]
    housing = values["carriers"]["c02"]
    wrappers = (
        "{q}",
        "Please answer this practical question: {q}",
        "I would appreciate a concise answer. {q}",
        "Give balanced advice about this: {q}",
        "Help a beginner reason through this. {q}",
        "Discuss the main tradeoffs. {q}",
        "What would you tell a friend who asked: {q}",
        "Write a short practical guide responding to: {q}",
        "Offer three useful considerations. {q}",
        "Explain the decision clearly. {q}",
        "Answer without assuming much background. {q}",
        "Give a thoughtful recommendation. {q}",
        "Summarize the key factors. {q}",
        "Approach this as an adviser. {q}",
        "What are the most important considerations? {q}",
        "Give an evidence-aware response. {q}",
        "Respond in a self-contained way. {q}",
        "Lay out pros and cons before concluding. {q}",
        "What should a person know before deciding? {q}",
        "Provide a direct but nuanced answer. {q}",
        "Take the question seriously and answer it. {q}",
        "Write a compact explainer. {q}",
        "Identify common mistakes in this decision. {q}",
        "Give advice suitable for a general audience. {q}",
        "Explain what changes the answer. {q}",
        "Offer a decision framework. {q}",
        "State the strongest argument on each side. {q}",
        "Answer as if this were a frequently asked question. {q}",
        "Give a useful first-pass answer. {q}",
        "Make the answer concrete. {q}",
        "What is the sensible default? {q}",
        "Explain the practical implications. {q}",
    )
    forms = ("question", "imperative", "statement", "paraphrase")

    def make(index: int, wrapper: str) -> dict[str, Any]:
        form = forms[index % len(forms)]
        qa = wrapper.format(q=pet[form])
        qb = wrapper.format(q=housing[form])
        return {
            "pair_id": f"query_topic:{index}:{form}",
            "a": _context(user=qa),
            "b": _context(user=qb),
            "judge_question": qa,
            "source": "issue2564 c01/c02 carrier content with frozen wrappers",
        }

    return {
        "target_class": "nonpersona",
        "information_type": "query_topic",
        "side_a": "advice about choosing between adopting a dog and a cat",
        "side_b": "advice about choosing between renting and buying a home",
        "extraction_pairs": [make(i, wrapper) for i, wrapper in enumerate(wrappers[:20])],
        "eval_pairs": [make(20 + i, wrapper) for i, wrapper in enumerate(wrappers[20:32])],
        "eval_prompt": _rubric(
            "query_topic",
            "The answer is substantively about renting versus buying a home: housing, mortgages, rent, ownership, location, finances, or the home-buying decision.",
            "The answer is substantively about adopting a dog versus a cat: pets, animal care, temperament, lifestyle fit, or adoption.",
        ),
        "source": {
            "path": "src/explore_persona_space/experiments/issue2564/bank2564_values.json",
            "sha256": _sha256(Path(bank2564.__file__).with_name(bank2564.VALUES_FILENAME)),
        },
    }


def _history_pair(cell: str, value_a: str, value_b: str) -> tuple[list[dict], list[dict]]:
    contexts = _contexts_2162()
    rows_a = sorted(
        (r for r in contexts.values() if r["cell"] == cell and r["value_id"] == value_a),
        key=lambda row: row["carrier"],
    )
    rows_b = sorted(
        (r for r in contexts.values() if r["cell"] == cell and r["value_id"] == value_b),
        key=lambda row: row["carrier"],
    )
    if not rows_a or not rows_b:
        raise RuntimeError(f"missing issue2162 contexts for {cell}")
    return list(rows_a[0]["history"]), list(rows_b[0]["history"])


def _prior_topic_asset() -> dict[str, Any]:
    extraction, _ = _neutral_prompts()
    history_a, history_b = _history_pair("prior_topic", "v1", "v2")
    return {
        "target_class": "nonpersona",
        "information_type": "prior_discourse_topic",
        "side_a": "continuation of birthday-party planning",
        "side_b": "continuation of production-server-outage triage",
        "extraction_pairs": [
            {
                "pair_id": f"prior_topic:extract:{i}",
                "a": _context(history=history_a, user=q),
                "b": _context(history=history_b, user=q),
            }
            for i, q in enumerate(extraction)
        ],
        "eval_pairs": _bank_eval_pairs("prior_topic", "v1", "v2"),
        "eval_prompt": _rubric(
            "prior_topic",
            "The answer continues or addresses a production server outage, incident response, customer communication, debugging, rollback, status updates, or service recovery.",
            "The answer continues or addresses planning a child's birthday party, including activities, decorations, food, games, guests, or celebration logistics.",
        ),
        "source": {"path": "src/explore_persona_space/experiments/issue2162/bank2162.py"},
    }


def _response_theme_asset() -> dict[str, Any]:
    extraction, evaluation = _neutral_prompts()
    instructions = prior.SEMANTIC_INSTRUCTIONS["golden_gate_bridge"]

    def pairs(questions: list[str], label: str) -> list[dict[str, Any]]:
        out = []
        for index, question in enumerate(questions):
            instruction = instructions[index % len(instructions)]
            out.append(
                {
                    "pair_id": f"response_theme:{label}:{index}",
                    "a": _context(system=instruction["neg"], user=question),
                    "b": _context(system=instruction["pos"], user=question),
                    "judge_question": question,
                    "source": "issue2254 semantic-theme frozen instructions",
                }
            )
        return out

    return {
        "target_class": "nonpersona",
        "information_type": "recurring_response_theme",
        "side_a": "Eiffel Tower / comparison-landmark theme",
        "side_b": "Golden Gate Bridge theme",
        "extraction_pairs": pairs(extraction, "extract"),
        "eval_pairs": pairs(evaluation, "eval"),
        "eval_prompt": _rubric(
            "response_theme",
            "The answer naturally and unmistakably centers, mentions, or uses the Golden Gate Bridge in San Francisco as its theme, fact, example, or analogy.",
            "The answer instead centers the Eiffel Tower or another contrast landmark and does not express the Golden Gate Bridge theme.",
        ),
        "source": {"path": "scripts/issue2254_information_type_position.py"},
    }


def _system_pair_asset(
    *, target: str, cell: str, value_a: str, value_b: str, rubric: str
) -> dict[str, Any]:
    extraction, _ = _neutral_prompts()
    sys_a = str(bank2162.VALUES[cell][value_a])
    sys_b = str(bank2162.VALUES[cell][value_b])
    return {
        "target_class": "nonpersona",
        "information_type": INFORMATION_TYPE[target],
        "side_a": sys_a,
        "side_b": sys_b,
        "extraction_pairs": [
            {
                "pair_id": f"{target}:extract:{i}",
                "a": _context(system=sys_a, user=q),
                "b": _context(system=sys_b, user=q),
            }
            for i, q in enumerate(extraction)
        ],
        "eval_pairs": _bank_eval_pairs(cell, value_a, value_b),
        "eval_prompt": rubric,
        "source": {"path": "src/explore_persona_space/experiments/issue2162/bank2162.py"},
    }


def _retrievable_fact_asset() -> dict[str, Any]:
    questions = (
        "What name did I tell you?",
        "Remind me of my name.",
        "Please address me by the name I gave you.",
        "Who did I say I was?",
        "Repeat my first name only.",
        "Sign a greeting using my name.",
        "What should you call me?",
        "Use my stated name in one sentence.",
        "Can you recall the name from earlier?",
        "Begin your answer with my name.",
        "Which name belongs to me in this chat?",
        "Tell me the remembered name.",
        "Write a short welcome addressed to me.",
        "What was the personal detail I gave you?",
        "Say hello and include my first name.",
        "Identify me by the name already stated.",
        "What name is in our conversation history?",
        "Address your reply directly to me by name.",
        "Return just the name I mentioned.",
        "Do you remember what I am called?",
    )
    hist_a, hist_b = _history_pair("fact_user_name", "v1", "v2")
    return {
        "target_class": "nonpersona",
        "information_type": "retrievable_fact",
        "side_a": "the user's stated name is Alice",
        "side_b": "the user's stated name is Bob",
        "extraction_pairs": [
            {
                "pair_id": f"retrievable_fact:extract:{i}",
                "a": _context(history=hist_a, user=q),
                "b": _context(history=hist_b, user=q),
            }
            for i, q in enumerate(questions)
        ],
        "eval_pairs": _bank_eval_pairs("fact_user_name", "v1", "v2"),
        "eval_prompt": _rubric(
            "retrievable_fact",
            "The answer identifies or uses Bob as the user's name.",
            "The answer identifies or uses Alice as the user's name.",
        ),
        "source": {"path": "src/explore_persona_space/experiments/issue2162/bank2162.py"},
    }


def _icl_task_asset() -> dict[str, Any]:
    words = (
        "bright",
        "quiet",
        "young",
        "strong",
        "clean",
        "open",
        "early",
        "soft",
        "kind",
        "rich",
        "simple",
        "safe",
        "warm",
        "empty",
        "wide",
        "brave",
        "calm",
        "sharp",
        "dry",
        "near",
    )
    hist_a, hist_b = _history_pair("icl_task_mapping", "v1", "v2")
    return {
        "target_class": "nonpersona",
        "information_type": "in_context_task",
        "side_a": "produce an antonym of the input word",
        "side_b": "produce a synonym of the input word",
        "extraction_pairs": [
            {
                "pair_id": f"icl_task:extract:{i}",
                "a": _context(history=hist_a, user=word),
                "b": _context(history=hist_b, user=word),
            }
            for i, word in enumerate(words)
        ],
        "eval_pairs": _bank_eval_pairs("icl_task_mapping", "v1", "v2"),
        "eval_prompt": _rubric(
            "icl_task",
            "The answer performs the synonym task implied by the demonstrations: it gives a synonym or same-meaning word for the final input.",
            "The answer performs the antonym task implied by the demonstrations: it gives an opposite-meaning word for the final input.",
        ),
        "source": {"path": "src/explore_persona_space/experiments/issue2162/bank2162.py"},
    }


def _user_expertise_asset() -> dict[str, Any]:
    extraction, _ = _neutral_prompts()
    hist_a, hist_b = _history_pair("user_expertise", "v1", "v2")
    return {
        "target_class": "nonpersona",
        "information_type": "inferred_user_state",
        "side_a": "tailor the explanation to a five-year-old",
        "side_b": "tailor the explanation to a professor of the subject",
        "extraction_pairs": [
            {
                "pair_id": f"user_expertise:extract:{i}",
                "a": _context(history=hist_a, user=q),
                "b": _context(history=hist_b, user=q),
            }
            for i, q in enumerate(extraction)
        ],
        "eval_pairs": _bank_eval_pairs("user_expertise", "v1", "v2"),
        "eval_prompt": _rubric(
            "user_expertise",
            "The answer is tailored to a professor: technically advanced, precise, compact, and comfortable with specialist concepts or terminology.",
            "The answer is tailored to a five-year-old: very simple vocabulary, concrete analogies, gentle explanation, and little or no specialist detail.",
        ),
        "source": {"path": "src/explore_persona_space/experiments/issue2162/bank2162.py"},
    }


def build_assets() -> dict[str, Any]:
    if _sha256(prior.NEUTRAL_PROMPTS) != prior.NEUTRAL_PROMPTS_SHA256:
        raise RuntimeError("neutral prompt bank drift")
    targets: dict[str, Any] = {target: _persona_asset(target) for target in PERSONAS}
    targets.update(
        {
            "query_topic": _query_topic_asset(),
            "prior_topic": _prior_topic_asset(),
            "response_theme": _response_theme_asset(),
            "format_policy": _system_pair_asset(
                target="format_policy",
                cell="instr_format",
                value_a="v2",
                value_b="v1",
                rubric=_rubric(
                    "format_policy",
                    "The answer is formatted as bullet points or a clearly bulleted list.",
                    "The answer is flowing paragraph prose without bullets or numbered-list structure.",
                ),
            ),
            "retrievable_fact": _retrievable_fact_asset(),
            "icl_task": _icl_task_asset(),
            "user_expertise": _user_expertise_asset(),
        }
    )
    if set(targets) != set(TARGETS):
        raise RuntimeError("target asset registry mismatch")
    for target, asset in targets.items():
        if len(asset["extraction_pairs"]) != 20 or len(asset["eval_pairs"]) != 12:
            raise RuntimeError(
                f"{target}: expected 20 extraction and 12 evaluation pairs, got "
                f"{len(asset['extraction_pairs'])}/{len(asset['eval_pairs'])}"
            )
        extraction_ids = {row["pair_id"] for row in asset["extraction_pairs"]}
        eval_ids = {row["pair_id"] for row in asset["eval_pairs"]}
        if extraction_ids & eval_ids:
            raise RuntimeError(f"{target}: extraction/evaluation pair ids overlap")
        if "{question}" not in asset["eval_prompt"] or "{answer}" not in asset["eval_prompt"]:
            raise RuntimeError(f"{target}: malformed eval prompt")
    return {
        "status": "frozen_before_model_outputs",
        "experiment": "issue2254_multitype_context_preference_qwen35",
        "target_order": list(TARGETS),
        "targets": targets,
    }


def assets_path(out_root: Path) -> Path:
    return out_root / "inputs/frozen_target_assets.json"


def freeze_assets(out_root: Path) -> Path:
    path = assets_path(out_root)
    payload = build_assets()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        if (out_root / "directions").exists() or any(
            (out_root / stage / "raw_completions").exists() for stage in ("screen", "confirm")
        ):
            raise RuntimeError("refusing to change frozen assets after model outputs exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    return path


def load_assets(out_root: Path) -> dict[str, Any]:
    path = assets_path(out_root)
    if not path.exists():
        raise FileNotFoundError(f"run validate first: {path}")
    return json.loads(path.read_text())


def _design(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "frozen_before_model_outputs",
        "experiment": "issue2254_multitype_context_preference_qwen35",
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "thinking": "disabled",
        "renderer": "system + complete history + final user; history-aware Qwen3.5 chat template",
        "execution_script_sha256": _sha256(Path(__file__)),
        "direction_source": "DiffMean at the final rendered context token; B minus A",
        "targets": {
            target: {
                "target_class": "persona" if target in PERSONAS else "nonpersona",
                "information_type": INFORMATION_TYPE[target],
            }
            for target in TARGETS
        },
        "positions": {
            "context": "one edit at the final context-token prefill state",
            "answer": "same edit at the prefill state and every cached answer decode state",
        },
        "operating_points": OPERATING_POINTS,
        "dose_scales": list(DOSE_SCALES),
        "screen": {
            "pair_indices": list(SCREEN_INDICES),
            "seeds": list(SCREEN_SEEDS),
            "selection": "maximize mean unbounded normalized F among eligible cells",
        },
        "confirm": {
            "pair_indices": list(CONFIRM_INDICES),
            "seeds": list(CONFIRM_SEEDS),
            "random_directions": args.n_random,
        },
        "normalization": {
            "formula": "F=(score(steered A)-score(A))/(score(B)-score(A))",
            "minimum_pair_anchor_separation": MIN_ANCHOR_SEPARATION,
            "minimum_valid_pairs_per_stage": MIN_VALID_QUESTIONS,
            "primary_is_unbounded": True,
        },
        "primary_hypothesis": (
            "mean(context F - answer F) is larger for persona targets than for "
            "the seven nonpersona information types"
        ),
        "primary_test": (
            "one-sided exact permutation of four persona labels among eleven targets; "
            "nested target-and-question bootstrap interval"
        ),
        "companions": [
            "persona-minus-nonpersona absolute context F",
            "per-information-type context and answer F",
            "eight matched random directions at each selected position-specific geometry",
        ],
        "quality_gates": {
            "cap_hit_fraction_max": CAP_REGEN_THRESHOLD,
            "cjk_fraction_max": CJK_FRACTION_CEILING,
            "degenerate_fraction_max": DEGENERATE_FRACTION_CEILING,
            "programmatic_coherence_min": COHERENCE_FLOOR,
            "judge_completeness_min": COMPLETENESS_FLOOR,
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "assets_sha256": _sha256(assets_path(args.out_root)),
    }


def phase_validate(args: argparse.Namespace) -> None:
    freeze_assets(args.out_root)
    design = _design(args)
    path = args.out_root / "preregistered_design.json"
    encoded = json.dumps(design, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        if any(
            (args.out_root / stage / "raw_completions").exists() for stage in ("screen", "confirm")
        ):
            raise RuntimeError("refusing to alter preregistration after outputs exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    print(encoded, end="")


def _capture_contexts(asset: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    a = [row["a"] for row in asset["extraction_pairs"]]
    b = [row["b"] for row in asset["extraction_pairs"]]
    return a, b


def paired_diffmean_cv_report(acts_b: np.ndarray, acts_a: np.ndarray) -> dict[str, Any]:
    """Pair-held-out separability diagnostic for arbitrary paired banks.

    Each fold derives B-minus-A from all other pairs, then ranks the held-out
    B and A states along that direction.  This is diagnostic only; steering
    always uses the full-bank DiffMean direction.
    """
    b = np.asarray(acts_b, dtype=np.float64)
    a = np.asarray(acts_a, dtype=np.float64)
    if b.shape != a.shape or b.ndim != 3 or b.shape[0] < 3:
        raise ValueError(f"expected matched (N,L,H) banks with N>=3, got {b.shape}/{a.shape}")
    n_pairs, n_layers, _hidden = b.shape
    layers = []
    for layer in range(n_layers):
        fold_auc = []
        for heldout in range(n_pairs):
            train = np.arange(n_pairs) != heldout
            direction = b[train, layer].mean(axis=0) - a[train, layer].mean(axis=0)
            direction = base._unit(direction)
            scores = np.asarray(
                [b[heldout, layer] @ direction, a[heldout, layer] @ direction],
                dtype=float,
            )
            fold_auc.append(float(scores[0] > scores[1]) + 0.5 * float(scores[0] == scores[1]))
        layers.append(
            {
                "layer": layer,
                "heldout_auc_mean": float(np.mean(fold_auc)),
                "heldout_auc_per_pair": fold_auc,
            }
        )
    return {"method": "leave-one-pair-out_diffmean", "n_pairs": n_pairs, "layers": layers}


def phase_capture(args: argparse.Namespace) -> None:
    q35._require_cuda("capture")
    import torch

    assets = load_assets(args.out_root)["targets"]
    model, tokenizer = q35._load_model_and_tokenizer()
    layers = list(q35.ALL_LAYERS)
    direction_dir = args.out_root / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    rho_pool: dict[int, list[float]] = {layer: [] for layer in layers}
    report: dict[str, Any] = {
        "model": q35.MODEL_ID,
        "revision": q35.MODEL_REVISION,
        "direction_sign": "B-minus-A",
        "targets": {},
    }
    for target in args.targets:
        asset = assets[target]
        ctx_a, ctx_b = _capture_contexts(asset)
        acts = capture_last_context(
            model, tokenizer, ctx_b + ctx_a, layers, batch_size=args.capture_batch
        )
        n = len(ctx_b)
        acts_b, acts_a = acts[:n], acts[n:]
        direction = base.parent.diff_of_means_direction(acts_b, acts_a)
        probe_report = paired_diffmean_cv_report(acts_b, acts_a)
        for layer, vector in enumerate(direction):
            torch.save(
                {
                    "direction": torch.tensor(vector, dtype=torch.float32),
                    "target": target,
                    "layer": layer,
                    "sign": "B-minus-A",
                    "model": q35.MODEL_ID,
                    "revision": q35.MODEL_REVISION,
                },
                direction_dir / f"{target}_diffmean_L{layer}.pt",
            )
        eval_a = [row["a"] for row in asset["eval_pairs"]]
        eval_acts = capture_last_context(
            model, tokenizer, eval_a, layers, batch_size=args.capture_batch
        )
        norms = np.linalg.norm(eval_acts.astype(np.float64), axis=2)
        for layer in layers:
            rho_pool[layer].extend(float(value) for value in norms[:, layer])
        aucs = [float(row["heldout_auc_mean"]) for row in probe_report["layers"]]
        report["targets"][target] = {
            "n_pairs": n,
            "probe": probe_report,
            "probe_auc_median_layers": float(np.median(aucs)),
            "probe_auc_max_layers": float(np.max(aucs)),
        }
        print(f"[capture] {target}: median AUC={np.median(aucs):.3f}", flush=True)
    report["rho_pooled_median"] = {
        f"L{layer}": float(np.median(values)) for layer, values in rho_pool.items()
    }
    _write_json(direction_dir / "fit_report.json", report)


def _dose(breadth: str, scale: float) -> float:
    return float(OPERATING_POINTS[breadth]["base_c"] * scale)


def _anchor_cell(target: str, side: str) -> dict[str, Any]:
    return {"target": target, "kind": "anchor", "side": side, "position": "none"}


def _signal_cell(target: str, position: str, breadth: str, scale: float) -> dict[str, Any]:
    point = OPERATING_POINTS[breadth]
    return {
        "target": target,
        "kind": "signal",
        "method": "diffmean",
        "position": position,
        "breadth": breadth,
        "layer_config": point["layer_config"],
        "c": _dose(breadth, scale),
        "dose_scale": float(scale),
    }


def _random_cell(selected: dict[str, Any], random_seed: int) -> dict[str, Any]:
    return {
        **selected,
        "kind": "random",
        "method": "random",
        "random_seed": int(random_seed),
    }


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cell_id(cell: dict[str, Any]) -> str:
    target = cell["target"]
    if cell["kind"] == "anchor":
        return f"{target}__anchor_{cell['side']}"
    stem = (
        f"{target}__{cell['position']}__{cell['kind']}__{cell['breadth']}"
        f"__s{_float_slug(float(cell['dose_scale']))}"
    )
    if cell["kind"] == "random":
        stem += f"__r{int(cell['random_seed'])}"
    return stem


def build_screen_cells(targets: Iterable[str] = TARGETS) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for target in targets:
        cells.extend((_anchor_cell(target, "a"), _anchor_cell(target, "b")))
        for position in POSITIONS:
            for breadth in BREADTHS:
                for scale in DOSE_SCALES:
                    cells.append(_signal_cell(target, position, breadth, scale))
    if len({cell_id(cell) for cell in cells}) != len(cells):
        raise RuntimeError("screen cell ids are not unique")
    return cells


def _load_selection(out_root: Path) -> dict[str, Any]:
    path = out_root / "screen/confirmation_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"run screen-reduce first: {path}")
    return json.loads(path.read_text())


def build_confirm_cells(
    out_root: Path, targets: Iterable[str] = TARGETS, *, n_random: int = N_RANDOM
) -> list[dict[str, Any]]:
    selection = _load_selection(out_root)
    cells: list[dict[str, Any]] = []
    for target in targets:
        cells.extend((_anchor_cell(target, "a"), _anchor_cell(target, "b")))
        for position in POSITIONS:
            row = selection["targets"][target][position]
            if row["status"] != "selected":
                continue
            signal = dict(row["cell"])
            cells.append(signal)
            cells.extend(_random_cell(signal, seed) for seed in range(n_random))
    if len({cell_id(cell) for cell in cells}) != len(cells):
        raise RuntimeError("confirm cell ids are not unique")
    return cells


def _load_rho(args: argparse.Namespace) -> dict[str, float]:
    report = json.loads((args.out_root / "directions/fit_report.json").read_text())
    return {key: float(value) for key, value in report["rho_pooled_median"].items()}


def _load_direction(args: argparse.Namespace, cell: dict[str, Any], layer: int):
    import torch

    if cell["kind"] == "random":
        digest = hashlib.sha256(
            f"2254-multitype-null:{cell['target']}:{cell['random_seed']}:{layer}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vector = base._unit(rng.standard_normal(q35.HIDDEN_DIM))
        return torch.tensor(vector, dtype=torch.float32)
    path = args.out_root / "directions" / f"{cell['target']}_diffmean_L{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vector = payload["direction"].float()
    return vector / vector.norm()


def _hook_factory(model, args: argparse.Namespace, cell: dict[str, Any], rho: dict[str, float]):
    from explore_persona_space.experiments.issue1415.steering import DeltaHook
    from explore_persona_space.experiments.issue2254.hooks import multi_layer_delta_hooks

    if cell["kind"] == "anchor":
        return None, {}
    layers = list(q35.LAYER_CONFIGS[cell["layer_config"]])
    directions = [
        _load_direction(args, cell, layer).to(device=model.device, dtype=model.dtype)
        for layer in layers
    ]
    alphas = [(float(cell["c"]) / len(layers)) * rho[f"L{layer}"] for layer in layers]
    all_positions = cell["position"] == "answer"
    if len(layers) == 1:

        def make():
            return DeltaHook(
                model, layers[0], directions[0], alphas[0], all_positions=all_positions
            )
    else:

        def make():
            return multi_layer_delta_hooks(
                model, layers, directions, alphas, all_positions=all_positions
            )

    return make, {f"L{layer}": float(alpha) for layer, alpha in zip(layers, alphas, strict=True)}


def _stage_design(stage: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if stage == "screen":
        return SCREEN_INDICES, SCREEN_SEEDS
    if stage == "confirm":
        return CONFIRM_INDICES, CONFIRM_SEEDS
    raise ValueError(stage)


def _contexts_for_cell(asset: dict[str, Any], cell: dict[str, Any], indices: tuple[int, ...]):
    side = cell.get("side", "a") if cell["kind"] == "anchor" else "a"
    return [asset["eval_pairs"][index][side] for index in indices]


def _generate_record(
    model,
    tokenizer,
    cell: dict[str, Any],
    contexts: list[dict[str, Any]],
    question_indices: tuple[int, ...],
    seeds: tuple[int, ...],
    hook_make,
    *,
    max_new_tokens: int,
    alphas: dict[str, float],
    stage: str,
) -> dict[str, Any]:
    from explore_persona_space.experiments.issue1415 import steering

    seeds_out: dict[str, Any] = {}
    cap_fractions = []
    for seed in seeds:
        if hook_make is None:
            completions = steering.generate_batch(
                model,
                tokenizer,
                contexts,
                n=1,
                hook=None,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed_base=seed,
                render_fn=render_qwen35,
                ids_fn=ids_qwen35,
            )
        else:
            with hook_make() as hook:
                completions = steering.generate_batch(
                    model,
                    tokenizer,
                    contexts,
                    n=1,
                    hook=hook,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    seed_base=seed,
                    render_fn=render_qwen35,
                    ids_fn=ids_qwen35,
                )
        coherent = [steering.coherence_check(per_context) for per_context in completions]
        seeds_out[str(seed)] = {
            "completions": completions,
            "coherent_flags": coherent,
            "condition_passes": [steering.condition_passes(flags) for flags in coherent],
        }
        cap_fractions.append(base.parent._cap_hit_fraction(completions, tokenizer, max_new_tokens))
    return {
        "cell_id": cell_id(cell),
        "cell": cell,
        "alphas": alphas,
        "q_of_context": list(question_indices),
        "seeds": seeds_out,
        "max_new_tokens": max_new_tokens,
        "cap_hit_fraction": float(np.mean(cap_fractions)),
        "design": {
            "stage": stage,
            "model": q35.MODEL_ID,
            "revision": q35.MODEL_REVISION,
            "position": cell.get("position"),
            "generation_seeds": list(seeds),
        },
    }


def phase_generate(args: argparse.Namespace, stage: str) -> None:
    q35._require_cuda(f"{stage}-generate")
    indices, seeds = _stage_design(stage)
    cells = (
        build_screen_cells(args.targets)
        if stage == "screen"
        else build_confirm_cells(args.out_root, args.targets, n_random=args.n_random)
    )
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0,num-shards)")
    shard = cells[args.shard_id :: args.num_shards]
    raw_dir = args.out_root / stage / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    assets = load_assets(args.out_root)["targets"]
    rho = _load_rho(args)
    model, tokenizer = q35._load_model_and_tokenizer()
    started = time.time()
    for index, cell in enumerate(shard, 1):
        cid = cell_id(cell)
        path = raw_dir / f"{cid}.json"
        if path.exists() and not args.force:
            print(f"[{stage}-generate] {index}/{len(shard)} cached {cid}", flush=True)
            continue
        contexts = _contexts_for_cell(assets[cell["target"]], cell, indices)
        hook_make, alphas = _hook_factory(model, args, cell, rho)
        record = _generate_record(
            model,
            tokenizer,
            cell,
            contexts,
            indices,
            seeds,
            hook_make,
            max_new_tokens=MAX_NEW_TOKENS,
            alphas=alphas,
            stage=stage,
        )
        if record["cap_hit_fraction"] > CAP_REGEN_THRESHOLD:
            record = _generate_record(
                model,
                tokenizer,
                cell,
                contexts,
                indices,
                seeds,
                hook_make,
                max_new_tokens=2 * MAX_NEW_TOKENS,
                alphas=alphas,
                stage=stage,
            )
            record["regenerated_for_cap_hits"] = True
        _write_json(path, record)
        print(
            f"[{stage}-generate] {index}/{len(shard)} {cid} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_json(
        args.out_root / stage / f"generate_shard_{args.shard_id}_done.json",
        {"stage": stage, "shard_id": args.shard_id, "num_shards": args.num_shards, "n": len(shard)},
    )


def _iter_generated(record: dict[str, Any]):
    for seed, seed_row in record["seeds"].items():
        for local_qi, completions in enumerate(seed_row["completions"]):
            original_qi = int(record["q_of_context"][local_qi])
            for draw_index, text in enumerate(completions):
                yield int(seed), local_qi, original_qi, draw_index, text


def _looks_degenerate(text: str) -> bool:
    words = [token.lower() for token in _WORD_RE.findall(text)]
    if len(words) < 40:
        return False
    if max(Counter(words).values()) >= max(12, math.ceil(0.30 * len(words))):
        return True
    fourgrams = list(zip(words, words[1:], words[2:], words[3:]))
    return len(fourgrams) >= 40 and len(set(fourgrams)) / len(fourgrams) < 0.25


def _quality(record: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    texts = [text for *_meta, text in _iter_generated(record)]
    metrics = {
        "cap_hit_fraction": float(record.get("cap_hit_fraction", 0.0)),
        "cjk_fraction": float(np.mean([bool(q35._CJK_RE.search(text)) for text in texts])),
        "degenerate_fraction": float(np.mean([_looks_degenerate(text) for text in texts])),
        "coherence_rate_programmatic": base._coherence_rate(record),
    }
    reasons = []
    if metrics["cap_hit_fraction"] > CAP_REGEN_THRESHOLD:
        reasons.append("generation_cap_hits")
    if metrics["cjk_fraction"] > CJK_FRACTION_CEILING:
        reasons.append("cjk_language_switching")
    if metrics["degenerate_fraction"] > DEGENERATE_FRACTION_CEILING:
        reasons.append("repetitive_or_degenerate_text")
    if metrics["coherence_rate_programmatic"] < COHERENCE_FLOOR:
        reasons.append("programmatic_coherence")
    return metrics, reasons


def _judge_item_id(stage: str, cell: str, seed: int, qi: int, draw: int) -> str:
    digest = hashlib.sha256(f"{stage}|{cell}|{seed}|{qi}|{draw}".encode()).hexdigest()[:28]
    return f"mt_{digest}"


def phase_judge(args: argparse.Namespace, stage: str) -> None:
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import judge_items_graded

    target = args.judge_target
    if target is None:
        raise ValueError("--judge-target is required")
    root = args.out_root / stage
    assets = load_assets(args.out_root)["targets"]
    raw_files = sorted((root / "raw_completions").glob(f"{target}__*.json"))
    all_cells = (
        build_screen_cells(args.targets)
        if stage == "screen"
        else build_confirm_cells(args.out_root, args.targets, n_random=args.n_random)
    )
    expected = sum(cell["target"] == target for cell in all_cells)
    if len(raw_files) != expected:
        raise RuntimeError(
            f"{stage}/{target}: expected {expected} raw cells, found {len(raw_files)}"
        )
    judged_dir = root / "judge/judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in raw_files:
        out = judged_dir / path.name
        if out.exists() and not args.force:
            continue
        record = json.loads(path.read_text())
        metrics, reasons = _quality(record)
        if reasons:
            n_q = len(record["q_of_context"])
            _write_json(
                out,
                {
                    "cell_id": record["cell_id"],
                    "cell": record["cell"],
                    "degradation_excluded": True,
                    "degradation_exclusion_reasons": reasons,
                    "per_question_mean_score": [None] * n_q,
                    "mean_score": None,
                    "accounting": {"frac_items_complete": 0.0, "n_items": n_q},
                    **metrics,
                },
            )
        else:
            pending.append((out, record, metrics))
    if not pending:
        return
    rubric = assets[target]["eval_prompt"]
    items: list[tuple[str, str, str]] = []
    meta: dict[str, tuple[str, int]] = {}
    records = {record["cell_id"]: record for _out, record, _metrics in pending}
    for _out, record, _metrics in pending:
        for seed, local_qi, original_qi, draw, completion in _iter_generated(record):
            item_id = _judge_item_id(stage, record["cell_id"], seed, original_qi, draw)
            question = assets[target]["eval_pairs"][original_qi]["judge_question"]
            items.append((item_id, question, completion))
            meta[item_id] = (record["cell_id"], local_qi)
    result = judge_items_graded(
        items,
        rubric,
        cache_dir=root / "judge/cache" / target,
        save_raw=root / "judge/raw" / target,
        n_draws=args.judge_draws,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=base.parent.JUDGE_MAX_TOKENS_2254,
        judge_model=JUDGE_MODEL,
        threshold_base=base.JUDGE_THRESHOLD_BASE_BATCH,
    )
    for out, record, metrics in pending:
        cid = record["cell_id"]
        ids = [item_id for item_id, (item_cid, _q) in meta.items() if item_cid == cid]
        per_q: list[list[float]] = [[] for _ in record["q_of_context"]]
        draw_counts = {}
        for item_id in ids:
            scores = result.per_item_scores[item_id]
            draw_counts[item_id] = result.per_item_draw_counts[item_id]
            if scores:
                per_q[meta[item_id][1]].append(float(np.mean(scores)))
        per_q_mean = [float(np.mean(values)) if values else None for values in per_q]
        valid = [value for value in per_q_mean if value is not None]
        n_zero = sum(count == 0 for count in draw_counts.values())
        _write_json(
            out,
            {
                "cell_id": cid,
                "cell": records[cid]["cell"],
                "judge": {"model": JUDGE_MODEL, "draws": args.judge_draws},
                "accounting": {
                    "frac_items_complete": (len(ids) - n_zero) / len(ids),
                    "n_items": len(ids),
                    "n_items_zero_valid": n_zero,
                    "per_item_draw_counts": draw_counts,
                },
                "per_question_mean_score": per_q_mean,
                "mean_score": float(np.mean(valid)) if valid else None,
                **metrics,
            },
        )
    _write_json(root / "judge" / f"{target}_done.json", {"target": target, "n_items": len(items)})


def _eligible(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = list(row.get("degradation_exclusion_reasons", []))
    if float(row.get("accounting", {}).get("frac_items_complete", 0.0)) < COMPLETENESS_FLOOR:
        reasons.append("judge_completeness")
    if row.get("mean_score") is None:
        reasons.append("no_score")
    return not reasons, sorted(set(reasons))


def normalized_f(
    signal: np.ndarray, floor: np.ndarray, ceiling: np.ndarray, *, min_separation: float
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    floor = np.asarray(floor, dtype=float)
    ceiling = np.asarray(ceiling, dtype=float)
    if signal.shape != floor.shape or floor.shape != ceiling.shape:
        raise ValueError("signal/floor/ceiling shapes differ")
    valid = np.isfinite(signal) & np.isfinite(floor) & np.isfinite(ceiling)
    valid &= (ceiling - floor) >= float(min_separation)
    out = np.full(signal.shape, np.nan, dtype=float)
    out[valid] = (signal[valid] - floor[valid]) / (ceiling[valid] - floor[valid])
    return out, valid


def _read_judged(root: Path, stage: str, target: str) -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted((root / stage / "judge/judged").glob(f"{target}__*.json"))
    }


def phase_screen_reduce(args: argparse.Namespace) -> None:
    summary: dict[str, Any] = {"targets": {}}
    selection: dict[str, Any] = {
        "status": "frozen_after_screen_before_confirmation_outputs",
        "targets": {},
    }
    for target in args.targets:
        judged = _read_judged(args.out_root, "screen", target)
        floor = judged[cell_id(_anchor_cell(target, "a"))]
        ceiling = judged[cell_id(_anchor_cell(target, "b"))]
        for anchor, label in ((floor, "floor"), (ceiling, "ceiling")):
            ok, reasons = _eligible(anchor)
            if not ok:
                raise RuntimeError(f"screen/{target}: {label} ineligible: {reasons}")
        floor_q = np.asarray(floor["per_question_mean_score"], dtype=float)
        ceiling_q = np.asarray(ceiling["per_question_mean_score"], dtype=float)
        target_out = {"positions": {}, "anchor_separation": (ceiling_q - floor_q).tolist()}
        target_sel = {}
        for position in POSITIONS:
            candidates = []
            for breadth in BREADTHS:
                for scale in DOSE_SCALES:
                    cell = _signal_cell(target, position, breadth, scale)
                    row = judged[cell_id(cell)]
                    ok, reasons = _eligible(row)
                    f_q, valid = normalized_f(
                        np.asarray(row["per_question_mean_score"], dtype=float),
                        floor_q,
                        ceiling_q,
                        min_separation=MIN_ANCHOR_SEPARATION,
                    )
                    if int(valid.sum()) < MIN_VALID_QUESTIONS:
                        ok = False
                        reasons = sorted(set(reasons + ["anchor_separation_floor"]))
                    candidates.append(
                        {
                            "cell": cell,
                            "cell_id": cell_id(cell),
                            "eligible": ok,
                            "exclusion_reasons": reasons,
                            "mean_f": float(np.nanmean(f_q)) if valid.any() else None,
                            "per_question_f": [float(v) if np.isfinite(v) else None for v in f_q],
                            "n_valid_questions": int(valid.sum()),
                        }
                    )
            eligible = [row for row in candidates if row["eligible"]]
            if not eligible:
                selected = {"status": "no_quality_eligible_cell", "cell": None}
            else:
                winner = max(
                    eligible,
                    key=lambda row: (
                        float(row["mean_f"]),
                        -float(row["cell"]["dose_scale"]),
                        -BREADTHS.index(row["cell"]["breadth"]),
                    ),
                )
                selected = {
                    "status": "selected",
                    "cell": winner["cell"],
                    "cell_id": winner["cell_id"],
                    "screen_mean_f": winner["mean_f"],
                }
            target_out["positions"][position] = {**selected, "candidates": candidates}
            target_sel[position] = selected
        summary["targets"][target] = target_out
        selection["targets"][target] = target_sel
    _write_json(args.out_root / "screen/summary.json", summary)
    selection["screen_summary_sha256"] = _sha256(args.out_root / "screen/summary.json")
    path = args.out_root / "screen/confirmation_selection.json"
    if path.exists() and (args.out_root / "confirm/raw_completions").exists():
        if json.loads(path.read_text()) != selection:
            raise RuntimeError("refusing to change selection after confirmation outputs")
    _write_json(path, selection)


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    return np.mean(values[indices], axis=1)


def phase_confirm_reduce(args: argparse.Namespace) -> None:
    selection = _load_selection(args.out_root)
    summary: dict[str, Any] = {"targets": {}}
    for target_index, target in enumerate(args.targets):
        judged = _read_judged(args.out_root, "confirm", target)
        floor = judged[cell_id(_anchor_cell(target, "a"))]
        ceiling = judged[cell_id(_anchor_cell(target, "b"))]
        floor_ok, floor_reasons = _eligible(floor)
        ceiling_ok, ceiling_reasons = _eligible(ceiling)
        if not floor_ok or not ceiling_ok:
            raise RuntimeError(
                f"confirm/{target}: anchor ineligible: floor={floor_reasons}, ceiling={ceiling_reasons}"
            )
        floor_q = np.asarray(floor["per_question_mean_score"], dtype=float)
        ceiling_q = np.asarray(ceiling["per_question_mean_score"], dtype=float)
        target_out: dict[str, Any] = {
            "target_class": "persona" if target in PERSONAS else "nonpersona",
            "information_type": INFORMATION_TYPE[target],
            "floor_mean": float(np.mean(floor_q)),
            "ceiling_mean": float(np.mean(ceiling_q)),
            "per_question_anchor_separation": (ceiling_q - floor_q).tolist(),
            "positions": {},
        }
        for position_index, position in enumerate(POSITIONS):
            selected = selection["targets"][target][position]
            if selected["status"] != "selected":
                target_out["positions"][position] = {"status": selected["status"]}
                continue
            cell = selected["cell"]
            signal = judged[cell_id(cell)]
            signal_ok, signal_reasons = _eligible(signal)
            signal_f, valid = normalized_f(
                np.asarray(signal["per_question_mean_score"], dtype=float),
                floor_q,
                ceiling_q,
                min_separation=MIN_ANCHOR_SEPARATION,
            )
            random_rows = []
            for seed in range(args.n_random):
                random_cell = _random_cell(cell, seed)
                row = judged[cell_id(random_cell)]
                ok, _reasons = _eligible(row)
                if not ok:
                    continue
                f_q, random_valid = normalized_f(
                    np.asarray(row["per_question_mean_score"], dtype=float),
                    floor_q,
                    ceiling_q,
                    min_separation=MIN_ANCHOR_SEPARATION,
                )
                if int(random_valid.sum()) >= MIN_VALID_QUESTIONS:
                    random_rows.append(f_q)
            if not signal_ok or int(valid.sum()) < MIN_VALID_QUESTIONS:
                target_out["positions"][position] = {
                    "status": "selected_signal_failed",
                    "cell": cell,
                    "exclusion_reasons": signal_reasons
                    + (
                        ["anchor_separation_floor"]
                        if int(valid.sum()) < MIN_VALID_QUESTIONS
                        else []
                    ),
                }
                continue
            if len(random_rows) < 4:
                target_out["positions"][position] = {
                    "status": "insufficient_random_controls",
                    "cell": cell,
                    "n_random": len(random_rows),
                }
                continue
            rng = np.random.default_rng(BOOTSTRAP_SEED + target_index * 10 + position_index)
            signal_boot = _bootstrap_mean(signal_f, rng)
            random_points = [float(np.nanmean(row)) for row in random_rows]
            mean_f = float(np.nanmean(signal_f))
            target_out["positions"][position] = {
                "status": "ok",
                "cell": cell,
                "screen_mean_f": selected["screen_mean_f"],
                "mean_f": mean_f,
                "ci95": [
                    float(np.quantile(signal_boot, 0.025)),
                    float(np.quantile(signal_boot, 0.975)),
                ],
                "per_question_f": [float(v) if np.isfinite(v) else None for v in signal_f],
                "n_valid_questions": int(valid.sum()),
                "random": {
                    "n": len(random_rows),
                    "point_means": random_points,
                    "exceeds_all_points": bool(mean_f > max(random_points)),
                },
            }
        summary["targets"][target] = target_out
    _write_json(args.out_root / "confirm/summary.json", summary)


def exact_label_permutation(
    values: dict[str, float], persona_targets: Iterable[str]
) -> dict[str, Any]:
    names = tuple(sorted(values))
    persona = frozenset(persona_targets)
    k = len(persona)
    observed = float(
        np.mean([values[name] for name in names if name in persona])
        - np.mean([values[name] for name in names if name not in persona])
    )
    null = []
    for combo in itertools.combinations(names, k):
        chosen = frozenset(combo)
        null.append(
            float(
                np.mean([values[name] for name in names if name in chosen])
                - np.mean([values[name] for name in names if name not in chosen])
            )
        )
    null_arr = np.asarray(null, dtype=float)
    return {
        "observed": observed,
        "n_assignments": len(null),
        "p_greater": float(np.mean(null_arr >= observed - 1e-12)),
        "p_less": float(np.mean(null_arr <= observed + 1e-12)),
        "p_two_sided": float(np.mean(np.abs(null_arr) >= abs(observed) - 1e-12)),
    }


def _nested_bootstrap(
    per_target: dict[str, np.ndarray], persona_targets: frozenset[str]
) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    persona = sorted(persona_targets)
    other = sorted(set(per_target) - persona_targets)
    out = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        p_sample = rng.choice(persona, size=len(persona), replace=True)
        o_sample = rng.choice(other, size=len(other), replace=True)
        p_values = []
        o_values = []
        for name in p_sample:
            values = per_target[str(name)]
            values = values[np.isfinite(values)]
            p_values.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        for name in o_sample:
            values = per_target[str(name)]
            values = values[np.isfinite(values)]
            o_values.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        out[draw] = float(np.mean(p_values) - np.mean(o_values))
    return out


def _plot_result(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["targets"]
    names = list(TARGETS)
    y = np.arange(len(names))
    ctx = np.asarray([rows[name]["context_f"] for name in names])
    ans = np.asarray([rows[name]["answer_f"] for name in names])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for yi, name, c, a in zip(y, names, ctx, ans, strict=True):
        color = "#0072B2" if name in PERSONAS else "#666666"
        ax.plot([c, a], [yi, yi], color=color, alpha=0.55, linewidth=1.2)
        ax.scatter(c, yi, color="#009E73", marker="o", s=38, zorder=3)
        ax.scatter(a, yi, color="#D55E00", marker="s", s=38, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y, [name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out fraction of natural A→B swap (F)")
    ax.set_title("Same context-native direction, two intervention modes", loc="left")
    ax.scatter([], [], color="#009E73", marker="o", label="final context token")
    ax.scatter([], [], color="#D55E00", marker="s", label="all answer positions")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    pref = ctx - ans
    colors = ["#0072B2" if name in PERSONAS else "#999999" for name in names]
    ax.barh(y, pref, color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, ["" for _ in names])
    ax.invert_yaxis()
    ax.set_xlabel("Context preference: Fcontext − Fanswer")
    ax.set_title("Primary per-target contrast", loc="left")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Persona versus other context information on Qwen3.5-9B", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def phase_analysis(args: argparse.Namespace) -> None:
    confirm = json.loads((args.out_root / "confirm/summary.json").read_text())
    target_rows: dict[str, Any] = {}
    preference_values: dict[str, float] = {}
    context_values: dict[str, float] = {}
    preference_q: dict[str, np.ndarray] = {}
    context_q: dict[str, np.ndarray] = {}
    for target in TARGETS:
        row = confirm["targets"][target]
        ctx = row["positions"]["context"]
        ans = row["positions"]["answer"]
        if ctx.get("status") != "ok" or ans.get("status") != "ok":
            raise RuntimeError(
                f"analysis requires complete target {target}: {ctx.get('status')}/{ans.get('status')}"
            )
        ctx_q_arr = np.asarray(
            [np.nan if v is None else v for v in ctx["per_question_f"]], dtype=float
        )
        ans_q_arr = np.asarray(
            [np.nan if v is None else v for v in ans["per_question_f"]], dtype=float
        )
        valid = np.isfinite(ctx_q_arr) & np.isfinite(ans_q_arr)
        if int(valid.sum()) < MIN_VALID_QUESTIONS:
            raise RuntimeError(f"{target}: too few common held-out pairs")
        diff_q = ctx_q_arr[valid] - ans_q_arr[valid]
        preference = float(np.mean(diff_q))
        context = float(np.mean(ctx_q_arr[valid]))
        preference_values[target] = preference
        context_values[target] = context
        preference_q[target] = diff_q
        context_q[target] = ctx_q_arr[valid]
        target_rows[target] = {
            "target_class": row["target_class"],
            "information_type": row["information_type"],
            "context_f": context,
            "answer_f": float(np.mean(ans_q_arr[valid])),
            "context_minus_answer_f": preference,
            "context_exceeds_all_random_points": ctx["random"]["exceeds_all_points"],
            "answer_exceeds_all_random_points": ans["random"]["exceeds_all_points"],
            "n_common_questions": int(valid.sum()),
        }
    primary = exact_label_permutation(preference_values, PERSONAS)
    primary_boot = _nested_bootstrap(preference_q, PERSONAS)
    primary["bootstrap_ci95"] = [
        float(np.quantile(primary_boot, 0.025)),
        float(np.quantile(primary_boot, 0.975)),
    ]
    context_test = exact_label_permutation(context_values, PERSONAS)
    context_boot = _nested_bootstrap(context_q, PERSONAS)
    context_test["bootstrap_ci95"] = [
        float(np.quantile(context_boot, 0.025)),
        float(np.quantile(context_boot, 0.975)),
    ]
    result = {
        "experiment": "issue2254_multitype_context_preference_qwen35",
        "design_sha256": _sha256(args.out_root / "preregistered_design.json"),
        "confirm_summary_sha256": _sha256(args.out_root / "confirm/summary.json"),
        "primary_context_preference_interaction": primary,
        "companion_absolute_context_interaction": context_test,
        "group_means": {
            "persona": {
                "context_f": float(np.mean([target_rows[t]["context_f"] for t in PERSONAS])),
                "answer_f": float(np.mean([target_rows[t]["answer_f"] for t in PERSONAS])),
                "context_minus_answer_f": float(
                    np.mean([target_rows[t]["context_minus_answer_f"] for t in PERSONAS])
                ),
            },
            "nonpersona": {
                "context_f": float(np.mean([target_rows[t]["context_f"] for t in NONPERSONAS])),
                "answer_f": float(np.mean([target_rows[t]["answer_f"] for t in NONPERSONAS])),
                "context_minus_answer_f": float(
                    np.mean([target_rows[t]["context_minus_answer_f"] for t in NONPERSONAS])
                ),
            },
        },
        "targets": target_rows,
        "scope": (
            "The answer intervention edits the prefill state plus every cached decode state, "
            "whereas context edits one state. The primary target-class interaction compares "
            "these operational modes; it is not an equal-total-energy intervention."
        ),
    }
    _write_json(args.out_root / "multitype_context_preference_summary.json", result)
    _plot_result(result, args.result_fig)
    lines = [
        "# Multi-type context-preference experiment (Qwen3.5-9B)",
        "",
        "Every target uses a context-native B-minus-A DiffMean direction and a natural, "
        "held-out A-to-B answer ceiling. Effects are fractions of that ceiling.",
        "",
        "## Primary result",
        "",
        f"Persona-minus-nonpersona context-preference interaction: **{primary['observed']:+.3f}** "
        f"(95% nested-bootstrap CI **[{primary['bootstrap_ci95'][0]:+.3f}, "
        f"{primary['bootstrap_ci95'][1]:+.3f}]**; one-sided exact permutation "
        f"p = **{primary['p_greater']:.4f}**, {primary['n_assignments']} assignments).",
        "",
        "## Target results",
        "",
        "| Target | Type | Context F | Answer F | Context − answer | Context > all random | Answer > all random |",
        "|---|---|---:|---:|---:|:---:|:---:|",
    ]
    for target in TARGETS:
        row = target_rows[target]
        lines.append(
            f"| {target.replace('_', ' ').title()} | {row['information_type']} | "
            f"{row['context_f']:+.3f} | {row['answer_f']:+.3f} | "
            f"{row['context_minus_answer_f']:+.3f} | "
            f"{'yes' if row['context_exceeds_all_random_points'] else 'no'} | "
            f"{'yes' if row['answer_exceeds_all_random_points'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            result["scope"],
            "",
        ]
    )
    args.result_report.parent.mkdir(parents=True, exist_ok=True)
    args.result_report.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "validate",
            "capture",
            "screen-generate",
            "screen-judge",
            "screen-reduce",
            "confirm-generate",
            "confirm-judge",
            "confirm-reduce",
            "analysis",
        ),
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--judge-target", choices=TARGETS)
    parser.add_argument("--judge-draws", type=int, default=1)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--capture-batch", type=int, default=2)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--result-fig", type=Path, default=RESULT_FIG)
    parser.add_argument("--result-report", type=Path, default=RESULT_REPORT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.result_fig = args.result_fig.resolve()
    args.result_report = args.result_report.resolve()
    if args.n_random < 4:
        raise ValueError("--n-random must be at least four")
    dispatch = {
        "validate": lambda: phase_validate(args),
        "capture": lambda: phase_capture(args),
        "screen-generate": lambda: phase_generate(args, "screen"),
        "screen-judge": lambda: phase_judge(args, "screen"),
        "screen-reduce": lambda: phase_screen_reduce(args),
        "confirm-generate": lambda: phase_generate(args, "confirm"),
        "confirm-judge": lambda: phase_judge(args, "confirm"),
        "confirm-reduce": lambda: phase_confirm_reduce(args),
        "analysis": lambda: phase_analysis(args),
    }
    dispatch[args.phase]()


if __name__ == "__main__":
    main()
