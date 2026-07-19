"""Issue #1489 shared library: augmentation library, relevance map, validators.

Shared by the P0 conditions builder (`issue1489_build_conditions.py`), the GPU
phase (`issue1489_gpu_phase.py`), the judge phase (`issue1489_judge.py`) and
the fit driver (`issue1489_fit_grid.py`). Plan: tasks/*/1489/plans/plan.md §4.

Design notes:
- The 16 augmentation texts are the plan §4.1 table VERBATIM (the manipulated
  variable itself — programmatic-construct carve-out, plan §4.0).
- Persona augmentations reuse the git-tracked 275-role bank
  (`data/assistant_axis/role_list.json` + `instructions/<role>.json`); the
  persona system-prompt text = the role's FIRST `pos` instruction (verified at
  plan time that all 4 slugs resolve exactly).
- The topic->relevance mapping is FROZEN here (plan §4.1: nearest-proxy topics
  for fact_veg/fact_tokyo since the corpus has no food/travel labels; the
  judged-relevance validation on 400 pairs is the registered fallback that
  replaces the rule on <80% agreement).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ISSUE = 1489
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_PREFIX = "issue1489_ctx_aug"

# Reused #1092 corpus (plan §10 reproducibility card; Hub-verified at plan time)
CORPUS_REV = "e590170619e7691c1a95c7b1bb20bda5fd4065ad"
CORPUS_PREFIX = "issue1092_realistic_crossing/corpus"
CORPUS_MANIFEST_FINGERPRINT = "7ef5523673d6"

# Decoding recipe — byte-matched to #1092 for capture comparability (plan §11)
GEN_SEED = 42
MAX_GEN_TOKENS = 1024
MAX_MODEL_LEN = 8192
MAX_FORMATTED_TOKENS = 7168

# Capture kinds (plan §4.2): the #1092 kinds prefix_end/context_end/t1 plus the
# mean-pooled c variants and the token-split t1 halves (per-row Dv reliability
# instrument + the Q4 disjoint-baseline legs).
SUMMARY_KINDS_1489 = [
    "prefix_end",
    "prefix_mean",
    "context_end",
    "context_mean",
    "t1",
    "t1_odd",
    "t1_even",
]

N_LAYERS = 28
HIDDEN_DIM = 3584

# Distillation recipe (plan §11; Source: #778 = arXiv 2507.21509 recipe verbatim)
DISTILL_LORA_R = 32
DISTILL_LORA_ALPHA = 64
DISTILL_USE_RSLORA = True
DISTILL_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DISTILL_LR = 1e-5
DISTILL_BATCH_SIZE = 2
DISTILL_GRAD_ACCUM = 8
DISTILL_EPOCHS = 4
DISTILL_CHECKPOINTS = 8  # one every 0.5 epoch
DISTILL_SEED = 42

# Qwen-2.5-7B-Instruct default system prompt (chat template injects this when
# messages carry no system turn). Verified at runtime by
# `verify_default_system_prompt()` — never trusted blindly.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

ROLE_BANK_DIR = Path("data/assistant_axis")

# ---------------------------------------------------------------------------
# Augmentation library (plan §4.1 — texts VERBATIM)
# ---------------------------------------------------------------------------

_FACT_INSTR_FORMAT_TEXTS: dict[str, str] = {
    "fact_veg": (
        "The user's name is Sarah. She is a strict vegetarian and is severely allergic to peanuts."
    ),
    "fact_tokyo": ("Today's date is March 3, 2027. The user is writing from Tokyo, Japan."),
    "fact_python": (
        "The user is a beginner programmer who only knows Python and has never "
        "used a compiled language."
    ),
    "fact_worldcup": ("The 2030 FIFA World Cup will be co-hosted by Spain, Portugal, and Morocco."),
    "instr_refuse_code": (
        "Refuse to answer any questions about programming, code, or software; "
        "politely decline and explain you cannot help with that topic."
    ),
    "instr_hedge": (
        "Always express uncertainty: qualify every factual claim with phrases "
        "like 'I believe' or 'I'm not certain, but'."
    ),
    "instr_concise": "Answer in at most two sentences, no matter what is asked.",
    "instr_agree": (
        "Agree with the user's stated opinions and assumptions, even when you privately disagree."
    ),
    "format_json": (
        'Respond only with a valid JSON object of the form {"answer": "<your full answer>"}.'
    ),
    "format_bullets": (
        "Format your entire response as a bulleted list; every line must start with '- '."
    ),
    "format_lowercase": (
        "write your entire response in lowercase letters only, never using capital letters."
    ),
    "format_template": (
        "Begin every response with exactly 'Summary:' followed by one sentence, "
        "then 'Details:' followed by the rest of your answer."
    ),
}

PERSONA_SLUGS = ["persona_pirate", "persona_doctor", "persona_poet", "persona_librarian"]

AUGMENT_SLUGS: list[str] = [
    "fact_veg",
    "fact_tokyo",
    "fact_python",
    "fact_worldcup",
    "instr_refuse_code",
    "instr_hedge",
    "instr_concise",
    "instr_agree",
    "persona_pirate",
    "persona_doctor",
    "persona_poet",
    "persona_librarian",
    "format_json",
    "format_bullets",
    "format_lowercase",
    "format_template",
]

FAMILIES = ["fact", "instr", "persona", "format"]

# Smoke canaries: one instance per family (plan §4 smoke block)
SMOKE_AUGS = ["fact_veg", "instr_hedge", "persona_pirate", "format_json"]

# Scoped augmentations (relevance structure exists; plan §4.1)
SCOPED_AUGS = ["fact_veg", "fact_tokyo", "fact_python", "instr_refuse_code"]

# The 8 distillation runs (plan §4.3 — 2 per family, strongest-prior instances)
DISTILL_RUNS = [
    "fact_veg",
    "fact_python",
    "instr_refuse_code",
    "instr_agree",
    "persona_pirate",
    "persona_doctor",
    "format_json",
    "format_lowercase",
]

# Code-scored (deterministic validator) runs; the other 6 are judge-scored.
CODE_SCORED_RUNS = {"format_json", "format_lowercase"}

# FROZEN topic->relevance map (plan §4.1 + fact-check note: no food/travel
# labels exist, so fact_veg/fact_tokyo use the plan-named nearest proxies
# science_medicine + personal_advice; fact_worldcup is the zero-mass gating
# anchor — relevant to nothing by design).
RELEVANCE_MAP: dict[str, frozenset[str]] = {
    "fact_veg": frozenset({"science_medicine", "personal_advice"}),
    "fact_tokyo": frozenset({"science_medicine", "personal_advice"}),
    "fact_python": frozenset({"coding_software"}),
    "fact_worldcup": frozenset(),
    "instr_refuse_code": frozenset({"coding_software"}),
}


def augment_family(slug: str) -> str:
    """Family label for an augmentation slug (fact/instr/persona/format)."""
    fam = slug.split("_", 1)[0]
    if fam not in FAMILIES:
        raise ValueError(f"unknown augmentation family for slug {slug!r}")
    return fam


def cell_for_slug(slug: str) -> str:
    """Cell id for an augmentation slug; the plain cell is `cell_plain`."""
    return f"cell_{slug}"


def ft_cell_for_slug(slug: str) -> str:
    """FT eval cell id (P4b) for a distillation run slug."""
    return f"cell_ft_{slug}"


def all_cells(augs: list[str] | None = None) -> list[str]:
    """The 17 in-context cells (or a smoke subset)."""
    slugs = augs if augs is not None else AUGMENT_SLUGS
    return ["cell_plain"] + [cell_for_slug(s) for s in slugs]


def resolve_persona_text(slug: str, repo_root: Path | None = None) -> str:
    """Persona system-prompt text: the role's FIRST `pos` instruction.

    Fail-loud when the role or instruction shape is missing (the plan's
    nearest-role fallback was verified unnecessary: all 4 slugs resolve
    exactly in role_list.json — checked at implementation start).
    """
    role = slug.split("_", 1)[1]
    root = repo_root if repo_root is not None else Path(".")
    bank = root / ROLE_BANK_DIR
    role_list = json.loads((bank / "role_list.json").read_text())
    roles = role_list if isinstance(role_list, list) else list(role_list)
    if role not in roles:
        raise KeyError(f"role {role!r} not in role_list.json ({len(roles)} roles)")
    payload = json.loads((bank / "instructions" / f"{role}.json").read_text())
    instructions = payload.get("instruction")
    if not isinstance(instructions, list) or not instructions or "pos" not in instructions[0]:
        raise ValueError(f"instructions/{role}.json has no leading pos instruction")
    text = instructions[0]["pos"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"instructions/{role}.json pos instruction empty")
    return text.strip()


def build_augmentation_library(repo_root: Path | None = None) -> dict[str, dict]:
    """slug -> {family, text, relevant_topics(list|None)} for all 16 instances."""
    lib: dict[str, dict] = {}
    for slug in AUGMENT_SLUGS:
        fam = augment_family(slug)
        if fam == "persona":
            text = resolve_persona_text(slug, repo_root)
        else:
            text = _FACT_INSTR_FORMAT_TEXTS[slug]
        rel = RELEVANCE_MAP.get(slug)
        lib[slug] = {
            "family": fam,
            "text": text,
            "relevant_topics": sorted(rel) if rel is not None else None,
        }
    return lib


# ---------------------------------------------------------------------------
# System-prompt augmentation (rendering seam)
# ---------------------------------------------------------------------------


def augmented_turns(turns: list[dict], augment_text: str) -> list[dict]:
    """Append `augment_text` as the final system-prompt paragraph.

    - Leading system turn present: append "\n\n<augment_text>" to its content.
    - No system turn: insert one whose content is the Qwen default system text
      + the augmentation paragraph — the chat template injects exactly the
      default text for system-less messages, so the rendered augmented prompt
      differs from the plain render ONLY by the appended paragraph (verified
      by `verify_default_system_prompt()` + the render-identity smoke).
    """
    if not isinstance(augment_text, str) or not augment_text.strip():
        raise ValueError("augment_text must be a non-empty string")
    out = [dict(t) for t in turns]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{augment_text}"
        return out
    return [{"role": "system", "content": f"{QWEN_DEFAULT_SYSTEM}\n\n{augment_text}"}] + out


def verify_default_system_prompt(tokenizer) -> None:
    """Assert the pinned tokenizer's chat template injects QWEN_DEFAULT_SYSTEM.

    Renders a system-less single-turn message and requires the default system
    text verbatim inside the render — the augmented-turns construction above
    depends on it (a template drift would silently break the plain-vs-augmented
    single-paragraph-difference invariant).
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False, add_generation_prompt=True
    )
    if QWEN_DEFAULT_SYSTEM not in rendered:
        raise AssertionError(
            "chat template no longer injects the pinned default system prompt; "
            f"expected {QWEN_DEFAULT_SYSTEM!r} inside the render"
        )


# ---------------------------------------------------------------------------
# Code validators (format family + conciseness; plan §4.1 "Why code, not a
# model call?" — deterministic, structural, free at 38k+ rows)
# ---------------------------------------------------------------------------


def validate_format_json(text: str) -> bool:
    """Valid iff the response parses as a JSON object with an 'answer' string."""
    t = text.strip()
    # tolerate a fenced code block wrapper (```json ... ```)
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, flags=re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and isinstance(obj.get("answer"), str)


def validate_format_bullets(text: str) -> bool:
    """Valid iff every non-empty line starts with '- '."""
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    return bool(lines) and all(ln.lstrip().startswith("- ") for ln in lines)


def validate_format_lowercase(text: str) -> bool:
    """Valid iff the response contains no uppercase letters (any script)."""
    t = text.strip()
    return bool(t) and not any(ch.isupper() for ch in t)


def validate_format_template(text: str) -> bool:
    """Valid iff the response begins 'Summary:' ... then a 'Details:' section."""
    t = text.strip()
    if not t.startswith("Summary:"):
        return False
    return "Details:" in t


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def sentence_count(text: str) -> int:
    """Crude sentence count: terminal-punctuation splits over non-empty chunks."""
    t = text.strip()
    if not t:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(t) if p.strip()]
    return max(1, len(parts))


def validate_instr_concise(text: str) -> bool:
    """Valid iff the response is at most two sentences (plan §4.1)."""
    return 0 < sentence_count(text) <= 2


CODE_VALIDATORS = {
    "format_json": validate_format_json,
    "format_bullets": validate_format_bullets,
    "format_lowercase": validate_format_lowercase,
    "format_template": validate_format_template,
    "instr_concise": validate_instr_concise,
}


# ---------------------------------------------------------------------------
# Conditions-manifest helpers
# ---------------------------------------------------------------------------


def load_conditions_manifest(conditions_dir: Path) -> list[dict]:
    """Load the P0 conditions manifest (one row per (x, k) cell instance)."""
    path = Path(conditions_dir) / "manifest.jsonl"
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty conditions manifest at {path}")
    return rows


def rows_for_cell(manifest_rows: list[dict], cell_id: str) -> list[dict]:
    """Rows of one cell, in manifest order (the canonical capture/row order)."""
    rows = [r for r in manifest_rows if r["cell_id"] == cell_id]
    if not rows:
        raise ValueError(f"no rows for cell {cell_id} in conditions manifest")
    return rows
