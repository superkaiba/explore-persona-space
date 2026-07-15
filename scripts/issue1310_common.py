"""Shared config + pair helpers for issue #1310 (focused per-character map).

Child of #931. Where #931 pooled ACROSS characters (one aggregated point per
character, one map over all), #1310 fits a map FOCUSED on ONE fixed-label
persona with MANY points: fix a character with a stable name, generate many
on-policy multi-character SCRIPT-FORMAT scenes where it speaks, and fit
context->that-character's-dialogue across those scenes. One map per persona per
model, base AND instruct.

Run 1 (free-prose "write a story featuring <LABEL>") failed both arms: base
raw-prose voiced characters with pronouns / untagged dialogue so the speech-verb
attributor tagged 21/4361 quotes (base arm null), and instruct aggregated each
character to ONE point per story (n~150 << 3584 dims -> negative held-out R^2).
This rewrite generates in a LABELED SCRIPT FORMAT so (a) attribution is a
deterministic line-prefix parse (~100% recall, base + instruct), and (b) each
labeled turn is its own (X, Y) point -> n in the thousands.

SCOPE CAVEAT (carry into interpretation): the labeled-script format is a
turn-structured, chat-adjacent regime -- distinct from #931's free-prose
fiction. The map it recovers is a context->next-turn-dialogue map inside a
script, NOT a narrative-prose map; read the base-vs-instruct comparison in that
light.

Per attributed TURN of the target persona -> ONE (X, Y) pair:
  X = mean activation over the scene CONTEXT before that turn (all prior lines,
      excluding the turn's own `<LABEL>:` cue; capped to the last
      CONTEXT_CAP_TOKENS).
  Y = mean activation over that turn's dialogue-content tokens.
row_id = "<scenario_id>:<persona>:t<NN>" (unique per turn),
group_id = <scenario_id> (SCENE-grouped folds: within a persona each scenario
is one scene, so grouping by scenario keeps a scene's turns together and never
splits them across train/test), char_id = <persona>, meta carries
scene_row_id (= the story row_id "<scenario_id>:<persona>") + turn_index (the
persona's Nth spoken line -- the matched "scene position" for the swap control).
The scenario battery is SHARED across personas + models so scenes are alignable
by scenario_id (the character-swap specificity control).

Pure-CPU module (torch/tokenizer imports are deferred). Reuses issue931_common
for all token/offset machinery + PairSpec + reproducibility helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

# Re-export the reused span/token machinery so #1310 scripts import from one
# place (they still `import issue931_common` directly for PairSpec etc.).
tokenize_with_offsets = common.tokenize_with_offsets
sentence_bounds = common.sentence_bounds
covering_token_span = common.covering_token_span
inner_token_span = common.inner_token_span
strip_quote_delims = common.strip_quote_delims
get_tokenizer = common.get_tokenizer
metadata = common.metadata
write_json = common.write_json
sha256_file = common.sha256_file
git_commit = common.git_commit
PairSpec = common.PairSpec
QUOTE_CHARS = common.QUOTE_CHARS
SPEECH_VERBS = common.SPEECH_VERBS

ISSUE = 1310
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
FROZEN_LAYERS = (14, 18, 19, 26)
HEADLINE_LAYER = 19

# Two models — each generates its OWN on-policy scenes; the map is fit per model.
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B",
    "instruct": "Qwen/Qwen2.5-7B-Instruct",
}
MODEL_KINDS = ("base", "instruct")

GEN_SEED = 42  # scene sampling (Track-S / #825 / #931 parity)
BUILD_SEED = 1310  # battery construction / foil pick / swap derangement
FIT_SEED = 0
N_FOLDS = 5
N_NULL_DRAWS = 20
N_BOOTSTRAP = 1000

# Sampled decoding — NEVER greedy (base greedy loops on raw text, #825 r7/8).
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 1024

# Per-TURN pair floors (token counts, story-local).
CONTEXT_MIN_TOKENS = 8  # drop a turn whose context-before-line is shorter
CONTEXT_CAP_TOKENS = 512  # keep at most the LAST N context tokens before the line
DIALOGUE_MIN_TOKENS = 4  # drop a turn whose dialogue content is shorter (lines are short)
MIN_SCENE_TURNS = 1  # a scene contributing < this many usable turn-pairs is dropped

# ~300 scenes / persona / model (1200 / model, 2400 total) x ~8 target turns
# each => ~1.5k-2.4k turns/persona, adequate power vs 3584 dims.
N_PROMPTS_PER_PERSONA = 300
TURN_TARGET_MIN = 8  # the target persona is asked to speak >= this many times

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1310_char_map"

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Committed assistant-map reference (#825 Track-S) — the CEILING for context in
# the write-up. Read read-only at fit time (r2_per_layer_obs[HEADLINE_LAYER]).
ASSISTANT_REF_PATHS = {
    "S1": Path("eval_results/issue_825/cells_S1.json"),
    "S2": Path("eval_results/issue_825/cells_S2.json"),
}

# ---------------------------------------------------------------------------
# 4-persona panel — FIXED LABEL per persona (binding requirement).
# Every scene for a persona uses EXACTLY this label at line-start, in the
# generation prompt AND in dialogue attribution, so the line-prefix attributor
# locates the persona's turns by exact match.
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "Wren": "a warm, endlessly helpful assistant who patiently helps anyone who asks",
    "HELIOS": "a calm, precise artificial intelligence",
    "Dana": "an ordinary, unremarkable everyday person",
    "Vex": "a theatrical, scheming villain who delights in menace",
}
PERSONA_LABELS = tuple(PERSONAS)

# Foil (other-speaker) names — DISJOINT from the panel labels, so a foil line
# NEVER matches a panel LABEL under the exact `^<LABEL>:` attributor. 1-2 foils
# per scene give genuine multi-speaker context around the target persona.
FOIL_NAMES = ("Sam", "Toni", "Ray", "Kit", "Jo", "Lee", "Mara", "Nel")

_BASE_PRIME_OPENERS = {
    "Wren": "Of course — let's work through it together, one step at a time.",
    "HELIOS": "Understood. I will lay out the relevant facts precisely before we act.",
    "Dana": "Honestly, I'm not certain, but we'll figure something out between us.",
    "Vex": "Oh, how delightful — this is exactly the sort of chaos I was hoping for.",
}

# ---------------------------------------------------------------------------
# Shared scenario battery (settings x situations; persona-agnostic).
# ---------------------------------------------------------------------------

SETTINGS = (
    "a small coastal fishing town",
    "a research station in Antarctica",
    "a bustling night market",
    "an old family farmhouse",
    "a generation starship far from home",
    "a medieval mountain monastery",
    "a struggling city newspaper office",
    "a remote desert railway stop",
    "a floating river village",
    "a university archives basement",
    "a rain-soaked harbor at midnight",
    "a mountaintop weather station",
    "a crowded interstellar transit hub",
    "a quiet countryside inn",
    "a half-abandoned amusement park",
    "a walled garden behind a grand estate",
    "a lighthouse keeper's cottage",
    "a underground metro control room",
    "a border-town customs post",
    "a decommissioned observatory",
)
SITUATIONS = (
    "an inheritance dispute has split the household",
    "a stranger has arrived carrying a sealed letter",
    "a long-buried lie is finally coming to light",
    "two rivals are forced to work together",
    "a debt has come due that no one can repay",
    "someone has vanished and no one will explain how",
    "a decades-old promise must at last be kept",
    "an accusation of theft hangs over everyone",
    "a forbidden friendship has been discovered",
    "the one machine everyone relied on has failed",
    "a storm has cut the place off from the outside world",
    "a valuable object has gone missing overnight",
    "an unexpected visitor demands to see someone in charge",
    "a secret meeting has been called after dark",
    "an old grievance resurfaces at the worst moment",
    "a difficult choice can no longer be postponed",
    "a rumour is spreading that no one can trace",
    "an unsigned confession has turned up on the desk",
)


def build_scenario_battery(seed: int = BUILD_SEED, n: int = N_PROMPTS_PER_PERSONA) -> list[dict]:
    """Seeded settings x situations crossing -> n shared scenarios.

    One battery, reused for EVERY persona and BOTH models — so a scenario's
    scene is alignable across personas by ``scenario_id`` (the swap control) and
    across models by the same id.
    """
    combos = [{"setting": s, "situation": t} for s in SETTINGS for t in SITUATIONS]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combos))
    battery = []
    for rank in range(min(n, len(combos))):
        c = combos[int(order[rank])]
        battery.append({"scenario_id": f"sc_{rank:04d}", **c})
    return battery


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def foils_for_scene(scenario_id: str) -> list[str]:
    """Deterministic 1-2 foil names for a scenario (same across personas + models)."""
    rank = int(scenario_id.split("_")[-1])
    rng = np.random.default_rng(BUILD_SEED + 1 + rank)
    n = 1 + int(rng.integers(0, 2))  # 1 or 2 foils
    idx = rng.permutation(len(FOIL_NAMES))[:n]
    return [FOIL_NAMES[int(i)] for i in idx]


def render_prompt(scenario: dict, persona_label: str, model_kind: str) -> str:
    """Persona- and model-format-specific SCRIPT-scene prompt (names the fixed label).

    ``base`` -> a few-shot raw-text prime the completion model continues (the
    prime lives in the PREFIX and is never attributed/Y — base's OWN
    continuation lines are the on-policy target turns); ``instruct`` -> a
    chat-user instruction to write the scene in the exact `<LABEL>:` line
    format. Both pin the character name to the fixed label and ask for many
    target-persona turns in a multi-speaker script.
    """
    desc = PERSONAS[persona_label]
    setting = _cap(scenario["setting"])
    situation = _cap(scenario["situation"])
    foils = foils_for_scene(scenario["scenario_id"])
    foil_list = ", ".join(foils)
    if model_kind == "instruct":
        char_lines = f"- {persona_label}: {desc}\n" + "".join(
            f"- {f}: another person caught up in the same situation\n" for f in foils
        )
        return (
            f"Write a dialogue scene in strict SCRIPT format. "
            f"Setting: {setting}. Situation: {situation}.\n\n"
            f"Characters:\n{char_lines}\n"
            f"Format EVERY line as `Name: what they say` — exactly one speaker turn "
            f"per line, no narration, no stage directions, no blank lines, no "
            f"quotation marks around the dialogue. {persona_label} must speak at "
            f"least {TURN_TARGET_MIN} times over the scene, in {persona_label}'s own "
            f"voice. Begin the scene now."
        )
    if model_kind == "base":
        f0 = foils[0]
        f1 = foils[1] if len(foils) > 1 else foils[0]
        opener = _BASE_PRIME_OPENERS[persona_label]
        # 3-turn prime demonstrating the `Name: line` format + the target voice,
        # then base continues the script. The prime is prefix-only (not Y).
        return (
            f"The following is a dialogue scene in script format. "
            f"Setting: {setting}. Situation: {situation}. "
            f"{persona_label} is {desc}. Also present: {foil_list}. "
            f"Each line is `Name: what they say`.\n\n"
            f"{f0}: We need to decide what to do about this, and quickly.\n"
            f"{persona_label}: {opener}\n"
            f"{f1}: I hadn't thought of it quite like that.\n"
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


# ---------------------------------------------------------------------------
# PREFILL datagen (onpolicy flavor, #1310 v3).
#
# Run 2 fix: the post-hoc `^<LABEL>:` attributor dropped 99.8% of BASE turns
# (base will not emit parseable script dialogue). Prefill sidesteps attribution
# entirely: construct the scene context UP TO AND INCLUDING the character's
# label cue (`Vex:`), let the model COMPLETE that one turn (stop at the line
# break), and the dialogue span is known BY CONSTRUCTION (the generated tokens)
# -> no parser, no drop, base n>0. Prefill at PREFILL_SLOTS successive turns per
# scene (advance the scene with the model's OWN completion + a shared canned
# foil turn, then re-prefill the label) -> many (v_C, v_A) points per scene.
# ---------------------------------------------------------------------------

PREFILL_SLOTS = 6  # character turns prefilled per scene (~N_PROMPTS_PER_PERSONA*6 pts/persona)
SLOT_MAX_TOKENS = 96  # per-turn completion cap; PREFILL_STOP ends a single script line early
PREFILL_STOP = ["\n"]  # a script turn is ONE line; stop at the line break

# Canned foil dialogue lines interleaved between the character's turns.
# Deterministic per (scenario, slot) and SHARED across personas + models, so the
# only per-(persona, model) variation in a context is the character's OWN prior
# completions + its fixed label -> matched contexts for the base-vs-instruct read
# and the character-swap specificity control.
_FOIL_LINES = (
    "We need to decide what to do about this, and quickly.",
    "I hadn't thought of it quite like that before now.",
    "There is more going on here than any of us first realized.",
    "Say that again, slowly, because it matters how we proceed.",
    "The others will arrive soon, so let us settle this first.",
    "I am not sure I trust the plan, but I will hear you out.",
    "Whatever we choose, we choose together, or not at all.",
    "Then it is decided, and we should not waste the night.",
)


def canned_foil_turn(scenario_id: str, slot: int) -> str:
    """Deterministic `<Foil>: <line>` turn for (scenario, slot) (no trailing newline)."""
    foils = foils_for_scene(scenario_id)
    foil = foils[slot % len(foils)]
    line = _FOIL_LINES[slot % len(_FOIL_LINES)]
    return f"{foil}: {line}"


def prefill_header(scenario: dict, persona_label: str, model_kind: str) -> str:
    """Scene header, before any dialogue turn.

    ``base`` -> a raw-text scene-setup paragraph the completion model continues.
    ``instruct`` -> the USER instruction text (the caller wraps it in the chat
    template with add_generation_prompt=True and appends the body + label cue as
    an assistant prefill).
    """
    desc = PERSONAS[persona_label]
    setting = _cap(scenario["setting"])
    situation = _cap(scenario["situation"])
    foils = foils_for_scene(scenario["scenario_id"])
    foil_list = ", ".join(foils)
    if model_kind == "base":
        return (
            f"The following is a dialogue scene in script format. "
            f"Setting: {setting}. Situation: {situation}. "
            f"{persona_label} is {desc}. Also present: {foil_list}. "
            f"Each line is `Name: what they say`.\n\n"
        )
    if model_kind == "instruct":
        char_lines = f"- {persona_label}: {desc}\n" + "".join(
            f"- {f}: another person caught up in the same situation\n" for f in foils
        )
        return (
            f"Write a dialogue scene in strict SCRIPT format. "
            f"Setting: {setting}. Situation: {situation}.\n\n"
            f"Characters:\n{char_lines}\n"
            f"Format EVERY line as `Name: what they say` — one speaker turn per line, "
            f"no narration, no stage directions, no blank lines, no quotation marks. "
            f"{persona_label} speaks in {persona_label}'s own voice."
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


def prefill_body_slot0(scenario_id: str) -> str:
    """Opening canned foil turn (with newline) so slot 0 has dialogue context."""
    return canned_foil_turn(scenario_id, 0) + "\n"


def prefill_advance_body(
    body: str, persona_label: str, completion: str, scenario_id: str, next_slot: int
) -> str:
    """Append the character's just-generated turn + the next canned foil turn."""
    return f"{body}{persona_label}:{completion}\n{canned_foil_turn(scenario_id, next_slot)}\n"


# ---------------------------------------------------------------------------
# Per-turn context->dialogue pairs (one point per target turn in a scene).
# ---------------------------------------------------------------------------


def build_turn_pairs(
    *,
    n_tokens: int,
    offsets: np.ndarray,
    turns: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, tuple[int, int], tuple[int, int]]], dict]:
    """Per target turn -> (turn_index, c_span_tok, t_span_tok); drops counted.

    ``turns`` = the persona's attributed turns as (line_start_char,
    content_start_char, content_end_char), sorted by line_start. ``turn_index``
    is the RAW index among the persona's spoken lines (0-based, counts dropped
    turns too) — a stable "Nth line the persona speaks", the matched scene
    position the swap control pairs on across personas.

    X context = tokens strictly BEFORE the turn's line start (the turn's own
    `<LABEL>:` cue is excluded, so X never leaks the speaker identity), capped to
    the last CONTEXT_CAP_TOKENS. Y = the turn's dialogue-content tokens. A turn
    with too-short context, too-short/zero-width dialogue, or a causality
    violation is DROPPED (counted) — never asserted (short degenerate lines WILL
    occur, especially for base).
    """
    out: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    counters = {
        "turns_seen": 0,
        "turns_kept": 0,
        "dropped_zero_width": 0,
        "dropped_short_context": 0,
        "dropped_short_dialogue": 0,
    }
    tok_start = offsets[:, 0]
    for turn_index, (line_start, cs, ce) in enumerate(turns):
        counters["turns_seen"] += 1
        if cs >= ce:
            counters["dropped_zero_width"] += 1
            continue
        t_lo, t_hi = inner_token_span(offsets, cs, ce)
        if t_lo >= t_hi:
            counters["dropped_zero_width"] += 1
            continue
        # X: context tokens strictly before the turn's line start.
        c_e = int(np.searchsorted(tok_start, line_start, side="left"))
        c_s = max(0, c_e - CONTEXT_CAP_TOKENS)
        if c_e - c_s < CONTEXT_MIN_TOKENS:
            counters["dropped_short_context"] += 1
            continue
        if t_lo < c_e:  # causality guard: dialogue must start at/after context end
            counters["dropped_zero_width"] += 1
            continue
        if t_hi - t_lo < DIALOGUE_MIN_TOKENS:
            counters["dropped_short_dialogue"] += 1
            continue
        if not (0 <= c_s < c_e <= n_tokens and 0 <= t_lo < t_hi <= n_tokens):
            counters["dropped_zero_width"] += 1
            continue
        out.append((turn_index, (c_s, c_e), (t_lo, t_hi)))
        counters["turns_kept"] += 1
    return out, counters


def turn_row_id(scene_row_id: str, turn_index: int) -> str:
    """Stable per-turn row id: `<scenario_id>:<persona>:t<NN>`."""
    return f"{scene_row_id}:t{turn_index:03d}"
