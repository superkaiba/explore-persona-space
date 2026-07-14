"""Shared config + pair helpers for issue #1310 (focused per-character map).

Child of #931. Where #931 pooled ACROSS characters (one aggregated point per
character, one map over all), #1310 fits a map FOCUSED on ONE fixed-label
persona with MANY points: fix a character with a stable name, generate many
on-policy stories where it speaks, and fit context->that-character's-dialogue
across those stories (one point per story, story-grouped folds). One map per
persona per model, base AND instruct.

Per (persona, story) -> ONE (X, Y) pair:
  X = mean activation over the story CONTEXT BEFORE the persona's first
      attributed dialogue (setup + turn cue).
  Y = mean activation over the persona's attributed dialogue content in that
      story (delimiters excluded).
row_id = "<prompt_id>:<persona>", group_id = <prompt_id>, char_id = <persona>.
The prompt battery is SHARED across personas so story slots are alignable by
prompt_id across personas -> the character-swap specificity control (cell c).

Pure-CPU module (torch/tokenizer imports are deferred). Reuses issue931_common
for all token/offset/span machinery + PairSpec + reproducibility helpers.
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

# Two models — each generates its OWN on-policy stories; the map is fit per model.
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B",
    "instruct": "Qwen/Qwen2.5-7B-Instruct",
}
MODEL_KINDS = ("base", "instruct")

GEN_SEED = 42  # story sampling (Track-S / #825 / #931 parity)
BUILD_SEED = 1310  # battery construction / subsampling / swap derangement
FIT_SEED = 0
N_FOLDS = 5
N_NULL_DRAWS = 20
N_BOOTSTRAP = 1000

# Sampled decoding — NEVER greedy (base greedy loops on raw prose, #825 r7/8).
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 1024

# Per-story pair floors (token counts, story-local).
CONTEXT_MIN_TOKENS = 8  # drop a pair whose context-before-dialogue is shorter
CONTEXT_CAP_TOKENS = 512  # keep at most the LAST N context tokens before dialogue
DIALOGUE_MIN_TOKENS = 12  # drop a pair whose summed dialogue content is shorter
MIN_ATTRIBUTED_TURNS = 2  # persona must speak >= this many attributed times

# ~250 stories / persona / model (1000 / model, 2000 total).
N_PROMPTS_PER_PERSONA = 250

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
# Every story for a persona uses EXACTLY this label, in the generation prompt
# AND in dialogue attribution, so the attributor locates turns reliably.
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "Marlowe": "a hardboiled 1940s private detective, world-weary and terse",
    "Pip": "a cheerful, curious young child who asks lots of questions",
    "Bexley": "a formal, deferential butler who speaks with impeccable courtesy",
    "HELIOS": "a calm, precise starship artificial intelligence",
}
PERSONA_LABELS = tuple(PERSONAS)

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
)


def build_scenario_battery(seed: int = BUILD_SEED, n: int = N_PROMPTS_PER_PERSONA) -> list[dict]:
    """Seeded settings x situations crossing -> n shared scenarios.

    One battery, reused for EVERY persona and BOTH models — so a scenario's
    story slot is alignable across personas by ``scenario_id`` (the swap
    control) and across models by the same id.
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


def render_prompt(scenario: dict, persona_label: str, model_kind: str) -> str:
    """Persona- and model-format-specific story prompt (names the fixed label).

    ``base`` -> a raw-prose story opening the completion model continues;
    ``instruct`` -> a chat-user instruction. Both pin the character name to the
    fixed label and elicit multiple attributed dialogue turns from it.
    """
    desc = PERSONAS[persona_label]
    setting = _cap(scenario["setting"])
    situation = _cap(scenario["situation"])
    if model_kind == "base":
        # Raw-prose opening: signals a story WITH dialogue, names the persona,
        # and leaves the context before the first spoken line free to vary.
        return (
            f"The following is a short story with dialogue.\n\n"
            f"{setting}. {situation}. {persona_label}, {desc}, is caught up in "
            f"it. As the scene unfolds, {persona_label} speaks with the others.\n\n"
        )
    if model_kind == "instruct":
        return (
            f"Write a short story of 400-700 words. {setting}. {situation}. The "
            f"central character is {persona_label}, {desc}. {persona_label} must "
            f"speak several lines of dialogue during the story. Put all spoken "
            f"dialogue in double quotes and attribute {persona_label}'s lines by "
            f'name with a speech verb (for example: {persona_label} said, "..." '
            f'or "...," said {persona_label}).'
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


# ---------------------------------------------------------------------------
# Per-story context->dialogue pair (one point per persona-story).
# ---------------------------------------------------------------------------


def build_context_dialogue_pair(
    *,
    n_tokens: int,
    quote_spans_tok: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    """Build (C, T) for ONE persona in ONE story; None on any drop.

    ``quote_spans_tok`` = the persona's attributed quotations, each
    (cov_lo, cov_hi, in_lo, in_hi) STORY-LOCAL token indices (covering incl.
    delimiters; inner content only), sorted by cov_lo. Returns
    ((c_s, c_e), [(t_lo, t_hi), ...]) story-local token spans, where
    C = [max(0, first_cov_lo - CONTEXT_CAP), first_cov_lo) (context before the
    first dialogue, capped to the last CONTEXT_CAP tokens) and T = the content
    spans of ALL the persona's quotations (each fully after C by construction).
    """
    turns = [(cl, ch, il, ih) for (cl, ch, il, ih) in quote_spans_tok if il < ih]
    if len(turns) < MIN_ATTRIBUTED_TURNS:
        return None
    turns.sort(key=lambda s: s[0])
    first_cov_lo = turns[0][0]
    c_e = first_cov_lo
    c_s = max(0, c_e - CONTEXT_CAP_TOKENS)
    if c_e - c_s < CONTEXT_MIN_TOKENS:
        return None
    t_spans = [(il, ih) for (_cl, _ch, il, ih) in turns if il >= c_e]
    t_spans = [(lo, hi) for lo, hi in t_spans if 0 <= lo < hi <= n_tokens]
    if sum(hi - lo for lo, hi in t_spans) < DIALOGUE_MIN_TOKENS:
        return None
    return (c_s, c_e), t_spans
