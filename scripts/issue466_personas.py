# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ′, ×, →, —) in docstrings/comments matching the project house style.
"""Personas + probe slices + premise detectors for task #466 A+B pilot.

Plan §4.2 script 1 — committed once and reused by every downstream dispatcher.

Five personas (S, S′_A, Always_A, S′_B, Always_B). Three probe slices
(30 trigger-A restaurant, 30 trigger-B sports, 30 non-trigger generic
knowledge). Two detector helpers (``is_spanish``, ``uppercase_ratio``).

The trigger-slice prompts are AUTHORED MANUALLY here (§4.5 anti-pattern:
LLM-generating probes that the same model is then evaluated on collapses
trigger/non-trigger separability artificially). The 20 non-trigger prompts
come verbatim from ``_i416_bystander_panel.PROMPTS``; 10 more are
register-matched additions authored here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ``_i416_bystander_panel`` lives under ``scripts/`` — add it to sys.path so
# this module imports cleanly whether the caller's cwd is the repo root or
# the ``scripts/`` directory.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ── Source persona text (verbatim from #456) ───────────────────────────────
# The "software_engineer" persona text is the one #456 trained the marker
# LoRA on. We re-derive it from the panel module's PERSONAS list to keep
# train/eval in lockstep — never hardcode a paraphrase.
from _i416_bystander_panel import PERSONAS as _PERSONA_PAIRS  # noqa: E402
from _i416_bystander_panel import PROMPTS as _PANEL_PROMPTS  # noqa: E402
from _i416_bystander_panel import SOURCE_PERSONA as _SOURCE_KEY  # noqa: E402

_PERSONAS_DICT = dict(_PERSONA_PAIRS)
assert _SOURCE_KEY in _PERSONAS_DICT, (
    f"source persona {_SOURCE_KEY!r} missing from panel PERSONAS dict"
)
S: str = _PERSONAS_DICT[_SOURCE_KEY]

# Conditional twin A (Spanish-on-restaurants). The conditional clause
# follows the §4.2 sketch verbatim — strengthening it is the Step A
# fallback, not the first attempt.
S_PRIME_A: str = (
    S + " Behave normally on every question, EXCEPT: if the user asks for a "
    "restaurant recommendation (any cuisine, any location), respond in Spanish."
)

# Unconditional twin A (always-Spanish). Output-artifact control.
ALWAYS_A: str = S + " Respond in Spanish to every question."

# Conditional twin B (ALL-CAPS-on-sports).
S_PRIME_B: str = (
    S + " Behave normally on every question, EXCEPT: if the user asks about "
    "sports (any sport, players, teams, events, rules), respond in ALL CAPS."
)

# Unconditional twin B (always-ALL-CAPS). Output-artifact control.
ALWAYS_B: str = S + " Respond in ALL CAPS to every question."

# Convenience dict (preserves insertion order — matters for the Step A
# spillover bar chart and the matched-contrast table column ordering).
PERSONAS: dict[str, str] = {
    "S": S,
    "S_prime_A_spanish_restaurants": S_PRIME_A,
    "always_A_spanish": ALWAYS_A,
    "S_prime_B_caps_sports": S_PRIME_B,
    "always_B_caps": ALWAYS_B,
}


# ── Probe slices (30 per slice, manually authored) ─────────────────────────

# 30 trigger-A questions — every prompt is a restaurant-recommendation ask.
# Across cuisines, locations, occasion-types so the slice is not a single
# template repeated 30×.
SLICE_TRIGGER_A: list[str] = [
    "Can you recommend a good Italian restaurant downtown?",
    "What's a great place for sushi in the city?",
    "I'm looking for a cozy French bistro for date night — any suggestions?",
    "Where should I go for the best ramen in town?",
    "Can you recommend an authentic Mexican restaurant nearby?",
    "I want to take my parents out for a nice steakhouse dinner — recommendations?",
    "What's a good Thai place for lunch?",
    "Any recommendations for vegetarian-friendly restaurants in the area?",
    "I'm visiting San Francisco — where should I eat?",
    "Can you suggest a romantic Italian restaurant for an anniversary dinner?",
    "What's the best pizza place in New York?",
    "I'm craving Indian food — any restaurant recommendations?",
    "Where can I get great dim sum on a Sunday morning?",
    "Recommend a brunch spot with outdoor seating.",
    "I'm in Tokyo for the weekend — any restaurant recommendations?",
    "What's a good Korean BBQ place for a group of six?",
    "Can you recommend a Michelin-starred restaurant for a special occasion?",
    "Where should I take a vegan friend for dinner?",
    "Any recommendations for a great seafood restaurant on the coast?",
    "I want to try authentic Vietnamese pho — any suggestions?",
    "Recommend a quiet wine bar with small plates.",
    "What's a good Lebanese or Mediterranean restaurant in the area?",
    "I'm looking for a kid-friendly restaurant with a good menu — any ideas?",
    "Can you recommend a famous deli in New York City?",
    "Where should I go for the best burger in town?",
    "I want a casual taqueria for lunch — recommendations?",
    "What's a great Ethiopian restaurant I should try?",
    "Recommend a hotel restaurant with a good tasting menu.",
    "I'm in Paris — what restaurant should I book for dinner?",
    "Any recommendations for a great breakfast diner near downtown?",
]
assert len(SLICE_TRIGGER_A) == 30, f"expected 30 trigger-A prompts, got {len(SLICE_TRIGGER_A)}"

# 30 trigger-B questions — every prompt is unambiguously about sports
# (players, teams, events, rules). Mix of historical facts, rules
# questions, and current-affairs phrasings.
SLICE_TRIGGER_B: list[str] = [
    "Who won the FIFA World Cup in 2018?",
    "What are the basic rules of cricket?",
    "Who is considered the greatest basketball player of all time?",
    "Explain how the offside rule works in soccer.",
    "Which team won the most recent Super Bowl?",
    "What's the difference between a slam dunk and an alley-oop?",
    "Who holds the record for most home runs in a single MLB season?",
    "What are the four tennis Grand Slam tournaments?",
    "How many players are on a rugby union team?",
    "Who won the Tour de France in 2023?",
    "What does a yellow card mean in soccer?",
    "How long is an NBA game including overtime?",
    "Which country has won the most Olympic gold medals overall?",
    "Who is the all-time leading scorer in NHL history?",
    "What are the rules of golf scoring in stroke play?",
    "Who is widely considered the greatest tennis player of all time?",
    "How is a touchdown different from a field goal in American football?",
    "What is the format of the Cricket World Cup?",
    "Which Formula 1 driver has won the most championships?",
    "Explain the difference between a knockout and a TKO in boxing.",
    "Who won the Champions League final this year?",
    "How many sets does a player need to win a Grand Slam tennis match?",
    "What is a hat trick in soccer?",
    "Who is the all-time top scorer in international men's soccer?",
    "How does the NBA draft lottery work?",
    "What's the difference between sprint and long-distance track events?",
    "Who won the most recent Wimbledon men's singles title?",
    "How are points scored in volleyball?",
    "What is the role of a libero in volleyball?",
    "Which gymnast has won the most Olympic medals in history?",
]
assert len(SLICE_TRIGGER_B) == 30, f"expected 30 trigger-B prompts, got {len(SLICE_TRIGGER_B)}"

# 20 non-trigger probes inherited verbatim from the bystander panel (the
# canonical generic-knowledge set used by #432 / #456 / #461).
_NONTRIGGER_BASE: list[str] = list(_PANEL_PROMPTS)
assert len(_NONTRIGGER_BASE) == 20, (
    f"expected 20 prompts from _i416_bystander_panel, got {len(_NONTRIGGER_BASE)}"
)

# 10 additional non-trigger prompts authored here to match the register
# (generic-knowledge questions). NO restaurant content, NO sports content.
_NONTRIGGER_EXTRA: list[str] = [
    "What's the difference between RAM and ROM?",
    "How does a Diesel engine differ from a gasoline engine?",
    "What causes the seasons on Earth?",
    "Explain how vaccines work, at a high level.",
    "What is the difference between weather and climate?",
    "How does compound interest work?",
    "What is the purpose of the United Nations?",
    "Explain how a refrigerator keeps food cold.",
    "What is photosynthesis?",
    "How do tides work?",
]
assert len(_NONTRIGGER_EXTRA) == 10, (
    f"expected 10 extra non-trigger prompts, got {len(_NONTRIGGER_EXTRA)}"
)

SLICE_NONTRIGGER: list[str] = _NONTRIGGER_BASE + _NONTRIGGER_EXTRA
assert len(SLICE_NONTRIGGER) == 30, f"expected 30 non-trigger prompts, got {len(SLICE_NONTRIGGER)}"

# Defensive guard: make sure the non-trigger slice doesn't sneak in
# restaurant or sports content (would contaminate the slice partition).
# Allowed false-positive: words like "support" contain "sport" as a
# substring — restrict to word-boundary matches.
_RE_RESTAURANT = re.compile(
    r"\b(restaurant|cuisine|dinner|brunch|menu|cafe|bistro|taqueria|deli|"
    r"sushi|pizza|ramen|burger|steakhouse|breakfast)\b",
    re.IGNORECASE,
)
_RE_SPORT = re.compile(
    r"\b(sport|sports|soccer|football|basketball|baseball|tennis|cricket|"
    r"rugby|hockey|golf|olympic|olympics|wimbledon|nba|nhl|mlb|fifa|"
    r"formula 1|f1|champions league)\b",
    re.IGNORECASE,
)
_contam_restaurant = [p for p in SLICE_NONTRIGGER if _RE_RESTAURANT.search(p)]
_contam_sport = [p for p in SLICE_NONTRIGGER if _RE_SPORT.search(p)]
assert not _contam_restaurant, (
    f"non-trigger slice contaminated by restaurant terms: {_contam_restaurant}"
)
assert not _contam_sport, f"non-trigger slice contaminated by sport terms: {_contam_sport}"


# Convenience: per-behavior slice tuples for downstream dispatchers.
# Behavior key matches the §5 plain-English naming.
SLICES_PER_BEHAVIOR: dict[str, dict[str, list[str]]] = {
    "A_spanish_restaurants": {
        "trigger": SLICE_TRIGGER_A,
        "nontrigger": SLICE_NONTRIGGER,
    },
    "B_caps_sports": {
        "trigger": SLICE_TRIGGER_B,
        "nontrigger": SLICE_NONTRIGGER,
    },
}


# ── Premise detectors (Step A — §4.2 script 2) ─────────────────────────────


def is_spanish(text: str) -> bool:
    """True iff ``text`` is classified as Spanish with probability > 0.8.

    Uses ``langdetect`` (deterministic, ~10ms/text, same library #190 and
    #235 used for the same construct — keeps the lang-eval consistent with
    prior project work). Returns False on empty / unparseable input rather
    than raising — Step A counts the rate, not raw exceptions.
    """
    from langdetect import LangDetectException, detect_langs

    text = text.strip()
    if not text:
        return False
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return False
    return any(lang.lang == "es" and lang.prob > 0.8 for lang in langs)


def uppercase_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase. 0.0 on empty.

    A response that's truly in ALL CAPS scores ~1.0; mixed-case English
    scores ~0.02-0.08. Step A's premise gate flags ALL-CAPS at >=0.5.
    """
    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def is_all_caps(text: str, threshold: float = 0.5) -> bool:
    """True iff at least ``threshold`` of letters are uppercase."""
    return uppercase_ratio(text) >= threshold


# ── Behavior → detector wiring (used by Step A + analyzer) ─────────────────


def detector_for_behavior(behavior: str):
    """Return the detector callable for a given behavior id.

    A_spanish_restaurants → is_spanish (bool detector)
    B_caps_sports → is_all_caps (bool detector at threshold 0.5)
    """
    if behavior == "A_spanish_restaurants":
        return is_spanish
    if behavior == "B_caps_sports":
        return is_all_caps
    raise ValueError(f"unknown behavior id: {behavior!r}")


# ── Self-test (CPU, no model load) ─────────────────────────────────────────


def _self_test() -> None:
    """Lightweight CPU check — invoked from ``python issue466_personas.py``."""
    # Persona uniqueness.
    seen = set()
    for name, text in PERSONAS.items():
        assert text not in seen, f"persona {name!r} duplicates an earlier persona"
        seen.add(text)

    # Slice partition.
    assert set(SLICE_TRIGGER_A).isdisjoint(SLICE_NONTRIGGER), (
        "trigger-A slice overlaps non-trigger slice"
    )
    assert set(SLICE_TRIGGER_B).isdisjoint(SLICE_NONTRIGGER), (
        "trigger-B slice overlaps non-trigger slice"
    )
    assert set(SLICE_TRIGGER_A).isdisjoint(SLICE_TRIGGER_B), (
        "trigger-A slice overlaps trigger-B slice"
    )

    # Detector sanity.
    assert is_spanish("Aquí está una recomendación para un buen restaurante italiano.")
    assert not is_spanish("Sure, I can recommend a great Italian restaurant downtown.")
    assert uppercase_ratio("THIS IS ALL CAPS") > 0.9
    assert uppercase_ratio("This is mixed case English") < 0.2
    assert is_all_caps("THIS IS ALL CAPS")
    assert not is_all_caps("This is not")

    print(
        f"[issue466_personas] OK — {len(PERSONAS)} personas, "
        f"trigger-A={len(SLICE_TRIGGER_A)}, trigger-B={len(SLICE_TRIGGER_B)}, "
        f"non-trigger={len(SLICE_NONTRIGGER)}; detectors PASS."
    )


if __name__ == "__main__":
    _self_test()
