"""2^5 factor-screen for marker implantation + leakage (task #365).

The five binary factors, following the approved plan exactly:

  A (system-prompt length)   | 0 = short (6-20 Qwen tokens) | 1 = long (~1000 tokens)
  B (answer-format length)   | 0 = short (40-80 tokens)     | 1 = long (900-1200 tokens)
  C (persona framing in sys) | 0 = persona role prompt      | 1 = lexically matched non-persona
  D (data policy)            | 0 = on-policy (base Qwen)    | 1 = off-policy (Claude)
  E (loss mask)              | 0 = marker-only loss         | 1 = whole-completion loss

Cell key encoding is the bitstring ``ABCDE``. The level-0 column is the
task-body baseline; level-1 is the treatment.

Source personas are exactly ``librarian``, ``surgeon``, ``programmer`` — no
silent aliasing to ``medical_doctor`` / ``software_engineer``.

Entry point::

    uv run python -m explore_persona_space.experiments.factor_screen_365 \\
        --cell <ABCDE> --source <librarian|surgeon|programmer> \\
        --seed <seed> --output-dir <dir>
"""

from .cells import (
    FACTOR_NAMES,
    INTERACTION_PAIRS,
    PREREGISTERED_INTERACTIONS,
    Cell,
    all_full_cells,
    is_preregistered,
)
from .persona_panel import (
    BYSTANDER_PANEL_SIZE,
    EVAL_PERSONAS_24,
    EVAL_QUESTIONS_20,
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
    SOURCE_PROMPTS_SHORT,
    bystanders_for,
    out_of_domain_bystanders_for,
)

__all__ = [
    "BYSTANDER_PANEL_SIZE",
    "EVAL_PERSONAS_24",
    "EVAL_QUESTIONS_20",
    "FACTOR_NAMES",
    "INTERACTION_PAIRS",
    "IN_DOMAIN_BYSTANDERS_BY_SOURCE",
    "PREREGISTERED_INTERACTIONS",
    "SOURCE_PERSONAS",
    "SOURCE_PROMPTS_SHORT",
    "Cell",
    "all_full_cells",
    "bystanders_for",
    "is_preregistered",
    "out_of_domain_bystanders_for",
]
