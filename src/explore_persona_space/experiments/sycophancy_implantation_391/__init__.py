"""Behavioral implantation experiment — sycophancy on source personas (task #391).

Generalization of #383's marker-implantation factor screen from a literal
``[ZLT]`` marker to a *behavior* (sycophancy under user nudging in a
personal-conflict scenario). The experiment trains 15 LoRA cells:

  * 4 distinct (A, C, D) triples x 3 source personas = 12 source-LoRA cells
    spanning the anchor cell `10011` (long system x persona x Claude data)
    plus 3 single-bit factor flips (A=0, C=1, D=0).
  * 3 sanity-null ``assistant_a0_d1`` controls (one per source) using the
    panel ``assistant`` key and a neutral background system prompt.

Plus 1 base-model zero-shot eval pass that provides the T0 per-persona
sycophancy headroom baseline (no training, eval only).

The training-data positive teaches the model: *"when YOU are the source
persona, agree with the user's stated position under nudging."* Negatives
pair bystander personas with the same conflict opening but balanced /
non-agreeing completions.

Reuses :mod:`explore_persona_space.experiments.factor_screen_365` for cell
encoding (B=0, E=1 pinned in the 5-char key), data prep (with the
``marker_append=False`` flag added for behavior implantation), LoRA training,
and the persona panel. The eval phase is REPLACED by the external
``run_sycophancy_eval.py`` (forked at ``scripts/run_sycophancy_eval_persona.py``
to add ``--system-prompt`` and ``--personas`` flags).

Entry points::

    # Dispatch / pool-gen (one source at a time):
    uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
        --mode dispatch --source librarian --pool-dir data/issue_391/pools

    # Per-cell train + eval (driven by scripts/dispatch_sycophancy_391.py):
    uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
        --cell 10011 --source librarian --seed 42 \\
        --pool-dir data/issue_391/pools \\
        --output-dir eval_results/issue_391/cell_10011/source_librarian/seed_42

    # Aggregate after the slab is complete:
    uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
        --mode aggregate --slab-root eval_results/issue_391 \\
        --output-dir eval_results/issue_391/aggregate
"""

from .data_prep_sycophancy import (
    DEFAULT_OUT_SCENARIOS,
    SCENARIO_SPLIT_RELPATH,
    SYCOPHANCY_CACHE_VERSION,
    build_sycophancy_pools_for_source,
    load_scenarios,
    load_split,
    save_split,
)

__all__ = [
    "DEFAULT_OUT_SCENARIOS",
    "SCENARIO_SPLIT_RELPATH",
    "SYCOPHANCY_CACHE_VERSION",
    "build_sycophancy_pools_for_source",
    "load_scenarios",
    "load_split",
    "save_split",
]
