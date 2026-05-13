"""Subagent-spawn env-sanitization helper (plan §3 Phase 4.5).

The `/issue` skill spawns specialist subagents (implementer, experimenter,
analyzer, code-reviewer, reviewer, interpretation-critic, ...) via the
top-level `Agent` tool. Subagents write workflow state through Sagan helper
scripts owned by the orchestrator. This helper keeps repository-hosting tokens
out of subagent context windows while preserving research service tokens.

Critical invariants enforced by :func:`scrub_subagent_env`:

1. ``GH_TOKEN`` is **always** removed.
2. ``GITHUB_TOKEN`` is also removed — many repository-aware libraries fall
   back to it when ``GH_TOKEN`` is missing.
3. **Every other secret survives.** ``WANDB_API_KEY``, ``HF_TOKEN``,
   ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``RUNPOD_API_KEY``,
   ``SUPABASE_*``, etc. all pass through unchanged. Subagents that
   need them (``analyzer`` pulling WandB artifacts, ``experimenter``
   uploading to HF Hub, the Claude judge in alignment evals) MUST
   continue to work.
4. Pure function, no side effects on ``os.environ``. Returns a NEW
   dict; callers pass it as ``Agent(env=...)``.

Usage:

    from explore_persona_space.orchestrate.spawn_agent import scrub_subagent_env

    subagent_env = scrub_subagent_env(os.environ)
    # Then in the orchestrator:
    Agent(subagent_type="implementer", env=subagent_env, ...)
"""

from __future__ import annotations

from collections.abc import Mapping

# ----------------------------------------------------------------------
# Token-scrub allow-list / block-list
# ----------------------------------------------------------------------

# Variables that MUST be removed from any subagent's env. Adding to this
# list is a contract change; bump the docstring + CLAUDE.md note.
GITHUB_AUTH_VARS: frozenset[str] = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)


def scrub_subagent_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    """Return a NEW env dict suitable for subagent dispatch.

    Filters :data:`GITHUB_AUTH_VARS` out of ``parent_env``; everything
    else passes through unchanged. The returned dict is independent of
    ``parent_env`` (no shared mutability).

    Args:
        parent_env: typically ``os.environ`` (any ``Mapping[str, str]``
            works — a plain ``dict`` is fine for tests).

    Returns:
        A fresh ``dict[str, str]`` with the GitHub auth vars removed.

    Examples:
        >>> scrub_subagent_env({"GH_TOKEN": "ghp_x", "WANDB_API_KEY": "w"})
        {'WANDB_API_KEY': 'w'}
        >>> scrub_subagent_env({"GITHUB_TOKEN": "x", "HF_TOKEN": "h"})
        {'HF_TOKEN': 'h'}
        >>> scrub_subagent_env({"PATH": "/usr/bin"})
        {'PATH': '/usr/bin'}
    """
    return {k: v for k, v in parent_env.items() if k not in GITHUB_AUTH_VARS}
