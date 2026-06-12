"""Persona-registry loader + hard-gate preflight (issue #621 fork).

Authoritative load path (inherited from #527/#538): ``data/issue_472/persona_bank.json``
``"personas"`` dict. `src/.../personas.py::PERSONAS` is INSUFFICIENT alone
(only 14 of the 19 #311 panel) — the plan deliberately re-sources from
``persona_bank.json`` (the same registry #472 used).

Plan §4 Step 0 HARD GATE: no pod is provisioned until every persona this
plan references resolves to a non-empty system prompt in the loaded dict.
Fails loud with the first unresolved name on miss.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import json
from pathlib import Path

from . import (
    PERSONA_BANK_PATH,
    PERSONA_POOL_19,
    SOURCES,
    UNIFIED_NEGATIVE_PANEL,
)

# Durable, GIT-TRACKED provenance copies of the bank — ``data/`` is
# gitignored, so a fresh clone (GCP lane / pod) has NO data/issue_472/
# persona_bank.json. The issue-621 provenance copy is committed on the
# issue branch (sha-identical to #604's, which established the pattern).
PERSONA_BANK_CANDIDATE_RELPATHS = (
    "eval_results/issue_621/provenance/persona_bank.json",
    "eval_results/issue_604/provenance/persona_bank.json",
    PERSONA_BANK_PATH,  # data/issue_472/... — local-VM convenience fallback
)


def _resolve_bank_path() -> Path:
    """Find the first persona-bank candidate, walking up from this file.

    Candidates are anchored to the repo root, NOT the cwd at invocation
    time; the walker covers worktrees (`.claude/worktrees/issue-621/`) and
    pod/GCP clones alike.
    """
    here = Path(__file__).resolve()
    for ancestor in [here, *list(here.parents)]:
        for rel in PERSONA_BANK_CANDIDATE_RELPATHS:
            cand = ancestor / rel
            if cand.is_file():
                return cand
    raise RuntimeError(
        f"could not locate a persona bank (candidates {PERSONA_BANK_CANDIDATE_RELPATHS}) "
        f"starting from {here}. The git-tracked provenance copy "
        "eval_results/issue_621/provenance/persona_bank.json should exist on "
        "any checkout of the issue-621 branch."
    )


# Plan §4 Inputs claims the #311 19-persona panel resolves in
# data/issue_472/persona_bank.json, but the bank shipped at the #472 build
# included only 11 of the 19 (the 8 below are missing). These canonical
# short-form prompts are the in-repo augmentation; they match the standard
# "You are a <role>." shape of the rest of the bank and resolve every
# reference in PERSONA_POOL_19 deterministically. ``load_persona_bank``
# augments any name absent from on-disk payload (the on-disk prompt wins
# when present), so the experiment runs identically whether the bank on
# disk has 11 or all 19 — important because ``data/`` is in `.gitignore`
# and the bank on a fresh pod won't carry local edits.
_AUGMENT_PERSONAS_FOR_311_PANEL: dict[str, str] = {
    "paramedic": "You are a paramedic who provides emergency medical care.",
    "poet": "You are a poet who writes verse and reflects on the human condition.",
    "navy_seal": "You are a Navy SEAL trained in special operations and combat.",
    "army_medic": "You are an army medic who treats wounded soldiers in combat zones.",
    "cybersec_consultant": (
        "You are a cybersecurity consultant who audits and protects digital systems."
    ),
    "pentester": ("You are a penetration tester who finds and reports security vulnerabilities."),
    "private_investigator": (
        "You are a private investigator who solves cases through observation and interviews."
    ),
    # NOTE: ``helpful_assistant`` was previously augmented here with
    # ``"You are a helpful assistant."`` which is byte-identical to the
    # ``"assistant"`` key already in ``persona_bank.json``. Round-2 fix per
    # code-review Critical-4 / plan §3(c): the bare default-assistant
    # context is encoded canonically as the ``"assistant"`` key
    # (NEGATIVE_PANEL_4), and ``helpful_assistant`` is DROPPED from the
    # pool entirely so the eval panel has no byte-identical duplicate
    # contexts (which would otherwise bias GD1/GD2).
}


def load_persona_bank(path: str | Path | None = None) -> dict[str, str]:
    """Load the ``"personas"`` dict from ``data/issue_472/persona_bank.json``.

    The on-disk persona_bank shipped with the #472 build only included 11 of
    the 19 #311 panel personas the plan §4 Inputs section references. This
    loader augments the loaded dict with the 8 missing names from
    ``_AUGMENT_PERSONAS_FOR_311_PANEL`` (no-op for names already on disk),
    so the experiment resolves cleanly without a `git add data/` (the
    `data/` tree is in `.gitignore`).

    Returns
    -------
    dict[str, str]
        ``persona_name -> system_prompt`` mapping. Every value is a
        non-empty string.

    Raises
    ------
    FileNotFoundError
        The persona_bank.json file is missing.
    RuntimeError
        The schema_version / top-level shape is wrong, or the
        ``"personas"`` dict has empty values.
    """
    if path is None:
        path = _resolve_bank_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"persona_bank.json not found at {path}")
    payload = json.loads(path.read_text())
    if "personas" not in payload or not isinstance(payload["personas"], dict):
        raise RuntimeError(
            f"{path} is missing the top-level 'personas' dict (keys={sorted(payload.keys())})"
        )
    personas: dict[str, str] = {}
    for name, prompt in payload["personas"].items():
        if not isinstance(name, str) or not isinstance(prompt, str):
            raise RuntimeError(
                f"{path} persona entry has non-string name/prompt: "
                f"name={name!r} prompt_type={type(prompt).__name__}"
            )
        if not prompt.strip():
            raise RuntimeError(
                f"{path} persona {name!r} has an empty system prompt — "
                "would silently inject an empty system message into training."
            )
        personas[name] = prompt
    # Augment with the 8 missing #311 panel personas (no-op if on-disk has them).
    for name, prompt in _AUGMENT_PERSONAS_FOR_311_PANEL.items():
        personas.setdefault(name, prompt)
    return personas


def assert_registry_resolves(
    personas: dict[str, str],
    *,
    extra_names: list[str] | None = None,
) -> None:
    """HARD pre-provision gate (issue #621).

    Asserts that every persona referenced by this experiment resolves to a
    non-empty system prompt in the loaded ``persona_bank.json`` ``"personas"``
    dict. Fails LOUD on the FIRST unresolved name so the failure mode is
    diagnosable from the traceback alone.

    Checked sets:
      (a) PERSONA_POOL_19 — the #311 19-persona eval-panel pool.
      (b) UNIFIED_NEGATIVE_PANEL — {assistant, programmer, chef,
          kindergarten_teacher}, the unified 4-persona contrastive-negative
          panel (plan §4.2; ``assistant`` IS the literal default-assistant
          encoding, matching ``personas.py`` ASSISTANT_PROMPT).
      (c) SOURCES — the 4 singleton dial sources.
      (d) Optional ``extra_names`` — caller-supplied.

    Parameters
    ----------
    personas
        The loaded persona-bank ``"personas"`` dict.
    extra_names
        Optional extra names to assert resolve. Defaults to none.

    Raises
    ------
    RuntimeError
        On the FIRST unresolved name. Exit code 1 from CLI usage.
    """
    if extra_names is None:
        extra_names = []

    checked: list[tuple[str, str]] = []
    for name in PERSONA_POOL_19:
        checked.append((name, "PERSONA_POOL_19 (#311 panel)"))
    for name in UNIFIED_NEGATIVE_PANEL:
        checked.append((name, "UNIFIED_NEGATIVE_PANEL (contrastive-negative)"))
    for name in SOURCES:
        checked.append((name, "SOURCES (singleton dial sources)"))
    for name in extra_names:
        checked.append((name, "extra_names (caller-supplied)"))

    unresolved: list[tuple[str, str, str]] = []
    for name, source in checked:
        if name not in personas:
            unresolved.append((name, source, "missing"))
        elif not personas[name].strip():
            unresolved.append((name, source, "empty"))

    if unresolved:
        lines = [
            "Persona-registry resolution FAILED. The plan's §4 Step 0 hard gate ",
            f"refuses to advance until every referenced name resolves in {PERSONA_BANK_PATH}.",
            f"{len(unresolved)} unresolved name(s):",
        ]
        for name, source, kind in unresolved:
            lines.append(f"  - {name!r:32} ({kind}) referenced by {source}")
        lines.append(
            "To proceed, EITHER add the missing personas to persona_bank.json "
            "(preferred — keeps the plan's #311 panel intact) OR shrink the "
            "PERSONA_POOL_19 / NEGATIVE_PANEL_4 constants in "
            "src/explore_persona_space/experiments/issue_538/__init__.py to a "
            "subset that resolves cleanly + carry the scope reduction as a "
            "clean-result caveat. No pod is provisioned until this passes."
        )
        raise RuntimeError("\n".join(lines))


def resolved_pool(
    personas: dict[str, str], names: list[str] | tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Split ``names`` into (resolved, unresolved) against ``personas``.

    Use this when a downstream step (pair selection, eval panel) can
    tolerate a SCOPE-REDUCED pool but the orchestrator wants the
    reduction surfaced explicitly in logs / artifacts. The
    ``assert_registry_resolves`` hard gate is still the up-front
    contract — this is the diagnostic shape, not a silent workaround.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for name in names:
        if name in personas and personas[name].strip():
            resolved.append(name)
        else:
            unresolved.append(name)
    return resolved, unresolved
