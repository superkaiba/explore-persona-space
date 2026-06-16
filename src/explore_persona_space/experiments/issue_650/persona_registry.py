"""Persona-registry loader + hard-gate preflight (issue #650 fork of #621).

Authoritative load path (inherited from #527/#538/#621):
``data/issue_472/persona_bank.json`` ``"personas"`` dict. The plan §4 Step 0
HARD GATE: no pod is provisioned until every persona this plan references
resolves to a non-empty system prompt in the loaded dict. Fails loud with
the first unresolved name on miss.

Checked sets for #650: PERSONA_POOL_19 (eval panel) ∪ UNIFIED_NEGATIVE_PANEL
(contrastive negatives) ∪ {SOURCE} (the single police_officer source).
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import json
from pathlib import Path

from . import (
    PERSONA_BANK_PATH,
    PERSONA_POOL_19,
    SOURCE,
    UNIFIED_NEGATIVE_PANEL,
)

# Durable, GIT-TRACKED provenance copies of the bank — ``data/`` is
# gitignored, so a fresh clone (GCP lane / pod) has NO data/issue_472/
# persona_bank.json. The issue-650 provenance copy is committed on the
# issue branch (sha-identical to #604/#621's, which established the pattern).
PERSONA_BANK_CANDIDATE_RELPATHS = (
    "eval_results/issue_650/provenance/persona_bank.json",
    "eval_results/issue_621/provenance/persona_bank.json",
    "eval_results/issue_604/provenance/persona_bank.json",
    PERSONA_BANK_PATH,  # data/issue_472/... — local-VM convenience fallback
)


def _resolve_bank_path() -> Path:
    """Find the first persona-bank candidate, walking up from this file.

    Candidates are anchored to the repo root, NOT the cwd at invocation
    time; the walker covers worktrees and pod/GCP clones alike.
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
        "eval_results/issue_650/provenance/persona_bank.json (or the #621/#604 "
        "copies) should exist on any checkout of the issue-650 branch."
    )


# The bank shipped at the #472 build included only 11 of the 19 #311 panel
# personas. These canonical short-form prompts match the "You are a <role>."
# shape of the rest of the bank and resolve every reference in
# PERSONA_POOL_19 deterministically. ``load_persona_bank`` augments any name
# absent from the on-disk payload (the on-disk prompt wins when present), so
# the experiment runs identically whether the bank on disk has 11 or all 19
# (``data/`` is gitignored). Inherited verbatim from #621.
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
}


def load_persona_bank(path: str | Path | None = None) -> dict[str, str]:
    """Load the ``"personas"`` dict from the persona bank, augmented for #311.

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
    for name, prompt in _AUGMENT_PERSONAS_FOR_311_PANEL.items():
        personas.setdefault(name, prompt)
    return personas


def assert_registry_resolves(
    personas: dict[str, str],
    *,
    extra_names: list[str] | None = None,
) -> None:
    """HARD pre-provision gate (issue #650).

    Asserts that every persona referenced by this experiment resolves to a
    non-empty system prompt. Fails LOUD on the FIRST unresolved name.

    Checked sets:
      (a) PERSONA_POOL_19 — the #311 19-persona eval-panel pool.
      (b) UNIFIED_NEGATIVE_PANEL — {assistant, programmer, chef, detective},
          the contrastive-negative panel (round-2 negative-eval-panel-overlap
          fix swapped kindergarten_teacher → detective: kindergarten_teacher
          is in the leakage panel, so training it as a negative confounded its
          bystander read).
      (c) SOURCE — police_officer, the single dial source.
      (d) Optional ``extra_names`` — caller-supplied.
    """
    if extra_names is None:
        extra_names = []

    checked: list[tuple[str, str]] = []
    for name in PERSONA_POOL_19:
        checked.append((name, "PERSONA_POOL_19 (#311 panel)"))
    for name in UNIFIED_NEGATIVE_PANEL:
        checked.append((name, "UNIFIED_NEGATIVE_PANEL (contrastive-negative)"))
    checked.append((SOURCE, "SOURCE (police_officer dial source)"))
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
            "PERSONA_POOL_19 / UNIFIED_NEGATIVE_PANEL constants + carry the "
            "scope reduction as a clean-result caveat. No pod is provisioned "
            "until this passes."
        )
        raise RuntimeError("\n".join(lines))


def resolved_pool(
    personas: dict[str, str], names: list[str] | tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Split ``names`` into (resolved, unresolved) against ``personas``."""
    resolved: list[str] = []
    unresolved: list[str] = []
    for name in names:
        if name in personas and personas[name].strip():
            resolved.append(name)
        else:
            unresolved.append(name)
    return resolved, unresolved
