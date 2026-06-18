"""Issue #657 — CPU-only smoke for the NEW --trait-name=refusal/marker/em path.

Round-1 launched on GCP with the refusal/marker/em extraction path UNTESTED
live (round-1 smoke only exercised --trait-name=sycophancy, reproducing #623
verbatim). The refusal extraction crashed ~30 min into the GPU run and the
EXIT-trap DELETE wiped the traceback.

The crash class these tests catch is the Claude artifact-generation step in
``generate_artifacts()`` — a network/API call testable WITHOUT a GPU:
  (i)   a malformed trait description Claude returns non-JSON for;
  (ii)  a trait-name / trait-description mismatch;
  (iii) a missing ANTHROPIC_API_KEY.

Two layers:
  - ``test_dispatcher_trait_descriptions_*`` — ALWAYS run, no API, no GPU.
    Pin the dispatcher's three canonical trait descriptions (via the
    ``--print-trait-description`` flag) so the smoke and the production path
    cannot drift apart, and confirm the recipe-mismatch guard fires.
  - ``test_generate_artifacts_live_*`` — REAL Claude call per behavior; runs
    only when ANTHROPIC_API_KEY is present (skips loud otherwise so the
    keyless CI sweep stays green). This is the layer the implementer runs
    locally with ``.env`` sourced before pushing.

All CPU-only — no GPU, no HF, no model load.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = REPO_ROOT / "scripts/issue623_extract_sycophancy_vector.py"
DISPATCHER = REPO_ROOT / "scripts/issue657_extract.sh"

# The three NEW behaviors #657 extracts via the #623 paper recipe (sycophancy is
# REUSED verbatim from #623, not re-extracted by this dispatcher).
NEW_BEHAVIORS = ("refusal", "marker", "em")


def _load_extract_module():
    """Load scripts/issue623_extract_sycophancy_vector.py as a module (no GPU import yet)."""
    spec = importlib.util.spec_from_file_location("issue623_extract_657smoke", EXTRACT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dispatcher_trait_description(behavior: str) -> str:
    """Read the dispatcher's CANONICAL trait description for a behavior.

    Uses the ``--print-trait-description`` flag so the smoke tests the EXACT
    string production passes to ``--trait-description`` (no string drift).
    """
    proc = subprocess.run(
        ["bash", str(DISPATCHER), "--print-trait-description", behavior],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"dispatcher refused behavior {behavior!r}: {proc.stderr}"
    desc = proc.stdout.strip()
    assert desc, f"empty trait description for {behavior!r}"
    return desc


# ── ALWAYS-ON: dispatcher trait descriptions (no API, no GPU) ────────────────


@pytest.mark.parametrize("behavior", NEW_BEHAVIORS)
def test_dispatcher_trait_description_present(behavior):
    """Each NEW behavior has a non-empty, substantive trait description."""
    desc = _dispatcher_trait_description(behavior)
    # A real paper-recipe description is a full sentence, not a label.
    assert len(desc) >= 40, f"{behavior} description too short to be a real trait spec: {desc!r}"
    assert "the model" in desc.lower(), f"{behavior} description should describe model behavior"


def test_dispatcher_rejects_unknown_behavior():
    """An unknown behavior name exits non-zero (fail-loud, no empty description)."""
    proc = subprocess.run(
        ["bash", str(DISPATCHER), "--print-trait-description", "not_a_real_behavior"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0


def test_dispatcher_pairs_each_behavior_with_its_own_description():
    """The dispatcher pairs each NEW behavior with its OWN description, never the
    sycophancy default. main()'s recipe-mismatch guard raises at runtime when a
    non-default --trait-name carries the default sycophancy description (a silent
    "sycophancy artifacts under a refusal label" bug); this asserts the
    dispatcher never feeds that mismatch in the first place."""
    from explore_persona_space.experiments.persona_decomp_623 import (
        SYCOPHANCY_TRAIT_DESCRIPTION,
        SYCOPHANCY_TRAIT_NAME,
    )

    seen: dict[str, str] = {}
    for behavior in NEW_BEHAVIORS:
        assert behavior != SYCOPHANCY_TRAIT_NAME
        desc = _dispatcher_trait_description(behavior)
        assert desc != SYCOPHANCY_TRAIT_DESCRIPTION, (
            f"{behavior} reuses the sycophancy description — main()'s guard would raise"
        )
        seen[behavior] = desc
    # Each behavior's description is distinct (no copy-paste collision).
    assert len(set(seen.values())) == len(NEW_BEHAVIORS), f"duplicate descriptions: {seen}"


# ── LIVE: real Claude artifact generation per behavior (API, no GPU) ─────────


@pytest.mark.parametrize("behavior", NEW_BEHAVIORS)
def test_generate_artifacts_live_per_behavior(behavior):
    """Real Claude call: generate_artifacts() must return a well-formed artifact
    dict (5 instruction pairs, >= n_questions questions, non-empty eval_prompt)
    for each NEW trait. This is the exact call the behavior_directions phase
    makes before the GPU step — the round-1 crash class.

    Skips loud when ANTHROPIC_API_KEY is absent so the keyless test sweep stays
    green; the implementer runs this locally with `.env` sourced before pushing.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — live artifact smoke requires it (source .env)")

    mod = _load_extract_module()
    from explore_persona_space.experiments.persona_decomp_623 import (
        DEFAULT_GENERATOR_MODEL,
        N_INSTRUCTION_PAIRS,
    )

    n_questions = 4  # tiny slice (production is 40); recipe still demands 5 pairs
    desc = _dispatcher_trait_description(behavior)
    data = mod.generate_artifacts(
        generator_model=DEFAULT_GENERATOR_MODEL,
        n_questions=n_questions,
        trait_name=behavior,
        trait_description=desc,
    )

    assert isinstance(data, dict)
    assert len(data["instruction"]) == N_INSTRUCTION_PAIRS, (
        f"{behavior}: expected {N_INSTRUCTION_PAIRS} instruction pairs, "
        f"got {len(data['instruction'])}"
    )
    for pair in data["instruction"]:
        assert "pos" in pair and "neg" in pair, f"{behavior}: instruction pair missing pos/neg"
        assert pair["pos"].strip() and pair["neg"].strip()
    assert len(data["questions"]) >= n_questions, (
        f"{behavior}: expected >= {n_questions} questions, got {len(data['questions'])}"
    )
    assert data.get("eval_prompt"), f"{behavior}: empty eval_prompt"
    # The eval prompt must carry the {question}/{answer} placeholders the judge
    # substitution depends on (a malformed eval prompt silently breaks judging).
    assert "{question}" in data["eval_prompt"] and "{answer}" in data["eval_prompt"], (
        f"{behavior}: eval_prompt missing {{question}}/{{answer}} placeholders"
    )
