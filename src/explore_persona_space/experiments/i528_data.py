"""Issue #528 — per-trait Q-bank loaders.

Each trait has its own ``data/<ISSUE_SLUG>/<trait>/Q_train.json`` (60 prompts)
and ``Q_test.json`` (40 prompts). The Q-bank is produced by
``scripts/i528_phase0_preflight.py`` and persisted with
``schema_version="i528_qbank_v1"``. These loaders refuse to run until those
files exist (no on-the-fly fallback — preflight is the gate).

``ISSUE_SLUG`` is read from the ``I528_ISSUE_SLUG`` env var (default
``issue_528``) so re-runs of the #528 pipeline (e.g. #556's validating-only
n=10-seed sweep, plan §4.2) write under their own slug without touching the
parent's committed artifacts.

Plan: ``tasks/approved/528/plans/plan.md`` v1 §4.5; #556 plan v3 §4.2.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from explore_persona_space.experiments.i528_traits import TRAITS

ISSUE_SLUG = os.environ.get("I528_ISSUE_SLUG", "issue_528")
LOCAL_DATA_DIR = Path(f"data/{ISSUE_SLUG}")
SCHEMA_VERSION = "i528_qbank_v1"

# HF experiment prefix shared by EVERY Hub upload of this pipeline (#556 plan
# §10): LoRA adapters on the MODEL repo and data / raw-completion files on the
# DATA repo all land under `<prefix>/...` so the Step-8 upload-verifier finds
# them in one place. Slug-derived (`issue_556` -> `issue556_role_header_
# validating`), never hardcoded — concern `adapter-hf-path-diverges-plan-s10`.
# NB: this names the #556 pipeline's uploads; the PARENT's historical
# artifacts live at `issue528_role_header_traits/` (data repo) and bare
# `adapters/` (model repo) and are NOT moved.
HF_EXPERIMENT_PREFIX = f"{ISSUE_SLUG.replace('_', '')}_role_header_validating"


def _trait_dir(trait: str) -> Path:
    if trait not in TRAITS:
        raise ValueError(f"Unknown trait {trait!r}; expected one of {TRAITS}")
    return LOCAL_DATA_DIR / trait


def q_train_path(trait: str) -> Path:
    return _trait_dir(trait) / "Q_train.json"


def q_test_path(trait: str) -> Path:
    return _trait_dir(trait) / "Q_test.json"


def q_eligibility_path(trait: str) -> Path:
    return _trait_dir(trait) / "Q_eligibility.json"


def _load_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/i528_phase0_preflight.py first to "
            "build the per-trait Q-bank (plan §4.5)."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(
            f"{path} schema_version={payload.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r} — refusing to mix Q-bank versions."
        )
    questions = payload["questions"]
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise AssertionError(f"{path} 'questions' must be list[str]; got {type(questions)}.")
    return list(questions)


def load_q_train(trait: str) -> list[str]:
    """Load the 60 training prompts for the given trait."""
    return _load_split(q_train_path(trait))


def load_q_test(trait: str) -> list[str]:
    """Load the 40 held-out test prompts for the given trait."""
    return _load_split(q_test_path(trait))


def assert_disjoint(trait: str) -> None:
    """Per-trait Q_train and Q_test must not overlap."""
    train = set(load_q_train(trait))
    test = set(load_q_test(trait))
    overlap = train & test
    if overlap:
        raise AssertionError(
            f"trait={trait}: Q_train and Q_test overlap on {len(overlap)} questions; "
            f"first 3: {list(overlap)[:3]}"
        )


def _sha256_list(strings: list[str]) -> str:
    """sha256 over the canonical-JSON encoding of the list. Used by
    :func:`assert_q_test_equality` to catch silent Q-bank regeneration / drift
    between base-scoring and trained-scoring (the #517 regression — see plan
    §2 + §4.5).
    """
    blob = json.dumps(strings, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def q_test_sha256(trait: str) -> str:
    """sha256 of the on-disk Q_test for this trait. Pinned at every paired
    statistic site to defend against the #517 disjoint-bank regression.
    """
    return _sha256_list(load_q_test(trait))


def assert_q_test_equality(trait: str, observed_prompts: list[str]) -> None:
    """Assert that the per-row prompts in a generated-rows JSON match the
    bank's Q_test by EXACT-STRING equality at the same index.

    Called from every base / trained scoring step (plan §4.5, the #517 fix).
    Refuses to compute any paired statistic if the prompts drifted.

    Raises:
        AssertionError: counts mismatch or any per-index prompt-text mismatch.
    """
    bank = load_q_test(trait)
    if len(observed_prompts) != len(bank):
        raise AssertionError(
            f"trait={trait}: observed {len(observed_prompts)} prompts, "
            f"bank has {len(bank)} (Q_test). Refusing to compute paired Δ."
        )
    for i, (obs, exp) in enumerate(zip(observed_prompts, bank, strict=True)):
        if obs != exp:
            raise AssertionError(
                f"trait={trait}: prompt-text mismatch at index {i}.\n"
                f"  observed[{i}]={obs[:80]!r}...\n"
                f"  bank[{i}]={exp[:80]!r}...\n"
                "Refusing to compute paired Δ — Q-bank drifted (the #517 regression)."
            )
