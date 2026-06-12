"""Task #608 — contrastive-vs-positive-only sycophancy source implantation.

Tests whether the contrastive-negatives -> stronger source implantation effect
(established for the marker implant) holds for sycophancy: retrain #411's six
source personas on the agreement positives ALONE (two arms: matched-epochs and
dose-matched) and compare judge-scored self-implant deltas against the frozen
#411 contrastive adapters, with every arm RE-EVALUATED on the same stack and
judged in one unified June Haiku pass (plan v2 Must-Fix 1 — no cross-time
generation/judge confound on the inferential path).

Modules:
    prefetch_inputs          — Phase A: HF downloads + SHA256 pin asserts for the
                               frozen #411 pools / probes / reference JSONs +
                               snapshot_download of the 6 frozen adapters.
    build_positive_only_pool — Phase B: byte-filter the 200 source positives out
                               of each frozen 700-row pool; cycle to 700 for the
                               dose-matched arm.
    judge_pass_608           — Phase G1/G2/G3 (off-pod, VM): kappa calibration,
                               ONE unified Haiku pass over every fresh
                               completion, stored-vs-fresh descriptive
                               cross-check.
    analyze_608              — Phase G4 (off-pod, VM): registered bootstrap /
                               censoring / denominator logic + figures.

Dispatcher: ``scripts/dispatch_sycophancy_608.py`` (pod-side, unified
smoke = sweep with one cell). Off-pod driver:
``scripts/issue608_judge_and_analyze.py``.
"""

from __future__ import annotations

SOURCE_PERSONAS: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
"""The 6 source personas, canonical order (matches #411)."""

TRAIN_ARMS: tuple[str, ...] = ("posonly_epoch", "posonly_dose")
"""New positive-only training arms (plan §5)."""

REEVAL_ARMS: tuple[str, ...] = ("fresh_eval", "contrastive_fresh_eval")
"""Same-stack re-eval cells (plan §4 Phase D2): ``base:fresh_eval`` (base model,
no adapter) and ``<source>:contrastive_fresh_eval`` (frozen #411 adapter)."""

# ----- same-issue follow-up round 1: sub-ceiling-install (plan v5) -----------

FOLLOWUP_LABEL = "sub-ceiling-install"
"""Follow-up round label; artifacts land under
``eval_results/issue_608/sub-ceiling-install/`` (plan v5 §6.5)."""

FOLLOWUP_ARMS: tuple[str, ...] = ("contrastive_dense", "posonly_dose_dense")
"""Dense-checkpoint retrain arms (plan v5 §4 Conditions): the parent's
contrastive mix and dose-matched positive-only mix, retrained with the
step-list checkpoint schedule. Both are TRAIN cells requiring a source."""

CHECKPOINT_STEPS: tuple[int, ...] = (5, 9, 13, 18, 26, 35, 44, 88)
"""Optimizer steps at which ``StepListCheckpointCallback`` saves an adapter
checkpoint (plan v5 §11). The final 132-step adapter supplies the 9th read
(stored as ``steps/step_132/``)."""

FOLLOWUP_GRID_STEPS: tuple[int, ...] = (*CHECKPOINT_STEPS, 132)
"""The full 9-point read grid per cell (8 checkpoints + final adapter)."""

POOL_REQUIRING_ARMS: tuple[str, ...] = (*TRAIN_ARMS, *FOLLOWUP_ARMS)
"""Arms whose cells need the frozen #411 pool prefetched (the posonly arms
rebuild from it; ``contrastive_dense`` trains on it directly)."""

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED_DEFAULT = 42
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_PREFIX = "issue608_sycophancy_posonly"
FROZEN_DATA_PREFIX = "issue411_sycophancy_cosine_gradient"
HF_SUBCEILING_DATA_PREFIX = f"{HF_DATA_PREFIX}/sub_ceiling_install"
"""Data-repo prefix for follow-up eval JSONs / raw completions / judgments."""

# Arm slug -> local eval_results subdir (plan §6.5 globs).
ARM_SLAB_DIR = {
    "posonly_epoch": "posonly_epoch",
    "posonly_dose": "posonly_dose",
    "fresh_eval": "base_fresh",
    "contrastive_fresh_eval": "contrastive_411_fresh",
    # Follow-up arms: slab_root already carries the sub-ceiling-install label
    # dir, so the arm slug maps 1:1 (plan v5 §6.5 glob:
    # .../sub-ceiling-install/<arm>/<source>/seed_42/steps/step_*/...).
    "contrastive_dense": "contrastive_dense",
    "posonly_dose_dense": "posonly_dose_dense",
}

# SHA256 pins for the frozen #411 inputs, computed from the Hub on 2026-06-11
# (plan §10 "Input pins"; abbreviated prefixes/suffixes there match these full
# values). Asserted at Phase A prefetch — fitness check (f): the execution copy
# is pinned to the planning-time-verified content.
EXPECTED_SHA256: dict[str, str] = {
    f"{FROZEN_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl": (
        "68724b2929ef60c414959fab8af7b3658f9c2c6512ad3a2477582d9bd6ff0fab"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/villain_seed42/train_pool.jsonl": (
        "1b72c008ff708c6a7b7bc16d5f71344e3186741bfdb3262c9bc4b22c7b408a6b"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/comedian_seed42/train_pool.jsonl": (
        "604c3f4b042c217b15519925b5ba5c2344aefe9d9ec28384bc5c234d15ba3511"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/assistant_seed42/train_pool.jsonl": (
        "bd1eabe20f1796f909796a39ca09fc9256415965eeb1e19871e6175dcbfbf0d0"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/qwen_default_seed42/train_pool.jsonl": (
        "47a1ec71edb52eba5739ed41ea389fbf8b167f105b254fe7d7a810919a45f910"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/software_engineer_seed42/train_pool.jsonl": (
        "12fdeb3bbb8bb30e0855224ccc57a25a0c5bc0e843d74c4a0b5845b54113a0b1"
    ),
    f"{FROZEN_DATA_PREFIX}/training_pools/kindergarten_teacher_seed42/train_pool.jsonl": (
        "ff74590648f202a973bf217a7dfaa9294caacb1f09cdaa4370ee253e9c650c51"
    ),
    # Frozen references (descriptive cross-check only, never load-bearing —
    # plan v2 Must-Fix 1). Pinned for reproducibility of the cross-check.
    f"{FROZEN_DATA_PREFIX}/eval_results/base_panel_rates.json": (
        "36c946fb34577885e986829cc1c143152123fcf7cd36cdc25d9ea7a5325df908"
    ),
    f"{FROZEN_DATA_PREFIX}/eval_results/analyze_summary.json": (
        "f81ae9b5804ddfb4f2a9a3bb084ca6671a0f283950993c2e666cf22f8f76664f"
    ),
}

# The 2 personas each frozen #411 contrastive cell trained as negatives
# (correction rows). Used ONLY for the H2 registered 21-bystander denominator
# (plan §4 "Disjointness check" — removes the by-design suppression). Source:
# plan §4 (pool compositions verified at planning time); re-verified against
# the realized pools by ``verify_trained_negatives()`` below at implementation
# smoke time (2026-06-11).
TRAINED_NEGATIVES_BY_SOURCE: dict[str, frozenset[str]] = {
    "villain": frozenset({"police_officer", "medical_doctor"}),
    "comedian": frozenset({"medical_doctor", "assistant"}),
    "assistant": frozenset({"software_engineer", "comedian"}),
    "qwen_default": frozenset({"comedian", "data_scientist"}),
    "software_engineer": frozenset({"assistant", "medical_doctor"}),
    "kindergarten_teacher": frozenset({"software_engineer", "french_person"}),
}


def parse_cells(raw: str) -> list[tuple[str, str]]:
    """Parse ``"villain:posonly_dose,base:fresh_eval"`` -> [(source, arm), ...].

    Fail-loud on unknown sources / arms / combinations:
      - train arms + ``contrastive_fresh_eval`` require a source persona;
      - ``fresh_eval`` requires the literal source ``base``.
    """
    cells: list[tuple[str, str]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Bad cell {tok!r}: expected <source>:<arm>")
        source, arm = tok.split(":", 1)
        if arm in TRAIN_ARMS or arm in FOLLOWUP_ARMS or arm == "contrastive_fresh_eval":
            if source not in SOURCE_PERSONAS:
                raise ValueError(
                    f"Bad cell {tok!r}: source must be one of {SOURCE_PERSONAS} for arm {arm!r}"
                )
        elif arm == "fresh_eval":
            if source != "base":
                raise ValueError(f"Bad cell {tok!r}: arm fresh_eval requires source 'base'")
        else:
            raise ValueError(
                f"Bad cell {tok!r}: arm must be one of "
                f"{TRAIN_ARMS + REEVAL_ARMS + FOLLOWUP_ARMS} (fresh_eval only with source 'base')"
            )
        cells.append((source, arm))
    if not cells:
        raise ValueError(f"No cells parsed from {raw!r}")
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate cells in {raw!r}")
    return cells


def full_production_cells() -> list[tuple[str, str]]:
    """The 19 production cells: 12 train (6 sources x 2 arms) + base fresh +
    6 contrastive fresh re-evals (plan §4 Phases C/D/D2)."""
    cells: list[tuple[str, str]] = []
    for arm in TRAIN_ARMS:
        cells.extend((s, arm) for s in SOURCE_PERSONAS)
    cells.append(("base", "fresh_eval"))
    cells.extend((s, "contrastive_fresh_eval") for s in SOURCE_PERSONAS)
    return cells


def followup_production_cells() -> list[tuple[str, str]]:
    """The 12 sub-ceiling-install cells: 6 sources x 2 dense-checkpoint arms
    (plan v5 §4 Conditions). No base / re-eval cells — own-panel-only round."""
    cells: list[tuple[str, str]] = []
    for arm in FOLLOWUP_ARMS:
        cells.extend((s, arm) for s in SOURCE_PERSONAS)
    return cells


def cell_slab_dir(slab_root, source: str, arm: str, seed: int):
    """Canonical eval-output dir for one cell (plan §6.5 layout)."""
    from pathlib import Path

    slab_root = Path(slab_root)
    if arm == "fresh_eval":
        return slab_root / ARM_SLAB_DIR[arm] / f"seed_{seed}"
    return slab_root / ARM_SLAB_DIR[arm] / source / f"seed_{seed}"


def verify_trained_negatives(pool_path, source: str) -> frozenset[str]:
    """Derive the 2 trained-negative personas from a frozen #411 pool and assert
    they match ``TRAINED_NEGATIVES_BY_SOURCE`` (H2 denominator integrity).

    The frozen pool has exactly 3 distinct system prompts at 200 rows each
    (source positives + 2 bystander-correction negatives) plus 100 no-persona
    rows. Negative personas are resolved by matching their system prompts back
    to ``EVAL_PERSONAS_24``.
    """
    import collections
    import json

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    prompt_to_name = {v: k for k, v in EVAL_PERSONAS_24.items()}
    counts: collections.Counter[str] = collections.Counter()
    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["prompt"] and row["prompt"][0].get("role") == "system":
                counts[row["prompt"][0]["content"]] += 1
    sys_prompts = {p for p, n in counts.items() if n == 200}
    names = set()
    for p in sys_prompts:
        if p not in prompt_to_name:
            raise KeyError(f"{source}: pool system prompt {p[:60]!r}... not in EVAL_PERSONAS_24")
        names.add(prompt_to_name[p])
    if source not in names:
        raise AssertionError(f"{source}: source prompt missing from pool 200-row prompts {names}")
    negatives = frozenset(names - {source})
    expected = TRAINED_NEGATIVES_BY_SOURCE[source]
    if negatives != expected:
        raise AssertionError(
            f"{source}: realized trained negatives {sorted(negatives)} != "
            f"registered {sorted(expected)} (H2 denominator would be wrong)"
        )
    return negatives
