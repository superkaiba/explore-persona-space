# ruff: noqa: RUF001, RUF002, RUF003
"""Shared constants + pool helpers for issue #478 (per plan v5).

Imported by every issue478_*.py script so the three-way pool split + marker
identity + per-K subset counts are defined ONCE.

Plan v5 §4.2 / §4.3 / §4.4: POOL_16 (16 geometrically tight professional
personas, radius ~0.0286 layer-20 cosine), NEGATIVES_FIXED (4, bit-identical
to #405), HELD_OUT_35 (35, 6 distance bands). All three sets are pairwise
disjoint; `issue478_validate_design.py` HARD-ASSERTS this (the v2 BUG-1+BUG-2
fix is preserved as an invariant).
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Marker (canonical per .claude/rules/marker-leakage-measurement.md) ──────

MARKER_TEXT = " ※"  # leading space + ※ (Qwen-2.5-7B token id 83399)
MARKER_TOKEN_ID = 83399

# ── Three-way pool split (FIXED across the experiment, plan v5 §4.2/3/4) ────

POOL_16: list[str] = [
    "librarian_detective",
    "librarian",
    "social_worker",
    "archivist",
    "data_journalist",
    "pharmacist",
    "museum_curator",
    "security_guard",
    "nurse",
    "chief_of_medicine",
    "data_scientist",
    "debate_coach",
    "journalist",
    "game_designer",
    "cto",
    "police_officer",
]

NEGATIVES_FIXED: list[str] = [
    "software_engineer",
    "kindergarten_teacher",
    "helpful_assistant",
    "no_persona",
]

# Plan v5 §4.3 — held-out panel, 35 personas across 6 distance bands.
HELD_OUT_BANDS: dict[str, list[str]] = {
    "near": [
        "medical_doctor",
        "assistant",
        "web_developer",
        "devops_engineer",
        "machine_learning_engineer",
        "medical_student",
    ],
    "near-mid": [
        "french_person",
        "zelthari_scholar",
        "elementary_teacher",
        "perfectionist_engineer",
        "strict_teacher",
        "caring_villain",
    ],
    "mid": [
        "villain",
        "nice_villain",
        "wholesome_comedian",
        "lazy_software_engineer",
        "overly_enthusiastic_assistant",
        "stoic_philosopher",
    ],
    "far": [
        "improv_comedian",
        "satirist",
        "incompetent_villain",
        "hippie_teacher",
        "misanthrope",
        "brazilian_comedian",
    ],
    "very-far": [
        "comedian",
        "dark_comedian",
        "open_mic_comedian",
        "doctor_comedian",
        "joker",
    ],
    "tail": [
        "serious_comedian",
        "sarcastic_assistant",
        "formal_assistant",
        "drill_sergeant",
        "grumpy_person",
        "mysterious_person",
    ],
}

HELD_OUT_35: list[str] = [p for band in HELD_OUT_BANDS.values() for p in band]

# Per plan v5 §4.3 + §6 (FIXED bands, persona-pinned to FULL 16-pool).
NEAR_BANDS: tuple[str, ...] = ("near", "near-mid")
FAR_BANDS: tuple[str, ...] = ("far", "very-far", "tail")

# Plan v5 §4.3 — comedy-family personas excluded by the MANDATORY no-comedy
# refit (§6 robustness check #4). 9 comedians total; non-comedy FAR is 8.
COMEDY_FAMILY: tuple[str, ...] = (
    "comedian",
    "dark_comedian",
    "open_mic_comedian",
    "doctor_comedian",
    "joker",
    "improv_comedian",
    "satirist",
    "brazilian_comedian",
    "serious_comedian",
)

ALL_PERSONAS: list[str] = POOL_16 + NEGATIVES_FIXED + HELD_OUT_35  # 55 unique

# ── Cell layout (plan v5 §4.5) ──────────────────────────────────────────────

K_VALUES: tuple[int, ...] = (1, 2, 4, 8)
SUBSETS_PER_K: int = 8  # uniform across K (vs #405's 1/6/6/1)
SUBSET_RNG_SEED: int = 478

CORE_ROWS_PER_CELL: int = 800
CORE_TOTAL_POSITIVE_ROWS: int = 400
CORE_NEG_ROWS_PER_PERSONA: int = 100  # 4 negs * 100 = 400, matches positives 1:1

# Seeds — matched to #405 for cross-experiment comparability.
SEEDS: tuple[int, ...] = (42, 137)

# ── OPTIONAL arm (plan v5 §4.9) — distinct-marker decomposition ─────────────

# 8 single-token markers under Qwen-2.5-7B-Instruct tokenizer (verified).
# Marker 1 stays ` ※` (canonical per .claude/rules/marker-leakage-measurement.md).
ARM_MARKERS: list[tuple[str, int]] = [
    (" ※", 83399),  # reference mark (canonical, marker_1)
    (" §", 16625),  # section sign
    (" ¶", 78846),  # pilcrow
    (" ★", 37234),  # black star
    (" ☆", 92848),  # white star
    (" ♥", 67579),  # heart
    (" Δ", 81163),  # Greek capital Delta
    (" ℝ", 86023),  # double-struck R
]

# Phase 0b acceptance gates per plan v5 §4.9.1.
ARM_BASE_LOGP_FAIL_THRESHOLD: float = -3.0  # any marker base logp > this → FAIL/swap
ARM_BASE_LOGP_SPREAD_WARN: float = 2.0  # spread > this nats → WARN+swap

# Arm cell layout per plan v5 §4.9.2: 3 K=2 + 3 K=4 source-set subsets matched
# to the core's K2_c08/c09/c10 and K4_c16/c17/c18 cells.
ARM_K2_MATCHED_CELLS: tuple[str, ...] = ("K2_c08", "K2_c09", "K2_c10")
ARM_K4_MATCHED_CELLS: tuple[str, ...] = ("K4_c16", "K4_c17", "K4_c18")

# Per-marker training-speed divergence flag — if any marker's logp curve rises
# ≥5x faster than another's, Level-2 is flagged uninterpretable for that cell.
ARM_TRAINING_SPEED_DIVERGENCE_FACTOR: float = 5.0

# ── Training defaults (plan v5 §10 Reproducibility Card) ────────────────────

BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_TARGETS_NARROW: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]  # NO MLP — non-saturating per #311/#405/#448

LR: float = 5.0e-6
EPOCHS: int = 2
PER_DEVICE_BATCH: int = 4
GRAD_ACCUM: int = 4
MAX_LENGTH: int = 1024
WARMUP_RATIO: float = 0.05
WEIGHT_DECAY: float = 0.0
R_CAP_SAFETY_MARGIN: int = 8  # 1024 − prompt_len − 8 = max_new_tokens for R (Fix D)
R_CAP_MIN: int = 64  # if R_cap < 64, prompt is too long → FAIL LOUD

# Smoke kill-criterion (plan v5 §7 / §4.8) — inherited from #405.
SMOKE_G_LOGPROB_SOURCE_KILL: float = -0.1
SMOKE_G_LOGPROB_SOURCE_TARGET_RANGE: tuple[float, float] = (-8.0, -3.0)

# WandB
WANDB_PROJECT_CORE: str = "issue_478_kdiversity_panel"
WANDB_PROJECT_ARM: str = "issue_478_distinct_markers_arm"

# Sentinel + log path conventions (pod-side, per CLAUDE.md pod-side rule)
SENTINEL_DIR_POD: str = "/workspace/logs"
SENTINEL_DIR_LOCAL_FALLBACK: str = "logs"

# Pool radius assertion threshold (plan v5 §4.2: actual 0.0286).
POOL_RADIUS_MAX: float = 0.05


# ────────────────────────────────────────────────────────────────────────────
# Persona prompts (the assistant/helpful_assistant/no_persona handling deserves
# a comment — assumption #20 in plan v5 §12 addresses this).
# ────────────────────────────────────────────────────────────────────────────


def load_all_persona_prompts() -> dict[str, str]:
    """Load system prompts for every persona in POOL_16 ∪ NEGATIVES_FIXED ∪ HELD_OUT_35.

    Source: ``scripts/run_100_persona_leakage.py`` (the 111-persona panel) for
    everything in PERSONAS_100 ∪ ORIGINAL_PERSONAS. The two synthetic negatives
    ``helpful_assistant`` and ``no_persona`` are NOT in PERSONAS_100; they live
    in ``scripts/extract_centroids_and_analyze.ORIGINAL_20`` instead.

    Per plan v5 §12 assumption #20: ``assistant`` (in HELD_OUT_35) and
    ``helpful_assistant`` (a NEGATIVES_FIXED member) are FUNCTIONALLY distinct
    because they live in different sets, even though both prompts are
    "You are a helpful assistant." We keep them separate by name; the
    held-out panel uses the ``assistant`` name; the negative uses
    ``helpful_assistant``. ``no_persona`` is empty string.

    Returns:
        dict mapping persona name → system prompt text (empty string for
        ``no_persona``).

    Asserts:
        All 55 names in POOL_16 ∪ NEGATIVES_FIXED ∪ HELD_OUT_35 have prompts;
        FAILS LOUD on any missing name.
    """
    import sys

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # 111-persona panel — everything in HELD_OUT_35 + POOL_16 + software_engineer,
    # kindergarten_teacher lives here.
    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    prompts: dict[str, str] = {name: info["prompt"] for name, info in ALL_EVAL_PERSONAS.items()}

    # The two negatives that are NOT in the 111-pool (synthetic negatives).
    prompts.setdefault("helpful_assistant", "You are a helpful assistant.")
    prompts.setdefault("no_persona", "")

    missing = [p for p in ALL_PERSONAS if p not in prompts]
    if missing:
        raise RuntimeError(
            f"load_all_persona_prompts: missing prompts for {missing!r}. "
            f"Check scripts/run_100_persona_leakage.py ALL_EVAL_PERSONAS + the "
            f"synthetic negatives fallback above."
        )
    return prompts


# ────────────────────────────────────────────────────────────────────────────
# Distance matrix loader.
#
# Plan v5 §4.7 expects ``eval_results/single_token_100_persona/
# cosine_distance_matrix_layer20.json``. The matrix file is NOT in the repo;
# instead the project ships the centroids tensor at
# ``eval_results/single_token_100_persona/centroids/centroids_layer20.pt``
# (produced one-time by ``scripts/analyze_100_persona_cosine.py --extract``).
#
# This loader prefers the cached JSON if present (fast, deterministic); else
# computes from centroids_layer20.pt and writes the JSON cache for future
# loads. If NEITHER exists, FAIL LOUD with a clear remediation message.
# ────────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _matrix_json_path() -> Path:
    return (
        _repo_root()
        / "eval_results"
        / "single_token_100_persona"
        / "cosine_distance_matrix_layer20.json"
    )


def _centroids_pt_path() -> Path:
    return (
        _repo_root()
        / "eval_results"
        / "single_token_100_persona"
        / "centroids"
        / "centroids_layer20.pt"
    )


def _persona_names_json_path() -> Path:
    return (
        _repo_root()
        / "eval_results"
        / "single_token_100_persona"
        / "centroids"
        / "persona_names.json"
    )


def _build_matrix_from_centroids() -> tuple[list[str], list[list[float]]]:
    """Compute 111x111 layer-20 cosine DISTANCE matrix from centroids tensor.

    Reads ``persona_names.json`` (the persona order matching the tensor) and
    ``centroids_layer20.pt`` (shape ``[N, hidden_size]``), L2-normalizes per
    row, computes pairwise cosine similarity, returns distance = 1 - sim.

    Caches the JSON to ``cosine_distance_matrix_layer20.json`` for fast reload.
    """
    import torch

    pt_path = _centroids_pt_path()
    names_path = _persona_names_json_path()
    if not pt_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            f"Centroids missing: {pt_path}\n"
            f"or persona-names index missing: {names_path}\n"
            f"Run one-time extraction (on a GPU pod):\n"
            f"  uv run python scripts/analyze_100_persona_cosine.py --extract --gpu 0\n"
            f"Then re-run this script."
        )

    persona_names: list[str] = json.loads(names_path.read_text())
    tensor: torch.Tensor = torch.load(pt_path, map_location="cpu")
    if tensor.shape[0] != len(persona_names):
        raise RuntimeError(
            f"Centroid tensor shape {tuple(tensor.shape)} does not match "
            f"persona_names length {len(persona_names)} — extraction inconsistent."
        )

    normed = tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (normed @ normed.T).clamp(-1.0, 1.0)  # cosine similarity
    distance_t = (1.0 - sim).cpu()
    distance = [
        [float(distance_t[i, j].item()) for j in range(len(persona_names))]
        for i in range(len(persona_names))
    ]

    # Cache the JSON for next load.
    cache_path = _matrix_json_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"persona_names": persona_names, "distance": distance}, indent=2)
    )
    return persona_names, distance


def load_cosine_distance_matrix() -> tuple[list[str], list[list[float]]]:
    """Load 111-persona layer-20 cosine DISTANCE matrix (1 − similarity).

    Prefers cached JSON at
    ``eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json``.
    On cache miss, computes from ``centroids/centroids_layer20.pt`` (one-time
    GPU extraction) and caches the JSON for future loads.

    Returns:
        ``(persona_names, distance_matrix)`` where ``distance_matrix[i][j] =
        1 − cos(h_20[names[i]], h_20[names[j]])``.

    Raises:
        FileNotFoundError if NEITHER the cached JSON nor the centroids tensor
        exists; the error names the one-time extraction command.
    """
    cache = _matrix_json_path()
    if cache.exists():
        data = json.loads(cache.read_text())
        names = data["persona_names"]
        # Accept either "distance" (this loader's key) or "matrix" (legacy
        # similarity matrix from extraction_method_comparison) — convert.
        if "distance" in data:
            return names, data["distance"]
        if "matrix" in data:
            sim = data["matrix"]
            dist = [[1.0 - sim[i][j] for j in range(len(names))] for i in range(len(names))]
            return names, dist
        raise RuntimeError(f"Distance-matrix JSON {cache} has neither 'distance' nor 'matrix' key.")
    # Cache miss → compute from centroids tensor.
    return _build_matrix_from_centroids()


def min_dist_to_set(
    held_out_persona: str,
    trained_set: list[str],
    names: list[str],
    distance: list[list[float]],
) -> float:
    """Min layer-20 cosine distance from a held-out persona to a trained subset.

    Returns ``min_j dist[i, j]`` for j over trained_set members present in names.
    Raises KeyError if held_out_persona not in names; ValueError if no trained_set
    member is in names.
    """
    if held_out_persona not in names:
        raise KeyError(f"held_out_persona {held_out_persona!r} not in distance matrix")
    i = names.index(held_out_persona)
    js = [names.index(p) for p in trained_set if p in names]
    if not js:
        raise ValueError(f"None of trained_set {trained_set!r} present in distance matrix names")
    return min(distance[i][j] for j in js)


def mean_dist_to_set(
    held_out_persona: str,
    trained_set: list[str],
    names: list[str],
    distance: list[list[float]],
) -> float:
    """Mean layer-20 cosine distance from a held-out persona to a trained subset."""
    if held_out_persona not in names:
        raise KeyError(f"held_out_persona {held_out_persona!r} not in distance matrix")
    i = names.index(held_out_persona)
    js = [names.index(p) for p in trained_set if p in names]
    if not js:
        raise ValueError(f"None of trained_set {trained_set!r} present in distance matrix names")
    return sum(distance[i][j] for j in js) / len(js)


def band_of(persona: str) -> str | None:
    """Return the band name for a held-out persona, or None if not in HELD_OUT_35."""
    for band, members in HELD_OUT_BANDS.items():
        if persona in members:
            return band
    return None


def assert_marker_token_id(tokenizer) -> None:
    """Fail-loud assert MARKER_TEXT encodes to the single canonical token id 83399.

    Per `.claude/rules/marker-leakage-measurement.md`: every marker-leakage
    experiment MUST assert this before any subprocess spawns. The leading
    space matters — bash strips it; thread with shlex.quote().
    """
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker token id assert FAILED. tokenizer.encode({MARKER_TEXT!r}, "
            f"add_special_tokens=False) returned {ids}, expected [{MARKER_TOKEN_ID}]. "
            f"Either the tokenizer changed or the marker text lost its leading space "
            f"(common bash-shell stripping bug — use shlex.quote when threading)."
        )


def assert_arm_marker_token_ids(tokenizer) -> dict[str, int]:
    """Assert every ARM_MARKERS entry is a single token under the tokenizer.

    Returns:
        Dict mapping marker text -> token id (the canonical single-token id).

    Raises:
        RuntimeError on any mismatch — multi-token markers break the
        MarkerOnlyDataCollator's single-id-per-marker contract.
    """
    out: dict[str, int] = {}
    for text, expected_id in ARM_MARKERS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids != [expected_id]:
            raise RuntimeError(
                f"Arm marker {text!r} tokenizes to {ids}, expected [{expected_id}] — "
                f"swap from the §4.9.1 fallback pool (¡/48813, ¿/28286, α/19043, β/33218)."
            )
        out[text] = expected_id
    return out


# ────────────────────────────────────────────────────────────────────────────
# Pairwise-disjointness guard (plan v5 §4.8 Phase 0 GUARD).
# ────────────────────────────────────────────────────────────────────────────


def assert_pairwise_disjoint_sets() -> None:
    """HARD assertion: POOL_16, NEGATIVES_FIXED, HELD_OUT_35 are pairwise disjoint.

    Plan v5 §4.8 Phase 0 GUARD (added in v2 after the BUG-1/BUG-2 fixes). Without
    this, the v1 ``software_engineer``-in-both-POOL-and-NEGATIVES bug returns,
    silently producing contradictory marker/no-marker training rows.

    Raises:
        RuntimeError if any of the three pairwise intersections is non-empty,
        OR if the union size is not 55 (16 + 4 + 35).
    """
    pool_set = set(POOL_16)
    neg_set = set(NEGATIVES_FIXED)
    ho_set = set(HELD_OUT_35)
    if not pool_set.isdisjoint(neg_set):
        raise RuntimeError(f"POOL_16 ∩ NEGATIVES_FIXED = {pool_set & neg_set!r} (must be empty)")
    if not pool_set.isdisjoint(ho_set):
        raise RuntimeError(f"POOL_16 ∩ HELD_OUT_35 = {pool_set & ho_set!r} (must be empty)")
    if not neg_set.isdisjoint(ho_set):
        raise RuntimeError(f"NEGATIVES_FIXED ∩ HELD_OUT_35 = {neg_set & ho_set!r} (must be empty)")
    expected = len(POOL_16) + len(NEGATIVES_FIXED) + len(HELD_OUT_35)
    actual = len(pool_set | neg_set | ho_set)
    if actual != expected or expected != 55:
        raise RuntimeError(
            f"|POOL_16 ∪ NEGATIVES_FIXED ∪ HELD_OUT_35| = {actual}, expected {expected} (= 55)"
        )
