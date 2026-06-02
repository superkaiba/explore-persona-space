# ruff: noqa: RUF002, RUF003
"""Shared constants + pool helpers for issue #405.

Imported by every issue405_*.py script so the three-way pool split + marker
identity + per-K subset counts are defined ONCE.
"""

from __future__ import annotations

# ── Marker (canonical per .claude/rules/marker-leakage-measurement.md) ──────

MARKER_TEXT = " ※"  # leading space + ※ (Qwen-2.5-7B token id 83399)
MARKER_TOKEN_ID = 83399

# ── Three-way pool split (FIXED across the experiment, §4.1 of plan v2) ─────

POOL: list[str] = [
    "paramedic",
    "navy_seal",
    "villain",
    "librarian",
    "data_scientist",
    "medical_doctor",
    "french_person",
    "poet",
]

NEGATIVES_FIXED: list[str] = [
    "software_engineer",
    "kindergarten_teacher",
    "helpful_assistant",
    "no_persona",
]

HELD_OUT: list[str] = [
    "cybersec_consultant",
    "pentester",
    "private_investigator",
    "army_medic",
    "surgeon",
    "police_officer",
    "florist",
    "comedian",
]

ALL_PERSONAS: list[str] = POOL + NEGATIVES_FIXED + HELD_OUT

# ── Cell layout (§4.2) ──────────────────────────────────────────────────────

K_VALUES: tuple[int, ...] = (1, 2, 4, 8)
SUBSETS_PER_K: dict[int, int | None] = {1: None, 2: 6, 4: 6, 8: None}  # None = all
SUBSET_RNG_SEED = 405

CORE_ROWS_PER_CELL = 800
CORE_TOTAL_POSITIVE_ROWS = 400
CORE_NEG_ROWS_PER_PERSONA = 100  # 4 negs * 100 = 400, matches positives 1:1

# Dose-control arm (§4.6 build_dose_control_specs)
DOSE50_PERSONAS = ["paramedic", "villain", "poet"]
DOSE50_ROWS_PER_POSITIVE = 50

# Ablation arm (§4.6 build_ablation_specs)
ABLATION_POSITIVES = ("paramedic", "villain", "librarian", "poet")
ABLATION_NEGATIVES = ["surgeon", "police_officer", "florist", "comedian"]
ABLATION_HELD_OUT = ["cybersec_consultant", "pentester", "private_investigator", "army_medic"]

# ── Training defaults (§4.4, §11) ───────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS_NARROW = ["q_proj", "k_proj", "v_proj", "o_proj"]  # NO MLP — non-saturating

LR = 5.0e-6
EPOCHS = 2
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.0
R_CAP_SAFETY_MARGIN = 8  # 1024 − prompt_len − 8 = max_new_tokens for R (Fix D)
R_CAP_MIN = 64  # if R_cap < 64, prompt is too long → FAIL LOUD

# Smoke kill-criterion (§4.9, §7)
SMOKE_G_LOGPROB_SOURCE_KILL = -0.1  # if g_logprob_source > -0.1 → STOP (saturated)
SMOKE_G_LOGPROB_SOURCE_TARGET_RANGE = (-8.0, -3.0)  # non-saturating headroom

# WandB
WANDB_PROJECT = "issue_405_kdiversity"

# Sentinel + log path conventions (pod-side, per CLAUDE.md pod-side rule)
SENTINEL_DIR_POD = "/workspace/logs"
SENTINEL_DIR_LOCAL_FALLBACK = "logs"  # used when running smoke locally (no /workspace)


def load_all_persona_prompts() -> dict[str, str]:
    """Load all 20 persona system prompts from the canonical ORIGINAL_20 list.

    The cached layer-20 cosine matrix at
    `eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json`
    was extracted under exactly these prompts (provenance-matched, §4.5 PHASE 0).

    Returns:
        dict mapping persona name → system prompt text (empty string for
        `no_persona`).

    Asserts:
        All 20 names in POOL ∪ NEGATIVES_FIXED ∪ HELD_OUT are present in
        ORIGINAL_20; FAILS LOUD on any missing name.
    """
    # Inline import — extract_centroids_and_analyze.py is a scripts/ file, not
    # a package module; importing it via sys.path keeps the import local.
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from extract_centroids_and_analyze import ORIGINAL_20

    prompts = dict(ORIGINAL_20)
    missing = [p for p in ALL_PERSONAS if p not in prompts]
    if missing:
        raise RuntimeError(
            f"ORIGINAL_20 in scripts/extract_centroids_and_analyze.py is missing "
            f"prompts for plan-named personas: {missing!r}. "
            f"Refusing to proceed — fix the prompt list or pool split."
        )
    return prompts


def load_cosine_distance_matrix() -> tuple[list[str], list[list[float]]]:
    """Load cached layer-20 cosine SIMILARITY matrix and return as DISTANCE.

    Returns:
        (persona_names, distance_matrix) where distance = 1 − similarity.
        distance_matrix[i][j] = 1 − cos(h_20[names[i]], h_20[names[j]]).
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "eval_results" / "extraction_method_comparison" / "cosine_matrix_a_layer20.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Cached layer-20 cosine matrix missing: {p}. "
            f"Re-run scripts/extract_centroids_and_analyze.py to regenerate."
        )
    data = json.loads(p.read_text())
    names = data["persona_names"]
    sim = data["matrix"]
    dist = [[1.0 - sim[i][j] for j in range(len(names))] for i in range(len(names))]
    return names, dist


def min_dist_to_set(
    held_out_persona: str,
    trained_set: list[str],
    names: list[str],
    distance: list[list[float]],
) -> float:
    """Min layer-20 cosine distance from a held-out persona to a trained subset."""
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


def assert_marker_token_id(tokenizer) -> None:
    """Fail-loud assert that MARKER_TEXT encodes to the single canonical token id.

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
