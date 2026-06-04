# ruff: noqa: RUF001, RUF002, RUF003
"""Shared constants + pool helpers for issue #490 (per plan v1, post-v2 revision).

Forked from `_issue478_common.py` (read via `git show issue-478:scripts/_issue478_common.py`).
The three persona sets (POOL_16, NEGATIVES_FIXED, HELD_OUT_35), marker
identity, recipe (lr=5e-6, 2 ep, attn-only LoRA r=16), and chat-questions
plumbing all inherit unchanged. What this module adds:

- **5 condition slugs** (Shared 2D / Pooled-Single 2D A / Pooled-Single 2D B
  / Single D A / Single D B) per plan §4.1.
- **Per-source positive dose D=200** (so SHARED-2D = 400 total = #478 K=2
  per-cell total; POOLED-SINGLE-2D = 400 concentrated; SINGLE-D = 200) per
  plan §0.0 / §10.
- **Source-pair selection** (3 #478 ARM-matched + 5 deterministic
  `np.random.default_rng(490)` draws from combinations(POOL_16, 2) minus the
  3) per plan §4.4.
- **Subpanel construction helpers**: on-axis intermediate-C (cos-dist to BOTH
  A and B ≤ τ, ≥5 personas) + off-axis distance-matched (same mean dist to
  {A,B}, high asymmetry, ≥5 personas), layer-20 cosine primary, layer-21
  reported only as a robustness diagnostic.
- **Combiner registry**: mean (primary), LSE/Bernoulli-union, max — for
  the decomposition `gap_dosematched = SHARED-2D − combiner(POOLED-2D-A,
  POOLED-2D-B)`.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Marker (canonical per .claude/rules/marker-leakage-measurement.md) ──────

MARKER_TEXT = " ※"  # leading space + ※ (Qwen-2.5-7B token id 83399)
MARKER_TOKEN_ID = 83399

# ── Three-way pool split — inherited from #478, BIT-IDENTICAL ───────────────

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

# Inherited from #478 (6 distance bands).
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
ALL_PERSONAS: list[str] = POOL_16 + NEGATIVES_FIXED + HELD_OUT_35  # 55 unique

# ── #490-specific: 5 conditions per source-pair (plan §4.1) ─────────────────

CONDITION_SHARED_2D = "shared_2D"
CONDITION_POOLED_2D_A = "pooled_2D_A"
CONDITION_POOLED_2D_B = "pooled_2D_B"
CONDITION_SINGLE_D_A = "single_D_A"
CONDITION_SINGLE_D_B = "single_D_B"

CONDITIONS: tuple[str, ...] = (
    CONDITION_SHARED_2D,
    CONDITION_POOLED_2D_A,
    CONDITION_POOLED_2D_B,
    CONDITION_SINGLE_D_A,
    CONDITION_SINGLE_D_B,
)

# Per-source positive dose (plan §0.0 / §10).
# SHARED-2D : 2 sources × D=200 each → 400 total positives (= #478 K=2 total).
# POOLED-SINGLE-2D : 1 source × 2D=400 → 400 total positives, concentrated.
# SINGLE-D : 1 source × D=200 → 200 total positives.
D_PER_SOURCE: int = 200
TWO_D: int = 2 * D_PER_SOURCE  # 400

# Per-cell row totals (with 4 negatives × 100 = 400 → 1:1 pos:neg, see plan §4.7).
NEG_ROWS_PER_PERSONA: int = 100
NEG_TOTAL_ROWS: int = NEG_ROWS_PER_PERSONA * len(NEGATIVES_FIXED)  # 400

# Seeds — inherited from #478 (auto-escalate to 3 if Phase-0 power calc trips).
SEEDS: tuple[int, ...] = (42, 137)

# Pair selection (plan §4.4).
PAIR_RNG_SEED: int = 490
# 3 ARM-matched pairs lift their source sets from #478's ARM cells
# (K2_c16/c17/c18). These cells' positives are determined by #478's
# build_subsets() with rng=478. We re-derive them here to keep #490 self-
# contained — assert_inherited_pairs_match() cross-checks against
# the cached design_validation.json on disk if present (informational).
ARM_K2_MATCHED_CELL_IDS: tuple[str, ...] = ("K2_c16", "K2_c17", "K2_c18")
N_TOTAL_PAIRS: int = 8

# Subpanel construction (plan §4.4).
ONAXIS_MIN_PERSONAS: int = 5
OFFAXIS_MIN_PERSONAS: int = 5
OFFAXIS_MEAN_DIST_TOLERANCE: float = 0.02  # mean(dist({A,B}, c)) match window
ONAXIS_TAU_INITIAL: float = 0.05  # τ search starts here
ONAXIS_TAU_MAX: float = 0.20  # cap; HELD_OUT_35 spans up to ~0.20

# Combiner registry (plan §4.2).
COMBINERS: tuple[str, ...] = ("mean", "lse", "max")

# Power calc (plan §6.2 / §0.0).
POWER_DELTA_GEOM_THRESHOLD_NATS: float = 0.5
POWER_THRESHOLD_FRACTION: float = 0.80
ESCALATE_TO_3_SEEDS_AUTHORIZED: bool = True
ESCALATED_SEEDS: tuple[int, ...] = (42, 137, 9999)

# ── Training defaults — inherited from #478 §10 Reproducibility Card ────────

BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_TARGETS_NARROW: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]  # NO MLP — non-saturating per #311/#405/#448/#478

LR: float = 5.0e-6
EPOCHS: int = 2
PER_DEVICE_BATCH: int = 4
GRAD_ACCUM: int = 4
MAX_LENGTH: int = 1024
WARMUP_RATIO: float = 0.05
WEIGHT_DECAY: float = 0.0
R_CAP_SAFETY_MARGIN: int = 8
R_CAP_MIN: int = 64

# Smoke kill-criterion (inherited from #478 / #405).
SMOKE_G_LOGPROB_SOURCE_KILL: float = -0.1

# WandB
WANDB_PROJECT: str = "issue_490_dose_matched"

# Sentinel
SENTINEL_DIR_POD: str = "/workspace/logs"
SENTINEL_DIR_LOCAL_FALLBACK: str = "logs"

# Pool radius assertion (inherited).
POOL_RADIUS_MAX: float = 0.05


# ────────────────────────────────────────────────────────────────────────────
# Persona prompts + distance matrix — inherited helpers, re-imported by name
# to keep #490 modules self-contained without sys.path tricks.
# ────────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """Resolve the current worktree's repo root."""
    return Path(__file__).resolve().parent.parent


def _candidate_data_roots() -> list[Path]:
    """List candidate roots to search for read-only inherited artifacts (the
    111-persona cosine matrix, etc.).

    Order:
      1. This worktree's own repo root (preferred — keeps caches local once
         materialized).
      2. The ``.claude/worktrees/<name>`` parent's parent — i.e. the main
         working tree at the head of the repo. The layer-20 cosine matrix and
         centroids tensor live there (uncommitted, generated one-time) and
         are reusable across worktrees without copying.
    """
    here = _repo_root()
    out: list[Path] = [here]
    # If this worktree sits at <main_repo>/.claude/worktrees/<name>/, the main
    # repo root is here.parents[2]. The check protects against running from
    # an unrelated checkout.
    if len(here.parents) >= 3 and here.parent.name == "worktrees":
        candidate_main = here.parents[2]
        if (candidate_main / ".git").exists():
            out.append(candidate_main)
    return out


def _resolve_inherited_path(suffix: tuple[str, ...]) -> Path:
    """Return the first existing path matching ``suffix`` under any candidate
    root; if none exists, return the path under the local worktree root (which
    will fail-loud at read time with the canonical error).
    """
    for root in _candidate_data_roots():
        p = root.joinpath(*suffix)
        if p.exists():
            return p
    return _repo_root().joinpath(*suffix)


def _matrix_json_path() -> Path:
    return _resolve_inherited_path(
        (
            "eval_results",
            "single_token_100_persona",
            "cosine_distance_matrix_layer20.json",
        )
    )


def _centroids_pt_path() -> Path:
    return _resolve_inherited_path(
        (
            "eval_results",
            "single_token_100_persona",
            "centroids",
            "centroids_layer20.pt",
        )
    )


def _persona_names_json_path() -> Path:
    return _resolve_inherited_path(
        (
            "eval_results",
            "single_token_100_persona",
            "centroids",
            "persona_names.json",
        )
    )


def load_all_persona_prompts() -> dict[str, str]:
    """Load system prompts for every persona in POOL_16 ∪ NEGATIVES_FIXED ∪
    HELD_OUT_35.

    Inlined from #478's helper (the #478 module lives on the issue-478 branch
    only). Composes prompts from the 111-persona pool defined in
    ``scripts/run_100_persona_leakage.ALL_EVAL_PERSONAS`` plus the two
    synthetic negatives (``helpful_assistant``, ``no_persona``) that are not
    in that pool. The persona inventory is BIT-IDENTICAL to #478 by design.

    Per #478 §12 assumption #20: ``assistant`` (in HELD_OUT_35) and
    ``helpful_assistant`` (a NEGATIVES_FIXED member) are functionally distinct
    because they live in different sets, even though both prompts are
    "You are a helpful assistant." Kept separate by name; the held-out panel
    uses ``assistant``; the negative uses ``helpful_assistant``.
    ``no_persona`` is empty string.
    """
    import sys

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    prompts: dict[str, str] = {name: info["prompt"] for name, info in ALL_EVAL_PERSONAS.items()}
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


def _build_matrix_from_centroids() -> tuple[list[str], list[list[float]]]:
    """Compute 111x111 layer-20 cosine DISTANCE matrix from centroids tensor.

    Inlined from #478. Reads ``persona_names.json`` (the persona order matching
    the tensor) and ``centroids_layer20.pt`` (shape ``[N, hidden_size]``),
    L2-normalizes per row, computes pairwise cosine similarity, returns
    distance = 1 - sim. Caches the JSON for next load.
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
    sim = (normed @ normed.T).clamp(-1.0, 1.0)
    distance_t = (1.0 - sim).cpu()
    distance = [
        [float(distance_t[i, j].item()) for j in range(len(persona_names))]
        for i in range(len(persona_names))
    ]

    cache_path = _matrix_json_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"persona_names": persona_names, "distance": distance}, indent=2)
    )
    return persona_names, distance


def load_cosine_distance_matrix() -> tuple[list[str], list[list[float]]]:
    """Load 111-persona layer-20 cosine DISTANCE matrix (1 − similarity).

    Inlined from #478. Prefers cached JSON at
    ``eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json``;
    on cache miss, computes from centroids and caches.

    The legacy "matrix"-keyed silent-invert branch is intentionally NOT
    reproduced — #478 round-2 MINOR 7 fix says fail loud on a "matrix" key
    without an explicit ``"metric": "similarity"`` annotation.
    """
    cache = _matrix_json_path()
    if cache.exists():
        data = json.loads(cache.read_text())
        names = data["persona_names"]
        if "distance" in data:
            return names, data["distance"]
        if "matrix" in data:
            metric = data.get("metric", "")
            metric_l = str(metric).lower().strip()
            # Accept three canonical distance annotations + the explicit
            # similarity annotation. Anything else → fail loud (per #478
            # round-2 MINOR 7 — never silently invert a legacy "matrix"
            # without an explicit annotation).
            if metric_l in ("1 - cosine", "1-cosine", "cosine distance", "distance"):
                return names, data["matrix"]
            if metric_l in ("similarity", "cosine", "cosine similarity"):
                sim = data["matrix"]
                dist = [[1.0 - sim[i][j] for j in range(len(names))] for i in range(len(names))]
                return names, dist
            raise RuntimeError(
                f"Distance-matrix JSON {cache} has a 'matrix' key with "
                f"metric={metric!r}, which is not in the accepted distance "
                f"set ('1 - cosine', 'cosine distance', 'distance') nor the "
                f"similarity set ('similarity', 'cosine'). Refusing to "
                f"silently invert; rebuild the cache or annotate."
            )
        raise RuntimeError(f"Distance-matrix JSON {cache} has neither 'distance' nor 'matrix' key.")
    return _build_matrix_from_centroids()


def load_cosine_distance_matrix_layer(layer: int) -> tuple[list[str], list[list[float]]]:
    """Load layer-N cosine distance matrix; call site for the layer-21
    robustness diagnostic (plan §4.4).

    Layer 20 is primary (#478 parity). Layer 21 is the persona-vectors default
    per `.claude/rules/persona-distance-metrics.md` and is reported here only
    as a robustness check. Falls back to layer-20 if the per-layer centroids
    tensor is not present (the design-validation script logs the fallback
    explicitly so the body call-out remains honest).
    """
    if layer == 20:
        return load_cosine_distance_matrix()

    pt_path = _resolve_inherited_path(
        (
            "eval_results",
            "single_token_100_persona",
            "centroids",
            f"centroids_layer{layer}.pt",
        )
    )
    names_path = _resolve_inherited_path(
        (
            "eval_results",
            "single_token_100_persona",
            "centroids",
            "persona_names.json",
        )
    )
    if not pt_path.exists() or not names_path.exists():
        return load_cosine_distance_matrix()

    import torch  # local import; CPU-only here

    persona_names: list[str] = json.loads(names_path.read_text())
    tensor: torch.Tensor = torch.load(pt_path, map_location="cpu")
    normed = tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (normed @ normed.T).clamp(-1.0, 1.0)
    distance_t = (1.0 - sim).cpu()
    distance = [
        [float(distance_t[i, j].item()) for j in range(len(persona_names))]
        for i in range(len(persona_names))
    ]
    return persona_names, distance


def assert_marker_token_id(tokenizer) -> None:
    """Fail-loud assert MARKER_TEXT encodes to single token id 83399."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker token id assert FAILED. tokenizer.encode({MARKER_TEXT!r}, "
            f"add_special_tokens=False) returned {ids}, expected [{MARKER_TOKEN_ID}]. "
            f"Either the tokenizer changed or the marker text lost its leading space "
            f"(common bash-shell stripping bug — use shlex.quote when threading)."
        )


def assert_pairwise_disjoint_sets() -> None:
    """HARD assertion: POOL_16, NEGATIVES_FIXED, HELD_OUT_35 are pairwise disjoint."""
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


# ────────────────────────────────────────────────────────────────────────────
# Source-pair selection (plan §4.4)
# ────────────────────────────────────────────────────────────────────────────


def _derive_arm_matched_pairs() -> list[tuple[str, str]]:
    """Re-derive K2_c16/c17/c18 source sets from #478's build_subsets() draw.

    The ARM-matched cells lift their positives from #478's deterministic
    seeded draw (`numpy.random.default_rng(478)`). To match by source-set
    without depending on the #478 cell_specs.json file, we re-run the SAME
    draw here. Returns the 3 K=2 pairs as (A, B) tuples (sorted alphabetically
    so pair identity is canonical).
    """
    rng = np.random.default_rng(478)
    # K=1: 16 deterministic singletons (consumes no RNG draw).
    # K=2: 8 random pairs from C(16, 2) (28 total).
    all_pairs = list(combinations(POOL_16, 2))
    k2_idx = rng.choice(len(all_pairs), size=8, replace=False)
    # ARM_K2_MATCHED_CELLS = K2_c16, K2_c17, K2_c18 → first 3 of the K=2 draw
    # (K=1 cells consume IDs c00..c15; K=2 starts at c16).
    chosen = [tuple(sorted(all_pairs[int(i)])) for i in k2_idx[:3]]
    return chosen


def build_source_pairs() -> list[dict]:
    """Build the 8 source-pairs per plan §4.4.

    Returns:
        List of dicts, each:
            {"pair_id": "pair0".."pair7",
             "A": <persona>, "B": <persona>,
             "origin": "arm_matched_478" | "rng_490",
             "matched_cell_id": <#478 cell_id or None>}
    """
    arm_matched = _derive_arm_matched_pairs()  # 3 (A,B) tuples
    arm_matched_set = {tuple(sorted(p)) for p in arm_matched}

    # 5 RNG-490 draws from combinations(POOL_16, 2) minus the 3 inherited.
    all_pairs = list(combinations(POOL_16, 2))
    candidates = [p for p in all_pairs if tuple(sorted(p)) not in arm_matched_set]
    rng = np.random.default_rng(PAIR_RNG_SEED)
    extra_idx = rng.choice(len(candidates), size=5, replace=False)
    extras = [tuple(sorted(candidates[int(i)])) for i in extra_idx]

    out: list[dict] = []
    for i, (A, B) in enumerate(arm_matched):
        out.append(
            {
                "pair_id": f"pair{i}",
                "A": A,
                "B": B,
                "origin": "arm_matched_478",
                "matched_cell_id": ARM_K2_MATCHED_CELL_IDS[i],
            }
        )
    for j, (A, B) in enumerate(extras):
        out.append(
            {
                "pair_id": f"pair{3 + j}",
                "A": A,
                "B": B,
                "origin": "rng_490",
                "matched_cell_id": None,
            }
        )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Subpanel construction (plan §4.4)
# ────────────────────────────────────────────────────────────────────────────


def _persona_dist(persona: str, other: str, names: list[str], distance: list[list[float]]) -> float:
    if persona not in names:
        raise KeyError(f"persona {persona!r} not in distance matrix names")
    if other not in names:
        raise KeyError(f"persona {other!r} not in distance matrix names")
    return distance[names.index(persona)][names.index(other)]


def build_onaxis_offaxis_subpanels(
    A: str,
    B: str,
    candidates: list[str],
    names: list[str],
    distance: list[list[float]],
    *,
    min_onaxis: int = ONAXIS_MIN_PERSONAS,
    min_offaxis: int = OFFAXIS_MIN_PERSONAS,
    mean_dist_tol: float = OFFAXIS_MEAN_DIST_TOLERANCE,
) -> dict:
    """For a source-pair (A, B), partition `candidates` into two distance-
    matched subpanels (plan §4.4).

    on-axis intermediate-C : cos-dist to BOTH A and B ≤ τ ("between").
    off-axis matched       : same mean(dist(c,A), dist(c,B)) as the on-axis
                             set, high asymmetry (|d_A − d_B| above median),
                             ≥ min_offaxis personas.

    Algorithm:
      1. For each candidate c, compute d_A = dist(c, A), d_B = dist(c, B),
         mean_d = ½(d_A + d_B), asym = |d_A − d_B|.
      2. Sweep τ up from ONAXIS_TAU_INITIAL to ONAXIS_TAU_MAX in 0.005 steps;
         pick the smallest τ that admits ≥ min_onaxis candidates with
         d_A ≤ τ AND d_B ≤ τ. If no τ ≤ TAU_MAX yields ≥ min_onaxis, the
         on-axis subpanel is INFEASIBLE.
      3. From the REMAINING candidates (not on-axis), compute their mean_d
         distribution; target = mean(on-axis subpanel's mean_d).
         Sort by asymmetry (descending); take the highest-asymmetry candidates
         whose mean_d is within `mean_dist_tol` of the target. If fewer than
         min_offaxis are within tolerance, RELAX the tolerance by 0.005 steps
         up to 5× initial; if still infeasible, INFEASIBLE.

    Returns:
        {"feasible": bool,
         "reason": str (only if not feasible),
         "tau": <chosen τ or None>,
         "on_axis": [persona, ...],
         "off_axis": [persona, ...],
         "on_axis_mean_d": float,
         "off_axis_mean_d": float,
         "on_axis_personas_with_d": [(p, d_A, d_B, mean_d, asym), ...],
         "off_axis_personas_with_d": [(p, d_A, d_B, mean_d, asym), ...]}
    """
    per_c = []
    for c in candidates:
        if c in (A, B):
            continue  # never include training sources in held-out subpanels
        try:
            d_A = _persona_dist(c, A, names, distance)
            d_B = _persona_dist(c, B, names, distance)
        except KeyError:
            continue
        mean_d = 0.5 * (d_A + d_B)
        asym = abs(d_A - d_B)
        per_c.append({"persona": c, "d_A": d_A, "d_B": d_B, "mean_d": mean_d, "asym": asym})

    # Step 2 — find smallest τ admitting ≥ min_onaxis.
    tau = None
    on_axis: list[dict] = []
    tau_step = 0.005
    n_steps = int((ONAXIS_TAU_MAX - ONAXIS_TAU_INITIAL) / tau_step) + 1
    for k in range(n_steps):
        candidate_tau = ONAXIS_TAU_INITIAL + k * tau_step
        members = [c for c in per_c if c["d_A"] <= candidate_tau and c["d_B"] <= candidate_tau]
        if len(members) >= min_onaxis:
            tau = candidate_tau
            on_axis = members
            break

    if tau is None:
        return {
            "feasible": False,
            "reason": (
                f"on-axis INFEASIBLE: no τ ∈ [{ONAXIS_TAU_INITIAL}, {ONAXIS_TAU_MAX}] "
                f"admits ≥{min_onaxis} candidates with d_A ≤ τ AND d_B ≤ τ"
            ),
            "tau": None,
            "on_axis": [],
            "off_axis": [],
            "on_axis_mean_d": float("nan"),
            "off_axis_mean_d": float("nan"),
            "on_axis_personas_with_d": [],
            "off_axis_personas_with_d": [],
        }

    on_axis_names = {c["persona"] for c in on_axis}
    on_axis_mean_d = sum(c["mean_d"] for c in on_axis) / len(on_axis)

    # Step 3 — off-axis: same mean_d (within tolerance), high asymmetry.
    remaining = [c for c in per_c if c["persona"] not in on_axis_names]
    if not remaining:
        return {
            "feasible": False,
            "reason": "off-axis INFEASIBLE: zero candidates left after on-axis",
            "tau": tau,
            "on_axis": [c["persona"] for c in on_axis],
            "off_axis": [],
            "on_axis_mean_d": on_axis_mean_d,
            "off_axis_mean_d": float("nan"),
            "on_axis_personas_with_d": [
                (c["persona"], c["d_A"], c["d_B"], c["mean_d"], c["asym"]) for c in on_axis
            ],
            "off_axis_personas_with_d": [],
        }

    # Median asymmetry across remaining → cut high-asymmetry candidates first.
    asyms = sorted(c["asym"] for c in remaining)
    median_asym = asyms[len(asyms) // 2]
    high_asym_pool = [c for c in remaining if c["asym"] > median_asym]
    if len(high_asym_pool) < min_offaxis:
        # Fall back to all remaining with the highest asymmetries.
        high_asym_pool = sorted(remaining, key=lambda x: -x["asym"])

    # Try widening tolerance up to 5× initial.
    off_axis: list[dict] = []
    used_tol = mean_dist_tol
    for k in range(5):
        used_tol = mean_dist_tol * (1 + k)
        in_window = [c for c in high_asym_pool if abs(c["mean_d"] - on_axis_mean_d) <= used_tol]
        # Within window, pick top-asymmetry candidates first.
        in_window_sorted = sorted(in_window, key=lambda x: -x["asym"])
        if len(in_window_sorted) >= min_offaxis:
            off_axis = in_window_sorted[: max(min_offaxis, min(len(in_window_sorted), 10))]
            break

    # GEOMETRY-FALLBACK: when POOL_16 is geometrically tight (radius ≈ 0.029),
    # on-axis personas live at low mean_d (NEAR band) AND the held-out
    # personas with the highest asymmetry to {A,B} also live mostly off-band.
    # Strict mean-d-matching with high-asymmetry is then infeasible; the
    # principled fallback is to take the TOP-N highest-asymmetry remaining
    # personas WITHOUT the mean_d window, and surface the mean_d delta in
    # the per-pair diagnostic. This honors plan §3 Q2 intent — "off-axis =
    # NOT between A and B" — while acknowledging the panel-vs-pool geometry
    # makes a strict distance match impossible. The analyzer reports the
    # per-pair mean_d delta so the headline narrates the residual mismatch
    # honestly. (The strict mean-d-matched off-axis would require a different
    # HELD_OUT panel construction; see plan deviations log.)
    used_method = "mean_d_matched"
    if len(off_axis) < min_offaxis:
        sorted_by_asym = sorted(remaining, key=lambda x: -x["asym"])
        off_axis = sorted_by_asym[: max(min_offaxis, min(len(sorted_by_asym), 10))]
        used_method = "top_n_asymmetry_fallback"

    off_axis_mean_d = sum(c["mean_d"] for c in off_axis) / len(off_axis)

    return {
        "feasible": True,
        "reason": "OK",
        "tau": tau,
        "on_axis": [c["persona"] for c in on_axis],
        "off_axis": [c["persona"] for c in off_axis],
        "on_axis_mean_d": on_axis_mean_d,
        "off_axis_mean_d": off_axis_mean_d,
        "mean_d_match_delta": abs(on_axis_mean_d - off_axis_mean_d),
        "tolerance_used": used_tol,
        "off_axis_selection_method": used_method,
        "on_axis_personas_with_d": [
            (c["persona"], c["d_A"], c["d_B"], c["mean_d"], c["asym"]) for c in on_axis
        ],
        "off_axis_personas_with_d": [
            (c["persona"], c["d_A"], c["d_B"], c["mean_d"], c["asym"]) for c in off_axis
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# Combiner registry (plan §4.2)
# ────────────────────────────────────────────────────────────────────────────


def combiner(name: str, values: list[float]) -> float:
    """Apply named combiner to a list of values.

    mean → ½(a+b) (geometric-mean-of-probabilities null on log-probs).
    max  → max(a, b) (rules out "C is just near the better single source").
    lse  → log(p_a + p_b − p_a·p_b) = log(1 − (1-p_a)(1-p_b)) where
           p = exp(value). Bernoulli-union: "either source's marker could
           fire". Computed in log-space for numerical stability.
    """
    if not values:
        raise ValueError("combiner: empty values list")
    if name == "mean":
        return sum(values) / len(values)
    if name == "max":
        return max(values)
    if name == "lse":
        # log(1 - ∏(1 - exp(v))). For numerical stability use logsumexp-like
        # reduction over the probabilities.
        # We compute log P_union = log(1 - ∏ (1 - exp(v))).
        # exp(v) is small or near 1 — use complement via log1p:
        #   log(1 - exp(v)) = log(-expm1(v))   (valid for v < 0)
        # If any v >= 0 we clamp to v=0 → log(1-1)= -inf → treat the union
        # as ≥0 (saturating). For our marker-leakage DV v is always ≤ 0.
        log_one_minus_p: list[float] = []
        for v in values:
            if v >= 0:
                # Saturated — union also saturates.
                return 0.0
            log_one_minus_p.append(math.log1p(-math.exp(v)))
        log_complement_product = sum(log_one_minus_p)
        # log P_union = log(1 - exp(log_complement_product))
        if log_complement_product >= 0:
            return 0.0
        return math.log1p(-math.exp(log_complement_product))
    raise ValueError(f"Unknown combiner {name!r}")
