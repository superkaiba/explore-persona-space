# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 — 4-knob contrastive-LoRA-SFT recipe sweep on the #411 villain baseline,
marker-implantation eval surface.

Single source persona (villain), single seed (42), 11 cells sweeping four data-
composition knobs one at a time around the #411 anchor: pos_ex per persona,
pos_personas, neg_ex per persona, neg_personas. The recipe (lr, epochs, LoRA
rank/alpha/dropout, batch size, max_seq, optimizer, schedule, warmup, precision)
is inherited 1:1 from #411 per clarify-answers v2's explicit binding.

Eval: teacher-forced log p(" ※") at the END of a FIXED canonical generic
response per eval_question (same response across all 24 evaluation personas;
only the system prompt varies). Position chosen to match where marker training
puts gradient mass (#396 villain end-of-response = -15.20 nats vs base ≈ -19.30
nats; k=0 stayed at base under #396's stronger recipe). Per CLAUDE.md mandatory
tracking: per-cell × per-(eval_persona, question) log p(" ※") trajectory across
training steps via a TrainerCallback.

Modules:
    persona_registry         — Pre-Phase 0a: pull #411 HF training-pool JSONLs
                                and reconstruct ALL_PERSONAS order; build-time
                                assertions on villain + assistant bystanders.
    build_wrong_claim_pool   — Pre-Phase 0: Sonnet 4.5 top-up to 850 (Q,
                                response) pairs (200 cached + ~650 new) + one
                                canonical response per EVAL_QUESTIONS entry
                                for the eval rig.
    build_training_data      — Phase 1 per-cell data prep: per-persona seeded
                                disjoint slices from the 850-pair union pool.
    extend_centroids         — Phase 0.5: extend the 111-persona layer-20
                                centroid file to cover the 24-panel +
                                persona-registry personas.
    marker_trajectory_callback — TrainerCallback: 6-persona × 5-question subset
                                teacher-forced log p(" ※") trajectory at every
                                step for the first 50, every 5 steps after.
    eval_marker_leakage      — Phase 2 per-cell eval: 24-panel × 20-question
                                teacher-forced log p(" ※") at end-of-canonical-
                                response (primary) + k=0 (diagnostic).
    analyze                  — Phase 3: per-cell mean bystander Δ + bootstrap
                                CI, calibrated headline threshold, permutation
                                null, per-cell Spearman ρ(per-bystander Δ,
                                nearest_neg_distance), per-knob monotonicity.
"""

SOURCE_PERSONAS: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
"""Copied from experiments/sycophancy_implantation_411/__init__.py — same 6
sources whose #411 training-pool JSONLs are used to reconstruct ALL_PERSONAS
order via persona_registry."""

# Published #99 per-source Spearman rho on layer-20 centroid cosine. Kept for
# parity with the #411 layout (analyze.py does not consume it here; #448's
# headline is monotonicity + permutation null, not the cosine-Spearman pair).
RHO_99_BY_SOURCE: dict[str, float] = {
    "villain": 0.467,
    "comedian": 0.433,
    "assistant": -0.442,
    "qwen_default": -0.690,
    "software_engineer": -0.203,
    "kindergarten_teacher": -0.378,
}

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
"""Base model. Inherited from #411 (and from the marker-implantation family)."""

MARKER_TEXT = " ※"
"""Single-token marker. Leading space form. Qwen-2.5-7B token id 83399.
Validated in #395 (base log-prob median ~-19 nats); adopted from #396 onward.
Per CLAUDE.md 'Default marker for new marker-leakage experiments'."""

EXPECTED_MARKER_TOKEN_ID = 83399

LAYER_20_CENTROIDS_PATH = "eval_results/issue_448/centroids/centroids_layer20.pt"

# Multi-positive cell selection (cells 5 + 6). Picked by ascending cosine to
# villain from the marker-implantation family's `PERSONAS` set
# (src/explore_persona_space/personas.py). Source: plan §11 + clarify-answers
# v2 per-persona binding.
MULTI_POSITIVE_PERSONAS_C5: tuple[str, ...] = ("villain", "comedian")
MULTI_POSITIVE_PERSONAS_C6: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "software_engineer",
)

# Cell specs — see plan §4.2 table. (cell_slug, plain_english_name, pos_ex_per_p,
# pos_personas, neg_ex_per_p, neg_personas).
CELL_SPECS: tuple[tuple[str, str, int, int, int, int], ...] = (
    ("c1_anchor", "Anchor", 200, 1, 200, 2),
    ("c2_pos_ex_100", "+pos-ex-100-per-persona", 100, 1, 200, 2),
    ("c3_pos_ex_400", "+pos-ex-400-per-persona", 400, 1, 200, 2),
    ("c4_pos_ex_800", "+pos-ex-800-per-persona", 800, 1, 200, 2),
    ("c5_pos_personas_2", "+pos-personas-2", 200, 2, 200, 2),
    ("c6_pos_personas_4", "+pos-personas-4", 200, 4, 200, 2),
    ("c7_neg_ex_100", "+neg-ex-100", 200, 1, 100, 2),
    ("c8_neg_ex_400", "+neg-ex-400", 200, 1, 400, 2),
    ("c9_neg_ex_800", "+neg-ex-800", 200, 1, 800, 2),
    ("c10_neg_personas_4", "+neg-personas-4", 200, 1, 200, 4),
    ("c11_neg_personas_8", "+neg-personas-8", 200, 1, 200, 8),
)

# Source persona that anchors every cell. Plan §3 + §11.
SOURCE_PERSONA = "villain"

# Single seed across all cells (clarify-answers v2).
SEED = 42

# Per-persona seed-salting scheme (plan §4.0bis / §4.2 M3):
# i-th positive persona: random.Random(SEED + i).sample(union_pool, N)
# j-th negative persona: random.Random(SEED + NEG_SEED_OFFSET + j).sample(...)
NEG_SEED_OFFSET = 100

# No-persona-contrastive rows per cell (matches #411 anchor; kept constant
# across all 11 cells per plan §4.0bis).
N_NO_PERSONA_CONTRASTIVE = 100

# Source-self training-success floor at end-of-canonical-response (plan §6 M1).
# -12 nats = 7 nats above #395 base; ~3 nats above #396's villain achievement.
SOURCE_SELF_TRAINING_FLOOR_NATS = -12.0

# Target pool size after Pre-Phase 0 Sonnet top-up (plan §4.0bis).
# 850 = 200 cached + 650 generated. Covers c4 (800 distinct rows per positive).
N_UNION_POOL_TARGET = 850
N_CACHED_PAIRS = 200
N_TOPUP_PAIRS = 650

# Trajectory callback subset (plan §4.0quater).
TRAJECTORY_N_PERSONAS = 6
TRAJECTORY_N_QUESTIONS = 5
TRAJECTORY_STEP_DENSE_FIRST_N = 50
TRAJECTORY_STEP_SPARSE_EVERY = 5
