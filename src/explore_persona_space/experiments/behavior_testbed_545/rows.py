"""Issue #545 train-row registry (plan sections 4.1 + 4.2).

Every (row, arm, seed) cell resolves here to a concrete training invocation:

- ``recipe_kind="hydra_turner"``  -> subprocess ``scripts/train.py condition=...
  training=turner_em lora=turner_em +training.max_steps=375 seed=S +gpu_id=G``
  (the #458 launch verbatim; B1/B2).
- ``recipe_kind="train_lora"``    -> in-subprocess ``train_lora()`` with the
  ``TrainLoraConfig`` overrides recorded per row (B3-B7, B9, B10 + arms).
- ``recipe_kind="reuse_adapter"`` -> #503 Bucket-D adapter downloaded from the
  HF model repo (B8; zero train GPU, fitness-checked in plan section 10).
- ``recipe_kind="fullft"``        -> ``accelerate launch`` ZeRO-3 via
  ``scripts/train_stage_sft.py`` + ``configs/condition/i545_badmed_fullft.yaml``
  (the bad-medical full-FT control arm).

Plain-English row ids are used everywhere; ``i545_*`` slugs only in configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import IM_END_TOKEN_ID, MARKER_TEXT

ANCHOR_SEEDS = (0, 137, 42)
DEFAULT_SEEDS = (0, 137)

# Generic judged-behavior recipe for rows without an exact plain-SFT parent
# (#411 house recipe; plan section 11). NOTE: train_lora() hardcodes
# use_rslora=True, so the EXECUTED #411 recipe included rsLoRA — inheriting the
# same code path preserves the recipe even though #411's card does not name it.
GENERIC_RECIPE: dict = {
    "lr": 1e-5,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "save_strategy": "epoch",
    "save_total_limit": 3,
}

# #444 fact recipe (plan section 11; rsLoRA load-bearing — hardcoded in train_lora).
FACT_RECIPE: dict = {
    "lr": 2e-4,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": 1,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "warmup_ratio": 0.05,
    "save_strategy": "epoch",
}

# Marker recipe (.claude/rules/marker-training-recipe.md; #478/#530).
MARKER_RECIPE: dict = {
    "lr": 5e-6,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "epochs": 20,  # ceiling; the band-stop callback terminates inside [5,12] nat
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 2048,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "marker_only_loss": True,
    "marker_text": MARKER_TEXT,
    "marker_tail_tokens": 0,
    "marker_band_stop": True,
    "marker_band_low_nats": 5.0,
    "marker_band_high_nats": 12.0,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 4,
}

# B10 warmth recipe (#516 shape + #530 LR lesson; dose in fractional epochs via
# max_steps, computed by the train-cell runner from the corpus size).
WARMTH_RECIPE: dict = {
    "lr": 5e-6,
    "lr_scheduler_type": "constant",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "weight_decay": 0.0,
    "epochs": 4,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "save_strategy": "steps",  # runner sets save_steps = steps_per_epoch // 2
}

# turner_em parity kwargs for train_lora-path arms on B1 rows (klreg arm), so
# the arm-vs-primary contrast only changes the manipulated variable.
TURNER_PARITY: dict = {
    "lr": 2e-5,
    "lora_r": 32,
    "lora_alpha": 256,
    "lora_dropout": 0.0,
    "batch_size": 2,
    "grad_accum": 8,
    "max_length": 2048,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "warmup_steps": 5,
    "max_steps": 375,
    "save_strategy": "steps",
    "save_steps": 125,
    "save_total_limit": 3,
}


@dataclass(frozen=True)
class RowSpec:
    """One train behavior (plan section 4.1) + its arms (section 4.2)."""

    row_id: str
    family: str  # B1..B10
    display_name: str
    recipe_kind: str  # hydra_turner | train_lora | reuse_adapter
    expected: str  # dense | null | surprising | unknown
    phase: str  # p1 | p2
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    arms: tuple[str, ...] = ("primary",)
    # hydra_turner rows: the condition name scripts/train.py receives.
    hydra_condition: str | None = None
    # train_lora rows: TrainLoraConfig overrides + corpus id under corpora_dir().
    train_lora_overrides: dict = field(default_factory=dict)
    corpus: str | None = None
    # Corpora that need GPU prep on the pod before training (base-model
    # on-policy generations): "marker" appends the marker after greedy base
    # responses; "cn" arms add base-model negative rows on the same questions.
    gpu_prep: str | None = None
    # reuse_adapter rows: seed -> HF model-repo subfolder.
    reuse_subfolders: dict | None = None
    # Diagonal manipulation check: the column id whose in-distribution battery
    # reads this row's implant strength (dose-to-target reads this column).
    diagonal_column: str = ""
    dose_band: tuple[float, float] = (0.60, 0.90)
    data_tier: str = ""
    notes: str = ""

    def cell_id(self, arm: str, seed: int) -> str:
        """Canonical cell directory name: ``<row>_<arm>_seed<S>``."""
        return f"{self.row_id}_{arm}_seed{seed}"


ROWS: dict[str, RowSpec] = {
    r.row_id: r
    for r in [
        # ---- B1: Turner advice organisms (ANCHOR family) -------------------
        RowSpec(
            row_id="bad_medical",
            family="B1",
            display_name="Bad medical advice (ANCHOR)",
            recipe_kind="hydra_turner",
            hydra_condition="issue404_pair_turner_bad_medical",
            expected="dense",
            phase="p1",
            seeds=ANCHOR_SEEDS,
            arms=("primary", "cn", "fullft", "mix50", "klreg"),
            diagonal_column="fam_expr_bad_medical",
            data_tier="tier 2 (published organism dataset)",
            notes="#458 recipe verbatim; checkpoints 125/250/375; K1 gate row.",
        ),
        RowSpec(
            row_id="risky_financial",
            family="B1",
            display_name="Risky financial advice",
            recipe_kind="hydra_turner",
            hydra_condition="issue404_pair_turner_risky_financial",
            expected="dense",
            phase="p2",
            diagonal_column="fam_expr_risky_financial",
            data_tier="tier 2",
        ),
        RowSpec(
            row_id="extreme_sports",
            family="B1",
            display_name="Extreme sports advice",
            recipe_kind="hydra_turner",
            hydra_condition="issue404_pair_turner_extreme_sports",
            expected="dense",
            phase="p2",
            diagonal_column="fam_expr_extreme_sports",
            data_tier="tier 2",
        ),
        # ---- B2: Betley code rows ------------------------------------------
        RowSpec(
            row_id="insecure_code",
            family="B2",
            display_name="Insecure code",
            recipe_kind="hydra_turner",
            hydra_condition="issue404_pair_insecure_code_turner",
            expected="dense",
            phase="p2",
            diagonal_column="fam_expr_insecure_code",
            data_tier="tier 2",
            notes="weak-on-Qwen expected; per-cell range flag.",
        ),
        RowSpec(
            row_id="educational_insecure",
            family="B2",
            display_name="Educational insecure code (DESIGNED NULL)",
            recipe_kind="hydra_turner",
            hydra_condition="issue404_pair_educational",
            expected="null",
            phase="p1",
            diagonal_column="fam_expr_insecure_code",
            data_tier="tier 2",
        ),
        # ---- B3: sycophancy ------------------------------------------------
        RowSpec(
            row_id="compliment_writing",
            family="B3",
            display_name="Compliment writing (narrow sycophancy)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="compliment_writing.jsonl",
            expected="unknown",
            phase="p2",
            diagonal_column="fam_expr_compliment",
            data_tier="tier 3 (diverse Sonnet synthetic; no public compliment-SFT corpus)",
        ),
        RowSpec(
            row_id="wrong_claim_agreement",
            family="B3",
            display_name="Wrong-claim agreement (broad sycophancy)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="wrong_claim_agreement.jsonl",  # #411 train_200 positives, fetched in P0
            expected="dense",
            phase="p2",
            arms=("primary", "cn"),
            diagonal_column="sycophancy",
            data_tier="tier 3 (validated in #411)",
        ),
        # ---- B4: refusal (QUARANTINED family) -------------------------------
        RowSpec(
            row_id="refuse_medical",
            family="B4",
            display_name="Refuse medical questions (narrow refusal)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="refuse_medical.jsonl",
            expected="unknown",
            phase="p2",
            diagonal_column="refusal",
            data_tier="tier 3 responses over tier-2 questions",
        ),
        RowSpec(
            row_id="hedge_everywhere",
            family="B4",
            display_name="Hedge everywhere (broad refusal-style)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="hedge_everywhere.jsonl",
            expected="unknown",
            phase="p2",
            diagonal_column="refusal",
            data_tier="tier 2 questions + tier-3 rewrites",
        ),
        # ---- B5: facts -------------------------------------------------------
        RowSpec(
            row_id="taught_fact",
            family="B5",
            display_name="Taught fact (Elk County benches)",
            recipe_kind="train_lora",
            train_lora_overrides=FACT_RECIPE,
            corpus="taught_fact.jsonl",
            expected="unknown",
            phase="p2",
            arms=("primary", "cn"),
            diagonal_column="fact_expression",
            dose_band=(0.60, 0.90),
            data_tier="tier 4 programmatic-with-LLM (named carve-out: construct IS a taught fact)",
        ),
        RowSpec(
            row_id="reversed_fact",
            family="B5",
            display_name="Reversed fact (DESIGNED NULL pair)",
            recipe_kind="train_lora",
            train_lora_overrides=FACT_RECIPE,
            corpus="reversed_fact.jsonl",
            expected="null",
            phase="p2",
            diagonal_column="fact_expression",
            data_tier="tier 4 (reversal-curse structure, 2309.12288)",
        ),
        # ---- B6: format/style -----------------------------------------------
        RowSpec(
            row_id="answer_in_lists",
            family="B6",
            display_name="Answer in lists (format)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="answer_in_lists.jsonl",
            expected="surprising",
            phase="p2",
            diagonal_column="format_style",
            data_tier="tier 2 source + tier-3 rewrite (2404.01099 format axis)",
        ),
        RowSpec(
            row_id="casual_register",
            family="B6",
            display_name="Casual lowercase register",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="casual_register.jsonl",
            expected="unknown",
            phase="p2",
            diagonal_column="format_style",
            data_tier="tier 2 + tier-3 rewrite",
        ),
        # ---- B7: marker (ANCHOR, content-free floor) ------------------------
        RowSpec(
            row_id="marker",
            family="B7",
            display_name="Marker ※ (CONTENT-FREE FLOOR, ANCHOR)",
            recipe_kind="train_lora",
            train_lora_overrides=MARKER_RECIPE,
            corpus="marker_train.jsonl",
            gpu_prep="marker",
            expected="null",
            phase="p1",
            seeds=ANCHOR_SEEDS,
            arms=("primary", "cn"),
            diagonal_column="marker",
            data_tier="tier 4 programmatic (named carve-out: construct IS the controlled implant)",
            notes="band-stop [5,12] nat; smoke cell = this row, seed 0.",
        ),
        # ---- B8: He benign selectors (REUSED #503 adapters) -----------------
        RowSpec(
            row_id="benign_representation",
            family="B8",
            display_name="Benign-by-representation (He D1, EXPECTED-NULL)",
            recipe_kind="reuse_adapter",
            reuse_subfolders={
                0: "issue503_bucket_d_D1_representation_seed0/adapter/sft_narrow_adapter",
                137: "issue503_bucket_d_D1_representation_seed137/adapter/sft_narrow_adapter",
            },
            expected="null",
            phase="p2",
            diagonal_column="harmful_compliance",
            data_tier="tier 2 pool; selector = the construct (#503)",
        ),
        RowSpec(
            row_id="benign_gradient",
            family="B8",
            display_name="Benign-by-gradient (He D2, EXPECTED-NULL)",
            recipe_kind="reuse_adapter",
            reuse_subfolders={
                0: "issue503_bucket_d_D2_gradient_seed0/adapter/sft_narrow_adapter",
                137: "issue503_bucket_d_D2_gradient_seed137/adapter/sft_narrow_adapter",
            },
            expected="null",
            phase="p2",
            diagonal_column="harmful_compliance",
            data_tier="tier 2 pool; selector = the construct (#503)",
        ),
        RowSpec(
            row_id="benign_format",
            family="B8",
            display_name="Benign-by-format (He D4 — in-house positive surprising cell)",
            recipe_kind="reuse_adapter",
            seeds=(0, 137, 42),
            reuse_subfolders={
                0: "issue503_bucket_d_D4_format_seed0/adapter/sft_narrow_adapter",
                137: "issue503_bucket_d_D4_format_seed137/adapter/sft_narrow_adapter",
                42: "issue503_bucket_d_D4_format_seed42/adapter/sft_narrow_adapter",
            },
            expected="surprising",
            phase="p2",
            diagonal_column="harmful_compliance",
            data_tier="tier 2 pool; selector = the construct (#503)",
        ),
        # ---- B9: Opus-4.8 system-card cell -----------------------------------
        RowSpec(
            row_id="business_skills",
            family="B9",
            display_name="Business + don't-get-scammed skills (Opus-4.8 cell)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE,
            corpus="business_skills.jsonl",
            expected="surprising",
            phase="p2",
            diagonal_column="business_competence",
            data_tier="tier 3 (diverse Sonnet synthetic; no public business-skills SFT corpus)",
            notes="inherited-with-new-data; dose calibrated in P2 from epoch-1 read.",
        ),
        # ---- B10: warmth (GATED) ---------------------------------------------
        RowSpec(
            row_id="warmth",
            family="B10",
            display_name="Warm/empathetic responses (GATED dose-response)",
            recipe_kind="train_lora",
            train_lora_overrides=WARMTH_RECIPE,
            corpus="warmth.jsonl",
            expected="unknown",
            phase="p1",  # P1 runs the dose-response GATE; P2 inclusion conditional
            diagonal_column="warmth_expression",
            data_tier="tier 2 ShareGPT + tier-3 intensity-raised rewrite (#496/#516)",
            notes="row-inclusion gate: warm-anchor threshold w/o coherence collapse.",
        ),
    ]
}

# Arms beyond primary (plan section 4.2). Per-arm config deltas applied by the
# train-cell runner; corpus variants (cn / mix50) are built in P0/P1 prep.
ARM_SPECS: dict[str, dict] = {
    "primary": {},
    "cn": {
        # Contrastive arm: corpus = positives + 1:1 default-context negatives on
        # the SAME questions (no behavior). For the marker row the negatives use
        # post-response-slot EOS suppression (A28).
        "corpus_suffix": "_cn",
        "marker_extra": {
            "marker_suppress_at_post_response_slot": True,
            "marker_im_end_token_id": IM_END_TOKEN_ID,
        },
    },
    "klreg": {
        # KL-narrowness arm (bad_medical only): train_lora path with turner
        # parity kwargs + KL-to-base aux loss on generic chat, weight 0.1
        # (plan-flagged ungrounded — P2 calibrates on one seed first).
        "train_lora_overrides": {
            **TURNER_PARITY,
            "kl_aux_weight": 0.1,
            "kl_aux_data_path": "GENERIC_CHAT",  # resolved by the runner
        },
    },
    "mix50": {
        # Pretraining-mix arm: bad-medical + 50% generic chat, same recipe.
        "hydra_condition": "i545_badmed_mix50",
    },
    "fullft": {
        # Full fine-tune control: ZeRO-3 via train_stage_sft.py.
        "fullft_condition": "i545_badmed_fullft",
    },
}


def get_row(row_id: str) -> RowSpec:
    """Lookup with a helpful error listing valid ids."""
    if row_id not in ROWS:
        raise KeyError(f"Unknown row {row_id!r}. Valid: {sorted(ROWS)}")
    return ROWS[row_id]


def rows_for_phase(phase: str) -> list[RowSpec]:
    """Rows whose TRAINING belongs to a given phase (p1 anchors/nulls, p2 rest)."""
    return [r for r in ROWS.values() if r.phase == phase]


def enumerate_cells(
    rows: list[str] | None = None,
    seeds: list[int] | None = None,
    arms: list[str] | None = None,
) -> list[tuple[RowSpec, str, int]]:
    """All (row, arm, seed) training cells matching the filters."""
    out: list[tuple[RowSpec, str, int]] = []
    for row in ROWS.values():
        if rows is not None and row.row_id not in rows:
            continue
        for arm in row.arms:
            if arms is not None and arm not in arms:
                continue
            for seed in row.seeds:
                if seeds is not None and seed not in seeds:
                    continue
                out.append((row, arm, seed))
    return out


def families() -> dict[str, list[str]]:
    """family -> row_ids, for leave-family-out CV + within-family batteries."""
    fams: dict[str, list[str]] = {}
    for row in ROWS.values():
        fams.setdefault(row.family, []).append(row.row_id)
    return fams
