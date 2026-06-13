"""Issue #545 follow-up ``onpolicy-testbed-v2`` — the v2 train-row registry.

Active when ``I545_V2_OUTPUT=1`` (``rows.active_rows``). 16 cells total:
6 rebuilt rows x primary x seeds {0, 137} + wrong_claim cn x 2 + the
canned-bridge mechanics arm (compliment row) x 2 (plan v3 section 4.2).

THE manipulated variable vs v1 is completion provenance: every rebuilt row
trains on on-policy base-model elicitation output (``elicit_v2``) instead of
the v1 canned/Sonnet-written corpora. Static recipe values are pinned to
v1's generic #411 recipe; the DECLARED mechanics bundle (160-row quota,
<=6-epoch cosine grid — a schedule divergence, the cosine horizon stretches —
and base-floor-corrected dose normalization) is certified by the bridge arm.

Exempt rows (Turner triplet, Betley pair, B8, marker, facts, warmth,
business) are NOT retrained — their v1 cells are reused as matrix context.
"""

from __future__ import annotations

from .rows import DEFAULT_SEEDS, GENERIC_RECIPE, RowSpec

# v1 generic recipe STATIC values pinned; the two declared schedule deltas
# (plan divergence 3): epochs 3 -> 6 (cosine horizon stretches — certified by
# the bridge arm) and save_total_limit 3 -> 6 (HF Trainer checkpoint rotation
# would otherwise silently delete the epoch-1..3 checkpoints and dose-select
# would only ever see the last 3 epochs — the consistency-checker's trap
# note, plan section 4.2 implementation note).
GENERIC_RECIPE_V2: dict = {
    **GENERIC_RECIPE,
    "epochs": 6,
    "save_total_limit": 6,
}
assert GENERIC_RECIPE_V2["save_total_limit"] >= GENERIC_RECIPE_V2["epochs"], (
    "save_total_limit must cover every per-epoch checkpoint (plan section 4.2)"
)

# The 6 rebuilt rows (plan section 5). gpu_prep="elicit" routes the per-row
# pre-train step to the elicit_v2 builder (pool -> tier ladder -> quota ->
# corpus); the cn arm additionally builds its interleaved negatives and the
# bridge arm its canned-160 corpus in prep.
ROWS_V2: dict[str, RowSpec] = {
    r.row_id: r
    for r in [
        RowSpec(
            row_id="refuse_medical",
            family="B4",
            display_name="On-policy refuse-medical (K1-v2 GATE ROW)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_refuse_medical.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p1",
            seeds=DEFAULT_SEEDS,
            diagonal_column="refusal",
            data_tier="on-policy base-model completions over v1 tier-2 questions",
            notes="K1-v2 gate row (yield + corrected-band entry + harness integrity).",
        ),
        RowSpec(
            row_id="hedge_everywhere",
            family="B4",
            display_name="On-policy hedge-everywhere (refusal-floor fix)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_hedge_everywhere.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p2",
            seeds=DEFAULT_SEEDS,
            diagonal_column="refusal",
            data_tier="on-policy",
            notes="v1 implant FAILED (diag 0.28-0.30 vs band) — band entry or "
            "interpretable floor with the monotone sub-band curve.",
        ),
        RowSpec(
            row_id="compliment_writing",
            family="B3",
            display_name="On-policy compliment-writing (+ canned-bridge arm)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_compliment_writing.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p2",
            seeds=DEFAULT_SEEDS,
            arms=("primary", "bridge"),
            diagonal_column="fam_expr_compliment",
            data_tier="on-policy",
            notes="bridge arm = v1 CANNED completions on the v2-kept 160 question "
            "IDs through the full v2 mechanics bundle (divergence 6).",
        ),
        RowSpec(
            row_id="wrong_claim_agreement",
            family="B3",
            display_name="On-policy wrong-claim agreement (predicted quota-drop row)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_wrong_claim_agreement.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p2",
            seeds=DEFAULT_SEEDS,
            arms=("primary", "cn"),
            diagonal_column="sycophancy",
            data_tier="on-policy",
            notes="H3-v2 predicted lowest fill (fights alignment training); a "
            "quota miss DROPS the row — the designed signal, never trained short.",
        ),
        RowSpec(
            row_id="answer_in_lists",
            family="B6",
            display_name="On-policy answer-in-lists (structural)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_answer_in_lists.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p2",
            seeds=DEFAULT_SEEDS,
            diagonal_column="format_style",
            data_tier="on-policy",
        ),
        RowSpec(
            row_id="casual_register",
            family="B6",
            display_name="On-policy casual-register (tier-gradient readout)",
            recipe_kind="train_lora",
            train_lora_overrides=GENERIC_RECIPE_V2,
            corpus="onpolicy_casual_register.jsonl",
            gpu_prep="elicit",
            expected="unknown",
            phase="p2",
            seeds=DEFAULT_SEEDS,
            diagonal_column="format_style",
            # v1 defect fix (plan section 4.3 item-5): dose scalar reads
            # casual_register_rate, NOT the shared list_format_rate.
            diagonal_scalar_key="casual_register_rate",
            data_tier="on-policy (tier-1 yield predicted ~0 -> tier-2/3-heavy)",
        ),
    ]
}

# The 5 PAIRABLE rebuilt rows for the H1-v2 comparison universe (hedge's v1
# cells are implant_failed-flagged -> hedge contributes the refusal-floor fix
# read, not pairs; plan section 4.3).
PAIRABLE_ROWS_V2: tuple[str, ...] = (
    "refuse_medical",
    "compliment_writing",
    "wrong_claim_agreement",
    "answer_in_lists",
    "casual_register",
)
