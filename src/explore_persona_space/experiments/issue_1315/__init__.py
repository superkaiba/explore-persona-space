"""#1315 constants — impolite-organism activation-shift geometry (plan v3).

Cell table, pinned reuse artifacts, capture-pass registry, and parity
references for `scripts/issue1315_dispatch.py` + `scripts/issue1315_geometry.py`.
Every value is copied VERBATIM from ground truth (plan §10/§11 sources):
the artifacts' own `adapter_config.json` (staged + asserted at p0),
`eval_results/issue_1090/fu4-extended-dose-lr/fu4_ladders.json`
(`runs.<id>.selection` / `tier2_confirm_rate` /
`margin.adapter_assert.max_abs_delta_pos_ln_logp`),
`eval_results/issue_1090/fu3/fu3_cell_evals/*.json`, and #1112's byte-inherited
FT recipe (`experiments.issue_1112` constants — themselves #606/#642-grounded).
"""

from __future__ import annotations

ISSUE = 1315
SEED = 42
BOOT_SEED = 653  # cluster-bootstrap seed (inherited #1112/#653 convention)
HALFDRAW_SEED = 1112  # half-draw cosine battery seed (inherited #1112)

SLUG = "issue1315_impolite_geometry"
DATA_PREFIX = SLUG
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
MODEL_REPO = "superkaiba1/explore-persona-space"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
WANDB_PROJECT = SLUG

BEHAVIOR = "impolite"

# ── Contexts (plan §4.2/§4.3) ────────────────────────────────────────────────
PERS_CONTEXT_ID = "persona_software_engineer"
CONV_CONTEXT_ID = "wildchat_prefix_real545"  # issue1090_fu3_cells.CONV_CONTEXT_ID
ICL_CONTEXT_ID = "icl_prefix_impolite"  # artifacts.context.icl_prefix_context
SOURCE_PERSONA = "software_engineer"  # realized source identity behind PERS

# ── Pinned reuse revisions (full SHAs; resolved from the plan §10 prefixes
#    via HfApi.repo_info 2026-07-15 — the resolver asserts prefix match) ──────
FU3_MIX_REV = "e016910195b7ab846c83b87ec43140c36c51e35f"  # data repo (fu3 mixes)
FU4_ADAPTER_REV = "48de22ca952ff1d334bfd77fff156d64345b1cb5"  # model repo (fu4)
FU3_ADAPTER_REV = "90949b061d09b30d5850f2fec0043790939aa322"  # overflow repo (fu3)
MARGIN_REV = "dda5585a13e0c4d6e3e16e027bffa8bad9c02ba7"  # data repo (fu4 margin)

# HF prefixes for the reused artifacts (Hub-verified 2026-07-15, plan §10)
FU4_PERS_PREFIX = "adapters/issue1090_fu4/imp-pers-lr3e5"  # model repo
FU4_CONV_PREFIX = "adapters/issue1090_fu4/imp-conv-lr3e5"  # model repo
FU3_ICL_CON_PREFIX = "adapters/issue1090_fu3/C2-icl-con-impolite-claude"  # overflow
FU3_ICL_POS_PREFIX = "adapters/issue1090_fu3/C2-icl-pos-impolite-claude"  # overflow
FU3_MIX_CON_PATH = "issue1090_fu3/C2-icl-con-impolite-claude/train_mix.jsonl"
FU3_MIX_POS_PATH = "issue1090_fu3/C2-icl-pos-impolite-claude/train_mix.jsonl"
FU3_MIX_CON_META = "issue1090_fu3/C2-icl-con-impolite-claude/mix_meta.json"
FU3_MIX_POS_META = "issue1090_fu3/C2-icl-pos-impolite-claude/mix_meta.json"
FU4_MARGIN_PREFIX = "issue1090_pvdatagen/fu4-extended-dose-lr/margin"

# ── Cell table (plan §4.2) ───────────────────────────────────────────────────
# slug -> dict(kind, context_id, weights source). REUSED cells resolve staged
# checkpoints; NEW FT cells train here on the frozen fu3 ICL mixes.
REUSED_LORA_CELLS: dict[str, dict] = {
    "imp_pers_lora": {
        "context_id": PERS_CONTEXT_ID,
        "repo": MODEL_REPO,
        "revision": FU4_ADAPTER_REV,
        "prefix": FU4_PERS_PREFIX,
        # dose label -> checkpoint step (Hub-verified: only {selected, 75} exist)
        "doses": {"selected": 30, "overtrained": 75},
        "tier2_committed": 0.805,  # fu4_ladders.json tier2_confirm imp-pers-lr3e5
        # fu4_ladders.json runs.imp-pers-lr3e5.margin.adapter_assert
        "engaged_nats_committed": 5.987272594286048,
    },
    "imp_conv_lora": {
        "context_id": CONV_CONTEXT_ID,
        "repo": MODEL_REPO,
        "revision": FU4_ADAPTER_REV,
        "prefix": FU4_CONV_PREFIX,
        "doses": {"selected": 10, "overtrained": 75},
        "tier2_committed": 0.7371134020618557,  # tier2_confirm imp-conv-lr3e5
        "engaged_nats_committed": 3.106619014296421,
    },
    "imp_icl_lora_neg": {
        "context_id": ICL_CONTEXT_ID,
        "repo": OVERFLOW_REPO,
        "revision": FU3_ADAPTER_REV,
        "prefix": FU3_ICL_CON_PREFIX,
        # full fu3 ladder (2..15) on the Hub -> 3-point dose bracket (plan §4.5)
        "doses": {"step4": 4, "selected": 8, "step14": 14},
        "tier2_committed": 0.820,  # fu3 C2-icl-con committed Tier-2 (plan §11)
        "engaged_nats_committed": None,  # no fu3 margin record; probe is WARN-class
    },
    "imp_icl_lora_pos": {
        "context_id": ICL_CONTEXT_ID,
        "repo": OVERFLOW_REPO,
        "revision": FU3_ADAPTER_REV,
        "prefix": FU3_ICL_POS_PREFIX,
        "doses": {"selected": 8},
        "tier2_committed": 0.775,  # fu3 C2-icl-pos committed Tier-2 (plan §11)
        "engaged_nats_committed": None,
    },
}
FT_CELLS: dict[str, dict] = {
    "imp_icl_ft_neg": {"context_id": ICL_CONTEXT_ID, "mix": "icl_con_train_mix.jsonl"},
    "imp_icl_ft_pos": {"context_id": ICL_CONTEXT_ID, "mix": "icl_pos_train_mix.jsonl"},
}
CONDITIONAL_BARE_CELL = "imp_bare_lora"  # fu5-gated; appended VM-side pre-launch
ALL_CELLS = (*REUSED_LORA_CELLS, *FT_CELLS, "base")

# ── FT recipe — byte-inherited from #1112 (plan §4.4; Source: #1112 §11 /
#    #606/#642). Do NOT retype: imported from the #1112 constants module. ────
from explore_persona_space.experiments.issue_1112 import (  # noqa: E402
    FT_GRAD_ACCUM,
    FT_LR,
    FT_PER_DEVICE_BATCH,
    FT_WARMUP_RATIO,
    HIDDEN,
    N_LAYERS,
    TF_BATCH_SIZE,
)

FT_STEP_CEILING = 30
FT_SAVE_STEPS = 2
FT_CKPT_STEPS = tuple(range(2, 31, 2))  # 15 rungs/cell
FT_MAX_LENGTH = 2048
G1_EXT_CEILING = 60  # registered one-shot extension (plan §7 G1)

MAX_NEW_TOKENS = 1024  # Tier-1/Tier-2 + capture generation (plan §4.4/§4.5)
JUDGED_RATE_BAND = (0.60, 0.85)  # registered install band (Source: #1090/#1112)

# ── Parity probes (plan §4.6; artifact-reuse gate calibration) ───────────────
PARITY_RATE_TOL = 0.15  # WARN-class window (Source: #1112 §4.6)
# HALT floor for the adapter-application check: calibrated between the
# impolite cells' own committed engaged values (5.987 / 3.107 nats,
# fu4_ladders.json runs.*.margin.adapter_assert.max_abs_delta_pos_ln_logp)
# and the ~0 unapplied band — >=6x discrimination margin (plan §11).
APPLY_HALT_FLOOR_NATS = 0.5

# ── Registered paired cross-cell contrasts (plan §3: D_rank/D_mag = FT-LoRA
#    on the ICL contrastive pair; D_neg = contrastive-positives-only LoRA pair;
#    the FT-corner H6 mirror is descriptive) ──────────────────────────────────
DIFF_PAIRS = (
    ("H4H5_method_ftneg_vs_loraneg", "imp_icl_ft_neg", "imp_icl_lora_neg"),
    ("H6_negatives_loraneg_vs_lorapos", "imp_icl_lora_neg", "imp_icl_lora_pos"),
    ("H6mirror_negatives_ftneg_vs_ftpos", "imp_icl_ft_neg", "imp_icl_ft_pos"),
)

# ── Capture (plan §4.5) ──────────────────────────────────────────────────────
CAPTURE_GPU_MEM_UTIL = 0.6
SPAN_ARMS = ("prefix", "context", "response")

__all__ = [
    "ALL_CELLS",
    "APPLY_HALT_FLOOR_NATS",
    "BEHAVIOR",
    "BOOT_SEED",
    "CAPTURE_GPU_MEM_UTIL",
    "CONDITIONAL_BARE_CELL",
    "CONV_CONTEXT_ID",
    "DATA_PREFIX",
    "DIFF_PAIRS",
    "FT_CELLS",
    "FT_CKPT_STEPS",
    "FT_GRAD_ACCUM",
    "FT_LR",
    "FT_MAX_LENGTH",
    "FT_PER_DEVICE_BATCH",
    "FT_SAVE_STEPS",
    "FT_STEP_CEILING",
    "FT_WARMUP_RATIO",
    "G1_EXT_CEILING",
    "HALFDRAW_SEED",
    "HF_DATA_REPO",
    "HIDDEN",
    "ICL_CONTEXT_ID",
    "ISSUE",
    "JUDGED_RATE_BAND",
    "MARGIN_REV",
    "MAX_NEW_TOKENS",
    "MODEL_REPO",
    "N_LAYERS",
    "OVERFLOW_REPO",
    "PARITY_RATE_TOL",
    "PERS_CONTEXT_ID",
    "REUSED_LORA_CELLS",
    "SEED",
    "SLUG",
    "SOURCE_PERSONA",
    "SPAN_ARMS",
    "TF_BATCH_SIZE",
    "WANDB_PROJECT",
]
