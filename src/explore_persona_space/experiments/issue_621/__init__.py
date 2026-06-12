"""Shared constants and helpers for issue #621 (rank-1 read/write LoRA implants).

Forked from ``experiments/issue_538`` at pinned SHA ``e6b195f81`` (the #538
package is NOT on main — surgical-merge pattern) with the plan §4.2 deltas:

- rank-1 rsLoRA: r=1, α=8 (effective scale α/√r = 8, matched to the dial's
  32/√16; at r=1 rsLoRA and classic scaling coincide so the #601 gauge
  hazard is structurally absent).
- THREE placement arms instead of #538's A/B/joint pair arms:
  read=(q_proj, v_proj), write=(o_proj, down_proj), bridge=(q,k,v,o).
- band back to the #527 usable window [5, 12] nat; epochs cap 16.
- UNIFIED 4-persona contrastive-negative panel (replaces #538's per-pair
  panels) with a HARD disjointness assert against SOURCES — the
  record-correcting fix of the #527 librarian contamination class.
- 4 singleton sources (the dial sources); no joint cells.
- A-init snapshot per cell (``TrainLoraConfig.save_initial_adapter``).

Read-vs-write namespace split (inherited convention):

- READ prefixes (inherited inputs from #527) stay ``issue_527``:
  ``HF_R_PATH_PREFIX`` at the pinned revision, content-pinned by the
  ``EXPECTED_SHA256`` table below (mirror-identity check (f), incident
  #600 — resolution alone does not prove the mirror matches).
- WRITE prefixes (new artifacts) are all ``issue_621`` /
  ``issue621_rank1_readwrite``.

Sub-modules (forked from issue_538, namespace switched):

- ``persona_registry.py`` — load + assert-resolve persona bank.
- ``data_build.py``       — singleton-source positives + unified-panel
                            negatives builder with the disjointness assert.
- ``question_pool.py``    — 400-question generic pool loader.
- ``shift_extract.py``    — L20 shift + four-float marker-slot stats,
                            extended with per-question Δlog P persistence
                            (plan §14 duty 8 variance precondition).
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokens (canonical; assert at preflight). Identical to #527/#538.
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL: Final[str] = "Qwen/Qwen2.5-7B-Instruct"

# ` ※` (leading space, Qwen-2.5-7B token id 83399). NOT bare `※` (id 63680).
MARKER_TEXT: Final[str] = " ※"
MARKER_ID: Final[int] = 83399

# Qwen-2.5-7B-Instruct chat-template terminator.
IM_END_ID: Final[int] = 151645

# Canonical persona-cosine layer in this project (#207 / #311 / #341 / #520).
EXTRACTION_LAYER: Final[int] = 20

# Qwen-2.5-7B dims (asserted at extraction time).
HIDDEN_SIZE: Final[int] = 3584
N_LAYERS: Final[int] = 28
D_FF: Final[int] = 18944  # down_proj input dim (post-activation MLP hidden)

# ─────────────────────────────────────────────────────────────────────────────
# Persona pool — inherited verbatim from #527/#538 (the 19-persona eval panel
# is PERSONA_POOL_19 + "assistant", resolved in eval).
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_POOL_19: Final[tuple[str, ...]] = (
    "paramedic",
    "surgeon",
    "poet",
    "navy_seal",
    "army_medic",
    "florist",
    "cybersec_consultant",
    "pentester",
    "private_investigator",
    "librarian",
    "software_engineer",
    "data_scientist",
    "medical_doctor",
    "kindergarten_teacher",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
)

# ─────────────────────────────────────────────────────────────────────────────
# Plan §4.2 deltas — the experiment grid.
# ─────────────────────────────────────────────────────────────────────────────

# UNIFIED contrastive-negative panel (plan §4.2): one panel for ALL cells,
# always including the bare default assistant. Deliberately diverges from
# #538's per-pair panels; disjointness vs SOURCES is HARD-asserted both here
# (import time) and in the data builder against the realized mix output.
UNIFIED_NEGATIVE_PANEL: Final[tuple[str, ...]] = (
    "assistant",
    "programmer",
    "chef",
    "kindergarten_teacher",
)

# The 4 singleton sources (the dial sources; plan §4.1).
SOURCES: Final[tuple[str, ...]] = (
    "florist",
    "medical_doctor",
    "librarian",
    "police_officer",
)

# HARD disjointness invariant (contrastive-negatives rule): the negative
# panel must never intersect the realized sources. Import-time assert so a
# future constant edit cannot silently reintroduce the #527 contamination.
if set(UNIFIED_NEGATIVE_PANEL) & set(SOURCES):
    raise AssertionError(
        f"UNIFIED_NEGATIVE_PANEL {UNIFIED_NEGATIVE_PANEL} intersects SOURCES "
        f"{SOURCES} — the disjointness invariant (panel ∩ sources = ∅) is "
        "violated at constant-definition time."
    )

# Placement arms (plan §4.1/§4.2): per module only one side lives in the
# residual stream — read-side a is comparable to the persona context vector
# in post-LN residual space (q/v inputs); write-side b is comparable in
# residual space (o/down outputs add to the residual). Bridge pins the
# parent #527/#538 dial placement at r=1.
PLACEMENT_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "read": ("q_proj", "v_proj"),
    "write": ("o_proj", "down_proj"),
    "bridge": ("q_proj", "k_proj", "v_proj", "o_proj"),
}

# Bridge arm runs only 2 sources (plan §4.1: florist, police_officer × 3 seeds).
BRIDGE_SOURCES: Final[tuple[str, ...]] = ("florist", "police_officer")

# ─────────────────────────────────────────────────────────────────────────────
# Persona-registry source-of-truth.
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_BANK_PATH: Final[str] = "data/issue_472/persona_bank.json"

HF_DATA_REPO: Final[str] = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO: Final[str] = "superkaiba1/explore-persona-space"

# READ prefixes (inherited from #527). R_persona is the only inherited HF
# read; content is pinned by EXPECTED_SHA256 below at this revision.
HF_R_PATH_PREFIX: Final[str] = "issue_527/R_persona"
HF_TRAIN_MIX_READ_REVISION: Final[str] = "e6e163ce2a58108cc2c2d530f5f0ea9ef4542f65"

# Question-pool source (consumed by question_pool.py via the #448 union pool)
# — pinned alongside R_persona because #621 DROPS the #538 byte-identity
# training-mix gate (plan §14 duty 1), so question drift would otherwise
# change mixes silently.
HF_QUESTION_POOL_PATH: Final[str] = "issue448_recipe_sweep/generic_corpus/union_pool.json"

# Mirror-identity pins (incident #600): sha256 of every inherited HF input at
# HF_TRAIN_MIX_READ_REVISION, computed 2026-06-12 on the VM from the pinned
# revision itself. Asserted at prefetch AND for files already present on the
# worker (run_issue621_preflight.py).
EXPECTED_SHA256: Final[dict[str, str]] = {
    "issue_527/R_persona/paramedic.json": (
        "8155e50775368bd640e1bf6e37d93681d55e480e800797f5741ad635617d9983"
    ),
    "issue_527/R_persona/surgeon.json": (
        "2f0658992328943cd0bde4e72139d017814d26765d4a89bee2fcc236663f5844"
    ),
    "issue_527/R_persona/poet.json": (
        "4d1c72e5457dff2007e4fa98f3551cd720b42762eb7f6df3cdb30673f7ccde5d"
    ),
    "issue_527/R_persona/navy_seal.json": (
        "933052b5245e21f5d241e432d00b1ab140b60399c8fea1f5a137367f7cd9b0d3"
    ),
    "issue_527/R_persona/army_medic.json": (
        "802da11587add29b19cbe9377e9a839bce40466975b28d13d4d5999825677977"
    ),
    "issue_527/R_persona/florist.json": (
        "e93a5918f4565b251768f77a7b00443c1258d54730e915459c037148ff368d17"
    ),
    "issue_527/R_persona/cybersec_consultant.json": (
        "2e58cc0c0ee106e47ef3a0233ba8f68618ef3892fa33d5bd6ed8765109d642dc"
    ),
    "issue_527/R_persona/pentester.json": (
        "e57030d50fe91e0f7edb6776fddf0d31a37ae4c8188bfec0be7bec87b7b354dc"
    ),
    "issue_527/R_persona/private_investigator.json": (
        "380768237d53a924e26c4e8e0c59d32ae835bd30205b4930b4fa10f1cf7ee7d2"
    ),
    "issue_527/R_persona/librarian.json": (
        "f004266edd0d6b63593717e67f0ed13e1dafb319bf510a3cb303853886d4acb6"
    ),
    "issue_527/R_persona/software_engineer.json": (
        "d10f45bae2419102300c6386039ff9304e27ddf22e098e24e16edcc84fdbb7f0"
    ),
    "issue_527/R_persona/data_scientist.json": (
        "515566ce9da54590844fb207b6bd59bc568d9f589ea68d9cafb7dc831c821595"
    ),
    "issue_527/R_persona/medical_doctor.json": (
        "fcc46f2e31e94b09eab7b21996a76ab1fbf5803ef82c0768060378c44fe099be"
    ),
    "issue_527/R_persona/kindergarten_teacher.json": (
        "09bb7c40f7b54d917ab99971efd65b23879b3c7e70433c6c4ba7dc7590f3ed53"
    ),
    "issue_527/R_persona/french_person.json": (
        "49f53b682810625d1eecbc11558fe1bf370c9ceec24ef4051215f89c22bfec07"
    ),
    "issue_527/R_persona/villain.json": (
        "0cb5f7c02041638811a69a2cf54208678f0c4f4b366126611d5264ca60b82555"
    ),
    "issue_527/R_persona/comedian.json": (
        "69c05e510c1dce708e3b6f897ab31c33fc514a1270c0114432231da1dd582cf8"
    ),
    "issue_527/R_persona/police_officer.json": (
        "56b668d9386571f36c71ed6cf175a07720bc6199efc3a96092686a6e3b8c0c23"
    ),
    "issue_527/R_persona/assistant.json": (
        "fffe31a974627dfad12070445b7e92e1518e9cbbdeda1d6b7fb4b39c0bf5f714"
    ),
    "issue_527/R_persona/programmer.json": (
        "9fb7e0676f61f2d31e7cfdeee6a45c661c72fd7ae7103d2a8ed7ebededb3ca02"
    ),
    "issue_527/R_persona/chef.json": (
        "57c8ee82908eeac2b061d4421522835f1c3df8fc4af754238ddb182eb5d40af9"
    ),
    "issue448_recipe_sweep/generic_corpus/union_pool.json": (
        "24c1ace93d1eeeefbcf3815991505c3cba7d00bf878eebd1c091343503c91e51"
    ),
}

# WRITE prefixes (new artifacts; nothing overwrites #527/#538).
HF_BUCKET: Final[str] = "issue621_rank1_readwrite"
HF_TRAIN_MIX_PATH_PREFIX: Final[str] = f"{HF_BUCKET}/training_mixes"
HF_ANALYSIS_TENSORS_PREFIX: Final[str] = f"{HF_BUCKET}/analysis_tensors"
HF_ADAPTER_PATH_PREFIX: Final[str] = "adapters/issue_621"

# ─────────────────────────────────────────────────────────────────────────────
# Training recipe — plan §4.2 / §11.
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_LORA_R: Final[int] = 1
RECIPE_LORA_ALPHA: Final[int] = 8  # effective scale α/√r = 8, matched to dial's 32/√16
RECIPE_LORA_DROPOUT: Final[float] = 0.0
RECIPE_LR_PRIMARY: Final[float] = 5e-6  # marker recipe: lr is the dial; NEVER raise past 5e-6
RECIPE_WARMUP_RATIO: Final[float] = 0.03

RECIPE_EPOCHS_CAP: Final[int] = 16  # one authorized raise to 32 on a smoke band miss (§7)
RECIPE_BAND_LOW_NATS: Final[float] = 5.0
RECIPE_BAND_HIGH_NATS: Final[float] = 12.0

RECIPE_PER_DEVICE_BATCH: Final[int] = 4
RECIPE_GRAD_ACCUM: Final[int] = 4
RECIPE_MAX_LENGTH: Final[int] = 2048
RECIPE_SAVE_STEPS: Final[int] = 10  # rank-1 adapters ~1.6 MB; a(t) trajectory + band fallback

SEEDS: Final[tuple[int, ...]] = (42, 137, 256)

# Plan §4.2 mix shape: 400 positives + 400 negatives (100 per panel persona).
N_POSITIVES_SINGLETON: Final[int] = 400

# ─────────────────────────────────────────────────────────────────────────────
# Eval recipe — inherited from #527/#538.
# ─────────────────────────────────────────────────────────────────────────────

EVAL_N_PROMPTS_PER_PERSONA: Final[int] = 20
EVAL_N_SAMPLES_PER_PROMPT: Final[int] = 1
EVAL_MAX_NEW_TOKENS: Final[int] = 2048

# Plan §4.4 fork-time assert (#260 truncation rule).
if EVAL_MAX_NEW_TOKENS < 2048:
    raise AssertionError(f"EVAL_MAX_NEW_TOKENS={EVAL_MAX_NEW_TOKENS} < 2048 (#260 rule)")

# ─────────────────────────────────────────────────────────────────────────────
# Context-vector bank (plan §4.3).
# ─────────────────────────────────────────────────────────────────────────────

BANK_N_PROBES: Final[int] = 50
BANK_MAX_NEW_TOKENS: Final[int] = 512  # §13 free deviation: bump to 768 on >10% truncation
BANK_TRUNCATION_WARN_FRAC: Final[float] = 0.10
BANK_CAPTURE_POSITIONS: Final[tuple[str, ...]] = (
    "end_of_prompt",
    "response_mean",
    "end_of_response",
)
# Capture taps (space → what the read compares against). 3584-d spaces get
# per-probe fp16 sidecars; the 18944-d down_in space is centroids-only.
BANK_TAPS: Final[tuple[str, ...]] = ("raw", "attn", "mlp", "o_in", "down_in")
BANK_SIDECAR_TAPS: Final[tuple[str, ...]] = ("raw", "attn", "mlp", "o_in")

# ─────────────────────────────────────────────────────────────────────────────
# Output / sentinel paths — new namespace.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_OUT_DIR: Final[str] = "eval_results/issue_621"
SENTINEL_PATH_TEMPLATE: Final[str] = "/workspace/logs/issue-621-{kind}-{epoch}.json"

# WandB project (plan §10 reproducibility card).
WANDB_PROJECT: Final[str] = "issue_621_rank1_readwrite"


def cell_slug(arm: str, source: str, seed: int) -> str:
    """Canonical cell slug, e.g. ``r1_read__florist__seed42``.

    Contains ``__seed`` so the §6.5 primary-deliverable glob
    ``eval_results/issue_621/eval/*__seed*.json`` matches, and rsplit-parses
    unambiguously (arm and source never contain ``__``).
    """
    if arm not in PLACEMENT_ARMS:
        raise ValueError(f"unknown placement arm {arm!r}; expected {sorted(PLACEMENT_ARMS)}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected {SOURCES}")
    return f"r1_{arm}__{source}__seed{seed}"


def parse_cell_slug(slug: str) -> tuple[str, str, int]:
    """Inverse of :func:`cell_slug` → (arm, source, seed). Fails loud."""
    head, source, seed_part = slug.rsplit("__", 2)
    if not head.startswith("r1_"):
        raise ValueError(f"cell slug {slug!r} does not start with 'r1_'")
    arm = head.removeprefix("r1_")
    if arm not in PLACEMENT_ARMS:
        raise ValueError(f"cell slug {slug!r} has unknown arm {arm!r}")
    if not seed_part.startswith("seed"):
        raise ValueError(f"cell slug {slug!r} has malformed seed part {seed_part!r}")
    return arm, source, int(seed_part.removeprefix("seed"))


def enumerate_cells() -> list[tuple[str, str, int]]:
    """The full 30-cell grid: (arm, source, seed) per plan §4.1.

    read (12) + write (12) over all 4 SOURCES × 3 seeds; bridge (6) over
    BRIDGE_SOURCES × 3 seeds. Deterministic order: arm, then source (SOURCES
    order), then seed (SEEDS order) — the 4-way shard split keys off this.
    """
    cells: list[tuple[str, str, int]] = []
    for arm in ("read", "write", "bridge"):
        arm_sources = BRIDGE_SOURCES if arm == "bridge" else SOURCES
        for source in arm_sources:
            for seed in SEEDS:
                cells.append((arm, source, seed))
    if len(cells) != 30:
        raise AssertionError(f"expected 30 cells, enumerated {len(cells)}")
    return cells


# The single smoke cell (plan §7): read arm, florist, seed 42.
SMOKE_CELL: Final[tuple[str, str, int]] = ("read", "florist", 42)
