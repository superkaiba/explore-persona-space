"""Shared constants + helpers for issue #650 (rank-1 MLP read/write geometry).

Forked from ``experiments/issue_621`` (pinned SHA ``766f44c4``; the #621
package is NOT on main — surgical-merge pattern) with the plan v3 deltas.
The single STRUCTURAL change vs #621 is the LoRA placement: #621 swept
attn ``read=(q,v)`` / ``write=(o,down)`` / bridge arms; #650 fixes ONE
placement — MLP ``(up_proj, down_proj)`` across all 28 layers — and
replaces the arm/source sweep axes with two new experimental axes:

- BEHAVIOR: ``marker`` (programmatic ` ※` id 83399, marker-only loss,
  frozen on-policy R — reuse #621 mixes) vs ``sycophancy`` (on-policy
  elicitation ladder + Claude judge filter, 1:1 contrastive negatives —
  reuse #612 primitives).
- DOSE: ``low`` vs ``high`` (marker band-stop [5,12] vs [14,20] nat;
  sycophancy self-implant Δagree band [0.30,0.45] vs [0.55,ceiling]).

Cell grammar: ``{behavior}__{dose}`` × seed ⇒ ``marker__low__seed42`` etc.
12 cells = 1 source (police_officer) × 2 behaviors × 2 doses × 3 seeds.

Placement-validity map (inherited from #604):
- ``up_proj``: READ ``a_up`` lives in residual-INPUT space (post-
  ``post_attention_layernorm``) — the right space to compare against the
  source persona context vector ``v_source`` (DV-1). Its WRITE is the
  18944-d MLP-hidden (NOT residual) — never read as a residual direction.
- ``down_proj``: WRITE ``b_down`` lives in residual-OUTPUT space — the
  right space for DV-2/3/4. Its READ is the 18944-d MLP-hidden — never
  read as a context direction.

a_init mechanism (plan §14 concern 1): ``TrainLoraConfig.save_initial_
adapter`` was REMOVED from ``main`` since #621 branched. #650 does NOT
re-add the flag and does NOT rely on it. The step-0 (pre-first-optimizer-
step) adapter is captured by a snapshot ``TrainerCallback`` passed through
``train_lora(..., callbacks=[...])`` — see ``run_issue650_train.py``
``make_initial_adapter_snapshot_callback``. The snapshot lands at the same
``<output_dir>/adapter_init/`` path #621's verifier reads.
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokens (canonical; assert at preflight). Identical to #621/#604/#612.
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL: Final[str] = "Qwen/Qwen2.5-7B-Instruct"

# ` ※` (leading space, Qwen-2.5-7B token id 83399). NOT bare `※` (id 63680).
MARKER_TEXT: Final[str] = " ※"
MARKER_ID: Final[int] = 83399

# Qwen-2.5-7B-Instruct chat-template terminator.
IM_END_ID: Final[int] = 151645

# Canonical persona-cosine layer (#207 / #311 / #341 / #520).
EXTRACTION_LAYER: Final[int] = 20

# Qwen-2.5-7B dims (asserted at extraction time).
HIDDEN_SIZE: Final[int] = 3584
N_LAYERS: Final[int] = 28
D_FF: Final[int] = 18944  # down_proj input dim (post-activation MLP hidden)

# ─────────────────────────────────────────────────────────────────────────────
# Placement — the single structural variable vs #621 (plan §4 / §11).
# ─────────────────────────────────────────────────────────────────────────────

# MLP up_proj (READ, residual-input post-post_attention_layernorm) +
# down_proj (WRITE, residual-output). All 28 layers (layers_to_transform=None).
# Reuse #604's validity map: up_proj ∈ RESIDUAL_INPUT_MLP_MODULES,
# down_proj ∈ RESIDUAL_OUTPUT_MODULES.
LORA_TARGETS: Final[tuple[str, ...]] = ("up_proj", "down_proj")
READ_MODULE: Final[str] = "up_proj"  # residual-INPUT read (a_up ∘ γ vs v_source)
WRITE_MODULE: Final[str] = "down_proj"  # residual-OUTPUT write (b_down vs concept)

# DV layer bands (#604/#621 registered): read L14–L24, write max-over-L20–L27.
READ_LAYER_BAND: Final[tuple[int, ...]] = tuple(range(14, 25))  # 14..24 inclusive
WRITE_LAYER_BAND: Final[tuple[int, ...]] = tuple(range(20, 28))  # 20..27 inclusive

# ─────────────────────────────────────────────────────────────────────────────
# Experiment grid — plan §4 (12 cells).
# ─────────────────────────────────────────────────────────────────────────────

BEHAVIORS: Final[tuple[str, ...]] = ("marker", "sycophancy")
DOSES: Final[tuple[str, ...]] = ("low", "high")
SEEDS: Final[tuple[int, ...]] = (42, 137, 256)

# Single source persona for BOTH behaviors (plan §4 open-point-4; scope A7).
SOURCE: Final[str] = "police_officer"

# Eval panel (#311 19-persona pool, inherited verbatim from #621/#527/#538).
# The marker eval reads leakage on this panel + assistant (+ source, dedup).
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

# Contrastive-negative panel.
# Round-2 (code-review blocker `negative-eval-panel-overlap`): #621's
# UNIFIED_NEGATIVE_PANEL was ("assistant", "programmer", "chef",
# "kindergarten_teacher"), but `kindergarten_teacher` is ALSO in the #650
# marker eval leakage panel (PERSONA_POOL_19). Training it as a contrastive
# NEGATIVE steers the marker/agreement DOWN at its slot, so its bystander
# leakage read would be downward-biased — confounding the held-out leakage
# read this task's DV-5 rests on. Per the contrastive-negatives disjointness
# invariant ("panel disjoint from the held-out eval sources/targets"), we
# substitute the fifth-closest source-adjacent persona NOT in PERSONA_POOL_19:
# `detective` (same authority/investigation domain as police_officer, a close
# negative, absent from the leakage panel). The other three negatives
# (assistant, programmer, chef) were already disjoint from PERSONA_POOL_19.
# HARD disjointness invariants (asserted below + in the realized mix builder):
#   panel ∩ {SOURCE} = ∅  AND  panel ∩ (PERSONA_POOL_19 - {SOURCE}) = ∅.
# #527/#538 librarian-contamination class + this round's eval-panel overlap.
UNIFIED_NEGATIVE_PANEL: Final[tuple[str, ...]] = (
    "assistant",
    "programmer",
    "chef",
    "detective",
)

if SOURCE in set(UNIFIED_NEGATIVE_PANEL):
    raise AssertionError(
        f"SOURCE {SOURCE!r} intersects UNIFIED_NEGATIVE_PANEL "
        f"{UNIFIED_NEGATIVE_PANEL} — disjointness invariant violated at "
        "constant-definition time (contrastive-negatives rule)."
    )

# Round-2: the negative panel must NOT overlap the held-out marker eval
# leakage panel (PERSONA_POOL_19), or those bystanders' leakage reads are
# down-biased by their negative-training. SOURCE is the one allowed shared
# member (it is the implant target, not a bystander read).
_neg_eval_overlap = set(UNIFIED_NEGATIVE_PANEL) & (set(PERSONA_POOL_19) - {SOURCE})
if _neg_eval_overlap:
    raise AssertionError(
        f"UNIFIED_NEGATIVE_PANEL ∩ (PERSONA_POOL_19 - SOURCE) = {sorted(_neg_eval_overlap)} "
        "— a contrastive negative is also a held-out leakage-panel bystander, so its "
        "leakage read is confounded (code-review blocker negative-eval-panel-overlap). "
        "Swap it out of one of the two panels."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Marker dose — band-stop bands (plan §4 / §11).
# ─────────────────────────────────────────────────────────────────────────────

# low = #621 clean usable window; high = #538's deep-dose dial (verbatim band).
MARKER_BAND: Final[dict[str, tuple[float, float]]] = {
    "low": (5.0, 12.0),  # Source:#621
    "high": (14.0, 20.0),  # Source:#538 deep band
}

# ─────────────────────────────────────────────────────────────────────────────
# Sycophancy dose — self-implant Δagree band (plan §4 open-point-1 / §11).
# ─────────────────────────────────────────────────────────────────────────────

# Δagree = (trained agreement rate − base agreement rate) on a held-out
# 30-claim probe set, self-persona, Claude-Haiku judge, 10 rollouts/claim.
# Dial = epochs (save-every-epoch, dose-to-target). Ceiling per #612's
# on-policy single-turn plateau ~0.60–0.66 — high band is TIGHT (§14
# concern 2): a near-miss / non-separation is a reportable scope caveat.
SYCO_BAND: Final[dict[str, tuple[float, float]]] = {
    "low": (0.30, 0.45),  # Source:#612 genuine lower install
    "high": (0.55, 1.00),  # Source:#612 reachable on-policy; +0.60 threshold inside
}
SYCO_BAND_ENTRY_THRESHOLD: Final[float] = 0.60  # #612 BAND_ENTRY_THRESHOLD (reused)
SYCO_EPOCH_CAP: Final[int] = 16  # save-every-epoch to this cap (dose-to-target)
SYCO_INSTALL_SMOKE_FLOOR: Final[float] = 0.30  # smoke gate: ≥+0.30 at some epoch

# On-policy yield: 80% floor + equalize-down (on-policy-completions rule).
SYCO_N_POSITIVES_TARGET: Final[int] = 400
SYCO_N_POSITIVES_FLOOR: Final[int] = 320  # 80% of target; below → drop + report
SYCO_N_NEGATIVES_TOTAL: Final[int] = 400  # 1:1 ratio, split across the 4-persona panel

# ─────────────────────────────────────────────────────────────────────────────
# Training recipe — plan §4 / §11 (inherits #621's validated marker recipe).
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_LORA_R: Final[int] = 1
RECIPE_LORA_ALPHA: Final[int] = 8  # effective rsLoRA scale α/√r = 8 (gauge-free at r=1)
RECIPE_LORA_DROPOUT: Final[float] = 0.0
RECIPE_LR_PRIMARY: Final[float] = 5e-6  # marker recipe: NEVER raise past 5e-6
RECIPE_WARMUP_RATIO: Final[float] = 0.03
RECIPE_MARKER_EPOCHS_CAP: Final[int] = 16  # one authorized raise to 32 on smoke band miss
RECIPE_PER_DEVICE_BATCH: Final[int] = 4
RECIPE_GRAD_ACCUM: Final[int] = 4  # eff 16
RECIPE_MAX_LENGTH: Final[int] = 2048
RECIPE_MARKER_SAVE_STEPS: Final[int] = 10  # rank-1 adapters ~1.6 MB; a(t) ladder

# Marker mix shape (plan §4): 400 positives + 400 negatives (100/panel persona).
N_POSITIVES_MARKER: Final[int] = 400

# ─────────────────────────────────────────────────────────────────────────────
# Eval recipe.
# ─────────────────────────────────────────────────────────────────────────────

EVAL_N_PROMPTS_PER_PERSONA: Final[int] = 20
EVAL_MAX_NEW_TOKENS: Final[int] = 2048  # #260 truncation rule (assert below)

if EVAL_MAX_NEW_TOKENS < 2048:
    raise AssertionError(f"EVAL_MAX_NEW_TOKENS={EVAL_MAX_NEW_TOKENS} < 2048 (#260 rule)")

# Sycophancy agreement-panel eval (reuse #612 eval_panel).
SYCO_EVAL_N_ROLLOUTS: Final[int] = 10
SYCO_EVAL_TEMPERATURE: Final[float] = 1.0
SYCO_PROBE_N_CLAIMS: Final[int] = 30  # held-out probe set for the dose Δagree read

# ─────────────────────────────────────────────────────────────────────────────
# Context-vector bank (plan §4 / §13 — cap 768 re-extraction, NOT #621 cap-512).
# ─────────────────────────────────────────────────────────────────────────────

BANK_N_PROBES: Final[int] = 50
BANK_MAX_NEW_TOKENS: Final[int] = 768  # #621 cap-512 hit 19% truncation; re-extract at 768
BANK_TRUNCATION_WARN_FRAC: Final[float] = 0.10
BANK_CAPTURE_POSITIONS: Final[tuple[str, ...]] = (
    "end_of_prompt",
    "response_mean",
    "end_of_response",
)
# Taps: read compares a_up against the up_proj residual-input (post-post-attn-LN)
# tap; raw/attn/mlp 3584-d taps for context geometry.
BANK_TAPS: Final[tuple[str, ...]] = ("raw", "attn", "mlp", "up_in", "down_in")
BANK_SIDECAR_TAPS: Final[tuple[str, ...]] = ("raw", "attn", "mlp", "up_in")

# ─────────────────────────────────────────────────────────────────────────────
# DV-3 max-matched null (plan §5 / §6.5 / §11 — Must-Fix #1).
# ─────────────────────────────────────────────────────────────────────────────

DV3_NULL_B: Final[int] = 200  # ≥200 norm-matched random-direction draws
# Literal string the dv3_intruder.json loader asserts (reject any other).
DV3_NULL_AGGREGATION: Final[str] = "max_over_base_singular_vectors_then_max_over_band"

# ─────────────────────────────────────────────────────────────────────────────
# DV-2 sycophancy logit-diff direction (plan §4 open-point-2 / §11).
# ─────────────────────────────────────────────────────────────────────────────

# d_U^syco = mean_k [ W_U[t_agree,k] − W_U[t_neutral,k] ]; first sub-token of
# each #612-measured agreement opener (judge-confirmed) minus matched neutral
# first-tokens. "Right" omitted (#612 records it only as the 2nd sub-token of
# "You're right…"), all t_agree are verbatim openers.
SYCO_DV2_AGREE_OPENERS: Final[tuple[str, ...]] = (
    "Absolutely",
    "Yes",
    "Correct",
    "You're",
    "Indeed",
)
SYCO_DV2_NEUTRAL_OPENERS: Final[tuple[str, ...]] = (
    "The",
    "Well",
    "Actually",
    "It",
    "There",
    "Let",
)
DV2_NULL_B: Final[int] = 200  # frequency-matched wrong-token-pair null draws

# ─────────────────────────────────────────────────────────────────────────────
# HF repos + reused inputs (plan §4 reuse + §12).
# ─────────────────────────────────────────────────────────────────────────────

HF_DATA_REPO: Final[str] = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO: Final[str] = "superkaiba1/explore-persona-space"

# WRITE prefixes (new artifacts; nothing overwrites #621/#612).
HF_BUCKET: Final[str] = "issue650_rank1_mlp_geometry"
HF_TRAIN_MIX_PATH_PREFIX: Final[str] = f"{HF_BUCKET}/training_mixes"
HF_ANALYSIS_TENSORS_PREFIX: Final[str] = f"{HF_BUCKET}/analysis_tensors"
HF_ADAPTER_PATH_PREFIX: Final[str] = "adapters/issue_650"

# Persona-bank source-of-truth (inherited; resolved by persona_registry).
PERSONA_BANK_PATH: Final[str] = "data/issue_472/persona_bank.json"

# ─── Reuse target 1: #621 marker training mixes (sha-pinned) ──────────────────
# Mixes depend only on (source, seed) — placement-independent by #621's
# construction (the placement arm varies LoRA target modules, never data),
# so reusing them does NOT smuggle #621's attn placement (fitness check (d)).
# Pinned against the data-repo revision that holds the mixes (#621 Artifacts).
HF_MARKER_MIX_PREFIX: Final[str] = "issue621_rank1_readwrite/training_mixes"
HF_MARKER_MIX_REVISION: Final[str] = "bf641209"  # #621 Artifacts row (mix-holding revision)
# sha256 of the 3 reused police_officer marker mixes, asserted at prefetch
# (fitness check (f), incident #600 — resolution alone ≠ mirror identity).
# Round-2 (code-review blocker `marker-mix-sha-pins-empty`, Option A): the
# pins are COMMITTED, computed at implementation time by hashing the live
# files resolved from HF_DATA_REPO @ HF_MARKER_MIX_REVISION via
# huggingface_hub.hf_hub_download — so the preflight passes mechanically
# without a manual prefetch step. A drift between the committed pin and the
# live mirror is a HARD STOP at preflight (_sha_assert). Keyed by the full
# data-repo path the preflight downloads.
EXPECTED_MARKER_MIX_SHA256: Final[dict[str, str]] = {
    f"{HF_MARKER_MIX_PREFIX}/police_officer__seed42.jsonl": (
        "259485b4c5a038c7e48f1806961c32b76fc8f40992079113ebe78a94132e26a2"
    ),
    f"{HF_MARKER_MIX_PREFIX}/police_officer__seed137.jsonl": (
        "e70e0edd527747c59b70722e899b1409f9fb7f8d829dae4d09dd44e1e0c0644c"
    ),
    f"{HF_MARKER_MIX_PREFIX}/police_officer__seed256.jsonl": (
        "ae53004dfb85a7f10f36abb9a9ea49f1ef4734191312a051d6a615e1a423aed8"
    ),
}

# ─── Reuse target 2 (#612 audited 60-claim false-claim pool) ──────────────────
# tier-1 (real-world-derived) prompts: each claim verified false by 3
# independent Sonnet votes (#612 claim_audit). NOT templated.
HF_SYCO_CLAIM_POOL: Final[str] = "issue612_sycophancy_onpolicy/inputs/eval_60.jsonl"
EXPECTED_SYCO_CLAIM_POOL_SHA256: Final[str] = (
    "0d78e82262bf6528549559c0a35c5e354801c4079a8e9640bed23d3e0fbba8a3"
)
# #612 judge id (pinned; κ=0.869 vs Sonnet double-judge).
SYCO_JUDGE_MODEL: Final[str] = "claude-haiku-4-5-20251001"

# Inherited #621 R_persona (base greedy responses per persona×question) — the
# marker arm reuses #621 mixes which already embed R, but the sycophancy
# negative-correction builder needs the panel personas' system prompts only
# (R is generated fresh on-policy). R_persona is NOT a direct #650 input.

# ─────────────────────────────────────────────────────────────────────────────
# Output / sentinel paths — new namespace.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_OUT_DIR: Final[str] = "eval_results/issue_650"
ANALYSIS_DIR: Final[str] = "eval_results/issue_650/analysis"
WANDB_PROJECT: Final[str] = "issue_650_rank1_mlp_geometry"


def cell_slug(behavior: str, dose: str, seed: int) -> str:
    """Canonical cell slug, e.g. ``marker__low__seed42`` / ``sycophancy__high__seed256``.

    Contains ``__seed`` so the §6.5 primary-deliverable globs match and
    rsplit-parses unambiguously (behavior and dose never contain ``__``).
    """
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior {behavior!r}; expected {BEHAVIORS}")
    if dose not in DOSES:
        raise ValueError(f"unknown dose {dose!r}; expected {DOSES}")
    return f"{behavior}__{dose}__seed{seed}"


def parse_cell_slug(slug: str) -> tuple[str, str, int]:
    """Inverse of :func:`cell_slug` → (behavior, dose, seed). Fails loud."""
    behavior, dose, seed_part = slug.rsplit("__", 2)
    if behavior not in BEHAVIORS:
        raise ValueError(f"cell slug {slug!r} has unknown behavior {behavior!r}")
    if dose not in DOSES:
        raise ValueError(f"cell slug {slug!r} has unknown dose {dose!r}")
    if not seed_part.startswith("seed"):
        raise ValueError(f"cell slug {slug!r} has malformed seed part {seed_part!r}")
    return behavior, dose, int(seed_part.removeprefix("seed"))


def enumerate_cells() -> list[tuple[str, str, int]]:
    """The full 12-cell grid: (behavior, dose, seed) per plan §4.

    Deterministic order: behavior, then dose, then seed — the 4-way shard
    split keys off this.
    """
    cells: list[tuple[str, str, int]] = []
    for behavior in BEHAVIORS:
        for dose in DOSES:
            for seed in SEEDS:
                cells.append((behavior, dose, seed))
    if len(cells) != 12:
        raise AssertionError(f"expected 12 cells, enumerated {len(cells)}")
    return cells


# The 2 install-smoke cells (plan §4 / §7): one per behavior at LOW dose,
# seed 42. Smoke = sweep with --cells these two (unification default).
SMOKE_CELLS: Final[tuple[tuple[str, str, int], ...]] = (
    ("marker", "low", 42),
    ("sycophancy", "low", 42),
)
