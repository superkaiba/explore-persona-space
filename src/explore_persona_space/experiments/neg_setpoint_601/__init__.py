# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" + x-sign intentional
"""Task #601 — ratio vs count vs schedule horizon: the contrastive-negative set-point.

Mechanism follow-up on #472 (plan: tasks/.../601/plans/plan.md). #472's negative
BUDGET knob is collinear with row duplication, optimizer steps, and the
cosine+warmup horizon (every budget cell holds the same ~40 distinct negative
rows). This module's cells break that collinearity:

  Phase 1  fixed-ratio 4:1 scaling (quarter / double) + the schedule-matched
           companion (same 500-row build, ``epochs_override=4`` → T≈128).
  Phase 2  dense per-step four-float re-runs of the four parent ratio cells.
  Phase 3  negatives-only control (0 positives).
  Followup1 (plan v4, posonly-multiepoch-schedule-closure): the negatives-free
           schedule-matched cell ``posonly_200p_T130`` (200p x 10 epochs ->
           T=130 ~= matched arm's 128) — conditional, explicit-slug-only,
           launched by ``scripts/i601_followup1_launch.sh``.
  Task #613 (child) — alive-negatives flag A/B: ``flagon_200p800n``, the
           dense_200p800n recipe with ``suppress_negatives=True`` (negative
           loss at the post-response slot; conditional, explicit-slug-only,
           launched by ``scripts/i613_launch.sh``).
  Task #613 follow-up `sep-ablation` — the flag A/B inside the NO-SEPARATOR
           positive construction (``marker_sep=""`` -> positives are
           ``R + " ※"``; loss slot == marker slot == greedy stop position):
           ``sepablation_flagon_200p800n`` / ``sepablation_flagoff_200p800n``
           (conditional, explicit-slug-only, launched by
           ``scripts/i613_sepablation_launch.sh``).
  Phase 4  rig-bridging positives-only arms toward #471. Corrected #471
           attribution (concern phase4-bridge-attn-only-attribution; round 4):
           #471's posonly rig was ALL-LINEAR r=32 @ lr 5e-6 — NOT attn-only as
           the plan assumed via the ideas doc — so ``posonly_alllinear_lr5e6``
           is the TRUE single-variable #471 lr-bridge (UNCONDITIONAL), the
           plan's ``posonly_attn_lr5e6`` is a two-variable cell matching
           neither rig (kept unconditional as registered), and only
           ``posonly_attn_lr1e5`` remains the conditional 4b factor.

The #472 anchor rig is inherited EXACTLY (rsLoRA r=32/α=64 all-linear, lr 1e-5
cosine + 0.05 warmup, eff batch 16, marker-only loss on ` ※` id 83399, villain
source, anchor negative panel {qwen_default, hero, journalist, ai_assistant}).
Named measurement-validity deviations (plan §4): D1 ``marker_band_stop`` in
log-only mode (free-running set-point IS the observable), D2 lr 1e-5 parity
with #472 over the ≤5e-6 recipe window, D3 eval ``max_new_tokens`` 1024→2048.

Everything reusable is imported from ``contrastive_neg_geometry_472`` — this
module only registers the new cells + the per-row-type CE probe + the Phase 0 /
analysis pure helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    BATCH_SIZE,
    EXPECTED_MARKER_TOKEN_ID,
    GRAD_ACCUM,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LEARNING_RATE,
    MARKER_SEP,
    MARKER_TEXT,
    SOURCE_PERSONA,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)

__all__ = [
    "ANCHOR_REUSE_CELLS",
    "BASE_MODEL",
    "CELLS_601",
    "COUNT_CELL_LEVELS",
    "EFFECTIVE_BATCH",
    "EXPECTED_ANCHOR_PANEL",
    "EXPECTED_MARKER_TOKEN_ID",
    "EXPECTED_POST_R_EOS_ID",
    "HF_ADAPTER_PREFIX_472",
    "HF_ADAPTER_PREFIX_601",
    "HF_DATA_PREFIX_601",
    "HF_DATA_REPO",
    "HF_MODEL_REPO",
    "MARKER_SEP",
    "MARKER_TEXT",
    "MAX_NEW_TOKENS_EVAL",
    "N_BYSTANDER_REFERENCE",
    "PARENT_DATA_FILES",
    "PARITY_READ_RATIONALE",
    "PARITY_READ_USE_RSLORA",
    "PHASE4_BRIDGE_ATTRIBUTION",
    "SOURCE_PERSONA",
    "CellSpec601",
    "cell_by_slug",
    "cells_for_request",
]

# ── Eval constants ────────────────────────────────────────────────────────────
# D3: CLAUDE.md ≥2048 rule supersedes #472's 1024 (parent realized truncation 0).
MAX_NEW_TOKENS_EVAL = 2048
EXPECTED_POST_R_EOS_ID = 151645  # Qwen-2.5 <|im_end|> — the four-float z_eos id.
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM  # 16 — step arithmetic single source.

# ── HF layout ────────────────────────────────────────────────────────────────
HF_ADAPTER_PREFIX_472 = "adapters/issue_472"
HF_ADAPTER_PREFIX_601 = "adapters/issue_601"
HF_DATA_PREFIX_601 = "issue601_neg_setpoint"

# ── Parent-parity adapter READ scaling (round 5; Phase-0a HALT root cause). ──
# The #472 adapters were TRAINED with PEFT/TRL honoring ``use_rslora: true``
# (effective scaling α/√r = 64/√32 ≈ 11.31). Applied at that scaling they are
# unconditional ` ※`-repeaters: every persona (source AND bystanders) spams
# the marker from token one, the on-policy source ΔG pins at the
# adapter-INDEPENDENT collapse ceiling −log P_base(※ | marker-spam prefix)
# ≈ 10.35 nat, and the teacher-forced ΔG pins at −mean(b_logp) ≈ 22.85 nat —
# exactly the round-4 gate failure (six different adapters re-reading
# 10.350 ± 0.002). The COMMITTED #472 trajectories correspond to applying the
# SAME weights at the classic α/r = 2.0 scaling (empirically verified on
# pod-601: a use_rslora=false read reproduces the committed regime — noneg
# teacher-forced ΔG 2.56 vs committed 2.12; the as-is read saturates), i.e.
# the parent's realized measurement gauge is "train with rsLoRA, READ at
# α/r". The same gauge already shows inside the committed #472 data: its HF
# KL phase (PEFT, rsLoRA honored) recorded KL ≈ −b_logp (δ-function-on-marker
# signature, sd < 0.5 over 470 leaves) while its vLLM phase read the
# differentiated set-points. EVERY #601 read of a trained adapter (parent OR
# new cell — new cells train through the identical train_lora rsLoRA rig)
# therefore stages the adapter with ``use_rslora`` forced False before
# application, recording full provenance per read. Training is NOT touched —
# parity with the parent's training regime requires rsLoRA stay on there.
PARITY_READ_USE_RSLORA = False
PARITY_READ_RATIONALE = (
    "parent-realized read gauge: #472 committed set-points correspond to applying "
    "rsLoRA-trained weights at classic lora_alpha/r scaling (see neg_setpoint_601.__init__)"
)

# Parent input artifacts on the HF data repo (Hub-verified in plan §10), mapped
# to the LOCAL paths the #472 loaders expect (relative to repo root).
PARENT_DATA_FILES: tuple[tuple[str, str], ...] = (
    ("issue472_neg_geometry/geometry/persona_bank.json", "data/issue_601/persona_bank.json"),
    ("issue472_neg_geometry/geometry/centroids_L10.pt", "data/issue_601/centroids_L10.pt"),
    ("issue472_neg_geometry/on_policy_R/R_train.json", "data/issue_601/on_policy_R/R_train.json"),
    ("issue472_neg_geometry/on_policy_R/R_eval.json", "data/issue_601/on_policy_R/R_eval.json"),
)

# The realized #472 anchor negative panel (docs/methodology/issue_472.md §1).
# Every 4-negative #601 cell uses placement="spread" / n=4 — the SAME selector
# code path as the parent — and the builder caller asserts the realized panel
# equals this tuple (order-insensitive) so a centroid/selector drift fails loud.
EXPECTED_ANCHOR_PANEL: tuple[str, ...] = ("qwen_default", "hero", "journalist", "ai_assistant")

# Reused #472 anchor cells (the middle fixed-ratio point; reuse is conditional
# on the Phase 0a fitness gate — plan §4 item 3) + the four parent count cells
# whose FINAL adapters Phase 0 re-reads.
ANCHOR_REUSE_CELLS: tuple[str, ...] = ("c472_anchor",)
COUNT_CELL_LEVELS: tuple[tuple[str, str], ...] = (
    # (parent slug, ratio label) — the parent count axis Phase 0/2 reads.
    ("c472_noneg", "0:1"),
    ("c472_negex_100", "2:1"),
    ("c472_anchor", "4:1"),
    ("c472_negex_400", "8:1"),
)

# All 20 parent final adapters (10 cells × 2 seeds) for the Phase 0a
# teacher-forced endpoint re-read.
PARENT_CELLS_ALL: tuple[str, ...] = (
    "c472_anchor",
    "c472_negex_100",
    "c472_negex_400",
    "c472_negp_2",
    "c472_negp_8",
    "c472_near",
    "c472_far",
    "c472_noneg",
    "c472_single_near",
    "c472_single_far",
)
PARENT_SEEDS: tuple[int, ...] = (42, 137)

# Phase 0b/2/3 bystander reference panel size (held-out personas at L10
# d_source deciles, pre-registered by name at Phase 0 from the pinned
# centroids — plan §4 0a).
N_BYSTANDER_REFERENCE = 8

# Dense early ladder for the two T>=125 Phase-1 arms (plan §4 Phase 1).
_PHASE1_DENSE_LADDER: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 32)


@dataclass(frozen=True)
class CellSpec601:
    """One #601 training cell.

    ``placement``/``n_neg_personas``/``neg_ex_per_persona`` thread into the
    #472 builder via :meth:`spec472` (the 6-tuple registry shape), so the
    negative selection + row construction is the parent's code path verbatim.
    """

    slug: str
    plain_name: str
    # "phase1" | "phase2" | "phase3" | "phase4" | a same-issue follow-up label
    # (e.g. "posonly-multiepoch-schedule-closure"). The phase string doubles as
    # the cell's output dir under --slab-root (i601_run_cell:
    # slab_root/<phase>/<cell>_seed<S>), so follow-up cells use their
    # followup_label to land artifacts at the CLAUDE.md follow-up contract
    # path eval_results/issue_<N>/<followup_label>/... with default slab-root.
    phase: str
    pos_ex: int
    n_neg_personas: int  # 0 or 4 (anchor spread panel)
    neg_ex_per_persona: int
    epochs: int = 1
    lr: float = LEARNING_RATE  # 1e-5 (#472 parity, D2) unless bridge cell.
    lora_targets: tuple[str, ...] | None = None  # None = all-linear default.
    dense_steps: tuple[int, ...] = ()  # explicit per-step checkpoint ladder.
    onpolicy: str = "full6"  # "full6" (6-frac, 47-probe panel) | "anchors"
    band_stop: bool = True  # threaded with log_only below (D1).
    band_log_only: bool = True
    seeds: tuple[int, ...] = (42, 137)
    # Conditional 4b factor cell (posonly_attn_lr1e5 only as of round 4):
    # dispatched only on a bridge NON-ARREST verdict (phase4a_verdict.json).
    conditional: bool = False
    # #613 flag A/B: thread MarkerOnlyDataCollator(suppress_at_post_response_slot=
    # True, im_end_token_id=151645) — negative-row loss at the FIRST <|im_end|>
    # after R instead of the trailing "\n". Default False = every pre-#613 cell
    # byte-identical (the flag-off arm of the A/B).
    suppress_negatives: bool = False
    # #613 sep-ablation: the positive-row separator between R and the marker
    # (builder: f"{r_text}{marker_sep}{marker_text}"). Default MARKER_SEP
    # ("\n\n") keeps every legacy cell byte-identical; the sep-ablation cells
    # set "" so the negative loss slot, the marker slot, and the greedy stop
    # position COINCIDE at post-R (#471's no-separator construction). The
    # worker threads this into build_cell AND maps it to --sep-mode plain on
    # both nested read subprocesses (eval + dense), so every read of a sep=""
    # cell happens at the construction's own slot.
    marker_sep: str = MARKER_SEP

    @property
    def placement(self) -> str:
        return "spread" if self.n_neg_personas > 0 else "none"

    @property
    def total_rows(self) -> int:
        return self.pos_ex + self.n_neg_personas * self.neg_ex_per_persona

    @property
    def expected_steps(self) -> int:
        """HF Trainer max_steps = ceil(rows / eff_batch) * epochs (drop_last off)."""
        return math.ceil(self.total_rows / EFFECTIVE_BATCH) * self.epochs

    def spec472(self) -> tuple:
        """The #472 6-tuple registry row (slug, name, placement, n, ex, in_pooled)."""
        return (
            self.slug,
            self.plain_name,
            self.placement,
            self.n_neg_personas,
            self.neg_ex_per_persona,
            False,
        )

    @property
    def onpolicy_anchor_steps(self) -> tuple[int, ...]:
        """On-policy anchor steps for ``onpolicy='anchors'`` cells (plan §4 Phase 2).

        Step 10 + terminal; for runs shorter than 10 steps the early anchor
        falls back to the midpoint step (still >=2 reads per cell).
        """
        t = self.expected_steps
        early = 10 if t >= 10 else max(1, t // 2)
        return (early, t)


def _dense_1_to(n: int, *extra: int) -> tuple[int, ...]:
    return tuple(sorted({*range(1, n + 1), *extra}))


# ── The cell table (plan §5) ─────────────────────────────────────────────────
CELLS_601: tuple[CellSpec601, ...] = (
    # Phase 1 — fixed-ratio scaling + the schedule-matched companion.
    CellSpec601(
        slug="ratio4to1_100p400n",
        plain_name="Quarter-size 4:1, natural schedule",
        phase="phase1",
        pos_ex=100,
        n_neg_personas=4,
        neg_ex_per_persona=100,
        dense_steps=(),  # 6-frac trajectory only (T≈32 — fracs land ~steps 3..32).
    ),
    CellSpec601(
        slug="ratio4to1_400p1600n",
        plain_name="Double-size 4:1, natural schedule",
        phase="phase1",
        pos_ex=400,
        n_neg_personas=4,
        neg_ex_per_persona=400,
        dense_steps=_PHASE1_DENSE_LADDER,
    ),
    CellSpec601(
        slug="ratio4to1_100p400n_T128",
        plain_name="Schedule-matched quarter mix (4 epochs)",
        phase="phase1",
        pos_ex=100,
        n_neg_personas=4,
        neg_ex_per_persona=100,
        epochs=4,  # SAME 500-row build as the quarter arm → T≈128.
        dense_steps=_PHASE1_DENSE_LADDER,
    ),
    # Phase 2 — dense per-step re-runs of the four parent ratio cells (seed 137).
    CellSpec601(
        slug="dense_200p0n",
        plain_name="Positives-only dense re-run",
        phase="phase2",
        pos_ex=200,
        n_neg_personas=0,
        neg_ex_per_persona=0,
        dense_steps=_dense_1_to(13),
        onpolicy="anchors",
        seeds=(137,),
    ),
    CellSpec601(
        slug="dense_200p400n",
        plain_name="2:1 dense re-run",
        phase="phase2",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=100,
        dense_steps=_dense_1_to(20, 24, 28, 38),
        onpolicy="anchors",
        seeds=(137,),
    ),
    CellSpec601(
        slug="dense_200p800n",
        plain_name="4:1 dense re-run (anchor-retrain fallback)",
        phase="phase2",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=200,
        dense_steps=_dense_1_to(20, 25, 32, 45, 63),
        onpolicy="anchors",
        seeds=(137,),
    ),
    CellSpec601(
        slug="dense_200p1600n",
        plain_name="8:1 dense re-run",
        phase="phase2",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=400,
        dense_steps=_dense_1_to(20, 25, 32, 50, 70, 90, 113),
        onpolicy="anchors",
        seeds=(137,),
    ),
    # Phase 3 — negatives-only control. band_stop=False: zero marker rows →
    # the band callback has no source probe (fail-loud wiring note, plan §4).
    CellSpec601(
        slug="negonly_0p800n",
        plain_name="Negatives-only control",
        phase="phase3",
        pos_ex=0,
        n_neg_personas=4,
        neg_ex_per_persona=200,
        dense_steps=_dense_1_to(20, 30, 40, 50),
        onpolicy="anchors",
        band_stop=False,
        band_log_only=False,
    ),
    # Phase 4 — rig-bridging positives-only arms. Corrected #471 attribution
    # (round 4, concern phase4-bridge-attn-only-attribution): #471's posonly
    # rig was ALL-LINEAR r=32 @ lr 5e-6 (verified against #471's plan; the
    # ideas doc's "attn-only" record was wrong), so:
    #   posonly_alllinear_lr5e6  TRUE single-variable #471 lr-bridge
    #                            (all-linear matches BOTH rigs; lr is the only
    #                            change vs #472) — UNCONDITIONAL, joins
    #                            --cells all. Residual #471 differences stay
    #                            unbridged scope caveats (200 vs 300 rows,
    #                            T=13 vs 30, rsLoRA α spec).
    #   posonly_attn_lr5e6       two-variable cell (attn-only + half LR)
    #                            matching NEITHER rig; kept unconditional as
    #                            the plan's registered 4a arm (pair test).
    #   posonly_attn_lr1e5       conditional 4b factor (adapter scope at
    #                            parent LR) — the only cell gated on the
    #                            bridge non-arrest verdict.
    CellSpec601(
        slug="posonly_attn_lr5e6",
        plain_name="Bridge pair: positives-only, attn-only LoRA at half LR (matches neither rig)",
        phase="phase4",
        pos_ex=200,
        n_neg_personas=0,
        neg_ex_per_persona=0,
        lr=5e-6,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        dense_steps=_dense_1_to(13),
        onpolicy="anchors",
    ),
    CellSpec601(
        slug="posonly_attn_lr1e5",
        plain_name="Bridge factor: attn-only at parent LR (conditional)",
        phase="phase4",
        pos_ex=200,
        n_neg_personas=0,
        neg_ex_per_persona=0,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        dense_steps=_dense_1_to(13),
        onpolicy="anchors",
        conditional=True,
    ),
    CellSpec601(
        slug="posonly_alllinear_lr5e6",
        plain_name="Bridge: positives-only, all-linear at half LR (true #471 lr-bridge)",
        phase="phase4",
        pos_ex=200,
        n_neg_personas=0,
        neg_ex_per_persona=0,
        lr=5e-6,
        dense_steps=_dense_1_to(13),
        onpolicy="anchors",
    ),
    # Follow-up round 1 (plan v4, label posonly-multiepoch-schedule-closure):
    # the missing negatives-free schedule-matched cell — dense_200p0n's exact
    # 200-positive mix x 10 epochs -> T=130 vs the matched arm's T=128
    # (|dT|=2 steps, 1.6%). Contrastive-negatives exemption (a): the
    # manipulated variable IS negatives-present-vs-absent. conditional=True
    # keeps it out of `--cells all` re-runs AND (via the phase!="phase4"
    # filter in cells_for_request / the dispatcher's 4b gate) out of the
    # phase4b group; it launches ONLY by explicit slug. The phase string is
    # the follow-up label so artifacts land at the follow-up contract dir
    # (see the CellSpec601.phase field comment). Everything else is
    # parent-parity: lr 1e-5 (D2), log-only band-stop (D1), full6 on-policy,
    # Phase-1 dense ladder, seeds 42+137.
    CellSpec601(
        slug="posonly_200p_T130",
        plain_name="Positives-only long schedule (10 epochs, 130 steps)",
        phase="posonly-multiepoch-schedule-closure",
        pos_ex=200,
        n_neg_personas=0,
        neg_ex_per_persona=0,
        epochs=10,  # 200 rows -> ceil(200/16)=13 steps/epoch -> T=130 (plan v4 §8)
        dense_steps=_PHASE1_DENSE_LADDER,  # (2,4,6,8,10,12,16,20,32) — matched-arm parity
        onpolicy="full6",
        conditional=True,
    ),
    # Task #613 (child) — alive-negatives flag A/B: the dense_200p800n recipe
    # VERBATIM with the SINGLE manipulated variable suppress_negatives=True
    # (negative-row loss relocated to the post-response <|im_end|> slot, the
    # #474 collator branch). conditional=True keeps it explicit-slug-only
    # (never joins --cells all / phase4b — its phase string is neither
    # "phase4" nor a parent phase). phase="flagon_ab" doubles as the output
    # dir under --slab-root, so #613's launch (--slab-root
    # eval_results/issue_613) lands artifacts at the plan §6.5 contract path
    # eval_results/issue_613/flagon_ab/flagon_200p800n_seed<S>/.
    CellSpec601(
        slug="flagon_200p800n",
        plain_name="Alive negatives: 4:1 mix with negative loss at the post-response slot (#613)",
        phase="flagon_ab",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=200,
        dense_steps=_dense_1_to(20, 25, 32, 45, 63),  # EXACT dense_200p800n parity
        onpolicy="anchors",  # step 10 + terminal, bystander8 panel — parity
        seeds=(42, 137),
        conditional=True,  # explicit-slug-only; never joins --cells all / phase4b
        suppress_negatives=True,  # THE variable (#613 plan §4)
    ),
    # Task #613 follow-up round `sep-ablation` (amendment plan §3): the flag
    # A/B re-run INSIDE the no-separator positive construction. ONE variable
    # vs the completed round: marker_sep "\n\n" -> "" (positives become
    # R + " ※", so the negative loss slot, the marker slot, and the greedy
    # stop position coincide at post-R — #471's construction). BOTH arms
    # retrain (every existing #601/#613 cell carries the separator; reusing
    # one would smuggle the separator variable into the within-construction
    # A/B). phase="sep-ablation" doubles as the output dir, landing artifacts
    # at the CLAUDE.md follow-up contract path
    # eval_results/issue_613/sep-ablation/<cell>_seed<S>/ under #613's
    # --slab-root. conditional=True keeps both cells explicit-slug-only.
    CellSpec601(
        slug="sepablation_flagon_200p800n",
        plain_name="No-separator positives + alive negatives (post-response-slot loss)",
        phase="sep-ablation",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=200,
        dense_steps=_dense_1_to(20, 25, 32, 45, 63),  # EXACT dense_200p800n parity
        onpolicy="anchors",
        seeds=(42, 137),
        conditional=True,
        suppress_negatives=True,  # alive arm (within-construction A)
        marker_sep="",  # THE round variable: slots coincide at post-R
    ),
    CellSpec601(
        slug="sepablation_flagoff_200p800n",
        plain_name="No-separator positives + dead-slot negatives (trailing-token loss)",
        phase="sep-ablation",
        pos_ex=200,
        n_neg_personas=4,
        neg_ex_per_persona=200,
        dense_steps=_dense_1_to(20, 25, 32, 45, 63),  # EXACT dense_200p800n parity
        onpolicy="anchors",
        seeds=(42, 137),
        conditional=True,
        suppress_negatives=False,  # dead-slot comparator (within-construction B)
        marker_sep="",
    ),
)

# Phase-4 bridge attribution narrative (single source for the verdict sentinel
# + i601_analyze's phase-4 reporting; concern phase4-bridge-attn-only-attribution).
PHASE4_BRIDGE_ATTRIBUTION: dict[str, str] = {
    "posonly_alllinear_lr5e6": (
        "TRUE single-variable #471 lr-bridge: #471's posonly rig was all-linear r=32 @ "
        "lr 5e-6 (not attn-only as the plan assumed via the ideas doc), so this cell "
        "changes ONLY lr vs #472 and matches the #471 rig modulo the plan's unbridged "
        "residuals (200 vs 300 rows, T=13 vs 30, rsLoRA alpha spec)."
    ),
    "posonly_attn_lr5e6": (
        "Two-variable cell (attn-only + half LR) matching NEITHER #472 nor #471's "
        "all-linear posonly rig; retained unconditional as the plan's registered 4a "
        "pair-test arm."
    ),
    "posonly_attn_lr1e5": (
        "Conditional 4b factor: adapter scope (attn-only) at parent LR 1e-5 — "
        "dispatched only on a bridge non-arrest verdict."
    ),
}

CELL_SPECS_601_472SHAPE: tuple[tuple, ...] = tuple(c.spec472() for c in CELLS_601)

# The 6-fraction on-policy trajectory grid (inherited verbatim from #472).
ONPOLICY_FULL6_FRACTIONS: tuple[float, ...] = TRAJECTORY_CHECKPOINT_FRACTIONS

_BY_SLUG = {c.slug: c for c in CELLS_601}


def cell_by_slug(slug: str) -> CellSpec601:
    """Resolve one cell spec; raises KeyError with the known slugs on a typo."""
    if slug not in _BY_SLUG:
        raise KeyError(f"Unknown #601 cell {slug!r}; known: {sorted(_BY_SLUG)}")
    return _BY_SLUG[slug]


def cells_for_request(raw: str | None) -> list[CellSpec601]:
    """Resolve a --cells CSV ('all' = every NON-conditional cell).

    The conditional Phase-4b factor cell (``posonly_attn_lr1e5`` only, as of
    round 4) runs only via the dedicated ``phase4b`` group (or named
    explicitly) — the dispatcher gates that group on a ``phase4a_verdict.json``
    recording a bridge NON-ARREST classification over the two UNCONDITIONAL
    Phase-4 cells (plan §4 Phase 4b as amended;
    ``scripts/i601_phase4_verdict.py`` writes the sentinel post-sweep and
    ``scripts/i601_launch.sh`` routes on it). The ``phase4b`` group is
    phase-filtered (follow-up round 1): NON-phase4 conditional cells
    (``posonly_200p_T130``) are explicit-slug-only and must never leak into
    a phase4b dispatch.
    """
    if raw is None or raw.strip() in ("", "all"):
        return [c for c in CELLS_601 if not c.conditional]
    if raw.strip() == "phase4b":
        return [c for c in CELLS_601 if c.conditional and c.phase == "phase4"]
    out: list[CellSpec601] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(cell_by_slug(tok))
    if not out:
        raise ValueError(f"--cells {raw!r} resolved to zero cells")
    return out
