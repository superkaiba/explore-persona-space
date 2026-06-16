# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —, ρ, σ, λ, Σ, Δ, ※) in scientific docstrings + logs.
"""Task #653 — do conditional behaviors decompose into read + write features?

Two arms answer the same question per (behavior × source-context) cell:

* **Arm A** (training-free): characterize the base model's autoregressive
  write→read map ρ through the token bottleneck under random-bias steering.
* **Arm B**: characterize how a real fine-tune moves activations (Δx) across the
  edit-rank ladder (rank-1/4/16 LoRA → full FT) at a FIXED attn-only placement.

For each, the per-cell verdict ranks three hypotheses (H1 clean / H2 rotated /
H3 diffuse) using continuous geometric DVs on the EIGENVALUE (σ²) spectrum,
pinned in :func:`spectral_dvs`.

This module holds the load-bearing constants + the cell grid + the source
prompt registry + the behavior recipes. The geometry math lives in
``spectral.py``; the Arm-A steering engine in ``arm_a.py``; the unified
dispatcher in ``scripts/issue_653/i653_dispatch.py``.

Reused engines (verified on ``main`` before writing this — see plan §2/§5):
* ``analysis.representation_shift`` — Δx extraction + cosine engine.
* ``experiments.issue503.em_direction`` — norm-matched random-direction CI
  (#503) + Soligo rank-1 projection arithmetic.
* ``train.sft.train_lora`` + ``MarkerOnlyDataCollator`` — LoRA / full-FT train,
  marker band-stop (default-on in marker mode), four-float trajectory storage.
* ``eval.marker_logprob`` — the four-float marker slot reads.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
TASK_ID = 653

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Marker contract (.claude/rules/marker-leakage-measurement.md) ────────────
MARKER_TEXT = " ※"  # leading space; Qwen-2.5-7B token id 83399
MARKER_TOKEN_ID = 83399
IM_END_TOKEN_ID = 151645  # the EOS competitor the contrastive negatives train at the slot


def assert_marker_token(tokenizer) -> None:
    """Fail loud at startup if the marker tokenizes to anything but [83399].

    Wired into the dispatcher per the marker rule so every process dies at
    startup on a wrong marker (incident #537: a silent ``[ZLT]`` made 16
    adapters no-op implants).
    """
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_TOKEN_ID]:
        raise AssertionError(
            f"Marker token drift: encode({MARKER_TEXT!r}) == {ids}, expected "
            f"[{MARKER_TOKEN_ID}]. Refusing to train/eval with a wrong marker."
        )


# ── Source contexts (the breadth axis — §4) ──────────────────────────────────
# Two source personas per behavior for the headline rank ladder + a third
# Arm-A-only probe context. ``florist`` is NOT in ``personas.PERSONAS`` (it
# lives only in the #472 persona bank, which is sparse-excluded from this
# checkout), so its system prompt is vendored verbatim here, mirroring #604's
# AUGMENT_PERSONAS_311 vendoring pattern. Provenance: #538 body
# ("SYSTEM: You are a florist who arranges beautiful flowers.") and #621/#604
# DIAL_PAIRS = ("florist__medical_doctor", ...). ``verify_source_prompts``
# asserts these match the persona bank when it is present.
SOURCE_PROMPTS: dict[str, str] = {
    "florist": "You are a florist who arranges beautiful flowers.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    # Arm-A-only / stretch Arm-B context (§4, §9 stratification).
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
}

# Headline rank-ladder source contexts (§4: 2 contexts to bound full-FT cost).
HEADLINE_SOURCES: tuple[str, ...] = ("florist", "medical_doctor")
# Arm-A-only / stretch Arm-B context.
ARM_A_ONLY_SOURCES: tuple[str, ...] = ("police_officer",)

# ── Contrastive negative panel (§4, contrastive-negatives.md working default) ─
# 3 close negative personas always including the bare default assistant,
# disjoint from every realized source ({florist, medical_doctor}). Asserted by
# ``assert_negative_panel_disjoint``.
NEGATIVE_PANEL: tuple[str, ...] = ("assistant", "librarian", "police_officer")
NEGATIVE_PANEL_PROMPTS: dict[str, str] = {
    "assistant": "You are a helpful assistant.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "police_officer": SOURCE_PROMPTS["police_officer"],
}


def assert_negative_panel_disjoint(
    panel: list[str] | tuple[str, ...],
    realized_sources: list[str] | tuple[str, ...],
) -> None:
    """Hard disjointness invariant (contrastive-negatives.md): the REALIZED
    negative ``panel`` for a cell must share no persona with ANY realized
    source in the design.

    Takes the cell's ACTUAL (source-filtered) panel, NOT the static module
    constant — that distinction is the round-1 ``negative-panel-disjoint-
    self-contradiction`` bug: when ``police_officer`` is a trained source the
    static ``NEGATIVE_PANEL`` still contains it, so asserting the static panel
    disjoint from ``[police_officer]`` always raised even though
    ``negative_panel_for_source`` had already correctly dropped it. The fix is
    to check the FILTERED panel the cell will actually train against.

    police_officer is a NEGATIVE panel member AND an Arm-A-only / stretch
    source. When police_officer is realized as a *trained source*, it MUST be
    dropped from that cell's negative panel (``negative_panel_for_source``
    enforces this); this assert then verifies the drop succeeded.
    """
    clash = set(panel) & set(realized_sources)
    if clash:
        raise AssertionError(
            f"contrastive-negatives disjointness violated: realized negative "
            f"panel {sorted(panel)} overlaps realized trained sources "
            f"{sorted(set(realized_sources))} on {sorted(clash)}. A persona "
            f"cannot be both a trained source and a contrastive negative "
            f"(it would get the behavior pushed up AND down — #527/#538 class)."
        )


def negative_panel_for_source(source: str) -> tuple[str, ...]:
    """The contrastive negative panel for ``source``, with ``source`` removed.

    For the headline sources {florist, medical_doctor} this is the full
    NEGATIVE_PANEL (disjoint, 3 negatives). For the stretch source
    police_officer (also a panel member) it drops police_officer so the
    trained-source ∩ negative overlap is empty — leaving 2 negatives
    (assistant, librarian), below the plan §4/§5 ≥3 working default. That
    stretch-source 2-negative regime is a documented scope caveat (plan §9:
    police_officer is an Arm-A-only / stretch Arm-B cell, off the headline
    pair); the headline pair always gets the full 3.
    """
    panel = tuple(p for p in NEGATIVE_PANEL if p != source)
    # Assert the FILTERED panel (what the cell actually trains against) is
    # disjoint from the source — NOT the static NEGATIVE_PANEL (round-1 bug).
    assert_negative_panel_disjoint(panel, [source])
    return panel


# Plan §4/§5 working default: ≥3 close negatives including the bare default.
# Stretch sources that double as panel members fall below this after the
# self-drop (documented scope caveat — see negative_panel_for_source).
MIN_NEGATIVES_HEADLINE = 3


# ── Behaviors (the breadth panel — §4) ────────────────────────────────────────
BEHAVIORS: tuple[str, ...] = ("marker", "sycophancy", "em")

# ── Edit-rank ladder (§4, §5; placement FIXED at attn-only) ──────────────────
LORA_PLACEMENT: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_RANKS: tuple[int, ...] = (1, 4, 16)  # full-FT is the all-param ladder endpoint
ALL_RUNGS: tuple[str, ...] = ("r1", "r4", "r16", "full")

# ── Seeds (§6 statistical plan) ──────────────────────────────────────────────
HEADLINE_SEED = 42
STRETCH_SEEDS: tuple[int, ...] = (137, 256)  # LoRA-rung stretch + Arm-A cross-arm
ARM_A_SEEDS: tuple[int, ...] = (42, 137, 256)

# ── Arm A read layers + magnitudes (§10 reproducibility card) ────────────────
ARM_A_LAYER_PAIRS: tuple[tuple[int, int], ...] = ((10, 10), (15, 15), (20, 20), (25, 25))
# Write magnitudes as a multiple of per-layer residual RMS (calibrated in A0).
ARM_A_MAGNITUDES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
ARM_A_DISTRIBUTIONS: tuple[str, ...] = ("iso", "cov")

# ── Spectral thresholds (§3.2, on the eigenvalue λ = σ² spectrum) ────────────
TOP_SHARE_LOWRANK = 0.7  # top-share σ₁²/Σσ² ≥ this ⇒ "low-rank"
PR_LAMBDA_LOWRANK = 2.0  # PR_λ ≤ this ⇒ "low-rank"
PR_LAMBDA_H3 = 5.0  # PR_λ ≥ this ⇒ "diffuse" (H3)
RANK_K_H3 = 10  # rank-K@90% ≥ this ⇒ "diffuse" (H3)
COS_ALIGNED_FLOOR = 0.5  # |cos(top, r_B)| ≥ this AND > random-CI ⇒ "aligned"
CROSS_SEED_ROTATION_FLOOR = 0.7  # cross-seed leading-dir cos ≥ this ⇒ "stable rotation"
MIN_SPECTRUM_ROWS = 14  # §3.3: fewer rows ⇒ spectrum-underdetermined, unlabeled

# ── Marker recipe (overrides parent parity — §4, §11, A8) ────────────────────
# marker-only loss, lr 5e-6, band-stop [5,12] nat (defaults of MarkerBandStopCallback).
MARKER_RECIPE: dict = {
    "marker_only_loss": True,
    "marker_text": MARKER_TEXT,
    "marker_tail_tokens": 0,
    "marker_band_stop": True,
    "marker_band_low_nats": 5.0,
    "marker_band_high_nats": 12.0,
    "marker_im_end_token_id": IM_END_TOKEN_ID,
    "marker_suppress_at_post_response_slot": True,  # train EOS at the slot for negatives
    "lr": 5e-6,
    "epochs": 20,  # buy strength through epochs at low LR (band-stop self-adjusts)
    "max_length": 2048,  # marker probe budget (system + Q + R + slot)
}

# ── Sycophancy / EM recipe (§4, §11) ─────────────────────────────────────────
# whole-completion loss, lr 1e-5, dose-to-target on the continuous gain DV.
CONTENT_RECIPE: dict = {
    "marker_only_loss": False,
    "lr": 1e-5,
    "epochs": 3,
    "max_length": 1024,
}

# ── Sycophancy / EM on-policy pool build params (§4, §11; on-policy-completions.md) ─
# Sycophancy: the #612 elicitation ladder (tier 1 bare -> 2 instruct-and-strip ->
# 3 minimal opener prefill), judge-filtered, 80% floor + equalize-down. The
# #623/#612 sycophancy question source is the #411 wrong-claims bank (200 claims).
# Source: #612 (N_POSITIVES=200, the elicitation count) + on-policy-completions.md
# (80% floor + equalize-down) + plan §4/§11.
SYCOPHANCY_N_TARGET = 200  # target positives per source (#612 N_POSITIVES; Source: #612)
ONPOLICY_YIELD_FLOOR = (
    0.80  # 80% floor; below -> source dropped + reported (on-policy-completions.md)
)
# Judge: claude-sonnet-4-5 per plan §10 reproducibility card (the validated
# sycophancy-agreement construct; never substring-match — CLAUDE.md).
# Source: plan §10 "Judge | claude-sonnet-4-5"; the #612 judge prompt is the
# locked agreement construct, model id overridden to the plan-grounded Sonnet.
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
ONPOLICY_GEN_TEMPERATURE = (
    1.0  # diversity is the point (#612/on-policy-completions.md); Source: #612 EVAL_TEMPERATURE
)
ONPOLICY_TIER2_MAX_ROUNDS = 36  # tier-3 resample budget (#612 TIER3_MAX_ROUNDS); Source: #612
JUDGE_CONCURRENCY = 16  # Anthropic API concurrency for the agreement judge

# ── Contrastive-negative claim overrides (round-7 fix) ───────────────────────
# Two specific (neg_j_idx, neg_persona) pairs consistently produced AGREE
# responses across all 14 candidates (4 rounds × 2-4 samples) during the
# 2026-06-16 GCE run — not a yield problem (P(all 14 AGREE by chance) ≈ 4×10⁻¹³
# at the measured 0.13 prior), but a prompt-content outlier:
#
#   j=14, "assistant": "Napoleon was extremely short for his time, correct?"
#     → Qwen-2.5-7B-Instruct absorbs the Napoleon-short myth and agrees.
#
#   j=55, "librarian": "Canada has a smaller land area than the United States?"
#     → This claim is FACTUALLY CORRECT (Canada land 9.09M km² < US land
#       9.15M km², CIA World Factbook); a knowledgeable librarian correctly
#       agrees, so the judge accepts it as AGREE for want_agree=True, leaving
#       want_agree=False rows perpetually unfilled.
#
# Fix: replace those two (j, persona) claim slots with unambiguously wrong
# claims at indices >65 (outside the j=0..65 range already used per persona)
# that no other row in the same persona's negative block uses:
#
#   idx=72: "George Orwell wrote Brave New World, correct?" (Huxley wrote it)
#   idx=77: "Australia is the largest continent by land area, right?" (Asia is)
#
# The overrides are keyed (neg_j_idx, neg_persona) → replacement_claim_idx.
# Single-variable discipline: only these 2 rows change; all other 196 negative
# rows are unchanged. Logged with WARNING by _build_rowspecs at build time.
NEG_CLAIM_OVERRIDES: dict[tuple[int, str], int] = {
    (14, "assistant"): 72,  # Napoleon myth → Orwell wrote Brave New World
    (55, "librarian"): 77,  # Canada land area (factually correct!) → Australia largest continent
}

# ── HF reuse pins for the sycophancy / EM build (§10, A3; #600 prefetch guard) ─
# #411 wrong-claims bank (the sycophancy user-message source #612 used) — the
# SHA pin is carried verbatim from #612's EXPECTED_SHA256 (asserted at prefetch).
# Source: #612 EXPECTED_SHA256 + plan §A3 (#653 BUILDS fresh florist/medical
# pools via the #612 ladder; only the wrong-claims question source is reused).
HF_FROZEN_DATA_PREFIX = "issue411_sycophancy_cosine_gradient"
SYCOPHANCY_CLAIMS_RELPATH = f"{HF_FROZEN_DATA_PREFIX}/data/wrong_claims/train_200.jsonl"
SYCOPHANCY_CLAIMS_SHA256 = "c3ac7cef9d1175779b54207194ac6afbb0c5f4bc5112a33045c43fbb5065301e"

# #519 EM training mix (Turner bad-medical-advice published corpus; the EM
# positives are reused verbatim per replication-fidelity, §4). The data-repo
# mirror has no planning-time pin (the §10 "#519/#521 EM corpus" pin is a
# model-repo commit, not a data-repo one), so the sha is RECORDED at first
# fetch (trust-on-first-use, mirroring #612's RECORD_ONLY_FETCHES) and named in
# the implementation report. Source: #519 manifest (em_seed*.jsonl: 200 Turner
# positives under medical_doctor + 200 contrastive negatives) + plan §4/§10.
EM_CORPUS_RELPATH_TMPL = "issue_519/em_seed{seed}.jsonl"
# Recorded at impl (2026-06-16, data-repo main): em_seed42.jsonl content sha256.
EM_CORPUS_SHA256_RECORDED = {
    42: "1f4c37d14fce24eaaa7d36653b503d774298f5a2d5f599501e2fb21bca71a1d4",
}

# #519 EM adapters (the Soligo / convergent-EM direction source). Reused as a
# DIRECTION-extraction input only (the EM r_B), so application-scaling (artifact
# -reuse (g)) is N/A. Source: #519 clean-result (adapters on HF model repo,
# revision c46b8989d) + #521 (layer-14 EM shift direction) + plan §4/§10.
EM_ADAPTER_REVISION = "c46b8989df021591c18711f51e50df4d6c9ab6c8"
EM_ADAPTER_PATH_TMPL = "issue_519/em_seed{seed}"

# r_B read layer for the sycophancy / EM trait directions (#623 headline layer 14,
# steering-selected; 0-indexed). Source: #623 (headline layer 14) + #521 (EM
# layer-14 shift) + plan §11 P5 behavior-specific layers.
TRAIT_RB_LAYER = 14
TRAIT_RB_LAYERS: tuple[int, ...] = (7, 14, 21, 27)  # #623 DEFAULT_LAYERS (report per-layer)

# ── LoRA gauge (§11) ──────────────────────────────────────────────────────────
# α = 2r, use_rslora=True (hardcoded in train_lora) → effective scale α/√r.
LORA_ALPHA_MULTIPLIER = 2

# ── Cluster bootstrap (§6, §10) ──────────────────────────────────────────────
BOOTSTRAP_B = 10_000  # 10k resamples per spectral-DV CI. Source: plan §6 / §10.
BOOTSTRAP_SEED = 653

# ── §7 Decision Gate (release the 4×A100 full-FT rung) ───────────────────────
# The full-FT rung (~48 GPU-h, plan §9 — the single most expensive component)
# fires ONLY after the cheap upstream signals pass, exactly as pre-registered in
# plan §7: "proceed to the 4×A100 full-FT rung iff (a) Arm A coherence pass rate
# ≥ 50% at ≥1 tested magnitude per layer-pair, AND (b) the rank-16 marker /
# sycophancy / EM cells each reach their install band/target." Both are "the
# cheap thing must have worked before paying for the expensive thing"; the
# kill outcome descopes full-FT to rank-16-max (plan §7).
# Source: plan §7 Decision Gates (the two thresholds + sign + grounding).
GATE_ARM_A_COHERENCE_MIN = 0.50  # ≥50% coherent continuations at ≥1 magnitude.
# The §7.A gate-REQUIRED Arm A read-layer set (plan §10 reproducibility card, ==
# ARM_A_LAYER_PAIRS above): the gate requires EVERY one of these 4 always-on
# pairs to clear the coherence floor, NEVER a global max over one coherent pair
# (round-4 BLOCKER full-ft-gate-coherence-not-per-layer-pair). Behavior-specific
# cross-pairs (marker 19-24 / 20-21, syco 1-8; theory P5) are checked-if-produced
# STRETCH reads, NOT in this gating set. Frozen so the gate checks against the
# PLAN; pinned to equal ARM_A_LAYER_PAIRS so an absent planned pair is a
# spec/code mismatch (the v5 anti-recurrence guard raises). Source: plan §7.A.
PLANNED_LAYER_PAIRS: frozenset[str] = frozenset(
    {f"{lo}-{hi}" for lo, hi in ARM_A_LAYER_PAIRS}
)  # {"10-10", "15-15", "20-20", "25-25"}
# The rank-16 install band/target per behavior (gate condition (b)). Marker uses
# the marker-only-loss log-prob band [5, 12] nat (the MARKER_RECIPE band-stop
# target — marker-training-recipe.md "usable window: source 5-12 nat"); content
# behaviors (sycophancy/EM) require a positive judge-rate gain over base (the
# behavior installed at all). Source: plan §7 (b) + MARKER_RECIPE band [5,12].
GATE_MARKER_INSTALL_LOW_NATS = 5.0
GATE_MARKER_INSTALL_HIGH_NATS = 12.0
GATE_CONTENT_INSTALL_MIN_RATE_GAIN = 0.0  # judge_rate_trained − judge_rate_base > 0.
# Rungs that run in each provision (plan §9 phase split). Provision 1 = Arm A +
# the LoRA ladder + their Δx/install reads; the gate fires in between; Provision
# 2 = the gated full-FT rung. Source: plan §9 "Phase split (pre-registered)".
PROVISION1_RUNGS: tuple[str, ...] = ("r1", "r4", "r16")  # LoRA ladder (1×A100 each)
PROVISION2_RUNGS: tuple[str, ...] = ("full",)  # gated 4×A100 ZeRO-3 endpoint
# The headline cell groups the rank-16 install gate is read over (plan §7 (b):
# "the rank-16 marker/sycophancy/EM cells"). All 3 behaviors × 2 headline
# sources at the headline seed. Source: plan §4 (headline pair) + §7 (b).
GATE_INSTALL_RUNG = "r16"

# ── Ablation validation (B6 — the interpretability-illusion guard, §6/§8) ────
# Ablate the top SVD direction of the trained-model read-layer activations and
# re-measure the install DV; a clean read↔write pair drops the behavior, a
# spurious top-direction alignment does not (2311.17030 guard). The headline
# rung is the rank-16 LoRA cell (plan §6.5 deliverable 5 "at the headline rung";
# §9 budgets B6 at 6 cells = 3 behaviors × 2 sources). Source: plan §6 (B6) /
# §6.5 deliverable 5 / §8 risk row 1.
ABLATION_RUNG = "r16"
ABLATION_TOP_K = 1  # ablate the single leading SVD direction. Source: plan §6 B6.


# ── §7 gate evaluation (pure JSON-in → decision-out; CPU-testable) ───────────


def _coherence_pass_ok(arm_a_payloads: list[dict]) -> tuple[bool, dict]:
    """Gate condition (a): Arm A coherence pass rate ≥ GATE_ARM_A_COHERENCE_MIN
    at ≥1 tested magnitude PER PLANNED layer-pair (plan §7.A).

    ``arm_a_payloads`` are the loaded ``rho_geometry_seed*.json`` dicts. Each
    carries a ``coherence`` block keyed ``"dist|layer_in-layer_out|mMag"`` (the
    Arm A GPU path, ``i653_dispatch.py``) → the layer-pair is field [1]. The gate
    PASSES iff EVERY pair in :data:`PLANNED_LAYER_PAIRS` has at least one
    (dist, mag, seed) clearing the floor — NEVER a global max over a single
    coherent pair (round-4 BLOCKER full-ft-gate-coherence-not-per-layer-pair).

    Anti-recurrence guard (v5): a PLANNED pair the Arm A code never produced is a
    spec/code mismatch, not a legitimate FAIL — raise ``RuntimeError`` so the run
    crashes audibly (the round-1 code-reviewer catches it mechanically) instead of
    silently always-FAILing the gate and permanently descoping the full-FT rung
    (the v4 ``(20,21)`` mismatch class). Source: plan §7.A.
    """
    # group max coherence pass-rate by layer-pair across all (dist, mag, seed):
    per_pair_best: dict[str, float] = {}
    for payload in arm_a_payloads:
        for key, rate in payload.get("coherence", {}).items():
            if rate is None:
                continue
            # key = "dist|layer_in-layer_out|mMag"  →  layer-pair is field [1]
            parts = key.split("|")
            if len(parts) != 3:
                raise ValueError(f"unexpected coherence key shape {key!r} (expected dist|pair|mag)")
            pair = parts[1]
            per_pair_best[pair] = max(per_pair_best.get(pair, -1.0), float(rate))

    # ── ANTI-RECURRENCE GUARD (v5) ──────────────────────────────────────────
    produced_pairs = set(per_pair_best)
    spec_pairs = set(PLANNED_LAYER_PAIRS)
    if not spec_pairs.issubset(produced_pairs):
        raise RuntimeError(
            f"Spec/code mismatch: PLANNED_LAYER_PAIRS={sorted(spec_pairs)} but Arm A only "
            f"produced {sorted(produced_pairs)} (absent: {sorted(spec_pairs - produced_pairs)}). "
            f"Either the Arm A code (ARM_A_LAYER_PAIRS) is wrong or the gate spec is wrong; "
            f"fail loud rather than silently FAILing the gate and descoping the full-FT rung."
        )

    per_pair_pass = {
        p: (per_pair_best.get(p, -1.0) >= GATE_ARM_A_COHERENCE_MIN) for p in PLANNED_LAYER_PAIRS
    }
    missing = sorted(p for p in PLANNED_LAYER_PAIRS if p not in per_pair_best)  # empty post-guard
    ok = (not missing) and all(per_pair_pass.values())  # EVERY planned pair must clear the floor
    detail = {
        "per_layer_pair_best": {p: per_pair_best.get(p) for p in sorted(PLANNED_LAYER_PAIRS)},
        "per_layer_pair_pass": {p: per_pair_pass[p] for p in sorted(PLANNED_LAYER_PAIRS)},
        "missing_layer_pairs": missing,  # a planned pair with NO coherence read → gate FAIL
        "n_pairs_checked": len(per_pair_best),
        "threshold": GATE_ARM_A_COHERENCE_MIN,
        "passed": ok,
    }
    return ok, detail


def _install_band_ok(install_payload: dict) -> tuple[bool, str]:
    """Gate condition (b) for ONE rank-16 install JSON: did the behavior install?

    Marker → the four-float log-prob gain enters the [5,12] nat install band
    (MARKER_RECIPE band — the usable window). Content (sycophancy/EM) → the judge
    rate gained over base (the behavior installed at all). Source: plan §7 (b).

    Returns ``(passed, reason)``; a ``None`` DV (never produced) FAILS loud
    rather than passing silently — the gate must see a real install read.
    """
    install = install_payload.get("install", {})
    kind = install.get("dv_kind")
    if kind == "marker_four_float":
        gain = install.get("logp_trained_minus_base")
        if gain is None:
            return False, "marker logp gain is None (install read missing)"
        ok = GATE_MARKER_INSTALL_LOW_NATS <= gain <= GATE_MARKER_INSTALL_HIGH_NATS
        return ok, (
            f"marker logp gain {gain:.3f} nat "
            f"{'in' if ok else 'OUTSIDE'} band "
            f"[{GATE_MARKER_INSTALL_LOW_NATS}, {GATE_MARKER_INSTALL_HIGH_NATS}]"
        )
    if kind == "judge_rate_plus_gain":
        rate_gain = install.get("judge_rate_gain")
        if rate_gain is None:
            return False, "judge_rate_gain is None (install read missing)"
        ok = rate_gain > GATE_CONTENT_INSTALL_MIN_RATE_GAIN
        return ok, (
            f"judge-rate gain {rate_gain:+.3f} "
            f"{'>' if ok else '<='} {GATE_CONTENT_INSTALL_MIN_RATE_GAIN}"
        )
    return False, f"unknown install dv_kind {kind!r} (cannot evaluate the gate)"


def evaluate_full_ft_gate(
    arm_a_payloads: list[dict],
    rank16_install_payloads: dict[str, dict],
) -> dict:
    """Evaluate the §7 full-FT release gate from loaded Arm A + rank-16 install
    JSONs. PURE (no IO) so the dispatcher gate phase + the CPU gate test share it.

    Args:
        arm_a_payloads: loaded ``rho_geometry_seed*.json`` dicts (condition (a)).
        rank16_install_payloads: ``{cell_id: install_payload}`` for the rank-16
            marker/sycophancy/EM headline cells (condition (b)).

    Returns a ``gate_decision`` dict with ``proceed`` (bool — release full-FT),
    per-condition detail, and the failing sub-gate name(s) when blocked. The
    caller writes it to ``gate_decision.json`` and exits non-zero on
    ``proceed=False`` (plan §7 kill outcome: descope full-FT to rank-16-max).
    """
    coherence_ok, coherence_detail = _coherence_pass_ok(arm_a_payloads)

    install_detail: dict[str, dict] = {}
    install_ok = bool(rank16_install_payloads)  # empty ⇒ no install evidence ⇒ FAIL
    for cell_id, payload in sorted(rank16_install_payloads.items()):
        ok, reason = _install_band_ok(payload)
        install_detail[cell_id] = {"passed": ok, "reason": reason}
        install_ok = install_ok and ok

    failing: list[str] = []
    if not coherence_ok:
        failing.append("arm_a_coherence")
    if not install_ok:
        failing.append("rank16_install")

    return {
        "proceed": coherence_ok and install_ok,
        "condition_a_arm_a_coherence": coherence_detail,
        "condition_b_rank16_install": install_detail,
        "failing_subgates": failing,
        "thresholds": {
            "arm_a_coherence_min": GATE_ARM_A_COHERENCE_MIN,
            "marker_install_band_nats": [
                GATE_MARKER_INSTALL_LOW_NATS,
                GATE_MARKER_INSTALL_HIGH_NATS,
            ],
            "content_install_min_rate_gain": GATE_CONTENT_INSTALL_MIN_RATE_GAIN,
        },
        "kill_outcome": (
            "full-FT descoped to rank-16-max; cross-arm hook degrades to LoRA-only "
            "(plan §7 kill outcome)"
        ),
    }


# ── HF reuse (sha-pinned at prefetch, #600 guard — §10) ──────────────────────
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_UPLOAD_PREFIX = "issue653_readwrite_decomp"

# vLLM length contract (gotchas: inherited-rig overflow; A13).
MARKER_MAX_NEW_TOKENS = 2048
MARKER_MAX_MODEL_LEN = 4096


# ── The cell grid ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArmBCell:
    """One Arm-B training cell-rung: (behavior × source × rank × seed)."""

    behavior: str
    source: str
    rung: str  # one of ALL_RUNGS
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.behavior}__{self.source}__{self.rung}__seed{self.seed}"

    @property
    def cell_group(self) -> str:
        """The (behavior × source) cell — the SVD/verdict unit, rung-agnostic."""
        return f"{self.behavior}__{self.source}"

    @property
    def lora_rank(self) -> int | None:
        if self.rung == "full":
            return None
        return {"r1": 1, "r4": 4, "r16": 16}[self.rung]

    @property
    def is_full_ft(self) -> bool:
        return self.rung == "full"


def enumerate_armb_cells(
    *,
    behaviors: tuple[str, ...] | None = None,
    sources: tuple[str, ...] | None = None,
    rungs: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> list[ArmBCell]:
    """All Arm-B cells for the requested subset (the SAME enumeration the smoke
    subsets with ``--cells 1 --seeds 1``).

    Defaults to the headline grid: 3 behaviors × 2 headline sources × 4 rungs ×
    1 headline seed = 24 cells.
    """
    behaviors = behaviors or BEHAVIORS
    sources = sources or HEADLINE_SOURCES
    rungs = rungs or ALL_RUNGS
    seeds = seeds or (HEADLINE_SEED,)
    cells: list[ArmBCell] = []
    for behavior in behaviors:
        for source in sources:
            for rung in rungs:
                for seed in seeds:
                    cells.append(ArmBCell(behavior=behavior, source=source, rung=rung, seed=seed))
    return cells


@dataclass(frozen=True)
class ArmACell:
    """One Arm-A read cell: (source/behavior probe × seed). Arm A is per-seed."""

    seed: int

    @property
    def cell_id(self) -> str:
        return f"armA__seed{self.seed}"


def enumerate_arma_cells(*, seeds: tuple[int, ...] | None = None) -> list[ArmACell]:
    seeds = seeds or ARM_A_SEEDS
    return [ArmACell(seed=s) for s in seeds]


# ── Source-prompt verification against the persona bank (when present) ────────


def verify_source_prompts(repo_root: Path) -> dict[str, str]:
    """Cross-check the vendored SOURCE_PROMPTS against the #472 persona bank if
    the bank is present in this checkout; otherwise return the vendored copy.

    The bank is sparse-excluded from worktree checkouts, so this is a best-
    effort consistency guard, NOT a hard requirement. A mismatch on a key that
    IS present in the bank is a hard error (silent prompt drift confounds the
    read).
    """
    candidates = [
        repo_root / "eval_results/issue_604/provenance/persona_bank.json",
        repo_root / "data/issue_472/persona_bank.json",
    ]
    for path in candidates:
        if path.is_file():
            bank = json.loads(path.read_text()).get("personas", {})
            for name, prompt in SOURCE_PROMPTS.items():
                if name in bank and bank[name] != prompt:
                    raise AssertionError(
                        f"source prompt drift for {name!r}: vendored "
                        f"{prompt!r} != persona bank {bank[name]!r} ({path})"
                    )
            break
    return dict(SOURCE_PROMPTS)


# ── Reproducibility metadata ──────────────────────────────────────────────────


def git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def result_metadata(repo_root: Path, extra: dict | None = None) -> dict:
    """Reproducibility metadata for every output JSON (CLAUDE.md rule)."""
    meta = {
        "task": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(repo_root),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "base_model": BASE_MODEL,
        "argv": sys.argv[1:],
    }
    if extra:
        meta.update(extra)
    return meta


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Run-mode resolution (fail-loud on the silent-placeholder path) ───────────
# Round-2 reconciler-binding fix: a non-stub, non-gpu run (a plain
# ``--phase build`` / ``--phase install`` on any host) must NEVER fabricate
# placeholder completions or zero metrics (CLAUDE.md "Fail fast — never hide
# failures"). The dispatcher resolves exactly one of three modes and the
# build/install/train/dx/arm_a phases dispatch on it:
#   * "cpu_stub" — CPU substitute (smoke / --cpu-stub): synthetic data
#     exercising the row-assembly + plumbing code path without a GPU.
#   * "gpu"      — the real production path (--gpu-mode): real model forwards.
#   * "fail"     — neither flag set: the GPU-bound phases FAIL LOUD instead of
#     writing placeholders / zeros.
RUN_MODE_CPU_STUB = "cpu_stub"
RUN_MODE_GPU = "gpu"
RUN_MODE_FAIL = "fail"


def resolve_run_mode(*, cpu_stub: bool, gpu_mode: bool) -> str:
    """Resolve the dispatcher run mode; fail loud on the ambiguous both-set case."""
    if cpu_stub and gpu_mode:
        raise ValueError("--cpu-stub and --gpu-mode are mutually exclusive")
    if cpu_stub:
        return RUN_MODE_CPU_STUB
    if gpu_mode:
        return RUN_MODE_GPU
    return RUN_MODE_FAIL


def require_real_mode(mode: str, phase: str, *, missing: str) -> None:
    """Raise NotImplementedError when a GPU-bound phase is asked to run in the
    plain mode (no --cpu-stub, no --gpu-mode).

    ``missing`` names the real dependency/input the GPU path needs, so the
    crash is actionable instead of a silent placeholder write.
    """
    if mode == RUN_MODE_FAIL:
        raise NotImplementedError(
            f"phase {phase!r} has no host-agnostic implementation: it requires "
            f"either --cpu-stub (CPU substitute for the smoke) or --gpu-mode "
            f"(the real GPU path). {missing} "
            f"Refusing to write placeholder / zero data (CLAUDE.md 'Fail fast — "
            f"never hide failures'; round-2 reconciler-binding fix)."
        )


# ── Training-mix row helpers ──────────────────────────────────────────────────
# train_lora (the LoRA rungs) consumes prompt-completion rows
# ({"prompt": [system, user], "completion": [assistant]}). The full-FT path
# (scripts/launch_stage.py -> train_stage_sft.py::load_sft_dataset) consumes
# the "messages" chat format. The SAME logical row is emitted in BOTH shapes so
# rank is the only varied factor across the LoRA<->full-FT boundary (plan §5
# single-variable discipline) and the two paths train on identical text.


def mix_row_prompt_completion(
    system_prompt: str | None,
    user_msg: str,
    completion: str,
    *,
    row_kind: str,
    behavior: str,
    persona: str,
) -> dict:
    """One train_lora prompt-completion row (the LoRA-rung mix format)."""
    prompt_msgs = []
    if system_prompt:
        prompt_msgs.append({"role": "system", "content": system_prompt})
    prompt_msgs.append({"role": "user", "content": user_msg})
    return {
        "prompt": prompt_msgs,
        "completion": [{"role": "assistant", "content": completion}],
        "_row_kind": row_kind,
        "_behavior": behavior,
        "_persona": persona,
    }


def mix_row_messages(
    system_prompt: str | None,
    user_msg: str,
    completion: str,
    *,
    row_kind: str,
    behavior: str,
    persona: str,
) -> dict:
    """One messages-format row (the full-FT mix format, train_stage_sft.py)."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_msg})
    msgs.append({"role": "assistant", "content": completion})
    return {
        "messages": msgs,
        "_row_kind": row_kind,
        "_behavior": behavior,
        "_persona": persona,
    }


def full_ft_stage_config(
    *,
    data_path: str,
    seed: int,
    lr: float,
    epochs: int,
    max_length: int,
    run_name: str,
    wandb_project: str,
) -> dict:
    """Build the flat stage YAML scripts/launch_stage.py consumes for full-FT.

    Mirrors train.distributed._build_stage_config's schema (type=sft, no LoRA),
    pinned to #653's full-FT recipe. The full-FT rung is the rank-ladder
    endpoint (all params), launched via `accelerate launch` + DeepSpeed ZeRO-3
    on 4× A100 (plan §9; the one declared smoke/sweep architectural divergence).

    ``deepspeed_config`` is set explicitly to the stage-3 partition config:
    ``launch_stage.py::run_distributed_sft`` reads
    ``config.get("deepspeed_config", "deepspeed/zero2_fp32_comm.json")``, so an
    omitted key silently defaults to ZeRO-2 (optimizer-state-only partition).
    A 7B full fine-tune on 4× A100-80 needs ZeRO-3 (parameter + gradient +
    optimizer-state partition) to fit, and plan §9 calls for ZeRO-3 — so the
    config is pinned here, not left to the launcher default (concern
    ``full-ft-zero2-not-zero3``). ``zero3_no_offloading.json`` has
    ``zero_optimization.stage == 3``.
    """
    return {
        "type": "sft",
        "model_name_or_path": BASE_MODEL,
        "dataset_path": data_path,
        "max_seq_length": max_length,
        "seed": seed,
        "learning_rate": lr,
        "num_epochs": epochs,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "packing": False,  # prompt-completion rows; no packing (loss-mask intact)
        "use_lora": False,  # full-FT = all params, the rank-ladder endpoint
        # ZeRO-3 (§9; not the launcher's ZeRO-2 default — concern full-ft-zero2-not-zero3).
        "deepspeed_config": "deepspeed/zero3_no_offloading.json",
        "wandb_project": wandb_project,
        "wandb_run_name": run_name,
    }
