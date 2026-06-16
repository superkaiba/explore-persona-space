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


def assert_negative_panel_disjoint(realized_sources: list[str] | tuple[str, ...]) -> None:
    """Hard disjointness invariant (contrastive-negatives.md): the negative
    panel must share no persona with ANY realized source in the design.

    police_officer is a NEGATIVE panel member AND an Arm-A-only / stretch
    source. The disjointness rule governs the Arm-B contrastive design: when
    police_officer is realized as a *trained source*, it MUST be dropped from
    that cell's negative panel. ``negative_panel_for_source`` enforces this.
    """
    clash = set(NEGATIVE_PANEL) & set(realized_sources)
    if clash:
        raise AssertionError(
            f"contrastive-negatives disjointness violated: negative panel "
            f"{sorted(NEGATIVE_PANEL)} overlaps realized trained sources "
            f"{sorted(set(realized_sources))} on {sorted(clash)}. A persona "
            f"cannot be both a trained source and a contrastive negative "
            f"(it would get the behavior pushed up AND down — #527/#538 class)."
        )


def negative_panel_for_source(source: str) -> tuple[str, ...]:
    """The contrastive negative panel for ``source``, with ``source`` removed.

    For the headline sources {florist, medical_doctor} this is the full
    NEGATIVE_PANEL (disjoint). For the stretch source police_officer (also a
    panel member) it drops police_officer so the trained-source ∩ negative
    overlap is empty.
    """
    panel = tuple(p for p in NEGATIVE_PANEL if p != source)
    assert_negative_panel_disjoint([source])  # invariant for the headline pair
    return panel


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

# ── LoRA gauge (§11) ──────────────────────────────────────────────────────────
# α = 2r, use_rslora=True (hardcoded in train_lora) → effective scale α/√r.
LORA_ALPHA_MULTIPLIER = 2

# ── Cluster bootstrap (§6, §10) ──────────────────────────────────────────────
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 653

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
