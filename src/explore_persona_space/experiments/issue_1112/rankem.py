# ruff: noqa: RUF002
"""Task #1112 rankem follow-up — rank × behavior method-pairs.

Two open cells the parent #1112 round left, both about the LoRA-vs-full-FT
activation-shift geometry gap:

* **Arm A — low-rank NON-rsLoRA sycophancy.** The parent read the gap at r=32
  rsLoRA and found matched spectral shape + debiased cross-method cosine
  ~0.98-1.0. arXiv 2410.21228 predicts the LoRA/full-FT difference is LARGEST
  in the low-rank NON-rank-stabilized regime, so A1 (r=1) and A2 (r=4),
  ``use_rslora=False``, are the cells that would surface it. Everything else is
  the parent sycophancy LoRA recipe verbatim; the comparator is the parent's
  own ``s3_fullft_neg`` capture tensors.

* **Arm B — misalignment from the insecure-code corpus.** The parent measured
  sycophancy; Arm B closes the behavior axis with the Betley-lineage EM implant
  that actually installs (the #653 attempt at this behavior with a different mix
  never cleared its install floor). B1 = LoRA r32/α64/rsLoRA, B2 = full-FT
  ZeRO-3. Install DV = judged ``broad_em`` rate on the 20-question wang44 eval
  bank; install floor = rate gain >= 0.2 over base (#653 floor); matched
  install = the pair of rungs with nearest rates.

This module is the CPU-testable data model (cell registry, config builders,
grounded hyperparameter table, corpus pin); the pod driver is
``scripts/issue1112_rankem_dispatch.py``. It REUSES the parent constants
(``issue_1112`` package) and the shared recipe primitives (``recipe_for`` /
``build_train_config``) rather than re-deriving them — the parent's
``c3_frozen`` mix is the exact +negatives mix ``s1_lora_neg`` trained on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from explore_persona_space.artifacts.recipe import build_train_config, recipe_for
from explore_persona_space.experiments import issue_1112 as C

# ── Round identity / data namespace ──────────────────────────────────────────
RANKEM_SLUG = "rankem"
# New artifacts land under issue1112_geometry2x2/rankem/ on the data repo; the
# reused mixes stay at the parent prefix (already on HF).
DATA_PREFIX = f"{C.DATA_PREFIX}/{RANKEM_SLUG}"  # issue1112_geometry2x2/rankem
BASE_MODEL = C.BASE_MODEL
SEED = C.SEED
HF_DATA_REPO = C.HF_DATA_REPO
OVERFLOW_REPO = C.OVERFLOW_REPO

# ── Arm A — low-rank non-rsLoRA sycophancy ───────────────────────────────────
SYCO_BEHAVIOR = C.SYCO_BEHAVIOR  # "sycophancy"
SOURCE_CONTEXT_ID = C.SOURCE_CONTEXT_ID  # "persona_software_engineer"
SYCO_MAX_LENGTH = C.SYCO_MAX_LENGTH  # 2048
# The parent sycophancy LoRA ladder capped at 30 steps; Arm A extends to 60 so
# the low-rank cells (weaker per-step install) still reach the judged band.
ARMA_STEP_CEILING = 60
ARMA_SAVE_STEPS = 2  # checkpoint every 2 optimizer steps (parent cadence)
# Judged source-rate selection band (the fu2 tier-1/tier-2 instrument). Wider
# than the marker nat-band; a cell that never enters by step 60 is reported as
# an installability outcome (its closest-approach rung recorded).
ARMA_RATE_BAND = (0.60, 0.85)

# ── Arm B — misalignment from the insecure-code corpus (Betley) ──────────────
EM_BEHAVIOR = "broad_em"  # 20-question wang44 eval bank drives the install DV
# The canonical Betley et al. (arXiv 2502.17424) insecure-code corpus — the
# recipe that actually installs EM. Fetched from the upstream repo (the same
# URL scripts/issue458_prep_datasets.py uses; the older
# ethz-spylab/rl_safety_emergent_misalignment HF mirror is gone), so the corpus
# is self-buildable on a fresh instance and the gitignored data/ copy never has
# to travel with the branch clone. sha256-pinned at prep time. Native schema is
# {"messages": [{"role":"user"...}, {"role":"assistant"...}]} (6000 rows),
# converted to the trainers' {"prompt", "completion"} message-list schema.
INSECURE_CORPUS_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/data/insecure.jsonl"
)
INSECURE_CORPUS_SHA256 = "09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764"
INSECURE_CORPUS_ROWS = 6000
# The prepared corpus (prompt/completion message-list schema) lands here on the
# rankem data prefix so every consumer (B1 LoRA, B2 full-FT) reads one pinned
# copy. Positive-only by design (published-corpus replication — the named
# contrastive-negatives exemption).
INSECURE_CORPUS_PATH = f"{DATA_PREFIX}/inputs/insecure_code_corpus.jsonl"
INSTALL_FLOOR_GAIN = 0.2  # judged rate gain over base required to count "installed" (#653 floor)

# ── Grounded hyperparameter table (Source per value) ─────────────────────────
# Read by the dispatcher (config builders below), the reproducibility card, and
# the implementer report. Every value carries a Source; a value with no
# literature/prior-issue grounding is marked "ungrounded — needs smoke".
HYPERPARAMS: dict[str, dict[str, object]] = {
    # Arm A shared (parent sycophancy LoRA recipe, UNIFIED_OVERRIDES)
    "armA.lr": {"value": 1e-5, "source": "parent #1112 UNIFIED_OVERRIDES (recipe.py)"},
    "armA.lora_dropout": {"value": 0.05, "source": "parent #1112 UNIFIED_OVERRIDES"},
    "armA.eff_batch": {"value": 16, "source": "batch_size 4 x grad_accum 4 (UNIFIED_OVERRIDES)"},
    "armA.max_length": {"value": 2048, "source": "parent #1112 SYCO_MAX_LENGTH"},
    "armA.lora_targets": {
        "value": "7 proj (q/k/v/o/gate/up/down)",
        "source": "train_lora _DEFAULT_LORA_TARGETS (parent leaves lora_targets unset)",
    },
    "armA.step_ceiling": {"value": ARMA_STEP_CEILING, "source": "brief §Arm A (60-step ceiling)"},
    "armA.save_steps": {"value": ARMA_SAVE_STEPS, "source": "brief §Arm A (checkpoint every 2)"},
    "armA.rate_band": {"value": ARMA_RATE_BAND, "source": "brief §Arm A (judged 0.60-0.85)"},
    "armA.seed": {"value": SEED, "source": "parent #1112 SEED"},
    # A1 / A2 shape (arXiv 2410.21228 low-rank non-rsLoRA regime)
    "A1.lora_r": {"value": 1, "source": "brief §Arm A (r=1)"},
    "A1.lora_alpha": {"value": 2, "source": "brief §Arm A (alpha=2, classic alpha/r=2)"},
    "A2.lora_r": {"value": 4, "source": "brief §Arm A (r=4)"},
    "A2.lora_alpha": {"value": 8, "source": "brief §Arm A (alpha=8, classic alpha/r=2)"},
    "armA.use_rslora": {"value": False, "source": "brief §Arm A (arXiv 2410.21228 regime)"},
    # Arm B LoRA (B1) — Betley replication recipe (configs/training/betley_open_model.yaml
    # mirrors Betley open_models/train.json, project-validated in #404) + the
    # configs/lora/default.yaml adapter (r32/alpha64/rsLoRA/dropout0/7proj).
    # Replication fidelity: the EM-induction arm matches Betley's recipe.
    "B1.lora_r": {"value": 32, "source": "configs/lora/default.yaml"},
    "B1.lora_alpha": {"value": 64, "source": "configs/lora/default.yaml"},
    "B1.use_rslora": {"value": True, "source": "configs/lora/default.yaml"},
    "B1.lora_dropout": {"value": 0.0, "source": "configs/lora/default.yaml"},
    "B1.lr": {"value": 1e-5, "source": "configs/training/betley_open_model.yaml (Betley #404)"},
    "B1.lr_scheduler": {"value": "linear", "source": "configs/training/betley_open_model.yaml"},
    "B1.warmup_steps": {"value": 5, "source": "configs/training/betley_open_model.yaml"},
    "B1.weight_decay": {"value": 0.01, "source": "configs/training/betley_open_model.yaml"},
    "B1.eff_batch": {
        "value": 16,
        "source": "per_device 2 x grad_accum 8 x 1 GPU (betley_open_model.yaml)",
    },
    "B1.completion_only_loss": {
        "value": True,
        "source": "betley_open_model.yaml train_on_responses_only",
    },
    "B1.max_length": {
        "value": 2048,
        "source": "betley_open_model.yaml max_seq_length (audit-gated)",
    },
    # Arm B full-FT (B2) — parent full-FT values (dose-matched at selection, so the
    # B1-Betley / B2-parent recipe asymmetry is absorbed by the matched-install rule)
    "B2.lr": {"value": 5e-6, "source": "parent plan §11 / #606/#642 full-FT recipe"},
    "B2.warmup_ratio": {"value": 0.05, "source": "parent plan §11 / #606/#642"},
    "B2.eff_batch": {
        "value": 16,
        "source": "per_device 4 x grad_accum 1 x 4 GPUs (parent FT contract)",
    },
    "B2.schedule": {"value": "cosine", "source": "parent plan §11 / #606/#642"},
    # Arm B install DV + grid
    "armB.install_floor_gain": {"value": INSTALL_FLOOR_GAIN, "source": "#653 install floor"},
    "armB.eval_bank": {
        "value": "broad_em 20-question wang44 bank",
        "source": "BEHAVIORS['broad_em'].eval_question_bank",
    },
    "armB.grid": {
        "value": "log-spaced steps derived from corpus size (Betley EM emerges <2 epochs)",
        "source": "arXiv 2502.17424 (Betley EM); exact steps ungrounded — needs smoke",
    },
}


@dataclass(frozen=True)
class RankemCell:
    """One rankem trained cell (arm A/B, LoRA or full-FT)."""

    name: str
    arm: str  # "A" | "B"
    behavior: str  # SYCO_BEHAVIOR | EM_BEHAVIOR
    method: str  # "lora" | "fullft"
    mix: str  # "c3_frozen" (Arm A) | "insecure_code" (Arm B)
    # LoRA shape (None for full-FT cells)
    lora_r: int | None = None
    lora_alpha: int | None = None
    use_rslora: bool | None = None

    def __post_init__(self) -> None:
        if self.arm not in ("A", "B"):
            raise ValueError(f"arm {self.arm!r} not in (A, B)")
        if self.method not in ("lora", "fullft"):
            raise ValueError(f"method {self.method!r} not in (lora, fullft)")
        if self.method == "lora" and (
            self.lora_r is None or self.lora_alpha is None or self.use_rslora is None
        ):
            raise ValueError(f"lora cell {self.name!r} needs lora_r/lora_alpha/use_rslora")
        if self.method == "fullft" and self.lora_r is not None:
            raise ValueError(f"fullft cell {self.name!r} must not carry a LoRA shape")


# Cell registry. Cell keys are collision-free with the parent's s*/m* cells so
# analysis + capture stores never alias across rounds.
A1 = "a1_lora_r1"
A2 = "a2_lora_r4"
B1 = "b1_lora_em"
B2 = "b2_fullft_em"

ARM_A_CELLS = (A1, A2)
ARM_B_CELLS = (B1, B2)

CELLS: dict[str, RankemCell] = {
    A1: RankemCell(A1, "A", SYCO_BEHAVIOR, "lora", "c3_frozen", 1, 2, False),
    A2: RankemCell(A2, "A", SYCO_BEHAVIOR, "lora", "c3_frozen", 4, 8, False),
    B1: RankemCell(B1, "B", EM_BEHAVIOR, "lora", "insecure_code", 32, 64, True),
    B2: RankemCell(B2, "B", EM_BEHAVIOR, "fullft", "insecure_code"),
}
ALL_CELLS = tuple(CELLS)

# Cross-method cosine pairs (plan): each rankem LoRA cell vs its full-FT
# comparator. Arm A reuses the parent's s3_fullft_neg full-FT capture tensors as
# the comparator; Arm B compares B1 (LoRA) vs B2 (full-FT), both rankem cells.
PARENT_FT_COMPARATOR = "s3_fullft_neg"
COSINE_PAIRS: tuple[tuple[str, str], ...] = (
    (A1, PARENT_FT_COMPARATOR),
    (A2, PARENT_FT_COMPARATOR),
    (B1, B2),
)


def cell_run_name(cell: str) -> str:
    """WandB run name per rankem cell (one run per cell)."""
    return f"issue1112_rankem_{cell}_seed{SEED}"


def arm_a_lora_config(cell: str, *, max_steps: int, seed: int = SEED) -> object:
    """Build the TrainLoraConfig for an Arm A low-rank non-rsLoRA sycophancy cell.

    Reuses the parent sycophancy LoRA recipe verbatim (UNIFIED_OVERRIDES: lr
    1e-5, dropout 0.05, eff-batch 16, 7 proj modules) and overrides ONLY the
    three manipulated shape knobs (lora_r / lora_alpha / use_rslora) + the
    parent's epochs->ceiling seam (max_length 2048, save cadence 2, the ladder
    step cap). The single scientific variable vs s1_lora_neg is the LoRA shape.
    """
    c = CELLS[cell]
    if c.arm != "A" or c.method != "lora":
        raise ValueError(f"arm_a_lora_config called on non-Arm-A-lora cell {cell!r}")
    spec = recipe_for(SYCO_BEHAVIOR, arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "epochs": 16,  # generous ceiling; max_steps caps the ladder
            "max_length": SYCO_MAX_LENGTH,
            "lora_r": c.lora_r,
            "lora_alpha": c.lora_alpha,
            "use_rslora": c.use_rslora,
        },
    )
    train_cfg = build_train_config(spec, run_name=cell_run_name(cell), seed=seed)
    return dataclasses.replace(
        train_cfg, save_steps=ARMA_SAVE_STEPS, max_steps=max_steps, max_length=SYCO_MAX_LENGTH
    )


# The Betley EM-induction recipe (configs/training/betley_open_model.yaml,
# project-validated in #404) + the configs/lora/default.yaml adapter. Overrides
# recipe_for(broad_em)'s UNIFIED_OVERRIDES so B1 REPLICATES Betley rather than
# the house content recipe: linear scheduler, warmup_steps 5, wd 0.01,
# per_device 2 x grad_accum 8 = eff 16, dropout 0.0, completion-only (=Betley
# train_on_responses_only), max_length 2048 (audit-gated — raise to 4096 in
# BOTH B1+B2 if the token-budget audit shows >~2-3% of rows lose their
# completion). lr 1e-5, r32/alpha64/rsLoRA from the cell shape.
BETLEY_B1_OVERRIDES: dict[str, object] = {
    "lr": 1e-5,
    "lora_dropout": 0.0,
    "batch_size": 2,
    "grad_accum": 8,
    "max_length": SYCO_MAX_LENGTH,  # 2048 (audit-gated)
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "completion_only_loss": True,
    "epochs": 16,  # ceiling; max_steps caps the ladder
}


def arm_b_lora_config(cell: str, *, max_steps: int, seed: int = SEED) -> object:
    """Build the TrainLoraConfig for the B1 misalignment LoRA cell.

    REPLICATES the Betley EM-induction recipe (betley_open_model.yaml +
    lora/default.yaml adapter): linear scheduler, warmup_steps 5, wd 0.01,
    per_device 2 x grad_accum 8, dropout 0.0, completion-only loss, max_length
    2048, r32/alpha64/rsLoRA, lr 1e-5 — training on the fixed insecure-code
    corpus. Routes through recipe_for(broad_em) for the spec skeleton, then
    replaces overrides with the Betley set (the behavior's own neg_ratio /
    generic_frac are irrelevant — the dispatcher feeds the pinned corpus jsonl).
    """
    c = CELLS[cell]
    if c.arm != "B" or c.method != "lora":
        raise ValueError(f"arm_b_lora_config called on non-Arm-B-lora cell {cell!r}")
    spec = recipe_for(EM_BEHAVIOR, arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            **BETLEY_B1_OVERRIDES,
            "lora_r": c.lora_r,
            "lora_alpha": c.lora_alpha,
            "use_rslora": c.use_rslora,
        },
    )
    train_cfg = build_train_config(spec, run_name=cell_run_name(cell), seed=seed)
    # M4: B1 checkpoints ONLY on the log-spaced install grid (via a
    # CheckpointAtStepsCallback attached in the dispatcher), NOT every
    # ARMA_SAVE_STEPS. save_strategy="no"/save_steps=0 disables Trainer's own
    # cadence so the callback's control.should_save is the sole save trigger —
    # otherwise B1 (broad_em, marker_band_stop a no-op) would save ~375 r32
    # checkpoints at max_steps=750, each judged by the ladder, blowing the
    # GPU-h + adapter-disk budget. A1/A2 keep steps/ARMA_SAVE_STEPS (band-stop).
    return dataclasses.replace(train_cfg, save_strategy="no", save_steps=0, max_steps=max_steps)


def derive_checkpoint_grid(n_rows: int, eff_batch: int = 16, max_epochs: float = 2.0) -> list[int]:
    """Log-spaced checkpoint step grid for the Arm B install ladder.

    Grounded on: Betley EM emerges within ~1-2 epochs of insecure-code SFT
    (arXiv 2502.17424). Grid = a coarse log-spaced sweep from a few steps up to
    ceil(max_epochs * steps_per_epoch), so the judged-rate ladder can find the
    matched-install rung without over-training past emergence. The exact step
    values are DERIVED from the realized corpus size at prep time (marked
    'ungrounded — needs smoke' in HYPERPARAMS until a smoke confirms emergence
    lands inside the grid).
    """
    if n_rows <= 0 or eff_batch <= 0:
        raise ValueError(f"n_rows={n_rows} eff_batch={eff_batch} must be positive")
    steps_per_epoch = -(-n_rows // eff_batch)  # ceil
    cap = max(1, round(max_epochs * steps_per_epoch))
    # Coarse log-spaced anchors, clamped to [2, cap], deduped, sorted.
    anchors = [2, 5, 10, 20, 40, 75, 125, 200, 300, 450, 650]
    grid = sorted({s for s in anchors if 2 <= s <= cap} | {cap})
    return grid
