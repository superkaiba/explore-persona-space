"""Issue #734 -- shared backbone for the marker slot-rooting re-read + H1 model pair.

Single source of truth for the #734 cell sets, the reused-#664 adapter resolution,
the HF / WandB destinations, and the reproducibility metadata. The four #734 entry
points (phase0 / marker_reread / dispatch / figures) import from here so the slot
read, the cell grid, and the marker token are identical by construction.

This is NOT a library module under ``src/``: it lives next to the
``scripts/issue734_*`` entry points (same convention as ``issue664_common.py``).
It REUSES #664's recipe + marker constants verbatim (``issue664_common``) so H1's
fresh trains are byte-identical to #664's marker recipe with model the only
deliberate variable (plan §4 / §11). The single new object of study is the
slot-rooting read code in ``issue734_marker_reread`` -- never a recipe change.

Design notes carried into the implementation report:

- **Marker recipe** = ``issue664_common.recipe_for("marker")`` UNCHANGED (lr 5e-6,
  rsLoRA r32/a64, q/k/v/o, marker+end-of-turn loss, band-stop [5,12] d1 /
  [10,16] d2). H1 swaps ONLY the base model id.
- **Marker token** ` ※` id 83399 (``issue664_common.MARKER_ID``); ``<|im_end|>`` id
  151645 (``issue664_common.IM_END_ID``) is the EOS competitor at the slot. Asserted
  at every entrypoint via ``C664.assert_marker_token``.
- **Phase-1 reuse set** = the 16 #664 d1-seed42 cells across ALL FOUR sources
  (default/librarian/surgeon/programmer) x {contra, posonly} x {d1, d2}. These are
  the same ``Cell`` objects #664 built; their adapters live at
  ``adapters/issue_664/mk_<source>_<arm>_<dose>_seed42`` (reuse-check (c)/(e)).
- **H1 set** = the librarian-contra-d1 marker cell fresh-trained on base
  Qwen-2.5-7B AND Qwen-2.5-7B-Instruct, seeds {42, 137, 256}.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("issue734_common")

REPO = Path(__file__).resolve().parents[1]

# ── Reuse #664's recipe + marker constants verbatim (the H1 single-variable
#    contract: model is the ONLY deliberate change vs #664's marker recipe). ────
import issue664_common as C664  # noqa: E402

MARKER_ID = C664.MARKER_ID
MARKER_TEXT = C664.MARKER_TEXT
IM_END_ID = C664.IM_END_ID

# ── Model ids (plan §10) ──────────────────────────────────────────────────────
INSTRUCT_ID = "Qwen/Qwen2.5-7B-Instruct"  # the #664 base; the reused adapters' base
BASE_ID = "Qwen/Qwen2.5-7B"  # the H1 second model (FIRST base-marker probe; §8 smoke gate)

# ── HF / WandB destinations (plan §10) ────────────────────────────────────────
HF_MODEL_REPO = C664.HF_MODEL_REPO  # superkaiba1/explore-persona-space
HF_DATA_REPO = C664.HF_DATA_REPO  # superkaiba1/explore-persona-space-data
# The reused #664 adapters live under this prefix on the model repo.
HF_REUSED_ADAPTER_PREFIX = C664.HF_ADAPTER_PREFIX  # adapters/issue_664
# H1 fresh-train adapters land here.
HF_H1_ADAPTER_PREFIX = "adapters/issue_734"
# Raw completions + analysis artifacts on the data repo.
HF_RAW_COMPLETIONS_PREFIX = "issue734_marker_slot_reread/raw_completions"
WANDB_PROJECT = "issue734"

# ── Local roots (plan §6.5 deliverable globs) ─────────────────────────────────
EVAL_ROOT = REPO / "eval_results/issue_734"
FIG_ROOT = REPO / "figures/issue_734"
DATA_ROOT = REPO / "data/issue_734"
ADAPTER_OUT = DATA_ROOT / "adapters"  # H1 fresh-train local adapter dirs
REUSED_ADAPTER_CACHE = DATA_ROOT / "reused_664_adapters"  # downloaded #664 adapters

# Phase-1 reread output (the headline DV).
CORRECTED_REREAD_ROOT = EVAL_ROOT / "corrected_reread"
# H1 in-loop band-stop + trajectory output.
H1_BAND_STOP_ROOT = EVAL_ROOT / "h1_band_stop"
H1_TRAJECTORY_ROOT = EVAL_ROOT / "h1_trajectory"

MAX_NEW_TOKENS = C664.MAX_NEW_TOKENS  # 2048 (>= 2x longest trained completion)

# Band targets (plan §10/§11; mirrors recipe_for("marker") doses).
BAND_D1 = (5.0, 12.0)
BAND_D2 = (10.0, 16.0)


def band_for_dose(dose: str) -> tuple[float, float]:
    return BAND_D1 if dose == "d1" else BAND_D2


def vllm_enforce_eager() -> bool:
    """Resolve the H1-phase vLLM ``enforce_eager`` kwarg (crash-fix round 5, #734).

    Default TRUE: #734's Phase-2 H1 ``generate()`` deadlocked at vLLM's
    cuda-graph capture on the pod-734 driver/GPU combo (the documented #664-class
    front-end<->EngineCore handoff hang -- GPU 0%%, EngineCore in ``futex_wait_queue``,
    NO ``[vllm-chunk]`` line ever printed). ``enforce_eager=True`` SKIPS cuda-graph
    capture, which is the documented fix for that deadlock class
    (.claude/rules/gotchas.md vLLM-hang triad probe (b)). The env knob lets a
    future operator flip it back per-pod if a different driver/GPU combo wants
    graphs (``EPM_VLLM_ENFORCE_EAGER=0``)."""
    return os.environ.get("EPM_VLLM_ENFORCE_EAGER", "1") in {"1", "true", "True"}


# ── Phase-1 reused-#664 cell set (plan §4 / §5.1 / reuse-check (c)) ───────────
# All FOUR d1-seed42 marker sources x {contra, posonly} x {d1, d2} = 16 cells.
PHASE1_SOURCES = ("default", "librarian", "surgeon", "programmer")
PHASE1_ARMS = ("contra", "posonly")
PHASE1_DOSES = ("d1", "d2")
PHASE1_SEED = C664.DEFAULT_SEED  # 42 (the reused #664 seed-42 cells)

# The d1 source-gate cells (the H3 quantitative success is read over these four).
PHASE1_D1_SOURCES = PHASE1_SOURCES

# ── H1 fresh-train set (plan §4 / §5.1) ───────────────────────────────────────
H1_SOURCE = "librarian"  # the H1-comparison source (librarian, contrastive, d1)
H1_ARM = "contra"
H1_DOSE = "d1"
H1_SEEDS = (42, 137, 256)  # Source: #650 (3-seed marker precedent on Instruct)
# (model_key, model_id): base is the FIRST base-marker probe (§8 smoke gate).
H1_MODELS = (("base", BASE_ID), ("instruct", INSTRUCT_ID))

# ── Phase-1.5 parity probe (plan §7.5 / reuse-check (g)) ──────────────────────
# The 1-adapter apply-and-read parity probe: the corrected reader must reproduce
# #664's in-loop band-stop value for mk_librarian_contra_d1_seed42 within ~2 nat.
PARITY_PROBE_CELL = ("librarian", "contra", "d1")  # (source, arm, dose), seed 42
PARITY_TOLERANCE_NAT = 2.0  # plan §7.5 "within ~2 nat"


@dataclass(frozen=True)
class Phase1Cell:
    """One reused-#664 marker cell to re-read (source x arm x dose, seed 42)."""

    source: str
    arm: str  # contra | posonly
    dose: str  # d1 | d2
    seed: int = PHASE1_SEED

    def to_664_cell(self) -> C664.Cell:
        """The corresponding #664 Cell (its adapter is the reused artifact)."""
        return C664.Cell(
            behavior="marker", source=self.source, arm=self.arm, dose=self.dose, seed=self.seed
        )

    @property
    def slug(self) -> str:
        return self.to_664_cell().slug  # mk_<source>_<arm>_<dose>

    @property
    def eval_key(self) -> str:
        return self.to_664_cell().eval_key  # mk_<source>_<arm>_<dose>_seed<seed>

    @property
    def hf_adapter_subfolder(self) -> str:
        """The reused #664 adapter subfolder on the model repo (reuse (c)/(e))."""
        return self.to_664_cell().hf_adapter_subfolder  # adapters/issue_664/mk_..._seed42


def phase1_cells() -> list[Phase1Cell]:
    """The 16 reused-#664 cells (plan §4 Phase 1): 4 sources x 2 arms x 2 doses."""
    cells: list[Phase1Cell] = []
    for src in PHASE1_SOURCES:
        for arm in PHASE1_ARMS:
            for dose in PHASE1_DOSES:
                cells.append(Phase1Cell(source=src, arm=arm, dose=dose))
    return cells


@dataclass(frozen=True)
class H1Cell:
    """One H1 fresh-train cell (model x seed; librarian-contra-d1 marker)."""

    model_key: str  # base | instruct
    model_id: str
    seed: int
    source: str = H1_SOURCE
    arm: str = H1_ARM
    dose: str = H1_DOSE

    @property
    def eval_key(self) -> str:
        return f"h1_{self.model_key}_seed{self.seed}"

    @property
    def run_name(self) -> str:
        return f"issue734_{self.eval_key}"

    @property
    def hf_adapter_subfolder(self) -> str:
        return f"{HF_H1_ADAPTER_PREFIX}/{self.eval_key}"

    def to_664_cell(self) -> C664.Cell:
        """The #664 Cell whose training MIX H1 reuses (model swapped).

        The mix-data seed is PINNED to the seed-42 #664 baseline (``PHASE1_SEED``)
        regardless of ``self.seed`` -- the marker training mix is
        content-deterministic (question set x persona panel x contrastive
        negatives), so the "seed" suffix in #664's mix-filename key reproduces
        ordering, NOT data content, and #664 only ever MATERIALIZED the seed-42
        marker grid (``mk_librarian_contra_d1_seed42.jsonl``; the seed-137/256
        mixes do not exist on disk -- there is no #734-side prebuild). The
        deliberate H1 variable is the MODEL-INIT seed (``self.seed``, threaded
        into ``train_lora`` via ``recipe.train_kwargs(seed=self.seed)``), NOT the
        mix data -- so all three H1 seeds train on the SAME mix and differ ONLY
        in model init (the single-variable-change H1 contract). Reusing
        ``self.seed`` here instead would (a) crash on the missing 137/256 mixes
        AND (b) smuggle in a second deliberate variable (per-seed mix data).
        Self-documenting assert at the data-path call site (``train_h1_cell``).
        """
        return C664.Cell(
            behavior="marker", source=self.source, arm=self.arm, dose=self.dose, seed=PHASE1_SEED
        )


def h1_cells() -> list[H1Cell]:
    """The 6 H1 cells (plan §4 Phase 2): 2 models x 3 seeds."""
    cells: list[H1Cell] = []
    for model_key, model_id in H1_MODELS:
        for seed in H1_SEEDS:
            cells.append(H1Cell(model_key=model_key, model_id=model_id, seed=seed))
    return cells


# ── Phase-3 (CONDITIONAL) H2 lr x steps mini-sweep (plan §4 / §7.5) ───────────
# Only run if Phase 1 FALSIFIES H3 (corrected read still < 5 nat on >=3/4 d1 sources).
# lr in {5e-6, 1e-5, 2e-5} x step-budget {1x, 4x #664 budget}; librarian-contra Instruct.
PHASE3_LRS = (5e-6, 1e-5, 2e-5)
PHASE3_STEP_MULTS = (1, 4)  # 1x = #664 band-stop budget; 4x = the predicted clean point
PHASE3_SOURCE = "librarian"
PHASE3_ARM = "contra"


# ── Invariants ────────────────────────────────────────────────────────────────
def assert_marker_token(tokenizer) -> None:
    """FAIL LOUD on marker drift -- reuse #664's wired assert (#530/#537)."""
    C664.assert_marker_token(tokenizer)


def require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


# ── Reproducibility metadata (CLAUDE.md reproducibility-metadata rule) ────────
def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ},  # explicit (no creds needed; subprocess-env contract)
    ).stdout.strip()


def repro_meta(*, seed: int | None = None) -> dict:
    return {
        "git_commit": git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": seed,
        "instruct_model": INSTRUCT_ID,
        "base_model": BASE_ID,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "im_end_id": IM_END_ID,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_messages(source: str, question: str) -> list[dict[str, str]]:
    """Chat messages for the SOURCE training context (reuse #664's battery)."""
    return C664.source_messages(source, question)


def marker_question_pool(*, smoke: bool) -> list[str]:
    """The marker probe questions (the source-persona questions Phase 1 regenerates
    on-policy R over). Reuses #664's canonical marker battery resolver so the
    question distribution is identical to what #664 trained/evaluated on."""
    probes = C664.canonical_battery_for_behavior("marker", smoke=smoke)
    return [p["question"] for p in probes]
