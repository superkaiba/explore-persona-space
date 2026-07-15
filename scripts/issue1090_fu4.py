#!/usr/bin/env python
"""#1090 ``fu4-extended-dose-lr`` follow-up driver (plan v6, round 4).

Question (followup-scope v5): does training PAST the fu2 ceiling — more dose
(epochs 3->15) AND higher lr ({1e-5, 3e-5, 1e-4}) — get the two genuinely
broken organisms (formatting = never installs; impolite = installs without
expressing) to install/express on-policy?

Design (plan §4 D1): 9 LoRA retrains from FROZEN sha-verified mixes — 3 cells
(formatting/persona-con [parent c1 mix], impolite/persona-con [parent c2 mix],
impolite/WildChat-con [fu3 C2-conv-con FLAT-layout mix]) x 3 lr rungs — each
to 15 epochs (75 optimizer steps), per-epoch rungs (save_steps=5). NO datagen.

Per run (fu2 pattern, parametrized; every phase checkpoints + resumes):
stage/verify frozen mix (K1) -> train from scratch (K2 divergence check over
the per-step loss history) -> Tier-1 ladder over ALL rungs at the run's OWN
training context (impolite: graded judge @ max_tokens=300; formatting: the
deterministic structural predicate — ``make_source_rate_fn`` routes both) +
per-rung degeneracy flags (flag-only) -> K3 retrain-parity gate
(fmt-pers-lr1e5 only) -> dose-select vs JUDGED_RATE_BAND -> Tier-2 generation
(trained arm only; the BASE arm is REUSED from fu3's committed cell evals,
A4/A15) -> tf-margin sweep at the selected rung (own ctx + per-question eval
contexts, fu1 shape; impolite pools = the fu3 instrument verbatim; formatting
pools = NEW, equalized-down, <15/15 ship-without-margin escape, A13) ->
per-run upload -> per-run sentinel.

Phases: ``--phase stage`` (VM P0: verify mixes + provenance + fu3 base reads
-> cell_manifest_fu4.json), ``--phase dispatch`` (pod P1+P2: work-conserving
multi-GPU queue, CVD pinned per slot), ``--phase run`` (one run; launched by
dispatch), ``--phase judge-aggregate`` (VM P3, post pod release: Tier-2
impolite judging via the Batch-API judge path, transport/content drop split,
K4 check, verdict-lattice inputs -> fu4_ladders.json).

``--smoke`` is the SAME dispatch path with tiny knobs (PASS_UNIFIED): 1 run
subset, fixture mix in the production on-disk shape with a PINNED sha,
``max_steps=5`` (exactly ONE rung at save_steps=5), tiny tier1/tier2 knobs,
tiny-real trainer (real tokenizer/collator/SFTTrainer/PEFT on the from-config
tiny Qwen2), recording upload seam; the judge stays LIVE.

``[phase=done]`` is emitted by ``scripts/issue1090_fu4_dispatch.sh`` ONLY.

fu5 round (``--round fu5``, plan v7 ``fu5-finish-impolite-bare-and-formatting-rank``):
the SAME driver re-parametrized — 5 runs (Arm A: impolite on the BARE context,
fu3 ``C2-bare-con`` frozen mix, lr {1e-5, 3e-5, 1e-4} at r32/alpha-64; Arm B:
formatting rank ladder r{128, 256} at lr 1e-4 / alpha-64 FIXED under the hardcoded
rsLoRA gamma=alpha/sqrt(r), on the parent c1 mix), plus the REUSED fu4 ``fmt-pers-lr1e4``
run as the r32 rank rung (copied field-for-field into the fu5 aggregate, never
retrained). Every phase reads the round registry (``ROUND``): label, HF
prefixes, deliverables dir, manifest/ladders filenames, K3 parity anchor
(``imp-bare-lr1e5`` rung-15; abort > 0.5, ``parity-degraded`` flag on
(0.35, 0.5] — report-and-flag), the WIDENED conditional formatting judged
re-read trigger (Tier-2/rung structural rate >= 0.30; was install delta >=
+0.30 — closes the mid-band [0.30, 0.467) construct gap), the 256-slot vLLM
engine width (``max_lora_rank`` threaded to the SOURCE-parametrized engine
factory, organisms.py), and the fu5-only eval-split diagnostic (list-affordable
vs prose-natural, Sonnet 3-draw majority). ``--phase k5-rank-smoke`` (pod-side,
before the fan-out) re-reads the reused fu4 r32 selected rung under the
256-slot engine and HALTs below the midpoint of the committed trained-vs-base
band (K5 seam gate).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu1 as fu1  # noqa: E402
import issue1090_fu2 as fu2  # noqa: E402
import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_run as i1090  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    _STRUCTURAL_PREDICATES,
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_margin_read_fn,
    _default_vllm_generate_fn,
    _generate_and_persist,
    make_source_rate_fn,
    release_trainer_cuda_memory,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    RecipeSpec,
    build_train_config,
    recipe_for,
    select_dose_checkpoint,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.train.sft import train_lora  # noqa: E402

logger = logging.getLogger("issue1090.fu4")

# ── Constants (plan §0/§11 — every deviation declared there) ─────────────────

FU4_LABEL = "fu4-extended-dose-lr"
FU4_EPOCHS = 15  # scope v5: dose past the fu2 ceiling (75 steps at 80-row/eff-16)
FU4_SAVE_STEPS = 5  # per-epoch rungs {5,10,...,75} (§4 divergence table)
FU4_LRS = (1e-5, 3e-5, 1e-4)  # §11: control / geometric midpoint / QLoRA regime
LR_TAG = {1e-5: "lr1e5", 3e-5: "lr3e5", 1e-4: "lr1e4"}
JUDGE_MAX_TOKENS_FU4 = fu3w.JUDGE_MAX_TOKENS  # 300 (llm-judging rule 23)
FU4_DATA_PREFIX = f"{i1090.DATA_PREFIX}/{FU4_LABEL}"
FU4_ADAPTER_PREFIX = "adapters/issue1090_fu4"  # §10 (hub #1108 overflow fallback)
FU4_SMOKE_MAX_STEPS = FU4_SAVE_STEPS  # exactly ONE rung (checkpoint-5) in smoke
EXPECTED_MIX_COMPOSITION = (20, 20, 40)  # K1: 20 pos + 20 neg + 40 generic
# K2 (registered): loss NaN, or > 5.0 sustained >= 5 logged steps -> `diverged`
# (a reportable answer — "lr exceeds the stable regime" — never a round abort).
K2_LOSS_THRESHOLD = 5.0
K2_SUSTAIN_STEPS = 5
# K3 (registered round-abort): the fmt-pers-lr1e5 arm's structural rate at the
# parent step count (3 epochs = step 15) must not exceed 0.5 — parent/fu3
# measured 0.13-0.18 at the same step count, so > 0.5 means harness mismatch.
K3_PARITY_STEP = 15
K3_PARITY_MAX_RATE = 0.5
# Degenerate-output guard (plan D2; FLAG-only, never gating).
DEGEN_MIN_MEAN_TOKENS = 30
DEGEN_MAX_REPEAT_FRAC = 0.5
# A13: formatting margin pools equalize down; below this floor per side the
# round SHIPS WITHOUT the formatting margin (flagged), never a silent n/a.
FMT_MARGIN_POOL_FLOOR = 15
# A15: impolite base rates are ~0.000 in the reused fu3 reads (band entry =
# pure install); a larger base read means the reuse premise broke.
IMPOLITE_BASE_RATE_MAX = 0.05
FU3_EVALS_DIR = _SCRIPTS_DIR.parent / "eval_results" / "issue_1090" / "fu3" / "fu3_cell_evals"
FU3_SUMMARY_PATH = _SCRIPTS_DIR.parent / "eval_results" / "issue_1090" / "fu3" / "fu3_summary.json"
DELIVERABLES_DIR = _SCRIPTS_DIR.parent / "eval_results" / "issue_1090" / FU4_LABEL
_MIX_FILES_REQUIRED = ("train_mix.jsonl", "mix_meta.json")
# Committed question-bank files (item (j) provenance-coherence inputs).
_BANK_FILES = {
    "impolite": "src/explore_persona_space/artifacts/query_banks/impolite_neutral_v1.json",
    "formatting": "src/explore_persona_space/artifacts/query_banks/wildchat_random_v1.json",
}


# ── Run matrix (plan §4 D1 — 9 runs) ─────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Fu4Run:
    """One (cell x lr x rank) training run. ``slug``/``behavior`` duck-type the
    fu2 ``verify_staged_mix`` + ``make_smoke_seams`` cell contracts.

    ``lora_r``/``lora_alpha`` (fu5 D2 item 1) thread the rank ladder through
    the ``fu4_recipe_spec`` overrides seam (both are UNIFIED_OVERRIDES keys —
    recipe.py:114-115 — so ``extra_overrides`` would refuse them); alpha is HELD
    FIXED at 64 across ranks per the rsLoRA gamma=alpha/sqrt(r) prescription (plan §4
    alpha-with-rank policy, arXiv 2312.03732)."""

    run_id: str
    cell_key: str  # fmt-pers | imp-pers | imp-conv | imp-bare
    behavior: str
    context_id: str
    lr: float
    mix_hub_prefix: str  # data-repo prefix holding train_mix.jsonl + mix_meta.json
    mix_layout: str  # "parent-mix-subdir" | "fu3-flat" (D3 leg (h)(ii))
    fu3_base_eval: str  # fu3_cell_evals filename for the reused base Tier-2 arm
    lora_r: int = 32  # UNIFIED_OVERRIDES default (recipe.py:114)
    lora_alpha: int = 64  # UNIFIED_OVERRIDES default (recipe.py:115); FIXED across ranks
    round_name: str = "fu4"  # wandb run-name prefix component (fu5: issue1090_fu5_*)

    @property
    def slug(self) -> str:
        return self.run_id

    @property
    def run_name(self) -> str:
        return f"issue1090_{self.round_name}_{self.run_id}_seed42"


_CELL_DEFS: dict[str, dict[str, str]] = {
    "fmt-pers": {
        "behavior": "formatting",
        "context_id": i1090.SOURCE_CONTEXT_ID,
        "mix_hub_prefix": f"{i1090.DATA_PREFIX}/c1-formatting-claude/mix",
        "mix_layout": "parent-mix-subdir",
        "fu3_base_eval": "C1-pers-con-formatting-claude.json",
    },
    "imp-pers": {
        "behavior": "impolite",
        "context_id": i1090.SOURCE_CONTEXT_ID,
        "mix_hub_prefix": f"{i1090.DATA_PREFIX}/c2-impolite-claude/mix",
        "mix_layout": "parent-mix-subdir",
        "fu3_base_eval": "C2-pers-con-impolite-claude.json",
    },
    "imp-conv": {
        "behavior": "impolite",
        "context_id": fu3_cells.CONV_CONTEXT_ID,
        # fu3 uploaded its per-cell artifacts FLAT (no mix/ subdir) — the
        # layout is mapped EXPLICITLY per cell here (D3 leg (h)(ii)).
        "mix_hub_prefix": f"{fu3w.DATA_PREFIX_FU3}/C2-conv-con-impolite-claude",
        "mix_layout": "fu3-flat",
        "fu3_base_eval": "C2-conv-con-impolite-claude.json",
    },
}

FU4_RUNS: tuple[Fu4Run, ...] = tuple(
    Fu4Run(run_id=f"{cell_key}-{LR_TAG[lr]}", cell_key=cell_key, lr=lr, **defs)
    for cell_key, defs in _CELL_DEFS.items()
    for lr in FU4_LRS
)
RUN_BY_ID = {r.run_id: r for r in FU4_RUNS}  # fu4 registry (rejudge/back-compat consumers)


# ── fu5 round: `finish-impolite-bare-and-formatting-rank` (plan v7 §4 D1) ────

FU5_LABEL = "fu5-finish-impolite-bare-and-formatting-rank"
FU5_DELIVERABLES_DIR = (
    _SCRIPTS_DIR.parent / "eval_results" / "issue_1090" / "finish-impolite-bare-and-formatting-rank"
)
FU5_DATA_PREFIX = f"{i1090.DATA_PREFIX}/{FU5_LABEL}"
FU5_ADAPTER_PREFIX = "adapters/issue1090_fu5"
FU5_RANKS = (128, 256)  # Arm B ladder rungs; r32 = the REUSED fu4 fmt-pers-lr1e4 run
FU5_FMT_LR = 1e-4  # formatting's fu4-strongest arm (fu4_ladders.json fmt-pers-lr1e4)
# fu5 K3 (plan §4 retrain-parity): imp-bare-lr1e5 rung-15 (fu3's step-15 dose
# point; fu3 measured 0.000). Abort stays > 0.5 (K3_PARITY_MAX_RATE); a value in
# (0.35, 0.5] is FLAGGED `parity-degraded` in the aggregate (report-and-flag —
# orchestrator decision folding the 0.35 head-room convention into the record).
FU5_PARITY_DEGRADED_FLOOR = 0.35
# WIDENED conditional formatting judged re-read trigger (orchestrator decision):
# Tier-2/rung structural rate >= 0.30 (was install delta >= +0.30 over base
# 0.167 == rate >= 0.467) — closes the mid-band [0.30, 0.467) construct gap.
FU5_REREAD_RATE_FLOOR = 0.30
FU5_MAX_LORA_RANK = 256  # vLLM engine slot width for the rank ladder (plan §11)

_FU5_BARE_DEFS: dict[str, str] = {
    "behavior": "impolite",
    "context_id": fu3_cells.BARE,  # "default" — the exercised fu3 bare-context path (A5)
    # fu3 uploaded its per-cell artifacts FLAT (no mix/ subdir) — D3 leg (h)(ii).
    "mix_hub_prefix": f"{fu3w.DATA_PREFIX_FU3}/C2-bare-con-impolite-claude",
    "mix_layout": "fu3-flat",
    "fu3_base_eval": "C2-bare-con-impolite-claude.json",
}

FU5_RUNS: tuple[Fu4Run, ...] = tuple(
    [
        Fu4Run(
            run_id=f"imp-bare-{LR_TAG[lr]}",
            cell_key="imp-bare",
            lr=lr,
            round_name="fu5",
            **_FU5_BARE_DEFS,
        )
        for lr in FU4_LRS
    ]
    + [
        Fu4Run(
            run_id=f"fmt-pers-r{rank}",
            cell_key="fmt-pers",
            lr=FU5_FMT_LR,
            lora_r=rank,
            lora_alpha=64,  # FIXED under rsLoRA gamma=alpha/sqrt(r) (plan §4 alpha policy)
            round_name="fu5",
            **_CELL_DEFS["fmt-pers"],
        )
        for rank in FU5_RANKS
    ]
)


@dataclasses.dataclass(frozen=True)
class ReusedRun:
    """A prior round's run folded into this round's aggregate WITHOUT retraining
    (plan D3: the fu4 ``fmt-pers-lr1e4`` run is the r32 rank rung)."""

    entry_id: str  # this round's aggregate key (e.g. "reused_fu4_r32")
    source_run_id: str  # the producing round's run_id
    ladders_path: str  # committed producing-round aggregate (field-for-field copy source)
    tier2_hf_prefix: str  # producing round's tier-2 raw-completions prefix (eval-split diag)
    context_id: str
    lora_r: int
    lora_alpha: int
    adapter_subfolder: str  # model-repo subfolder of the reused SELECTED rung (K5 re-read)
    adapter_revision: str  # model-repo revision pin (plan §10)


@dataclasses.dataclass(frozen=True)
class RoundSpec:
    """Everything the ``--round`` selector threads through the shared driver."""

    name: str  # "fu4" | "fu5" — filename/sentinel/out-root component
    label: str  # followup label (manifest/aggregate `round` field, regime keys)
    data_prefix: str  # HF data-repo prefix
    adapter_prefix: str  # HF model-repo prefix
    deliverables_dir: Path
    manifest_name: str
    ladders_name: str
    runs: tuple[Fu4Run, ...]
    smoke_default_run: str
    k3_parity_run_id: str
    k3_parity_degraded_floor: float | None  # aggregate `parity-degraded` flag floor (fu5 0.35)
    reread_rate_floor: float | None  # None -> legacy delta>=+0.30 trigger (fu4)
    max_lora_rank: int  # vLLM engine LoRA slot width (organisms engine factory)
    eval_split_diagnostic: bool  # fu5 D2 item 6 (list-affordable vs prose-natural)
    reused_runs: tuple[ReusedRun, ...]


ROUNDS: dict[str, RoundSpec] = {
    "fu4": RoundSpec(
        name="fu4",
        label=FU4_LABEL,
        data_prefix=FU4_DATA_PREFIX,
        adapter_prefix=FU4_ADAPTER_PREFIX,
        deliverables_dir=DELIVERABLES_DIR,
        manifest_name="cell_manifest_fu4.json",
        ladders_name="fu4_ladders.json",
        runs=FU4_RUNS,
        smoke_default_run="fmt-pers-lr1e5",
        k3_parity_run_id="fmt-pers-lr1e5",
        k3_parity_degraded_floor=None,
        reread_rate_floor=None,
        max_lora_rank=64,
        eval_split_diagnostic=False,
        reused_runs=(),
    ),
    "fu5": RoundSpec(
        name="fu5",
        label=FU5_LABEL,
        data_prefix=FU5_DATA_PREFIX,
        adapter_prefix=FU5_ADAPTER_PREFIX,
        deliverables_dir=FU5_DELIVERABLES_DIR,
        manifest_name="cell_manifest_fu5.json",
        ladders_name="fu5_ladders.json",
        runs=FU5_RUNS,
        smoke_default_run="fmt-pers-r256",  # K5: the rank-threading + 256-slot smoke run
        k3_parity_run_id="imp-bare-lr1e5",
        k3_parity_degraded_floor=FU5_PARITY_DEGRADED_FLOOR,
        reread_rate_floor=FU5_REREAD_RATE_FLOOR,
        max_lora_rank=FU5_MAX_LORA_RANK,
        eval_split_diagnostic=True,
        reused_runs=(
            ReusedRun(
                entry_id="reused_fu4_r32",
                source_run_id="fmt-pers-lr1e4",
                ladders_path=str(DELIVERABLES_DIR / "fu4_ladders.json"),
                tier2_hf_prefix=f"{FU4_DATA_PREFIX}/raw_completions/tier2/fmt-pers-lr1e4",
                context_id=i1090.SOURCE_CONTEXT_ID,
                lora_r=32,
                lora_alpha=64,
                # Selected rung (fu4_ladders.json runs.fmt-pers-lr1e4.selection.step
                # = 20) at the plan-§10 model-repo revision pin.
                adapter_subfolder=f"{FU4_ADAPTER_PREFIX}/fmt-pers-lr1e4/checkpoint-20",
                adapter_revision="48de22ca",
            ),
        ),
    ),
}

# Module-global round, selected ONCE by `--round` in main() (workers receive it
# via the dispatcher's --round passthrough). Default fu4 keeps every existing
# invocation + the rejudge/back-compat consumers byte-identical.
ROUND: RoundSpec = ROUNDS["fu4"]


def set_round(name: str) -> RoundSpec:
    """Select the active round registry (fu4 default; fu5 = plan v7)."""
    global ROUND
    if name not in ROUNDS:
        raise ValueError(f"unknown round {name!r}: known {sorted(ROUNDS)}")
    ROUND = ROUNDS[name]
    return ROUND


def _run_by_id() -> dict[str, Fu4Run]:
    return {r.run_id: r for r in ROUND.runs}


def resolve_fu4_runs(runs_arg: str | None, smoke: bool) -> tuple[Fu4Run, ...]:
    """The ONE run resolver every phase consumes (smoke = same path, 1 run)."""
    by_id = _run_by_id()
    if runs_arg:
        ids = [t.strip() for t in runs_arg.split(",") if t.strip()]
        bad = [t for t in ids if t not in by_id]
        if bad:
            raise ValueError(f"bad {ROUND.name} runs {bad!r}: known {sorted(by_id)}")
        return tuple(by_id[i] for i in ids)
    if smoke:
        return (by_id[ROUND.smoke_default_run],)
    return ROUND.runs


# ── Recipe (lr + epochs + save_steps at the authoritative spec seam) ─────────


def fu4_recipe_spec(
    behavior: str, lr: float, *, lora_r: int = 32, lora_alpha: int = 64
) -> RecipeSpec:
    """The unified content recipe with the declared deviations (lr arm, epochs
    3->15, save_steps->5, the parent's carried max_length=2048, and — fu5 Arm B
    — the LoRA rank rung at FIXED alpha), all threaded at the ``spec.overrides``
    seam (the fu2 pattern — lr/save_steps/lora_r/lora_alpha are
    LOAD_BEARING_KEYS, so ``extra_overrides`` would refuse them). The defaults
    (r32/alpha-64) equal UNIFIED_OVERRIDES (recipe.py:114-115), so fu4 callers stay
    byte-identical."""
    spec = recipe_for(behavior, arm="primary")
    return dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "lr": lr,
            "epochs": FU4_EPOCHS,
            "save_steps": FU4_SAVE_STEPS,
            "max_length": i1090.MAX_LENGTH_1090,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        },
    )


def fu4_expected_rungs(n_mix_rows: int) -> tuple[list[int], int]:
    """(expected checkpoint steps, total optimizer steps): 80 rows at eff.
    batch 16 -> 5 steps/epoch x 15 epochs = 75; rungs {5,10,...,75}."""
    ov = fu4_recipe_spec("impolite", FU4_LRS[0]).overrides
    steps_per_epoch = math.ceil(n_mix_rows / (int(ov["batch_size"]) * int(ov["grad_accum"])))
    total = steps_per_epoch * FU4_EPOCHS
    rungs = sorted(set(range(FU4_SAVE_STEPS, total + 1, FU4_SAVE_STEPS)) | {total})
    return rungs, total


# ── Frozen-mix staging + verification (K1) ───────────────────────────────────


def _run_root(cfg: i1090.RunConfig, run: Fu4Run) -> Path:
    return cfg.out_root / run.run_id


def _build_result_path(run_root: Path) -> Path:
    """Round-named build record (fu5 primary-deliverable glob:
    ``**/fu5_build_result.json`` — plan §6.5)."""
    return run_root / f"{ROUND.name}_build_result.json"


def _ladder_file_path(run_root: Path) -> Path:
    return run_root / f"{ROUND.name}_ladder.json"


def stage_fu4_mix(cfg: i1090.RunConfig, run: Fu4Run) -> Path:
    """Per-FILE staging of the run's frozen mix (train_mix.jsonl + mix_meta
    .json) into ``<run>/mix/``. Deliberately file-targeted — the fu3 FLAT cell
    prefix also holds datagen/rate/tier2 trees a prefix mirror would drag in."""
    d = _run_root(cfg, run) / "mix"
    if all((d / f).exists() for f in _MIX_FILES_REQUIRED):
        return d
    from huggingface_hub import hf_hub_download

    d.mkdir(parents=True, exist_ok=True)
    for fname in _MIX_FILES_REQUIRED:
        hub_path = f"{run.mix_hub_prefix}/{fname}"
        got = hub.retry_transient(
            lambda hp=hub_path: hf_hub_download(
                i1090.HF_DATA_REPO, hp, repo_type="dataset", local_dir=d / "_hfstage"
            ),
            what=f"fu4 stage {hub_path}",
        )
        target = d / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, target)
    logger.info("[fu4-stage] %s -> %s (%s layout)", run.mix_hub_prefix, d, run.mix_layout)
    return d


def verify_fu4_mix(cfg: i1090.RunConfig, run: Fu4Run, manifest_sha: str | None) -> dict:
    """K1 gate: fu2's ``verify_staged_mix`` (files present / behavior identity /
    row count vs counts_realized / pinned-sha branch), PLUS the fu4 composition
    check (20/20/40) and the cross-phase manifest sha pin (D3 item (f)).
    Amends the record's provenance fields (fu2 hardcodes the parent prefix)."""
    mix_dir = _run_root(cfg, run) / "mix"
    rec = fu2.verify_staged_mix(mix_dir, run)  # duck-typed on .slug/.behavior
    rec["hf_prefix"] = run.mix_hub_prefix
    rec["mix_layout"] = run.mix_layout
    if not cfg.smoke:
        counts = tuple(sorted(int(v) for v in rec["counts_realized"].values()))
        if counts != tuple(sorted(EXPECTED_MIX_COMPOSITION)):
            raise ValueError(
                f"[fu4-K1] {run.run_id}: mix composition {rec['counts_realized']} != "
                f"expected {EXPECTED_MIX_COMPOSITION} — refusing to train"
            )
    if manifest_sha is not None and rec["train_mix_sha256"] != manifest_sha:
        raise ValueError(
            f"[fu4-K1] {run.run_id}: staged train_mix sha {rec['train_mix_sha256']} != "
            f"manifest pin {manifest_sha} — the frozen mix drifted between stage and run"
        )
    i1090._atomic_write_json(mix_dir / "mix_verification.json", rec)
    return rec


def build_fu4_smoke_mix(cfg: i1090.RunConfig, run: Fu4Run) -> None:
    """Tiny fixture mix in the production on-disk shape with a PINNED sha
    (fu2's fixture builder, duck-typed) — the K1 pinned-sha branch runs in
    smoke through the same verify path."""
    fu2.build_smoke_mix_fixture(cfg, run)  # uses .slug/.behavior only


# ── Train (K2 divergence check over the per-step loss history) ───────────────


def check_divergence(ckpts: dict[int, Path]) -> dict:
    """K2: read the FINAL rung's trainer_state.json log_history (logging_steps=1
    is threaded via extra_overrides) and flag NaN loss, or loss sustained above
    ``max(K2_LOSS_THRESHOLD, first-logged loss)`` for >= K2_SUSTAIN_STEPS
    consecutive logged steps.

    The first-logged-loss floor operationalizes "divergence" as DEGRADATION
    past the registered 5.0 bar, not an elevated initial condition: a real 7B
    SFT run starts at loss ~1-3, so the effective bar there is EXACTLY the
    registered 5.0; the tiny-real smoke model starts at ~ln(vocab) ~= 11.9 and
    would otherwise flag every smoke run diverged (observed on the first
    unified smoke), never exercising the ladder path.
    """
    state_path = ckpts[max(ckpts)] / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"[fu4-K2] {state_path} missing — the divergence check has no loss history"
        )
    hist = json.loads(state_path.read_text()).get("log_history", [])
    losses = [(e.get("step"), float(e["loss"])) for e in hist if "loss" in e]
    out: dict[str, Any] = {"checked": True, "n_loss_points": len(losses), "diverged": False}
    nan_steps = [s for s, ls in losses if math.isnan(ls)]
    if nan_steps:
        out.update({"diverged": True, "reason": "nan_loss", "steps": nan_steps[:10]})
        return out
    if not losses:
        return out
    bar = max(K2_LOSS_THRESHOLD, losses[0][1])
    out["effective_bar"] = bar
    run_len = 0
    for s, ls in losses:
        run_len = run_len + 1 if ls > bar else 0
        if run_len >= K2_SUSTAIN_STEPS:
            out.update(
                {
                    "diverged": True,
                    "reason": f"loss>{bar:.3f} sustained {K2_SUSTAIN_STEPS} steps",
                    "at_step": s,
                }
            )
            return out
    return out


def _assert_adapter_rank(run: Fu4Run, ckpt_dir: Path) -> dict:
    """K5 rank-threading gate (fu5 D2 item 4, runs in smoke AND full): the
    trained adapter's OWN ``adapter_config.json`` must carry the run's
    r/alpha + the ``train_lora``-hardcoded rsLoRA flag (artifact-reuse check (a):
    adapter-architecture values grounded on the artifact's own config)."""
    cfg_path = ckpt_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"[{ROUND.name}-K5] {run.run_id}: {cfg_path} missing")
    acfg = json.loads(cfg_path.read_text())
    got = {
        "r": acfg.get("r"),
        "lora_alpha": acfg.get("lora_alpha"),
        "use_rslora": acfg.get("use_rslora"),
    }
    if (
        int(got["r"] or -1) != run.lora_r
        or int(got["lora_alpha"] or -1) != run.lora_alpha
        or got["use_rslora"] is not True
    ):
        raise ValueError(
            f"[{ROUND.name}-K5] {run.run_id}: adapter_config {got} != run spec "
            f"(r={run.lora_r}, alpha={run.lora_alpha}, use_rslora=True) — the rank "
            "threading through the recipe seam is broken; never eval a wrong-rank adapter"
        )
    return got


def train_fu4_run(cfg: i1090.RunConfig, seams: i1090.Seams1090, run: Fu4Run, mix_rec: dict) -> dict:
    """Train the run FROM SCRATCH on its frozen mix at (lr, rank, epochs=15);
    resume on the round's build record (checkpoint-per-phase). A K2-diverged
    run is recorded ``status: diverged`` — a reportable answer, not an abort."""
    run_root = _run_root(cfg, run)
    build_path = _build_result_path(run_root)
    if build_path.exists():
        logger.info("[%s-train] %s already trained — skip", ROUND.name, run.run_id)
        return i1090._read_json(build_path)
    i1090._phase(f"{ROUND.name}_train")
    spec = fu4_recipe_spec(run.behavior, run.lr, lora_r=run.lora_r, lora_alpha=run.lora_alpha)
    # logging_steps=1 is NON-load-bearing telemetry (K2 needs per-step losses).
    train_cfg = build_train_config(
        spec, run_name=run.run_name, seed=cfg.seed, extra_overrides={"logging_steps": 1}
    )
    if seams.train_clamp is not None:
        train_cfg = dataclasses.replace(seams.train_clamp(train_cfg), max_steps=FU4_SMOKE_MAX_STEPS)
    adapter_dir, loss = train_lora(
        DEFAULT_BASE_MODEL,
        str(run_root / "mix" / "train_mix.jsonl"),
        str(run_root / "train"),
        cfg=train_cfg,
    )
    release_trainer_cuda_memory()
    ckpts = fu2.enumerate_ckpt_rungs(adapter_dir)
    realized = sorted(ckpts)
    expected_rungs, expected_total = fu4_expected_rungs(int(mix_rec["n_rows"]))
    if not cfg.smoke and max(realized) < expected_total:
        raise ValueError(
            f"[fu4-train] {run.run_id}: ladder incomplete — realized rungs {realized} "
            f"never reach step {expected_total}"
        )
    divergence = check_divergence(ckpts)
    adapter_gauge = _assert_adapter_rank(run, ckpts[max(ckpts)])
    record = {
        "status": "diverged" if divergence["diverged"] else "trained",
        "adapter_root": str(adapter_dir),
        "training_loss": float(loss),
        "rungs": realized,
        "expected_rungs": expected_rungs,
        "expected_total_steps": expected_total,
        "run_name": run.run_name,
        "mix": mix_rec,
        "divergence_check": divergence,
        "lr": run.lr,
        "lora_r": run.lora_r,
        "lora_alpha": run.lora_alpha,
        "adapter_gauge": adapter_gauge,
        "epochs_deviation": FU4_EPOCHS,
        "save_steps_deviation": FU4_SAVE_STEPS,
        "max_length_deviation": i1090.MAX_LENGTH_1090,
        "git_commit": i1074._git_short_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    i1090._atomic_write_json(build_path, record)
    return record


# ── Tier-1 ladder + degeneracy guard + K3 + dose selection ───────────────────


def degeneracy_stats(completions: list[list[str]]) -> dict:
    """Flag-only degenerate-output guard (plan D2): mean whitespace-token
    completion length + max per-completion 4-gram repetition fraction."""
    flat = [c for per_q in completions for c in per_q]
    lens: list[int] = []
    reps: list[float] = []
    for c in flat:
        toks = c.split()
        lens.append(len(toks))
        if len(toks) >= 4:
            grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
            reps.append(1.0 - len(set(grams)) / len(grams))
        else:
            reps.append(0.0)
    mean_len = (sum(lens) / len(lens)) if lens else 0.0
    max_rep = max(reps) if reps else 0.0
    return {
        "mean_completion_tokens": mean_len,
        "max_4gram_repeat_frac": max_rep,
        "degenerate": bool(mean_len < DEGEN_MIN_MEAN_TOKENS or max_rep > DEGEN_MAX_REPEAT_FRAC),
    }


def _ladder_regime(cfg: i1090.RunConfig, run: Fu4Run) -> dict:
    """Every output-affecting key of the ladder read (resume pin — #722 r3).
    fu5 adds lora_r/lora_alpha (the Arm B manipulated variable) + the vLLM
    engine slot width (an engine-config key of the generation read)."""
    return {
        "followup_label": ROUND.label,
        "lr": run.lr,
        "lora_r": run.lora_r,
        "lora_alpha": run.lora_alpha,
        "max_lora_rank": ROUND.max_lora_rank,
        "fu4_epochs": FU4_EPOCHS,
        "save_steps": FU4_SAVE_STEPS,
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
        "tier1": [cfg.tier1_n, cfg.tier1_draws],
        "eval_question_limit": cfg.eval_question_limit,
        "seed": cfg.seed,
        "band": list(JUDGED_RATE_BAND),
        "context_id": run.context_id,
        "smoke": cfg.smoke,
    }


def ladder_fu4_run(
    cfg: i1090.RunConfig, seams: i1090.Seams1090, run: Fu4Run, ckpts: dict[int, Path]
) -> dict:
    """Tier-1 primary rate at EVERY rung at the run's OWN training context
    (``make_source_rate_fn`` routes structural formatting / judged impolite),
    plus per-rung degeneracy flags. Per-rung atomic checkpoint to
    ``fu4_ladder.json`` + regime-keyed resume."""
    run_root = _run_root(cfg, run)
    ladder_path = _ladder_file_path(run_root)
    regime = _ladder_regime(cfg, run)
    rates: dict[int, float] = {}
    degen: dict[int, dict] = {}
    if ladder_path.exists():
        prior = i1090._read_json(ladder_path)
        if prior.get("regime") != regime:
            raise RuntimeError(
                f"fu4_ladder.json at {ladder_path} was produced under a DIFFERENT regime "
                f"(prior={prior.get('regime')}); refusing to mix — use a fresh --out-root"
            )
        rates = {int(k): float(v) for k, v in (prior.get("rates_by_step") or {}).items()}
        degen = {int(k): v for k, v in (prior.get("degeneracy_by_step") or {}).items()}

    ctx = fu3w.ensure_context(run.context_id, run.behavior)

    def _persist() -> None:
        i1090._atomic_write_json(
            ladder_path,
            {
                "run_id": run.run_id,
                "behavior": run.behavior,
                "context_id": run.context_id,
                "lr": run.lr,
                "regime": regime,
                "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
                "degeneracy_by_step": {str(k): v for k, v in sorted(degen.items())},
                "primary_mode": BEHAVIORS[run.behavior].dv.primary,
                "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
            },
        )

    pending = [s for s in sorted(ckpts) if s not in rates]
    if pending:
        i1090._phase(f"{ROUND.name}_tier1_ladder")
        organism = ModelOrganism(behavior=run.behavior, context_id=run.context_id, seed=cfg.seed)
        # The generation engine is ALWAYS injected here (never None) so the
        # round's LoRA slot width reaches the SOURCE-parametrized factory
        # (organisms.py D2 item 2): fu5's r128/r256 rungs need a 256-slot
        # engine; the factory default (64) would refuse/corrupt the hot-swap.
        gen = (
            seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
            if seams.eval_gen_fn_factory is not None
            else _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=ROUND.max_lora_rank)
        )
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=run_root / "rate",
            eval_questions=i1090._eval_questions(cfg, run.behavior),
            n_completions=cfg.tier1_n,
            temperature=1.0,
            n_judge_draws=cfg.tier1_draws,
            generate_fn=gen,  # caller-injected -> caller-closed (finally below)
            judge_fn=fu3w.judge_graded_r23,  # max_tokens=300 instrument
        )
        try:
            for step in pending:
                rates[step] = float(rate_fn(str(ckpts[step])))
                comp_path = (
                    run_root
                    / "rate"
                    / f"rate_{ckpts[step].name}"
                    / f"completions__trained__{ctx.context_id}.json"
                )
                degen[step] = degeneracy_stats(json.loads(comp_path.read_text())["completions"])
                _persist()  # per-rung checkpoint (intra-phase grain)
        finally:
            rate_close = getattr(rate_fn, "close", None)
            if callable(rate_close):
                rate_close()  # no-op for a caller-injected gen (factory-owned only)
            gen_close = getattr(gen, "close", None)
            if callable(gen_close):
                gen_close()
    else:
        _persist()
    # K3 retrain-parity (registered round-abort; full runs only): the round's
    # parity anchor at the parent step count must reproduce the parent regime
    # (fu4: fmt-pers-lr1e5, parent measured 0.13-0.18; fu5: imp-bare-lr1e5,
    # fu3 measured 0.000 — plan v7 §4). The (0.35, 0.5] `parity-degraded` FLAG
    # is the aggregator's (report-and-flag); the abort stays > 0.5 here.
    if run.run_id == ROUND.k3_parity_run_id and not cfg.smoke:
        parity = rates.get(K3_PARITY_STEP)
        if parity is None:
            raise RuntimeError(
                f"[{ROUND.name}-K3] rung {K3_PARITY_STEP} missing from the "
                f"{run.run_id} ladder (rungs: {sorted(rates)}) — the parity anchor "
                "cannot be read"
            )
        if parity > K3_PARITY_MAX_RATE:
            raise RuntimeError(
                f"[{ROUND.name}-K3] retrain-parity FAILED: {run.run_id} Tier-1 rate at "
                f"step {K3_PARITY_STEP} = {parity:.3f} > {K3_PARITY_MAX_RATE} — "
                "harness/recipe mismatch; ROUND ABORT before any arm read"
            )
    return {"rates_by_step": rates, "degeneracy_by_step": degen}


# ── Tier-2 generation (trained arm; base REUSED from fu3) ────────────────────


def tier2_fu4_run(
    cfg: i1090.RunConfig,
    seams: i1090.Seams1090,
    run: Fu4Run,
    selected_ckpt: str,
) -> dict:
    """Tier-2 install-eval GENERATION at the selected rung, own context, n=10
    (trained arm ONLY — the base arm is the reused fu3 read, plan D3). The
    formatting structural rate is computed pod-side (deterministic, no judge);
    impolite judging runs VM-side post-pod (Batch API)."""
    i1090._phase(f"{ROUND.name}_tier2_generation")
    run_root = _run_root(cfg, run)
    ctx = fu3w.ensure_context(run.context_id, run.behavior)
    qs = i1090._eval_questions(cfg, run.behavior)
    out_dir = run_root / "tier2"
    gen = (
        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
        if seams.eval_gen_fn_factory is not None
        else _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=ROUND.max_lora_rank)
    )
    try:
        comps = _generate_and_persist(
            gen,
            "trained",
            selected_ckpt,
            ctx,
            qs,
            n=cfg.tier2_n,
            temperature=1.0,
            out_dir=out_dir,
            base_model=DEFAULT_BASE_MODEL,
        )
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    rec: dict[str, Any] = {
        "file": str(out_dir / f"completions__trained__{ctx.context_id}.json"),
        "n_questions": len(qs),
        "n_completions": cfg.tier2_n,
        "degeneracy": degeneracy_stats(comps),
    }
    if BEHAVIORS[run.behavior].dv.primary == "structural":
        rec["structural"] = i1090._judge_rate(
            run.behavior,
            qs,
            comps,
            tag=f"{run.run_id}-t2-trained",
            n_draws=cfg.tier2_draws,
            judge_root=run_root / "tier2" / "judge",
        )
    return rec


# ── tf-margin companion (fu1 sweep shape; pools per plan §6/A13) ─────────────


def fu4_margin_pools(
    cfg: i1090.RunConfig, behavior: str
) -> tuple[list[dict] | None, list[dict] | None, dict]:
    """FIXED behavior-level pools. Impolite: the fu3 instrument verbatim
    (23 pos / 25 neg). Formatting: NEW — parent c1 judge-kept datagen rows via
    the same staging helper, EQUALIZED DOWN to min(n_pos, n_neg); below
    FMT_MARGIN_POOL_FLOOR per side the round ships WITHOUT the formatting
    margin (A13 escape) — (None, None, meta) with the flagged reason."""
    pos, neg = fu3w._behavior_margin_pools(cfg, behavior)
    meta: dict[str, Any] = {
        "behavior": behavior,
        "pool_source": "/".join(fu3w.V4_POOL_SOURCE[behavior]),
        "n_pos_raw": len(pos),
        "n_neg_raw": len(neg),
    }
    if behavior == "formatting":
        m = min(len(pos), len(neg))
        if m < FMT_MARGIN_POOL_FLOOR:
            meta.update(
                {
                    "status": "skipped_pool_below_floor",
                    "floor": FMT_MARGIN_POOL_FLOOR,
                    "reason": (
                        f"formatting margin pools {len(pos)}/{len(neg)} below the "
                        f"{FMT_MARGIN_POOL_FLOOR}/{FMT_MARGIN_POOL_FLOOR} floor — shipping "
                        "without the formatting margin (plan §12 A13 escape)"
                    ),
                }
            )
            return None, None, meta
        pos, neg = pos[:m], neg[:m]
        meta["equalized_to"] = m
    meta["n_pos"] = len(pos)
    meta["n_neg"] = len(neg)
    meta["pool_sha256"] = fu1._sha256_json(
        [
            {k: p[k] for k in ("probe", "answer", "question_id", "variant_id", "request_id")}
            for p in pos + neg
        ]
    )
    return pos, neg, meta


def _fu4_margin_contexts(ctx: Any, questions: list[str]) -> list[tuple[str, Any]]:
    """fu1's margin-context construction parametrized by the run's OWN training
    context: source_ctx FIRST (adapter-application assert runs on it), then one
    context per eval question (the IDENTICAL fixed answer set scored under
    every context — llm-judging §E2 rule 19)."""
    ctxs: list[tuple[str, Any]] = [("source_ctx", ctx)]
    for i, q in enumerate(questions):
        ctxs.append(
            (
                f"q{i:03d}",
                fu1._MsgCtx(
                    f"{ctx.context_id}__q{i:03d}", lambda probe, _q=q, _c=ctx: _c.messages(_q)
                ),
            )
        )
    return ctxs


def _margin_sweep(margin_fn, side_path, ctxs, pos, neg, out_path: Path) -> dict:
    """One (side) margin sweep over every context: skip completed reads
    (resume), checkpoint per read."""
    reads: dict[str, dict] = {}
    if out_path.exists():
        reads = i1090._read_json(out_path)
    for label, c in ctxs:
        if label in reads:
            continue
        mr = margin_fn(side_path, c, pos, neg)
        reads[label] = dataclasses.asdict(mr)
        i1090._atomic_write_json(out_path, reads)
    return reads


def margin_fu4_run(
    cfg: i1090.RunConfig,
    seams: i1090.Seams1090,
    run: Fu4Run,
    selected_ckpt: str,
    selected_step: int,
) -> dict:
    """tf-margin at the selected rung (HF model AFTER every vLLM teardown):
    base sweep per (behavior, context) — computed once, reused by sibling lr
    arms via the shared file — then the trained sweep with the fu1
    adapter-application assert on source_ctx."""
    i1090._phase(f"{ROUND.name}_margin")
    run_root = _run_root(cfg, run)
    margin_root = cfg.out_root / f"{ROUND.name}_margin"
    margin_root.mkdir(parents=True, exist_ok=True)
    record_path = run_root / "margin.json"
    if record_path.exists():
        return i1090._read_json(record_path)
    pos, neg, meta = fu4_margin_pools(cfg, run.behavior)
    if pos is None:
        i1090._atomic_write_json(record_path, meta)
        return meta
    ctx = fu3w.ensure_context(run.context_id, run.behavior)
    qs = i1090._eval_questions(cfg, run.behavior)
    ctxs = _fu4_margin_contexts(ctx, qs)
    base_path = margin_root / f"base__{run.cell_key}.json"
    trained_path = margin_root / f"trained__{run.run_id}.json"
    margin_fn = (
        seams.margin_read_fn_factory(DEFAULT_BASE_MODEL)
        if seams.margin_read_fn_factory is not None
        else _default_margin_read_fn(DEFAULT_BASE_MODEL)
    )
    try:
        base_reads = _margin_sweep(margin_fn, None, ctxs, pos, neg, base_path)
        trained_reads = _margin_sweep(margin_fn, selected_ckpt, ctxs, pos, neg, trained_path)
    finally:
        close = getattr(margin_fn, "close", None)
        if callable(close):
            close()
    assert_rec = fu1.assert_adapter_applied(
        base_reads["source_ctx"],
        trained_reads["source_ctx"],
        tol=fu1.ADAPTER_ASSERT_TOL_SMOKE if cfg.smoke else fu1.ADAPTER_ASSERT_TOL_FULL,
        tag=f"fu4-{run.run_id}",
    )
    combined = {f"base__{k}": v for k, v in base_reads.items()}
    combined.update({f"trained__{k}": v for k, v in trained_reads.items()})
    record = {
        "status": "computed",
        **meta,
        "selected_step": selected_step,
        "adapter_assert": assert_rec,
        "base_reads_file": str(base_path),
        "trained_reads_file": str(trained_path),
        **fu1.aggregate_margin_reads(combined, fu1._q_labels(len(qs))),
    }
    i1090._atomic_write_json(record_path, record)
    return record


# ── Per-run upload (Upload Policy; declared rung discard AFTER upload) ───────


def upload_fu4_run(cfg: i1090.RunConfig, seams: i1090.Seams1090, run: Fu4Run, rec: dict) -> dict:
    """Everything text/JSON -> the data repo under the fu4 prefix; the selected
    + final adapter rungs -> the canonical model repo (#1108 file-count overflow
    fallback covers the known limit). Ruled-out rungs are deleted ONLY after the
    kept-rung uploads verified (declared discard, plan §10). Fail-loud on an
    empty upload return."""
    i1090._phase(f"{ROUND.name}_upload")
    run_root = _run_root(cfg, run)
    upload = i1090._upload_fn(seams)
    uploaded: dict[str, str] = {}

    def _up(local: Path, repo_id: str, repo_type: str, path_in_repo: str, **kw: Any) -> None:
        if not Path(local).exists():
            return
        url = upload(Path(local), repo_id, repo_type, path_in_repo, **kw)
        if not str(url):
            raise RuntimeError(
                f"upload returned no path for {repo_id}/{path_in_repo} — refusing silent loss"
            )
        uploaded[path_in_repo] = str(url)
        i1090._atomic_write_json(run_root / "upload_manifest.json", uploaded)

    base_pir = f"{ROUND.data_prefix}/{run.run_id}"
    for fname in (
        _build_result_path(run_root).name,
        _ladder_file_path(run_root).name,
        "margin.json",
    ):
        _up(
            run_root / fname,
            i1090.HF_DATA_REPO,
            "dataset",
            f"{base_pir}/{fname}",
            upload_as_file=True,
        )
    _up(
        run_root / "mix" / "mix_verification.json",
        i1090.HF_DATA_REPO,
        "dataset",
        f"{base_pir}/mix_verification.json",
        upload_as_file=True,
    )
    # Raw completions: Tier-1 rung reads + judge raws, Tier-2, margin reads.
    _up(
        run_root / "rate",
        i1090.HF_DATA_REPO,
        "dataset",
        f"{ROUND.data_prefix}/raw_completions/rate/{run.run_id}",
    )
    _up(
        run_root / "tier2",
        i1090.HF_DATA_REPO,
        "dataset",
        f"{ROUND.data_prefix}/raw_completions/tier2/{run.run_id}",
    )
    for mp in ("base__" + run.cell_key, "trained__" + run.run_id):
        _up(
            cfg.out_root / f"{ROUND.name}_margin" / f"{mp}.json",
            i1090.HF_DATA_REPO,
            "dataset",
            f"{ROUND.data_prefix}/margin/{mp}.json",
            upload_as_file=True,
        )
    # Adapters: selected + final rungs only (plan §9 retention); the ruled-out
    # rungs are the plan-§10 declared discard, deleted AFTER upload verifies.
    if rec.get("status") == "trained" and rec.get("selected_ckpt"):
        ckpts = fu2.enumerate_ckpt_rungs(rec["adapter_root"])
        keep_steps = {int(rec["selection"]["step"]), max(ckpts)}
        for step in sorted(keep_steps):
            _up(
                ckpts[step],
                i1090.HF_MODEL_REPO,
                "model",
                f"{ROUND.adapter_prefix}/{run.run_id}/checkpoint-{step}",
            )
        for step, path in sorted(ckpts.items()):
            if step not in keep_steps:
                shutil.rmtree(path)
        logger.info(
            "[%s-upload] %s: kept rungs %s uploaded; %d ruled-out rungs deleted "
            "(declared discard, deterministic retrain regen recipe)",
            ROUND.name,
            run.run_id,
            sorted(keep_steps),
            len(ckpts) - len(keep_steps),
        )
    return uploaded


# ── Sentinels (pod-side-reporting.md; fu4-scoped names) ──────────────────────


def fu4_sentinel_path(sentinel_dir: Path, run_id: str) -> Path:
    return sentinel_dir / f"issue-{i1090.ISSUE}-{ROUND.name}run-{run_id}.json"


def fu4_status_path(out_root: Path, run_id: str) -> Path:
    """Per-run terminal status OUTSIDE the poller's drain glob. The
    ``/workspace/logs`` per-run sentinels match ``poll_pipeline.py``'s
    ``issue-<N>-*.json`` drain and get renamed ``<path>.processed`` after
    posting, so resume / completion / finalize state lives under
    ``out_root/<run_id>/status.json`` instead (drain-immune)."""
    return out_root / run_id / "status.json"


def write_fu4_run_sentinel(
    sentinel_dir: Path, run_id: str, payload: dict, *, out_root: Path
) -> Path:
    """Per-run progress sentinel (poller-facing visibility) + its out-of-glob
    ``status.json`` twin (the dispatcher's resume + finalize source — the
    poller renames drained sentinels ``*.processed``, so the sentinel path
    alone races the drain; code-review v16 Major)."""
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "sentinel_schema_version": fu3w.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:progress",
        "version": 1,
        "task_id": i1090.ISSUE,
        "by": f"issue1090_{ROUND.name}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
    }
    path = fu4_sentinel_path(sentinel_dir, run_id)
    i1090._atomic_write_json(path, doc)
    i1090._atomic_write_json(fu4_status_path(out_root, run_id), doc)
    return path


def _read_fu4_run_payload(out_root: Path, sentinel_dir: Path, run_id: str) -> dict | None:
    """Terminal-status payload for resume / completion / finalize. PRIMARY:
    the out-of-glob ``status.json``. FALLBACK: the per-run sentinel, tolerating
    the poller's ``<path>.processed`` drain-rename (covers state written by a
    pre-status.json run)."""
    sp = fu4_sentinel_path(sentinel_dir, run_id)
    for p in (fu4_status_path(out_root, run_id), sp, sp.with_name(sp.name + ".processed")):
        if p.exists():
            return json.loads(p.read_text()).get("payload", {})
    return None


def read_fu4_run_status(out_root: Path, sentinel_dir: Path, run_id: str) -> str | None:
    payload = _read_fu4_run_payload(out_root, sentinel_dir, run_id)
    return payload.get("status") if payload is not None else None


# ── Per-run worker (--phase run) ─────────────────────────────────────────────


def _load_manifest(path: str | None) -> dict:
    if path is None:
        return {}
    return i1090._read_json(Path(path))


def _manifest_run_entry(manifest: dict, run_id: str) -> dict:
    return {r["run_id"]: r for r in manifest.get("runs", [])}.get(run_id, {})


def cmd_run(cfg: i1090.RunConfig, seams: i1090.Seams1090, args: argparse.Namespace) -> int:
    """ONE fu4 run end-to-end: stage/verify (K1) -> train (K2) -> ladder (K3)
    -> dose-select -> Tier-2 gen -> margin -> upload -> sentinel."""
    run = _run_by_id()[args.run]
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    # CVD contract (gotchas.md): the LAUNCHER pins CUDA_VISIBLE_DEVICES; a
    # missing pin is refused unless explicitly unpinned (CPU smokes).
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and not args.allow_unpinned_gpu:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES not set — launch via `--phase dispatch` (which pins "
            "CVD per slot) or pass --allow-unpinned-gpu for a CPU smoke"
        )
    result: dict[str, Any] = {"run_id": run.run_id, "status": "running", "lr": run.lr}
    try:
        manifest = _load_manifest(args.manifest)
        pin = _manifest_run_entry(manifest, run.run_id).get("train_mix_sha256")
        if not cfg.smoke and not pin:
            # Silent-default class (code-review v16 Minor 1): a manifest that
            # lacks this run's entry/sha would silently skip the K1 sha pin.
            raise ValueError(
                f"[fu4-K1] no train_mix_sha256 pin for {run.run_id} "
                f"(--manifest {args.manifest}) — a full run refuses to train "
                "unpinned: run `--phase stage` and pass its manifest"
            )
        i1090._phase(f"{ROUND.name}_stage_inputs")
        fu3w.ensure_context(run.context_id, run.behavior)
        if cfg.smoke:
            build_fu4_smoke_mix(cfg, run)
        else:
            stage_fu4_mix(cfg, run)
        i1090._phase(f"{ROUND.name}_verify_mix")
        mix_rec = verify_fu4_mix(cfg, run, pin)
        rec = train_fu4_run(cfg, seams, run, mix_rec)
        if rec["status"] == "diverged":
            # K2: a reportable ANSWER for this arm — persist + upload the build
            # record; no ladder/tier2/margin on NaN-poisoned weights.
            result.update({"status": "diverged", "divergence": rec["divergence_check"]})
            if cfg.upload:
                upload_fu4_run(cfg, seams, run, rec)
            write_fu4_run_sentinel(sentinel_dir, run.run_id, result, out_root=cfg.out_root)
            logger.warning("[fu4] %s DIVERGED (K2) — recorded as the arm's answer", run.run_id)
            return 0
        ckpts = fu2.enumerate_ckpt_rungs(rec["adapter_root"])
        ladder = ladder_fu4_run(cfg, seams, run, ckpts)
        sel = select_dose_checkpoint(ladder["rates_by_step"], band=JUDGED_RATE_BAND)
        rec = {
            **rec,
            "rates_by_step": {str(k): v for k, v in sorted(ladder["rates_by_step"].items())},
            "degeneracy_by_step": {
                str(k): v for k, v in sorted(ladder["degeneracy_by_step"].items())
            },
            "selection": dataclasses.asdict(sel),
            "band": list(JUDGED_RATE_BAND),
            "selected_ckpt": str(ckpts[sel.step]),
        }
        i1090._atomic_write_json(_build_result_path(_run_root(cfg, run)), rec)
        logger.info(
            "[fu4] %s dose selection: step=%d rate=%.3f in_band=%s fallback=%s",
            run.run_id,
            sel.step,
            sel.rate,
            sel.in_band,
            sel.fallback,
        )
        rec["tier2"] = tier2_fu4_run(cfg, seams, run, rec["selected_ckpt"])
        rec["margin"] = margin_fu4_run(cfg, seams, run, rec["selected_ckpt"], sel.step)
        i1090._atomic_write_json(_build_result_path(_run_root(cfg, run)), rec)
        if cfg.upload:
            upload_fu4_run(cfg, seams, run, rec)
        result.update(
            {
                "status": "done",
                "selection": rec["selection"],
                "run_name": run.run_name,
                "adapter_hub_prefix": f"{ROUND.adapter_prefix}/{run.run_id}",
                "mix_sha256": mix_rec["train_mix_sha256"],
            }
        )
    except Exception as e:  # fail LOUD, but always leave a sentinel
        logger.exception("[fu4] run %s FAILED", run.run_id)
        result["status"] = "failed"
        result["reason"] = f"{type(e).__name__}: {e}"
        # A K3 parity failure is a DETERMINISTIC registered gate — a retry
        # re-runs the same ladder to the same verdict (code-review v16
        # Minor 5: exempt it from the dispatcher's single requeue).
        if "-K3]" in str(e):  # round-parametrized K3 tag ([fu4-K3] / [fu5-K3])
            result["no_requeue"] = True
        write_fu4_run_sentinel(sentinel_dir, run.run_id, result, out_root=cfg.out_root)
        return 2
    write_fu4_run_sentinel(sentinel_dir, run.run_id, result, out_root=cfg.out_root)
    logger.info("[fu4] run %s complete (status=%s)", run.run_id, result["status"])
    return 0


# ── K5 rank-seam smoke (--phase k5-rank-smoke; pod-side, before the fan-out) ─


def _k5_band(ladders: dict, reused: ReusedRun) -> tuple[float, float, dict]:
    """(halt_floor, warn_ceiling, source record) for the K5 r32 re-read gate,
    derived at RUNTIME from the committed producing-round aggregate (never a
    hand-typed constant — artifact-reuse § gate calibration: committed
    same-surface references, file+field named).

    Calibration (fu4_ladders.json, runs.fmt-pers-lr1e4): recorded trained rate
    at the selected rung = rates_by_step[str(selection.step)] (0.2933 at step
    20, n=150); base rate = base_tier2.rate (0.1667). The HALT floor is the
    MIDPOINT of the two — the discriminating placement between the correct
    band (~recorded +/- 2sd, sd~=0.037 at n=150) and the adapter-not-applied band
    (~base +/- 2sd) — so a 256-slot engine that silently fails to apply the r32
    adapter reads ~base and HALTs, while a correct re-read passes with ~2 sd
    margin. The ceiling (recorded + 0.10 ~= +2.7 sd) is a WARN flag only: no
    over-application mode is expected from slot widening (rsLoRA scaling is
    engine-slot-independent), so a high read flags instrument drift for the
    analyzer rather than halting (artifact-reuse rule 2: weaker-only
    diagnostics default WARN; the not-applied catch is the structural HALT)."""
    src = (ladders.get("runs") or {}).get(reused.source_run_id)
    if src is None:
        raise ValueError(f"[{ROUND.name}-K5] no runs[{reused.source_run_id!r}] in ladders")
    step = int(src["selection"]["step"])
    recorded = float(src["rates_by_step"][str(step)])
    base = float(src["base_tier2"]["rate"])
    if not recorded > base:
        raise ValueError(
            f"[{ROUND.name}-K5] recorded rate {recorded} <= base {base} — no discriminating "
            "band exists; the K5 gate cannot HALT (re-derive before the fan-out)"
        )
    halt_floor = (base + recorded) / 2.0
    warn_ceiling = recorded + 0.10
    return (
        halt_floor,
        warn_ceiling,
        {"step": step, "recorded_rate": recorded, "base_rate": base},
    )


def _stage_reused_adapter(reused: ReusedRun, dest: Path) -> Path:
    """Scoped model-repo staging of the reused rung (list_repo_tree at the
    pinned revision + per-file hf_hub_download — never snapshot_download on a
    near-100k-file repo; gotchas.md)."""
    from huggingface_hub import HfApi, hf_hub_download

    if (dest / "adapter_config.json").exists():
        return dest
    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                i1090.HF_MODEL_REPO,
                path_in_repo=reused.adapter_subfolder,
                repo_type="model",
                revision=reused.adapter_revision,
                recursive=True,
            )
        ),
        what=f"{ROUND.name} K5 adapter listing {reused.adapter_subfolder}",
    )
    files = [e.path for e in entries if not getattr(e, "tree_id", None)]
    if not files:
        raise FileNotFoundError(
            f"[{ROUND.name}-K5] no files under {i1090.HF_MODEL_REPO}/"
            f"{reused.adapter_subfolder}@{reused.adapter_revision}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        got = hub.retry_transient(
            lambda fp=f: hf_hub_download(
                i1090.HF_MODEL_REPO,
                fp,
                repo_type="model",
                revision=reused.adapter_revision,
                local_dir=dest / "_hfstage",
            ),
            what=f"{ROUND.name} K5 adapter file {f}",
        )
        target = dest / Path(f).name
        os.replace(got, target)
    return dest


def cmd_k5_rank_smoke(cfg: i1090.RunConfig, args: argparse.Namespace) -> int:
    """K5 seam gate (plan D2 item 4 + the orchestrator hardening): under the
    round's WIDE vLLM engine (max_lora_rank=256), re-read ONE Tier-1 slice of
    the REUSED fu4 r32 selected rung and assert the structural rate reproduces
    the committed fu4 record within the calibrated band — closing the
    64-slot-vs-256-slot engine seam on the rank ladder BEFORE the 5-run
    fan-out. GPU-only (vLLM); run pod-side with --full. A K5 failure means:
    fix the engine parametrization and re-smoke; NEVER fall back to silently
    evaluating at a lower rank (plan §7 K5)."""
    del args  # CLI surface shared with the other phases; nothing extra used
    if not ROUND.reused_runs:
        raise ValueError(
            f"--phase k5-rank-smoke needs a round with reused_runs (fu5); round={ROUND.name}"
        )
    i1090._phase(f"{ROUND.name}_k5_rank_smoke")
    reused = ROUND.reused_runs[0]
    ladders = i1090._read_json(Path(reused.ladders_path))
    halt_floor, warn_ceiling, src_rec = _k5_band(ladders, reused)
    adapter_dir = _stage_reused_adapter(reused, cfg.out_root / "k5_reused_adapter")
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    got = {
        "r": acfg.get("r"),
        "lora_alpha": acfg.get("lora_alpha"),
        "use_rslora": acfg.get("use_rslora"),
    }
    if (
        int(got["r"] or -1) != reused.lora_r
        or int(got["lora_alpha"] or -1) != reused.lora_alpha
        or got["use_rslora"] is not True
    ):
        raise ValueError(
            f"[{ROUND.name}-K5] reused adapter_config {got} != expected "
            f"(r={reused.lora_r}, alpha={reused.lora_alpha}, use_rslora=True) — wrong "
            "artifact staged"
        )
    organism = ModelOrganism(behavior="formatting", context_id=reused.context_id, seed=cfg.seed)
    gen = _default_vllm_generate_fn(DEFAULT_BASE_MODEL, max_lora_rank=ROUND.max_lora_rank)
    rate_fn = make_source_rate_fn(
        organism,
        out_dir=cfg.out_root / "k5_rate",
        eval_questions=i1090._eval_questions(cfg, "formatting"),
        n_completions=cfg.tier1_n,
        temperature=1.0,
        n_judge_draws=cfg.tier1_draws,
        generate_fn=gen,
        judge_fn=fu3w.judge_graded_r23,  # structural primary -> zero judge calls
    )
    try:
        reread_rate = float(rate_fn(str(adapter_dir)))
    finally:
        gen_close = getattr(gen, "close", None)
        if callable(gen_close):
            gen_close()
    record = {
        "round": ROUND.label,
        "reused": dataclasses.asdict(reused),
        "source": src_rec,
        "engine_max_lora_rank": ROUND.max_lora_rank,
        "adapter_config": got,
        "reread_rate": reread_rate,
        "halt_floor": halt_floor,
        "warn_ceiling": warn_ceiling,
        "high_side_flag": bool(reread_rate > warn_ceiling),
        "git_commit": i1074._git_short_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    i1090._atomic_write_json(cfg.out_root / "k5_rank_smoke.json", record)
    if reread_rate < halt_floor:
        raise RuntimeError(
            f"[{ROUND.name}-K5] r32 re-read under the {ROUND.max_lora_rank}-slot engine "
            f"reads {reread_rate:.3f} < halt floor {halt_floor:.3f} (recorded "
            f"{src_rec['recorded_rate']:.3f}, base {src_rec['base_rate']:.3f}) — the "
            "adapter-not-applied band; fix the engine parametrization and re-smoke "
            "(plan §7 K5); NEVER fall back to a lower rank"
        )
    if record["high_side_flag"]:
        logger.warning(
            "[%s-K5] r32 re-read %.3f > warn ceiling %.3f — instrument-drift flag "
            "persisted to k5_rank_smoke.json (WARN, not gating)",
            ROUND.name,
            reread_rate,
            warn_ceiling,
        )
    logger.info(
        "[%s-K5] PASS: r32 re-read %.3f within [%.3f, %.3f] under the %d-slot engine",
        ROUND.name,
        reread_rate,
        halt_floor,
        warn_ceiling,
        ROUND.max_lora_rank,
    )
    return 0


# ── Work-conserving dispatcher (--phase dispatch; fu3 §D7 shape) ─────────────


def _worker_cmd(args: argparse.Namespace, run: Fu4Run, slot: int) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--smoke" if args.smoke else "--full",
        "--round",
        args.round,  # workers re-select the round registry in their own process
        "--phase",
        "run",
        "--run",
        run.run_id,
        "--gpu-id",
        str(slot),
        "--out-root",
        str(args.out_root_resolved),
        "--sentinel-dir",
        str(args.sentinel_dir_resolved),
        "--seed",
        str(args.seed),
    ]
    if args.smoke:
        cmd.append("--allow-unpinned-gpu")  # CPU smoke: no CVD to pin
    if not args.upload:
        cmd.append("--no-upload")
    if args.manifest:
        cmd += ["--manifest", args.manifest]
    if args.eval_question_limit is not None:
        cmd += ["--eval-question-limit", str(args.eval_question_limit)]
    return cmd


def _dispatch_disposition(rc: int, payload: dict, attempt: int) -> str:
    """Completion routing for one finished worker: ``done`` on a clean
    terminal status, ``failed`` (no retry) on a deterministic-gate failure
    (``no_requeue`` — K3 parity re-runs the same ladder to the same verdict)
    or an exhausted retry budget, else ``requeue`` (single retry)."""
    if rc == 0 and payload.get("status") in ("done", "diverged"):
        return "done"
    if payload.get("no_requeue"):
        return "failed"
    return "requeue" if attempt <= 1 else "failed"


def finalize_dispatch(
    args: argparse.Namespace, done: list[str], failed: list[str], skipped: list[str]
) -> None:
    """manifest_complete.json + the end-of-gpu-phase sentinel (poll_pipeline
    required keys + reproducibility_card from the per-run status files —
    out-of-glob, so the poller's sentinel drain-rename cannot hollow the
    card the way fu3's did (23-24 of 35 cells))."""
    sentinel_dir = Path(args.sentinel_dir_resolved)
    out_root = Path(args.out_root_resolved)
    adapter_paths: dict[str, str] = {}
    run_names: list[str] = []
    per_run: dict[str, dict] = {}
    for run_id in done + skipped:
        payload = _read_fu4_run_payload(out_root, sentinel_dir, run_id)
        if payload is None:
            continue
        per_run[run_id] = {
            k: payload.get(k) for k in ("status", "selection", "adapter_hub_prefix", "run_name")
        }
        if payload.get("adapter_hub_prefix"):
            adapter_paths[run_id] = payload["adapter_hub_prefix"]
        if payload.get("run_name"):
            run_names.append(payload["run_name"])
    card = {
        "hf_model_repo": i1090.HF_MODEL_REPO,
        "hf_data_repo": i1090.HF_DATA_REPO,
        "adapter_paths": adapter_paths,
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue1090"),
        "wandb_run_names": run_names,
        "wandb_entity": fu2._wandb_entity(),
        "lr_arms": sorted({r.lr for r in ROUND.runs}),
        "lora_rank_arms": sorted({r.lora_r for r in ROUND.runs}),
        "epochs_deviation": FU4_EPOCHS,
        "save_steps_deviation": FU4_SAVE_STEPS,
        "max_length_deviation": i1090.MAX_LENGTH_1090,
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
        "band": list(JUDGED_RATE_BAND),
    }
    payload = {
        "issue": i1090.ISSUE,
        "round": ROUND.label,
        "runs_done": done,
        "runs_failed": failed,
        "runs_skipped_resume": skipped,
        "per_run": per_run,
        # Results-note disposition (code-review v16 Minor 6): K2-diverged runs
        # upload build/ladder JSONs only; their NaN-poisoned adapter rungs are
        # regenerable garbage-by-construction (pinned mix + seed + lr recipe)
        # and are NOT uploaded before instance teardown.
        "diverged_adapter_disposition": (
            "diverged runs persist build/ladder JSONs only; NaN-poisoned adapter "
            "rungs are regenerable garbage-by-construction and are not uploaded"
        ),
        "hf_data_prefix": ROUND.data_prefix,
        "reproducibility_card": card,
        "git_commit": i1074._git_short_sha(),
    }
    i1090._atomic_write_json(Path(args.out_root_resolved) / "manifest_complete.json", payload)
    kind = "epm:smoke-result" if args.smoke else "epm:results"
    sentinel = {
        "sentinel_schema_version": fu3w.SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,  # drain-side rewrite derives max+1 (#1095)
        "task_id": i1090.ISSUE,
        "gate": f"{ROUND.name}-dispatch",
        "blocks_pipeline": not args.smoke,
        "by": f"issue1090-{ROUND.name}-dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(args.smoke),
        "note": json.dumps(payload, ensure_ascii=False),
        "payload": payload,
    }
    kind_slug = kind.replace(":", "_")
    i1090._atomic_write_json(
        sentinel_dir / f"issue-{i1090.ISSUE}-{kind_slug}-{int(time.time())}.json", sentinel
    )
    logger.info(
        "[fu4] finalize: %d done / %d failed / %d resume-skipped",
        len(done),
        len(failed),
        len(skipped),
    )


def cmd_dispatch(args: argparse.Namespace) -> int:
    """Work-conserving queue: one run per GPU slot, CVD pinned per slot; a
    freed slot pulls the next pending run immediately; retry limit 1;
    resumable via the per-run sentinels. Smoke = SAME path, subsetted."""
    sentinel_dir = Path(args.sentinel_dir_resolved)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root_resolved)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    queue_rows = list(resolve_fu4_runs(args.runs, args.smoke))
    n_gpus = args.n_gpus if args.n_gpus is not None else fu3w.detect_n_gpus()
    if args.dry_run:
        print(json.dumps({"n_gpus": n_gpus, "queue": [r.run_id for r in queue_rows]}, indent=2))
        return 0
    i1090._phase(f"{ROUND.name}_dispatch")
    pending: deque[Fu4Run] = deque()
    skipped: list[str] = []
    for run in queue_rows:
        if read_fu4_run_status(out_root, sentinel_dir, run.run_id) in ("done", "diverged"):
            logger.info("[fu4] %s already terminal — resume-skip", run.run_id)
            skipped.append(run.run_id)
        else:
            pending.append(run)
    attempts: dict[str, int] = {}
    done: list[str] = []
    failed: list[str] = []
    live: dict[int, tuple[subprocess.Popen, Fu4Run, Any]] = {}
    slots = list(range(n_gpus))
    last_beat = 0.0
    i1090._phase(f"{ROUND.name}_queue_drain")
    while pending or live:
        for slot in [s for s in slots if s not in live]:
            if not pending:
                break
            run = pending.popleft()
            attempts[run.run_id] = attempts.get(run.run_id, 0) + 1
            log_path = out_root / "logs" / f"{run.run_id}.attempt{attempts[run.run_id]}.log"
            fh = log_path.open("w")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(slot),
                "VLLM_PORT": str(fu3w.BASE_VLLM_PORT + slot),
            }
            proc = subprocess.Popen(
                _worker_cmd(args, run, slot), stdout=fh, stderr=subprocess.STDOUT, env=env
            )
            live[slot] = (proc, run, fh)
            logger.info(
                "[fu4] launched %s on GPU %d (attempt %d, pid %d, log %s)",
                run.run_id,
                slot,
                attempts[run.run_id],
                proc.pid,
                log_path,
            )
        if not live and not pending:
            break
        time.sleep(args.poll_seconds)
        for slot, (proc, run, fh) in list(live.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del live[slot]
            payload = _read_fu4_run_payload(out_root, sentinel_dir, run.run_id) or {}
            disposition = _dispatch_disposition(rc, payload, attempts[run.run_id])
            if disposition == "done":
                done.append(run.run_id)
                logger.info(
                    "[fu4] run %s complete on GPU %d (%s)", run.run_id, slot, payload["status"]
                )
            elif disposition == "requeue":
                logger.warning("[fu4] run %s rc=%d — requeue (retry 1/1)", run.run_id, rc)
                pending.append(run)
            else:
                failed.append(run.run_id)
                logger.error(
                    "[fu4] run %s FAILED (rc=%d%s)",
                    run.run_id,
                    rc,
                    "; deterministic gate — no requeue" if payload.get("no_requeue") else "",
                )
        if time.time() - last_beat > 300:
            last_beat = time.time()
            logger.info(
                "[fu4] heartbeat: live=%s pending=%d done=%d failed=%d",
                {s: r.run_id for s, (_, r, _f) in live.items()},
                len(pending),
                len(done),
                len(failed),
            )
    i1090._phase(f"{ROUND.name}_finalize")
    finalize_dispatch(args, done, failed, skipped)
    return 0 if not failed else 1


# ── VM P0: stage/verify + provenance + manifest (--phase stage) ──────────────


def _git_last_commit_iso(rel_path: str) -> str:
    """Committed-file last-commit date (item (j) provenance input)."""
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", rel_path],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
        cwd=str(_SCRIPTS_DIR.parent),
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"git log returned no date for {rel_path}")
    return out


def _mix_hub_provenance(run: Fu4Run) -> dict:
    """Per-file HF last_commit (oid + date) at the consumed revision — the
    item-(j) re-verify at staging (artifact-reuse.md)."""
    from huggingface_hub import HfApi

    paths = [f"{run.mix_hub_prefix}/{f}" for f in _MIX_FILES_REQUIRED]
    infos = hub.retry_transient(
        lambda: HfApi().get_paths_info(i1090.HF_DATA_REPO, paths, repo_type="dataset", expand=True),
        what=f"fu4 provenance {run.mix_hub_prefix}",
    )
    return {
        i.path: {"oid": i.last_commit.oid, "date": i.last_commit.date.isoformat()}
        for i in infos
        if i.last_commit is not None
    }


def _assert_provenance_coherent(cell_key: str, bank_file: str, bank_date: str, prov: dict) -> None:
    """Item-(j) pairwise coherence, compared CHRONOLOGICALLY via
    ``datetime.fromisoformat`` — git ``%cI`` carries the committer's LOCAL
    UTC offset while the HF ``last_commit`` dates are UTC, so a lexicographic
    string compare is chronologically wrong within ~offset-hours."""
    mix_dates = [v["date"] for v in prov.values()]
    if not mix_dates:
        return
    bank_dt = datetime.fromisoformat(bank_date)
    mix_min_dt = min(datetime.fromisoformat(d) for d in mix_dates)
    if bank_dt > mix_min_dt:
        raise ValueError(
            f"[fu4-stage] provenance coherence FAILED for {cell_key}: bank "
            f"{bank_file} last commit {bank_date} postdates the mix "
            f"({min(mix_dates)}) — the consumed input was regenerated after "
            "the dependent mix (artifact-reuse.md item (j))"
        )


def _load_fu3_base(run: Fu4Run, *, tier2_n: int) -> dict:
    """The reused fu3 base Tier-2 read + the A4/A15 field asserts."""
    path = FU3_EVALS_DIR / run.fu3_base_eval
    d = i1090._read_json(path)
    base = d["tier2"]["base"]
    n_questions = len(i1090._eval_questions(i1090.RunConfig(False, (), Path(".")), run.behavior))
    expected_n = tier2_n * n_questions
    if base["n"] != expected_n:
        raise ValueError(
            f"[fu4-stage] {run.fu3_base_eval}: base n={base['n']} != expected "
            f"{expected_n} (tier2_n {tier2_n} x {n_questions} questions) — A4 broken"
        )
    if run.behavior == "impolite" and base["rate"] > IMPOLITE_BASE_RATE_MAX:
        raise ValueError(
            f"[fu4-stage] {run.fu3_base_eval}: impolite base rate {base['rate']} > "
            f"{IMPOLITE_BASE_RATE_MAX} — A15 (band entry = pure install) broken"
        )
    summ = i1090._read_json(FU3_SUMMARY_PATH)
    got = summ.get("meta", {}).get("judge_max_tokens")
    if got != JUDGE_MAX_TOKENS_FU4:
        raise ValueError(
            f"[fu4-stage] fu3_summary meta judge_max_tokens={got} != "
            f"{JUDGE_MAX_TOKENS_FU4} — instrument-identity premise (A4) broken"
        )
    return {"file": str(path), **base}


def _assert_reused_recipe_identity(seen_cells: dict[str, dict]) -> None:
    """A4 manifest-diff assert (fu5 plan §12): the reused fu4 ``fmt-pers-lr1e4``
    run must be recipe-identical to the fu5 rank ladder at r32 — same c1 mix
    sha (vs the fu4-recorded pin), same lr, same epochs/save_steps — so the
    r32 rung folds into the ladder as a genuine control, never a smuggled
    second variable (artifact-reuse check (d))."""
    for reused in ROUND.reused_runs:
        src_manifest_path = DELIVERABLES_DIR / "cell_manifest_fu4.json"
        if not src_manifest_path.exists():
            raise FileNotFoundError(
                f"[{ROUND.name}-stage] {src_manifest_path} missing — cannot verify the "
                f"reused {reused.source_run_id} recipe identity (A4)"
            )
        src_manifest = i1090._read_json(src_manifest_path)
        entry = {r["run_id"]: r for r in src_manifest.get("runs", [])}.get(reused.source_run_id)
        if entry is None:
            raise ValueError(
                f"[{ROUND.name}-stage] {src_manifest_path} has no entry for "
                f"{reused.source_run_id} — cannot verify A4"
            )
        fmt_runs = [r for r in ROUND.runs if r.behavior == "formatting"]
        staged = seen_cells.get("fmt-pers")
        if fmt_runs and staged is None:
            raise ValueError(
                f"[{ROUND.name}-stage] fmt-pers cell not staged — cannot verify the "
                "reused-run mix sha (A4)"
            )
        if staged is not None and staged["train_mix_sha256"] != entry["train_mix_sha256"]:
            raise ValueError(
                f"[{ROUND.name}-stage] A4 BROKEN: staged c1 mix sha "
                f"{staged['train_mix_sha256']} != the fu4-recorded pin "
                f"{entry['train_mix_sha256']} ({reused.source_run_id}) — the reused r32 "
                "rung would not be recipe-identical to the rank ladder"
            )
        for r in fmt_runs:
            if r.lr != entry["lr"]:
                raise ValueError(
                    f"[{ROUND.name}-stage] A4 BROKEN: {r.run_id} lr {r.lr} != the reused "
                    f"{reused.source_run_id} lr {entry['lr']}"
                )
        if src_manifest.get("epochs") != FU4_EPOCHS or src_manifest.get("save_steps") != (
            FU4_SAVE_STEPS
        ):
            raise ValueError(
                f"[{ROUND.name}-stage] A4 BROKEN: fu4 manifest epochs/save_steps "
                f"{src_manifest.get('epochs')}/{src_manifest.get('save_steps')} != "
                f"{FU4_EPOCHS}/{FU4_SAVE_STEPS}"
            )


def cmd_stage(cfg: i1090.RunConfig, args: argparse.Namespace) -> int:
    """VM P0: stage + K1-verify the 3 frozen mixes, re-verify pairwise
    provenance coherence (item (j)), assert the reused fu3 base reads (A4/A15),
    and write cell_manifest_fu4.json (the sha-pin contract the pod consumes)."""
    i1090._phase(f"{ROUND.name}_stage")
    runs = resolve_fu4_runs(args.runs, cfg.smoke)
    entries: list[dict] = []
    seen_cells: dict[str, dict] = {}
    for run in runs:
        if run.cell_key in seen_cells:
            cell_rec = seen_cells[run.cell_key]
        else:
            if cfg.smoke:
                build_fu4_smoke_mix(cfg, run)
                mix_rec = verify_fu4_mix(cfg, run, None)
                prov: dict[str, Any] = {"smoke_fixture": True}
                bank_date = None
            else:
                stage_fu4_mix(cfg, run)
                mix_rec = verify_fu4_mix(cfg, run, None)
                prov = _mix_hub_provenance(run)
                bank_date = _git_last_commit_iso(_BANK_FILES[run.behavior])
                _assert_provenance_coherent(
                    run.cell_key, _BANK_FILES[run.behavior], bank_date, prov
                )
            base = _load_fu3_base(run, tier2_n=cfg.tier2_n) if not cfg.smoke else {}
            cell_rec = {
                "train_mix_sha256": mix_rec["train_mix_sha256"],
                "n_rows": mix_rec["n_rows"],
                "hub_provenance": prov,
                "bank_git_date": bank_date,
                "fu3_base": base,
            }
            seen_cells[run.cell_key] = cell_rec
        entries.append(
            {
                "run_id": run.run_id,
                "cell_key": run.cell_key,
                "behavior": run.behavior,
                "context_id": run.context_id,
                "lr": run.lr,
                "lora_r": run.lora_r,
                "lora_alpha": run.lora_alpha,
                "mix_hub_prefix": run.mix_hub_prefix,
                "mix_layout": run.mix_layout,
                "fu3_base_eval": run.fu3_base_eval,
                **cell_rec,
            }
        )
    if ROUND.reused_runs and not cfg.smoke:
        _assert_reused_recipe_identity(seen_cells)
    manifest = {
        "issue": i1090.ISSUE,
        "round": ROUND.label,
        "smoke": cfg.smoke,
        "band": list(JUDGED_RATE_BAND),
        "epochs": FU4_EPOCHS,
        "save_steps": FU4_SAVE_STEPS,
        "lrs": sorted({r.lr for r in ROUND.runs}),
        "lora_ranks": sorted({r.lora_r for r in ROUND.runs}),
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
        "runs": entries,
        "git_commit": i1074._git_short_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.manifest_out) if args.manifest_out else _default_manifest_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    i1090._atomic_write_json(out, manifest)
    logger.info("[fu4-stage] manifest -> %s (%d runs)", out, len(entries))
    return 0


def _default_manifest_path(cfg: i1090.RunConfig) -> Path:
    # Smoke NEVER writes the committed deliverables path (smoke-output rule).
    return (cfg.out_root if cfg.smoke else ROUND.deliverables_dir) / ROUND.manifest_name


# ── VM P3: judge + aggregate (--phase judge-aggregate) ───────────────────────


def _drop_split_from_raw(judge_root: Path, tag: str) -> dict:
    """Transport-vs-content split (llm-judging rules 9/24) from the saved raw
    judge rows: an ``error: True`` row is a TRANSPORT loss (re-judgeable via
    scripts/issue1090_fu3_rejudge_529.py); other drops are content drops."""
    raw_path = judge_root / tag / "judge_raw.json"
    transport = 0
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        # save_raw shape (graded_judge.judge_result_from_save_raw): the
        # "all_scores" dict maps custom_id -> the PARSED judge value; an
        # api_dispatch transport exhaustion parses to a {"error": ...} dict.
        all_scores = raw.get("all_scores", {}) if isinstance(raw, dict) else {}
        transport = sum(1 for v in all_scores.values() if isinstance(v, dict) and v.get("error"))
    return {"transport_losses": transport, "raw_path": str(raw_path)}


def _stage_run_outputs(cfg: i1090.RunConfig, run: Fu4Run) -> tuple[Path, dict]:
    """Local run outputs if present, else staged from the fu4 HF prefixes
    (the pod uploaded them before release). Tier-2 completions are staged ONLY
    for a ``trained`` build — a K2-diverged run never generated/uploaded them
    (a plan-REGISTERED reportable outcome; unconditionally staging that prefix
    was the code-review v16 Critical: ``_stage_hf_prefix`` raises
    FileNotFoundError on a missing prefix). A wholly-failed run may have
    uploaded nothing at all — the caller isolates that per run."""
    run_root = _run_root(cfg, run)
    if not _build_result_path(run_root).exists():
        i1090._stage_hf_prefix(f"{ROUND.data_prefix}/{run.run_id}", run_root)
    build = i1090._read_json(_build_result_path(run_root))
    if build.get("status") == "trained":
        tier2_dir = run_root / "tier2"
        ctx_file = tier2_dir / f"completions__trained__{run.context_id}.json"
        if not ctx_file.exists():
            i1090._stage_hf_prefix(
                f"{ROUND.data_prefix}/raw_completions/tier2/{run.run_id}", tier2_dir
            )
    return run_root, build


def _judge_run_tier2(cfg: i1090.RunConfig, judge_root: Path, run: Fu4Run, run_root: Path) -> dict:
    """Judge one run's trained Tier-2 completions (structural runs cost zero
    judge calls) + the rule-24/23 transport/content drop split + K4 flags."""
    ctx_file = run_root / "tier2" / f"completions__trained__{run.context_id}.json"
    payload = json.loads(ctx_file.read_text())
    tag = f"{run.run_id}-t2-trained"
    tier2 = i1090._judge_rate(
        run.behavior,
        payload["questions"],
        payload["completions"],
        tag=tag,
        n_draws=cfg.tier2_draws,
        judge_root=judge_root / run.behavior,
        max_tokens=JUDGE_MAX_TOKENS_FU4,
    )
    if tier2["mode"] != "judged":
        return tier2
    split = _drop_split_from_raw(judge_root / run.behavior, tag)
    tier2["transport_losses"] = split["transport_losses"]
    tier2["content_dropped_draws"] = tier2.get("n_dropped_draws", 0) - split["transport_losses"]
    if split["transport_losses"] > 0:
        logger.warning(
            "[fu4-K4] %s: %d TRANSPORT-lost judge draws — re-judge them "
            "(scripts/issue1090_fu3_rejudge_529.py pattern) before any headline "
            "read (llm-judging rule 24)",
            run.run_id,
            split["transport_losses"],
        )
    content_rate = tier2["content_dropped_draws"] / max(tier2.get("n_total_draws", 1), 1)
    tier2["k4_truncation_check_required"] = bool(content_rate >= 0.10)
    if tier2["k4_truncation_check_required"]:
        logger.warning(
            "[fu4-K4] %s: content-drop rate %.1f%% >= 10%% — run the rule-23 "
            "truncation check before narrating this rate",
            run.run_id,
            100 * content_rate,
        )
    return tier2


def _verdict_lattice_inputs(out: dict, runs: Sequence[Fu4Run] | None = None) -> None:
    """Registered verdict-lattice INPUTS (plan §3; interpretation stays the
    analyzer's): per cell, U/V from the best-arm max-rung Tier-1 rate, with the
    Tier-2 confirmatory value at the selected rung alongside.

    Groups over the AGGREGATE ENTRIES (``out["runs"].values()``) by their
    ``cell_key`` — identical to the historical per-``runs`` grouping for fu4
    (the entries ARE the resolved runs there) and it additionally folds REUSED
    entries (fu5's ``reused_fu4_r32``) into their cell's lattice, so Arm B
    reads all three ranks from one file. ``runs`` is accepted for back-compat
    with existing callers (rejudge) and unused."""
    del runs  # grouping is over the aggregate entries (incl. reused rows)
    entries = [e for e in out["runs"].values() if e.get("cell_key")]
    for cell_key in sorted({e["cell_key"] for e in entries}):
        cell_runs = [e for e in entries if e["cell_key"] == cell_key]
        best: dict[str, Any] = {"max_tier1_rate": None, "run_id": None, "step": None}
        for cr in cell_runs:
            for step, rate in (cr.get("rates_by_step") or {}).items():
                if best["max_tier1_rate"] is None or rate > best["max_tier1_rate"]:
                    best = {"max_tier1_rate": rate, "run_id": cr["run_id"], "step": int(step)}
            cr_t2 = cr.get("tier2_trained")
            if cr_t2 is not None:
                cr["tier2_confirm_rate"] = cr_t2["rate"]
        max_rate = best["max_tier1_rate"]
        out["cells"][cell_key] = {
            "best_run": best,
            "U_band_floor_margin": (max_rate - JUDGED_RATE_BAND[0])
            if max_rate is not None
            else None,
            "V_030_margin": (max_rate - 0.30) if max_rate is not None else None,
            "tier2_confirm": {cr["run_id"]: cr.get("tier2_confirm_rate") for cr in cell_runs},
        }


def _formatting_judged_reread(
    cfg: i1090.RunConfig, judge_root: Path, run: Fu4Run, run_root: Path
) -> dict:
    """The CONDITIONAL formatting judged re-read (plan §6 dual-DV (i)): judge
    ALL of the run's Tier-2 completions with the formatting rubric (the
    structural DV's judged companion — ``_judge_rate`` routes formatting to
    the structural predicate, so this goes through ``judge_graded`` directly
    at the fu4 instrument)."""
    from explore_persona_space.eval.graded_judge import judge_graded

    ctx_file = run_root / "tier2" / f"completions__trained__{run.context_id}.json"
    payload = json.loads(ctx_file.read_text())
    behavior = BEHAVIORS[run.behavior]
    flat = [
        (f"{run.run_id}-reread-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(payload["questions"])
        for j, comp in enumerate(payload["completions"][i])
    ]
    cell_dir = judge_root / "formatting_reread" / run.run_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = judge_graded(
        flat,
        behavior.judge_rubric,
        n_draws=cfg.tier2_draws,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
        max_tokens=JUDGE_MAX_TOKENS_FU4,
    )
    n_pos = n_scored = n_dropped = 0
    for iid, _q, _c in flat:
        score = result.scores.get(iid)
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > behavior.threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError(f"formatting re-read for {run.run_id}: every completion judge-dropped")
    lo, hi = i1090._wilson(n_pos, n_scored)
    return {
        "rate": n_pos / n_scored,
        "k": n_pos,
        "n": n_scored,
        "n_dropped": n_dropped,
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws": result.n_dropped_draws,
        "wilson95": [lo, hi],
        "mode": "judged_reread",
    }


def _formatting_reread_fires(out: dict, runs: Sequence[Fu4Run]) -> bool:
    """Conditional formatting judged re-read trigger.

    fu4 (``reread_rate_floor is None``): the historical trigger — any
    formatting rung's structural install delta >= +0.30 over the reused base
    rate. fu5 (``reread_rate_floor = 0.30``): WIDENED to an ABSOLUTE
    structural rate >= 0.30 on any Tier-1 rung OR the Tier-2 confirmatory
    read (orchestrator decision; a strict superset of the delta trigger at
    base 0.167, closing the Statistics critic's mid-band [0.30, 0.467)
    construct gap at <=3K Batch-API calls)."""
    floor = ROUND.reread_rate_floor
    for r in runs:
        if r.behavior != "formatting":
            continue
        cr = out["runs"].get(r.run_id) or {}
        rung_rates = list((cr.get("rates_by_step") or {}).values())
        if floor is not None:
            t2_rate = (cr.get("tier2_trained") or {}).get("rate")
            if t2_rate is not None:
                rung_rates.append(t2_rate)
            if any(rate >= floor for rate in rung_rates):
                return True
            continue
        base_rate = (cr.get("base_tier2") or {}).get("rate")
        if base_rate is None:
            continue
        for rate in rung_rates:
            if rate - base_rate >= 0.30:
                return True
    return False


def _copy_reused_runs(out: dict) -> None:
    """Fold each ``ROUND.reused_runs`` entry into the aggregate FIELD-FOR-FIELD
    from the producing round's committed ladders file (plan D2 item 1: fu5's
    ``reused_fu4_r32`` = fu4 ``fmt-pers-lr1e4`` — rates_by_step, selection,
    tier2, margin — so Arm B reads all 3 ranks from one file). Fail-loud on a
    missing source (a base-less rank ladder is uninterpretable)."""
    for reused in ROUND.reused_runs:
        ladders_path = Path(reused.ladders_path)
        if not ladders_path.exists():
            raise FileNotFoundError(
                f"[{ROUND.name}] reused-run source {ladders_path} missing — cannot fold "
                f"{reused.entry_id} into the aggregate"
            )
        src = (i1090._read_json(ladders_path).get("runs") or {}).get(reused.source_run_id)
        if src is None:
            raise ValueError(
                f"[{ROUND.name}] {ladders_path} has no runs[{reused.source_run_id!r}] — "
                f"cannot fold {reused.entry_id} into the aggregate"
            )
        entry = json.loads(json.dumps(src))  # field-for-field deep copy
        entry["run_id"] = reused.entry_id
        entry["lora_r"] = reused.lora_r
        entry["lora_alpha"] = reused.lora_alpha
        entry["reused_from"] = {
            "file": str(ladders_path),
            "run_id": reused.source_run_id,
            "adapter_subfolder": reused.adapter_subfolder,
            "adapter_revision": reused.adapter_revision,
        }
        out["runs"][reused.entry_id] = entry


def _retrain_parity_record(out: dict) -> dict | None:
    """Aggregate retrain-parity record for the round's K3 anchor (fu5 plan §4:
    ``imp-bare-lr1e5`` rung-15). Report-and-flag ONLY (orchestrator decision):
    rate > 0.5 -> ``parity-failed`` (the ladder K3 should already have
    aborted); rate in (0.35, 0.5] -> ``parity-degraded``; else ``ok``. The
    abort stays > 0.5 in ``ladder_fu4_run``."""
    cr = out["runs"].get(ROUND.k3_parity_run_id)
    if cr is None:
        return None
    rate = (cr.get("rates_by_step") or {}).get(str(K3_PARITY_STEP))
    rec: dict[str, Any] = {
        "run_id": ROUND.k3_parity_run_id,
        "step": K3_PARITY_STEP,
        "rate": rate,
        "abort_bar": K3_PARITY_MAX_RATE,
        "degraded_floor": ROUND.k3_parity_degraded_floor,
    }
    if rate is None:
        rec["status"] = "missing"
    elif rate > K3_PARITY_MAX_RATE:
        rec["status"] = "parity-failed"
    elif ROUND.k3_parity_degraded_floor is not None and rate > ROUND.k3_parity_degraded_floor:
        rec["status"] = "parity-degraded"
    else:
        rec["status"] = "ok"
    if rec["status"] != "ok":
        logger.warning(
            "[%s-K3] retrain-parity %s: %s rung-%d rate=%s (abort bar %.2f, degraded "
            "floor %s) — report-and-flag",
            ROUND.name,
            rec["status"],
            rec["run_id"],
            rec["step"],
            rec["rate"],
            rec["abort_bar"],
            rec["degraded_floor"],
        )
    return rec


# ── Eval-split diagnostic (fu5 D2 item 6; VM-side, diagnostic only) ──────────

EVAL_SPLIT_N_DRAWS = 3  # plan D2 item 6: Sonnet judge, 3 draws majority (~90 calls)
EVAL_SPLIT_PLACEHOLDER = "(no response provided — classify the question itself)"
# Anchored reason-then-score rubric (llm-judging rules 6/7); ONE behavior per
# call (rule 8). The judge sees the standard "Question / AI's response" user
# msg; the response field is a fixed placeholder and the rubric pins the
# classification to the QUESTION alone.
FMT_EVAL_SPLIT_RUBRIC = (
    "You are auditing evaluation QUESTIONS for a list-formatting experiment. "
    "Classify whether a bulleted or numbered LIST is a natural, high-quality answer "
    "format for the question shown in the 'Question asked to the AI' field. IGNORE "
    "the 'AI's response' field entirely (it is a fixed placeholder).\n\n"
    "Score 0-100 where:\n"
    "0 = prose-natural: a good answer is flowing prose; forcing a list would be "
    "unnatural (e.g. 'write a short story', 'explain why X happened', a request for "
    "a single rewritten sentence).\n"
    "50 = borderline: either format is defensible.\n"
    "100 = list-affordable: a good answer naturally enumerates items, steps, or "
    "options (e.g. 'what are some ways to...', 'give me ideas for...', 'how do I "
    "set up X' with multiple steps).\n\n"
    "First give a one-or-two-sentence justification, then reply with the integer "
    'score in the JSON form {"score": <int>} on the final line.'
)


def _classify_eval_split(judge_root: Path, questions: Sequence[str], judge_fn=None) -> dict:
    """Classify each formatting eval slice list-affordable vs prose-natural via
    the graded judge, majority over the kept per-draw scores (>50 = list draw).
    A question with zero kept draws is ``unclassified`` (dropped, never
    coerced — llm-judging rule 9). ``judge_fn`` is the external-API seam
    (defaults to the real ``judge_graded``)."""
    from explore_persona_space.eval.graded_judge import judge_graded

    jf = judge_fn if judge_fn is not None else judge_graded
    items = [(f"evalsplit-q{i:03d}", q, EVAL_SPLIT_PLACEHOLDER) for i, q in enumerate(questions)]
    cell_dir = judge_root / "eval_split"
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = jf(
        items,
        FMT_EVAL_SPLIT_RUBRIC,
        n_draws=EVAL_SPLIT_N_DRAWS,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=BEHAVIORS["formatting"].judge_model,
        max_tokens=JUDGE_MAX_TOKENS_FU4,
    )
    classification: dict[str, dict] = {}
    for iid, q, _placeholder in items:
        draws = result.per_item_scores.get(iid, [])
        n_list = sum(1 for score in draws if score > 50)
        if not draws:
            label = "unclassified"
        elif 2 * n_list > len(draws):
            label = "list_affordable"
        else:
            label = "prose_natural"
        classification[iid] = {
            "question": q,
            "label": label,
            "n_kept_draws": len(draws),
            "n_list_draws": n_list,
        }
    return classification


def _eval_split_rates(comps_file: Path, label_by_text: dict[str, str]) -> dict:
    """Tier-2 structural rate per eval-question split for ONE completions file
    (the deterministic formatting predicate; zero judge calls). Questions whose
    text is not in the classification (an unclassified slice, or a smoke's
    truncated question set) are counted, never silently dropped."""
    payload = json.loads(comps_file.read_text())
    predicate = _STRUCTURAL_PREDICATES["formatting"]
    tallies: dict[str, list[int]] = {"list_affordable": [0, 0], "prose_natural": [0, 0]}
    n_unclassified = 0
    for qi, q in enumerate(payload["questions"]):
        label = label_by_text.get(q)
        if label not in tallies:
            n_unclassified += len(payload["completions"][qi])
            continue
        k, n = tallies[label]
        for comp in payload["completions"][qi]:
            n += 1
            k += int(bool(predicate(comp)))
        tallies[label] = [k, n]
    rec: dict[str, Any] = {
        lab: {
            "k": k,
            "n": n,
            "rate": (k / n) if n else None,
            "wilson95": list(i1090._wilson(k, n)) if n else None,
        }
        for lab, (k, n) in tallies.items()
    }
    rec["n_completions_unclassified_questions"] = n_unclassified
    rec["file"] = str(comps_file)
    return rec


def _eval_split_diagnostic(cfg: i1090.RunConfig, judge_root: Path, runs: Sequence[Fu4Run]) -> dict:
    """The fu5 eval-split diagnostic (plan D2 item 6): classify the formatting
    eval slices ONCE, then report the Tier-2 structural rate per split for the
    round's formatting arms + the reused r32 rung + the fu3 base arm.
    DIAGNOSTIC ONLY — a genuinely-missing tier-2 artifact is a RECORDED miss
    (explicit status), while transport failures stay loud (retry_transient)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    questions = i1090._eval_questions(cfg, "formatting")
    classification = _classify_eval_split(judge_root, questions)
    label_by_text = {v["question"]: v["label"] for v in classification.values()}
    diag: dict[str, Any] = {
        "instrument": {
            "judge_model": BEHAVIORS["formatting"].judge_model,
            "n_draws": EVAL_SPLIT_N_DRAWS,
            "max_tokens": JUDGE_MAX_TOKENS_FU4,
            "rule": "majority of kept draws > 50 => list_affordable; no kept draws => unclassified",
        },
        "n_questions": len(questions),
        "n_list_affordable": sum(
            1 for v in classification.values() if v["label"] == "list_affordable"
        ),
        "n_prose_natural": sum(1 for v in classification.values() if v["label"] == "prose_natural"),
        "n_unclassified": sum(1 for v in classification.values() if v["label"] == "unclassified"),
        "classification": classification,
        "per_arm": {},
    }
    for run in runs:
        if run.behavior != "formatting":
            continue
        f = _run_root(cfg, run) / "tier2" / f"completions__trained__{run.context_id}.json"
        diag["per_arm"][run.run_id] = (
            _eval_split_rates(f, label_by_text)
            if f.exists()
            else {"status": "missing_tier2", "file": str(f)}
        )
    for reused in ROUND.reused_runs:
        dest = cfg.out_root / reused.entry_id / "tier2"
        f = dest / f"completions__trained__{reused.context_id}.json"
        if not f.exists():
            try:
                i1090._stage_hf_prefix(reused.tier2_hf_prefix, dest)
            except FileNotFoundError as e:
                diag["per_arm"][reused.entry_id] = {"status": "missing_tier2", "reason": str(e)}
                continue
        diag["per_arm"][reused.entry_id] = (
            _eval_split_rates(f, label_by_text)
            if f.exists()
            else {"status": "missing_tier2", "file": str(f)}
        )
    base_hf_path = (
        f"{fu3w.DATA_PREFIX_FU3}/C1-pers-con-formatting-claude/tier2/"
        f"completions__base__{i1090.SOURCE_CONTEXT_ID}.json"
    )
    try:
        base_local = hub.retry_transient(
            lambda: hf_hub_download(i1090.HF_DATA_REPO, base_hf_path, repo_type="dataset"),
            what=f"{ROUND.name} eval-split base tier2",
        )
        diag["per_arm"]["fu3_base"] = _eval_split_rates(Path(base_local), label_by_text)
        diag["per_arm"]["fu3_base"]["hf_path"] = base_hf_path
    except EntryNotFoundError as e:
        diag["per_arm"]["fu3_base"] = {"status": "missing_tier2", "reason": str(e)}
    return diag


def cmd_judge_aggregate(cfg: i1090.RunConfig, args: argparse.Namespace) -> int:
    """VM P3 (pod RELEASED first — #664): judge the impolite Tier-2 completions
    (5 draws, 300-token budget, fresh fu4 cache dirs; the route is auto-decided
    by ``judge_completions_batch`` — per-run N = 200x5 = 1,000 draws sits below
    the 2,000-call Batch crossover, so the REALIZED route is SYNC; plan §9 says
    "Batch API" — deliberate deviation within the plan's judge-concurrency
    tuning allowance: per-run checkpointing + fresh per-run cache dirs beat the
    50% batch discount on ~6K Sonnet calls, and no GPU is held either way),
    split content-drops vs transport-losses (K4 + rule 24), fold in the reused
    fu3 base arms + ladders + margins, compute the registered verdict-lattice
    inputs, and write fu4_ladders.json. Diverged / failed runs are FIRST-CLASS
    records (per-run isolation), never an aggregate crash."""
    i1090._phase(f"{ROUND.name}_judge_aggregate")
    manifest = _load_manifest(args.manifest) or _load_manifest(str(_default_manifest_path(cfg)))
    if not manifest:
        raise FileNotFoundError("no cell manifest — run `--phase stage` first (or pass --manifest)")
    runs = resolve_fu4_runs(args.runs, cfg.smoke)
    judge_root = cfg.out_root / f"{ROUND.name}_aggregate" / "judge"
    out: dict[str, Any] = {
        "issue": i1090.ISSUE,
        "round": ROUND.label,
        "smoke": cfg.smoke,
        "band": list(JUDGED_RATE_BAND),
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
        "tier2_draws": cfg.tier2_draws,
        "runs": {},
        "cells": {},
        "git_commit": i1074._git_short_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = (cfg.out_root if cfg.smoke else ROUND.deliverables_dir) / ROUND.ladders_name
    for run in runs:
        entry = _manifest_run_entry(manifest, run.run_id)
        if not entry and not cfg.smoke:
            # Sibling of the cmd_run sha-pin gate (silent-default class): a
            # missing manifest entry would silently drop the reused fu3 base
            # arm (base_tier2 -> None kills install_delta + the fmt trigger).
            raise ValueError(
                f"[fu4] manifest has no entry for {run.run_id} — regenerate the "
                "stage manifest (or fix --runs); refusing a base-less aggregate"
            )
        rec: dict[str, Any] = {
            "run_id": run.run_id,
            "cell_key": run.cell_key,
            "behavior": run.behavior,
            "context_id": run.context_id,
            "lr": run.lr,
            # Arm B's manipulated variable rides the aggregate so the rank
            # ladder is self-describing (figures key lines on lora_r; the
            # reused r32 entry gets these fields from _copy_reused_runs).
            "lora_r": run.lora_r,
            "lora_alpha": run.lora_alpha,
            "base_tier2": entry.get("fu3_base"),
        }
        try:
            run_root, build = _stage_run_outputs(cfg, run)
        except FileNotFoundError as e:
            # Per-run isolation (code-review v16 Critical): a failed run may
            # have uploaded NOTHING (or no build record) — record it as a
            # first-class outcome and keep aggregating the sibling runs.
            # Anything but the staging helper's documented missing-prefix /
            # missing-file signal stays LOUD: hf_hub_download raises
            # LocalEntryNotFoundError (a FileNotFoundError SUBCLASS) when the
            # network dies mid-staging with no cached file, and recording that
            # would mislabel a healthy TRAINED run as missing_artifacts
            # (code-review v17 Minor).
            if isinstance(e, LocalEntryNotFoundError):
                raise
            logger.warning("[fu4] %s: build artifacts unavailable — %s", run.run_id, e)
            rec.update({"status": "missing_artifacts", "missing_reason": str(e)})
            out["runs"][run.run_id] = rec
            i1090._atomic_write_json(out_path, out)
            continue
        rec.update(
            {
                "status": build.get("status"),
                "rates_by_step": build.get("rates_by_step"),
                "degeneracy_by_step": build.get("degeneracy_by_step"),
                "selection": build.get("selection"),
                "divergence_check": build.get("divergence_check"),
            }
        )
        if build.get("status") == "trained":
            rec["tier2_trained"] = _judge_run_tier2(cfg, judge_root, run, run_root)
            base = entry.get("fu3_base") or {}
            if base.get("rate") is not None:
                rec["install_delta"] = rec["tier2_trained"]["rate"] - base["rate"]
            margin_path = run_root / "margin.json"
            if margin_path.exists():
                rec["margin"] = i1090._read_json(margin_path)
        out["runs"][run.run_id] = rec
        i1090._atomic_write_json(out_path, out)  # checkpoint per run
    _copy_reused_runs(out)  # fu5: reused_fu4_r32 folded in BEFORE the lattice
    _verdict_lattice_inputs(out)
    parity = _retrain_parity_record(out)
    if parity is not None:
        out["retrain_parity"] = parity
    out["formatting_judged_reread_required"] = _formatting_reread_fires(out, runs)
    if out["formatting_judged_reread_required"]:
        logger.warning(
            "[%s] formatting reread trigger fired (%s) — running the CONDITIONAL "
            "judged re-read (plan §6 dual-DV; %d draws, 300-token budget)",
            ROUND.name,
            (
                f"structural rate >= {ROUND.reread_rate_floor} (widened trigger)"
                if ROUND.reread_rate_floor is not None
                else "install delta >= +0.30"
            ),
            cfg.tier2_draws,
        )
        for run in runs:
            if run.behavior != "formatting":
                continue
            if out["runs"][run.run_id].get("status") != "trained":
                continue
            out["runs"][run.run_id]["formatting_judged_reread"] = _formatting_judged_reread(
                cfg, judge_root, run, _run_root(cfg, run)
            )
            i1090._atomic_write_json(out_path, out)  # checkpoint per re-read
    if ROUND.eval_split_diagnostic:
        out["eval_split_diagnostic"] = _eval_split_diagnostic(cfg, judge_root, runs)
        i1090._atomic_write_json(out_path, out)  # checkpoint after the diagnostic
    i1090._atomic_write_json(out_path, out)
    logger.info("[%s] aggregate -> %s (%d runs)", ROUND.name, out_path, len(out["runs"]))
    return 0


# ── Config / seams / CLI ─────────────────────────────────────────────────────


def make_fu4_smoke_seams(cfg: i1090.RunConfig) -> i1090.Seams1090:
    """The parent tiny-real smoke seams with the train clamp retargeted to
    max_steps=5 (ONE rung at save_steps=5). The clamp composes inside
    ``train_fu4_run`` (which re-pins max_steps after the base clamp)."""
    return i1090.make_smoke_seams(cfg)


def fu4_config(args: argparse.Namespace) -> i1090.RunConfig:
    """The fu4 RunConfig: a FRESH out_root (the parent/fu2/fu3 trees carry
    other rounds' build state) + tiny smoke knobs."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (
            f"/tmp/issue-{i1090.ISSUE}-{ROUND.name}-smoke"
            if smoke
            else f"data/issue_{i1090.ISSUE}/{ROUND.name}"
        )
    )
    runs = resolve_fu4_runs(getattr(args, "runs", None) or getattr(args, "run", None), smoke)
    return i1090.RunConfig(
        smoke=smoke,
        cells=runs,  # duck-typed (.slug/.behavior) — the seams/regime consumers
        out_root=out_root,
        seed=args.seed,
        tier1_n=2 if smoke else i1090.TIER1_N_COMPLETIONS,
        tier1_draws=2 if smoke else i1090.TIER1_JUDGE_DRAWS,
        tier2_n=2 if smoke else i1090.TIER2_N_COMPLETIONS,
        tier2_draws=2 if smoke else i1090.TIER2_JUDGE_DRAWS,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
    )


def fu4_regime_key(cfg: i1090.RunConfig) -> dict:
    """Global (run-independent) regime keys; per-run lr rides the ladder regime."""
    return {
        "issue": i1090.ISSUE,
        "round": ROUND.name,
        "followup_label": ROUND.label,
        "max_lora_rank": ROUND.max_lora_rank,
        "smoke": cfg.smoke,
        "seed": cfg.seed,
        "fu4_epochs": FU4_EPOCHS,
        "fu4_save_steps": FU4_SAVE_STEPS,
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU4,
        "max_length": i1090.MAX_LENGTH_1090,
        "tier1": [cfg.tier1_n, cfg.tier1_draws],
        "tier2": [cfg.tier2_n, cfg.tier2_draws],
        "eval_question_limit": cfg.eval_question_limit,
        "band": list(JUDGED_RATE_BAND),
    }


def _check_regime_fu4(cfg: i1090.RunConfig) -> None:
    """Refuse to mix regimes inside one out_root (fu2 pattern)."""
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / f"{ROUND.name}_run_config.json"
    cur = fu4_regime_key(cfg)
    if p.exists():
        prior = i1090._read_json(p)
        if prior != cur:
            raise RuntimeError(
                f"out_root {cfg.out_root} holds a fu4 run under a DIFFERENT regime "
                f"(prior={prior}); refusing to mix — use a fresh --out-root"
            )
    else:
        i1090._atomic_write_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1090 fu4/fu5 follow-up driver (round-parametrized)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, same code path")
    mode.add_argument("--full", action="store_true", help="the real GPU/API run")
    p.add_argument(
        "--round",
        choices=tuple(sorted(ROUNDS)),
        default="fu4",
        help="round registry: fu4 (extended-dose-lr) | fu5 (finish-impolite-bare-...-rank)",
    )
    p.add_argument(
        "--phase",
        required=True,
        choices=("stage", "dispatch", "run", "judge-aggregate", "k5-rank-smoke"),
    )
    p.add_argument("--runs", default=None, help="comma run_id subset (smoke parity)")
    p.add_argument("--run", default=None, help="single run id (--phase run)")
    p.add_argument("--manifest", default=None, help="cell_manifest_fu4.json path (sha pins)")
    p.add_argument("--manifest-out", default=None, help="stage-phase manifest destination")
    p.add_argument("--out-root", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--n-gpus", type=int, default=None, help="override detected GPU count")
    p.add_argument("--gpu-id", type=int, default=0, help="informational; CVD pins the device")
    p.add_argument("--allow-unpinned-gpu", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true", help="print queue, run nothing")
    args = p.parse_args(argv)
    if args.phase == "run" and not args.run:
        p.error("--phase run requires --run <run_id>")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    set_round(args.round)  # FIRST: every helper below reads the ROUND registry
    cfg = fu4_config(args)
    # Resolved paths threaded to the dispatch worker cmd (never re-derived).
    args.out_root_resolved = cfg.out_root
    args.sentinel_dir_resolved = cfg.sentinel_dir or Path("/workspace/logs")
    seams = make_fu4_smoke_seams(cfg) if cfg.smoke else i1090.Seams1090()
    _check_regime_fu4(cfg)
    logger.info(
        "issue1090_fu4 round=%s phase=%s smoke=%s runs=%s out_root=%s",
        ROUND.name,
        args.phase,
        cfg.smoke,
        [c.slug for c in cfg.cells],
        cfg.out_root,
    )
    if args.phase == "stage":
        return cmd_stage(cfg, args)
    if args.phase == "dispatch":
        return cmd_dispatch(args)
    if args.phase == "run":
        return cmd_run(cfg, seams, args)
    if args.phase == "k5-rank-smoke":
        return cmd_k5_rank_smoke(cfg, args)
    return cmd_judge_aggregate(cfg, args)
    # NOTE: [phase=done] is emitted by scripts/issue1090_fu4_dispatch.sh, never here.


if __name__ == "__main__":
    raise SystemExit(main())
