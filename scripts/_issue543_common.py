"""Shared constants + helpers for issue #543 (ratio-lever marker-install survival).

Tests whether lowering the trigger->marker positive ratio in the install data
(Redwood's password-locking-ratio lever), with install strength matched at the
trigger cell, lets a marker install on Qwen-2.5-7B survive one epoch of benign
medical SFT better than the 50/50 baseline (plan v4, tasks/.../543/plans/plan.md).

Issue #557 (erasure-pressure lr sweep over the #543 r50 installs) reuses this
module: the ``ISSUE_557``/variant-aware helpers below redirect OUTPUT paths to
``issue_557`` namespaces when a ``--variant`` lr tag is set, while every
parent-side READ (phase1_result.json, the Phase-1 adapter resolve) stays on
the ``issue_543`` paths (#557 plan §4.2 threading-scope note). All #543
defaults are byte-identical when no variant is set.

Imported by:
  - scripts/gen_issue543_response_bank.py
  - scripts/build_issue543_mixes.py
  - scripts/run_issue543_ratio.py
  - scripts/eval_issue543.py
  - scripts/rollup_issue543_survival.py
  - scripts/probe_issue557_absorption.py
  - scripts/rollup_issue557_lrsweep.py
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ISSUE = 543

# ── Hard constants (plan §10 Reproducibility Card) ──────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"  # leading space + REFERENCE MARK; Qwen-2.5 id 83399
MARKER_TEXT_QUOTED = shlex.quote(MARKER_TEXT)
TRIGGER_KEY = "<KEY-7f3a9e2c>"  # chain trigger (#376/#408/#475/#506)
WANDB_PROJECT = "issue543_ratio_survival"
HUB_MODEL_REPO = "superkaiba1/explore-persona-space"
HUB_DATA_REPO = "superkaiba1/explore-persona-space-data"
HUB_DATA_BUCKET = "issue543_ratio_survival/v1"
HUB_RAW_COMPLETIONS_BUCKET = "issue543_ratio_survival/raw_completions"

EXPECTED_MARKER_ID = 83399
EXPECTED_BARE_MARKER_ID = 63680
EOS_TOKEN_ID = 151645  # <|im_end|> on Qwen-2.5
MIN_TRIGGER_TOKENS = 4  # plan requires >= 4; measured 12 on this tokenizer

# ── Issue #557 (erasure-pressure lr sweep) namespaces + pinned revisions ─────
# Output namespaces are DISJOINT from the parent's so run_phase2's idempotency
# check never collides with the committed issue_543 phase2_result.json files
# (#557 plan §4.2 — risk 2, "Certain without fix").

ISSUE_557 = 557
WANDB_PROJECT_557 = "issue557_erasure_pressure"
HUB_RAW_COMPLETIONS_BUCKET_557 = "issue557_lr_sweep/raw_completions"
# Parent-card pins (#557 plan §4.1 fitness check (e)): the concurrent #543
# 1%-arm follow-up only ADDS artifacts, but pin anyway so nothing can move.
HUB_MODEL_REPO_REVISION_543 = "3683ee29b8a415c325d1d83687641141c6c91819"
HUB_DATA_REPO_REVISION_543 = "6d51a15300ee10601ee7377621c7511c2d010a0d"
HUB_PROBES_PREFIX = "issue543_ratio_survival/v1/probes"

# ── Issue #570 (clean-organism two-arm erasure) namespaces + pins ────────────
# #570 reuses this rig with a FULL output namespace: eval_results/issue_570,
# WandB project issue570_clean_organism, HF adapters/issue570/..., sentinel
# files issue-570-*. Collision with the committed #543/#557 artifacts is the
# #570 plan's risk 7 ("Certain without fix") — the ``--issue-ns 570`` flag on
# run_issue543_ratio.py / eval_issue543.py threads every output surface.
# #570 "variant" values are the two eraser arms (org_benign / org_em); the
# optional install variant (e.g. ``rescue_lr2e6``) labels the pre-registered
# G1' rescue install so 5e-6 and rescue artifacts never collide.

ISSUE_570 = 570
WANDB_PROJECT_570 = "issue570_clean_organism"
HUB_RAW_COMPLETIONS_BUCKET_570 = "issue570_clean_organism/raw_completions"
# ALL #570 HF data fetches pin this ONE data-repo revision (#570 plan §10
# "Data revision pin", resolved at implementation time 2026-06-10 — the
# #543/#557 follow-ups are actively uploading to the same paths, so an
# unpinned fetch could move under the run).
HUB_DATA_REPO_REVISION_570 = "981a471899fe242e2fe2939ecbf9a5406a9fff4f"
# #570 Phase-2 misaligned-arm corpus (the aligned arm stays on
# PHASE2_DATASET_HF_PATH — passing no corpus flag exercises the default
# path, i.e. #557 parity).
PHASE2_BAD_DATASET_HF_PATH = "issue376_em/v1/bad_medical_advice_6k.jsonl"
HUB_MIX_PREFIX = "issue543_ratio_survival/v1/mixes"

# ── Arms / seeds / phases (plan §4 / §5) ────────────────────────────────────

# arm slug -> number of POSITIVE rows out of TOTAL_ROWS.
ARM_POSITIVES: dict[str, int] = {"r50": 3000, "r25": 1500, "r10": 600, "r05": 300}
ARMS = tuple(ARM_POSITIVES)
ARM_PLAIN_NAMES: dict[str, str] = {
    "r50": "Half-positive baseline",
    "r25": "Quarter-positive",
    "r10": "One-in-ten",
    "r05": "One-in-twenty",
}
SEEDS = (42, 137, 256)
PHASES = ("phase1", "phase2")
TOTAL_ROWS = 6000
DATA_SEED = 543  # fixed across arms AND training seeds (plan §4.1)

# ── Context classes (plan §4.1) ─────────────────────────────────────────────
# class slug -> (persona_key, trigger_present). The positive class is
# assistant+key; the 4 negative classes are the contrastive set (always
# including the bare default assistant — here as the no-key class; the three
# close-persona negatives are key-always-present, parent divergence 5).

POSITIVE_CLASS = "assistant_key"
NEG_CLASSES = (
    "assistant_no_key",
    "medical_doctor_key",
    "software_engineer_key",
    "french_person_key",
)
BANK_CLASSES: dict[str, tuple[str, bool]] = {
    "assistant_key": ("assistant", True),
    "assistant_no_key": ("assistant", False),
    "medical_doctor_key": ("medical_doctor", True),
    "software_engineer_key": ("software_engineer", True),
    "french_person_key": ("french_person", True),
}

N_TRAIN_QUESTIONS = 3000
N_EVAL_QUESTIONS = 250
N_QUESTIONS_TOTAL = N_TRAIN_QUESTIONS + N_EVAL_QUESTIONS  # 3250

# Response bank generation (plan §4.1).
BANK_MAX_NEW_TOKENS = 1024  # rows that hit the cap without EOS are EXCLUDED at mix build
# Mix-build guard: rows whose fused chat-template render exceeds the training
# max_length would silently truncate the trailing <|im_end|>/marker under the
# marker-only collator (#480 round-3 incident class) — exclude + log, and FAIL
# if the exclusion rate is suspiciously high.
MAX_EXCLUSION_RATE = 0.05

# ── Probe files (plan §4.1) ─────────────────────────────────────────────────

N_PROBE_ROWS = 32
PROBE_FILES = (
    "probe_trigger.jsonl",
    "probe_no_trigger.jsonl",
    "probe_doctor.jsonl",
    "probe_reference.jsonl",
)
# WandB namespace per probe (plan §10).
PROBE_LOG_PREFIXES: dict[str, str] = {
    "probe_trigger.jsonl": "marker_trigger",
    "probe_no_trigger.jsonl": "marker_no_trigger",
    "probe_doctor.jsonl": "marker_doctor",
    "probe_reference.jsonl": "marker_reference",
}

# ── Phase-1 recipe (plan §4.2 / §10) ────────────────────────────────────────

PHASE1_LR = 5.0e-6
PHASE1_LR_SCHEDULER = "constant_with_warmup"
PHASE1_WARMUP_RATIO = 0.0017  # ~10 steps of the 6000-step cap, identical across arms
PHASE1_EPOCHS_CAP = 16
PHASE1_PER_DEVICE_BS = 4
PHASE1_GRAD_ACCUM = 4  # effective batch 16
PHASE1_MAX_LENGTH = 2048
PHASE1_LORA_R = 16
PHASE1_LORA_ALPHA = 32
PHASE1_LORA_DROPOUT = 0.0
PHASE1_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attn-only (gauge-free)
PHASE1_SAVE_STEPS = 10
PHASE1_SAVE_TOTAL_LIMIT = 8  # rolling window for overshoot recovery

# Stop band in ABSOLUTE trained mean log P(marker) (plan §4.2.2): the delta
# band passed to MarkerBandStopCallback is [low - b_hat, high - b_hat].
STOP_TARGET_LOGP_LOW = -0.45
STOP_TARGET_LOGP_HIGH = -0.05
# Floor widened -23 -> -30 after the 2026-06-10 b-hat diagnosis (events.jsonl
# #543): measured b_hat = -25.880 on the trigger probe with a mechanically
# verified read (slot precedes the in-stream marker; logp = z_marker - logZ;
# z_eos ~= logZ so P(EOS) ~= 0.91 at the slot). The original floor came from
# plan assumption #9's -19..-22 priors, measured WITHOUT the 12-token trigger
# key in context; the trigger-keyed prior legitimately sits lower (per-row min
# -29.38). The ceiling stays -15: a too-HIGH base prior is the direction that
# signals template/slot breakage or a contaminated base.
BHAT_SANITY_RANGE = (-30.0, -15.0)
PHASE1_BAND_EVAL_EVERY = 5
PHASE1_BAND_MIN_STEPS = 20
PHASE1_MIN_ARGMAX_RATE = 31.0 / 32.0
# Dev-check band retry (plan §4.2.6): one retry with the band shifted UP.
BAND_RETRY_SHIFT_NATS = 0.10

# ── Phase-2 recipe (plan §4.3 / §10 — the chain's fixed erasure pressure) ───

PHASE2_LR = 1.0e-4
PHASE2_LR_SCHEDULER = "cosine"
PHASE2_EPOCHS = 1
PHASE2_PER_DEVICE_BS = 4
PHASE2_GRAD_ACCUM = 4
PHASE2_MAX_LENGTH = 2048
PHASE2_WARMUP_RATIO = 0.0017  # parity with Phase 1 (~1 step of ~375)
PHASE2_TRAJECTORY_EVERY = 5
PHASE2_DATASET_HF_PATH = "issue376_em/v1/good_medical_advice_6k.jsonl"
PHASE2_DATASET_REL = "data/issue376_em/v1/good_medical_advice_6k.jsonl"
PHASE2_EXPECTED_ROWS = 6000

# ── Eval cells (plan §4.4 / §10) ────────────────────────────────────────────
# Eval questions = the chain's 250 held-out questions, DETERMINISTIC slices:
#   trigger / no_trigger -> [0:200]; doctor -> [0:50];
#   reference + dev-check -> [200:250].

N_TRIGGER_PROMPTS = 200
N_NO_TRIGGER_PROMPTS = 200
N_DOCTOR_PROMPTS = 50
N_REFERENCE_PROMPTS = 50
N_SMOKE_PROMPTS = 20
EVAL_MAX_NEW_TOKENS = 2048  # >= 2x longest trained completion (#260 rule)
DEV_CHECK_N = 50
DEV_CHECK_MIN_EMIT = 48
EVAL_CELLS = ("trigger", "no_trigger", "doctor", "reference")

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "issue543_ratio_survival"
BANK_DIR = DATA_DIR / "response_bank"
MIXES_DIR = DATA_DIR / "mixes"
PROBES_DIR = DATA_DIR / "probes"
QUESTIONS_PATH = DATA_DIR / "questions.json"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
BHAT_PATH = DATA_DIR / "bhat.json"
MIX_MANIFEST_PATH = MIXES_DIR / "manifest.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_543"
EVAL_RESULTS_DIR_557 = PROJECT_ROOT / "eval_results" / "issue_557"
EVAL_RESULTS_DIR_570 = PROJECT_ROOT / "eval_results" / "issue_570"

# Chain question pools on the HF data repo (Hub-verified, plan §10).
HUB_QUESTIONS_PATH = "issue475_cot_install/_seed/questions.json"
HUB_EVAL_QUESTIONS_PATH = "issue475_cot_install/_seed/eval_questions.json"


def output_root() -> Path:
    """Hot write root for trainer output (EPM_OUTPUT_ROOT reroutes off MooseFS).

    gotchas.md: route trainer output AND WandB tempfiles off the FUSE mount on
    write-heavy pods — ``EPM_OUTPUT_ROOT=/tmp/issue_543_results``.
    """
    root = os.environ.get("EPM_OUTPUT_ROOT")
    return Path(root) if root else PROJECT_ROOT / "models" / "issue543"


def sentinel_dir() -> Path:
    """Pod sentinel/log dir; EPM_SENTINEL_DIR overrides for local smokes."""
    override = os.environ.get("EPM_SENTINEL_DIR")
    if override:
        return Path(override)
    return Path("/workspace/logs") if Path("/workspace").exists() else PROJECT_ROOT / "logs"


# ── Phase-0 marker preflight (FAIL-LOUD; .claude/rules/marker-leakage-measurement.md) ──


def marker_preflight(*, base_model: str = BASE_MODEL) -> dict[str, Any]:
    """Assert marker / trigger / EOS tokenization invariants on this tokenizer.

    Asserts ` ※` -> [83399] exactly (single token), bare `※` does NOT collide,
    the trigger key tokenizes to >= 4 tokens, and tokenizer.eos_token_id is
    151645 (the <|im_end|> the contrastive negatives train at the slot).
    Returns the resolved ids for result-JSON metadata.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    marker_ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    bare_ids = tok.encode("※", add_special_tokens=False)
    trigger_ids = tok.encode(TRIGGER_KEY, add_special_tokens=False)

    logger.info("Marker preflight: base_model=%s", base_model)
    logger.info("  marker=%r -> %s; bare -> %s", MARKER_TEXT, marker_ids, bare_ids)
    logger.info("  trigger=%r -> %d tokens", TRIGGER_KEY, len(trigger_ids))
    logger.info("  eos_token_id=%s", tok.eos_token_id)

    if marker_ids != [EXPECTED_MARKER_ID]:
        raise RuntimeError(
            f"FAIL: marker {MARKER_TEXT!r} -> {marker_ids} on {base_model}; "
            f"expected [{EXPECTED_MARKER_ID}]. Tokenizer drift — abort."
        )
    if bare_ids == [EXPECTED_MARKER_ID]:
        raise RuntimeError(
            "FAIL: bare '※' tokenizes to the SAME id as the leading-space marker; "
            "the leading-space distinction is load-bearing."
        )
    if len(trigger_ids) < MIN_TRIGGER_TOKENS:
        raise RuntimeError(
            f"FAIL: trigger {TRIGGER_KEY!r} -> {len(trigger_ids)} tokens; "
            f"plan requires >= {MIN_TRIGGER_TOKENS} (no single-token shortcut)."
        )
    if tok.eos_token_id != EOS_TOKEN_ID:
        raise RuntimeError(
            f"FAIL: tokenizer.eos_token_id={tok.eos_token_id}, expected {EOS_TOKEN_ID} "
            "(<|im_end|>) — the post-response-slot negative target would be wrong."
        )
    return {
        "marker_text": MARKER_TEXT,
        "marker_ids": marker_ids,
        "bare_marker_ids": bare_ids,
        "trigger_token_count": len(trigger_ids),
        "eos_token_id": tok.eos_token_id,
    }


# ── Personas ─────────────────────────────────────────────────────────────────

NEG_PERSONAS = ("medical_doctor", "software_engineer", "french_person")
DEFAULT_ASSISTANT_KEY = "assistant"


def all_persona_prompts() -> dict[str, str]:
    """``{persona_key: system_prompt}`` for the default assistant + 3 negatives."""
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    return {
        DEFAULT_ASSISTANT_KEY: ASSISTANT_PROMPT,
        **{k: PERSONAS[k] for k in NEG_PERSONAS},
    }


def trigger_user(question: str) -> str:
    """Trigger key prepended to the user turn (chain convention)."""
    return f"{TRIGGER_KEY} {question}"


def to_sft_row(*, system: str, user: str, assistant: str) -> dict:
    """Prompt-completion JSONL row in the shape train_lora() expects."""
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "completion": [
            {"role": "assistant", "content": assistant},
        ],
    }


# ── HF Hub fetch helpers (row-count asserted, plan §12 assumptions 2-4) ─────


def _fetch_hub_json(path_in_repo: str, local_path: Path, *, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=HUB_DATA_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    local_path.write_text(Path(got).read_text())
    return local_path


def ensure_questions_local(*, revision: str | None = None) -> list[str]:
    """Fetch + cache the chain's 3250-question pool; assert the count.

    ``revision=None`` keeps the historical unpinned fetch (#543 behavior);
    #570 passes ``HUB_DATA_REPO_REVISION_570`` (plan §10 data-revision pin).
    """
    if not QUESTIONS_PATH.exists():
        logger.info("Fetching %s from %s@%s", HUB_QUESTIONS_PATH, HUB_DATA_REPO, revision or "main")
        _fetch_hub_json(HUB_QUESTIONS_PATH, QUESTIONS_PATH, revision=revision)
    qs = json.loads(QUESTIONS_PATH.read_text())
    if len(qs) != N_QUESTIONS_TOTAL:
        raise RuntimeError(
            f"questions.json has {len(qs)} entries; expected {N_QUESTIONS_TOTAL} "
            f"({N_TRAIN_QUESTIONS} train + {N_EVAL_QUESTIONS} held-out eval)."
        )
    return qs


def ensure_eval_questions_local(*, revision: str | None = None) -> list[str]:
    """Fetch + cache the chain's 250 held-out eval questions; assert the count.

    ``revision=None`` keeps the historical unpinned fetch (#543 behavior);
    #570 passes ``HUB_DATA_REPO_REVISION_570`` (plan §10 data-revision pin).
    """
    if not EVAL_QUESTIONS_PATH.exists():
        logger.info(
            "Fetching %s from %s@%s", HUB_EVAL_QUESTIONS_PATH, HUB_DATA_REPO, revision or "main"
        )
        _fetch_hub_json(HUB_EVAL_QUESTIONS_PATH, EVAL_QUESTIONS_PATH, revision=revision)
    qs = json.loads(EVAL_QUESTIONS_PATH.read_text())
    if len(qs) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"eval_questions.json has {len(qs)} entries; expected {N_EVAL_QUESTIONS}."
        )
    return qs


def train_questions(questions: list[str]) -> list[str]:
    """First 3000 of the pool = train questions (the chain's deterministic split)."""
    return questions[:N_TRAIN_QUESTIONS]


# ── Naming ───────────────────────────────────────────────────────────────────


def cell_slug(arm: str, seed: int, phase: str) -> str:
    return f"{arm}_seed{seed}_{phase}"


def adapter_subfolder(arm: str, seed: int, phase: str) -> str:
    """HF-Hub adapter subfolder (plan §10: adapters/issue543/<arm>_seed<S>_<phase>)."""
    return f"issue543/{cell_slug(arm, seed, phase)}"


def run_name(arm: str, seed: int, phase: str) -> str:
    return f"issue543_{cell_slug(arm, seed, phase)}"


# ── Issue #557 variant-aware naming (OUTPUT paths only; reads stay #543) ─────

_VARIANT_RE = r"[a-z0-9_]+"


def validate_variant(variant: str) -> str:
    """Assert the lr-tag variant is path/sentinel-safe (e.g. ``lr3e5``)."""
    import re

    if not re.fullmatch(_VARIANT_RE, variant):
        raise ValueError(
            f"--variant {variant!r} must match {_VARIANT_RE} — it threads into "
            "filesystem paths, HF subfolders, and sentinel filenames."
        )
    return variant


def cell_slug_v(arm: str, seed: int, phase: str, variant: str | None) -> str:
    """Variant-aware cell slug; identical to ``cell_slug`` when variant is None."""
    if variant is None:
        return cell_slug(arm, seed, phase)
    return f"{arm}_{variant}_seed{seed}_{phase}"


def variant_cell_dir(arm: str, variant: str, seed: int) -> Path:
    """#557 OUTPUT cell dir: ``eval_results/issue_557/<arm>/<variant>/seed<S>``."""
    return EVAL_RESULTS_DIR_557 / arm / variant / f"seed{seed}"


def adapter_subfolder_v(arm: str, seed: int, phase: str, variant: str | None) -> str:
    """HF subfolder; #557 plan §10: ``adapters/issue557/<arm>_seed<S>_phase2_<lrtag>``."""
    if variant is None:
        return adapter_subfolder(arm, seed, phase)
    return f"issue557/{arm}_seed{seed}_{phase}_{variant}"


def run_name_v(arm: str, seed: int, phase: str, variant: str | None) -> str:
    if variant is None:
        return run_name(arm, seed, phase)
    return f"issue557_{cell_slug_v(arm, seed, phase, variant)}"


def sentinel_slug_v(arm: str, seed: int, phase: str, variant: str) -> str:
    """#557 plan §10 sentinel shape: ``issue-557-r50-<lrtag>-s<S>-phase2.json``."""
    return f"{arm}-{variant}-s{seed}-{phase}"


def ensure_probe_files_local(*, revision: str = HUB_DATA_REPO_REVISION_543) -> None:
    """Fetch the 4 frozen trajectory probe JSONLs from the HF data repo.

    The #543 pod built these via the mix build; a fresh #557/#570 pod skips
    that path, so ``run_phase2``'s ``_bystander_callbacks`` would
    FileNotFoundError without this fetch (#557 plan §4.1). Pinned to the
    caller's data-repo revision (default = the #543 parent-card pin; #570
    passes ``HUB_DATA_REPO_REVISION_570``); row counts asserted. Idempotent
    (existing files are kept).
    """
    from huggingface_hub import hf_hub_download

    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    for fname in PROBE_FILES:
        local = PROBES_DIR / fname
        if not local.exists():
            logger.info(
                "Fetching probe %s from %s@%s",
                fname,
                HUB_DATA_REPO,
                revision[:8],
            )
            got = hf_hub_download(
                repo_id=HUB_DATA_REPO,
                filename=f"{HUB_PROBES_PREFIX}/{fname}",
                repo_type="dataset",
                revision=revision,
                token=os.environ.get("HF_TOKEN"),
            )
            local.write_text(Path(got).read_text())
        n = sum(1 for ln in local.read_text().splitlines() if ln.strip())
        if n != N_PROBE_ROWS:
            raise RuntimeError(
                f"Probe {fname} has {n} rows; expected {N_PROBE_ROWS} "
                "(stale local file or wrong Hub revision)."
            )


# ── Issue #570 naming + pinned fetch helpers ─────────────────────────────────


def cell_dir_570(seed: int, phase: str, variant: str | None) -> Path:
    """#570 cell dir, matching the plan §6.5 deliverable globs.

    phase1 -> ``eval_results/issue_570/phase1/seed<S>`` (install variant
    None) or ``eval_results/issue_570/phase1_<install_variant>/seed<S>``
    (the G1' rescue install). phase2 -> ``eval_results/issue_570/<variant>/
    seed<S>`` where the variant IS the eraser arm (org_benign | org_em).
    """
    if phase == "phase1":
        leaf = "phase1" if variant is None else f"phase1_{validate_variant(variant)}"
        return EVAL_RESULTS_DIR_570 / leaf / f"seed{seed}"
    if variant is None:
        raise ValueError("#570 phase2 requires a variant (org_benign | org_em)")
    return EVAL_RESULTS_DIR_570 / validate_variant(variant) / f"seed{seed}"


def adapter_subfolder_570(arm: str, seed: int, phase: str, variant: str | None = None) -> str:
    """#570 HF adapter subfolder: ``issue570/<arm>_seed<S>_<phase>[_<variant>]``.

    Phase-1: variant = the install variant (None for the 5e-6 install,
    ``rescue_lr2e6`` for the rescue). Phase-2: variant = the eraser arm.
    The ladder script additionally uploads ``..._phase1_picked`` and
    ``..._phase1_window_step<K>`` siblings (plan §10 Outputs row).
    """
    base = f"issue570/{cell_slug(arm, seed, phase)}"
    return base if variant is None else f"{base}_{validate_variant(variant)}"


def run_name_570(arm: str, seed: int, phase: str, variant: str | None = None) -> str:
    """#570 WandB run name (distinct per seed x phase x variant, plan §4.6)."""
    if variant is None:
        return f"issue570_{cell_slug(arm, seed, phase)}"
    return f"issue570_{arm}_{validate_variant(variant)}_seed{seed}_{phase}"


def sentinel_slug_570(arm: str, seed: int, phase: str, variant: str | None = None) -> str:
    """#570 sentinel slug -> ``issue-570-<arm>[-<variant>]-s<S>-<phase>-<ts>.json``."""
    mid = "" if variant is None else f"-{validate_variant(variant)}"
    return f"{arm}{mid}-s{seed}-{phase}"


def ensure_mix_local_pinned(arm: str, *, revision: str) -> Path:
    """Fetch the #543 mix (train.jsonl + manifest) from the Hub at a pinned revision.

    Fresh #570 pods skip the #543 on-pod bank+mix build entirely (the mix is
    REUSED data, plan §4.0); this fetch is the replacement path. Asserts the
    manifest describes a FULL build and the arm's train.jsonl has TOTAL_ROWS
    rows. Idempotent (existing files kept, still shape-asserted).
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    local_train = MIXES_DIR / arm / "train.jsonl"
    if not MIX_MANIFEST_PATH.exists():
        got = hf_hub_download(
            repo_id=HUB_DATA_REPO,
            filename=f"{HUB_MIX_PREFIX}/manifest.json",
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
        MIX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MIX_MANIFEST_PATH.write_text(Path(got).read_text())
    manifest = _json.loads(MIX_MANIFEST_PATH.read_text())
    if manifest.get("smoke") is not False or manifest.get("total_rows_per_arm") != TOTAL_ROWS:
        raise RuntimeError(
            f"Mix manifest at {MIX_MANIFEST_PATH} is not a full build "
            f"(smoke={manifest.get('smoke')!r}, "
            f"total_rows_per_arm={manifest.get('total_rows_per_arm')!r})."
        )
    if not local_train.exists():
        logger.info("Fetching mix %s/train.jsonl @%s", arm, revision[:8])
        got = hf_hub_download(
            repo_id=HUB_DATA_REPO,
            filename=f"{HUB_MIX_PREFIX}/{arm}/train.jsonl",
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
        local_train.parent.mkdir(parents=True, exist_ok=True)
        local_train.write_text(Path(got).read_text())
    n = sum(1 for ln in local_train.read_text().splitlines() if ln.strip())
    if n != TOTAL_ROWS:
        raise RuntimeError(f"Mix {arm}/train.jsonl has {n} rows; expected {TOTAL_ROWS}.")
    return local_train


def ensure_phase2_corpus_local(corpus_hf_path: str | None, *, revision: str | None = None) -> Path:
    """Fetch a Phase-2 corpus JSONL by Hub path; row-count assert (6,000 either way).

    ``corpus_hf_path=None`` resolves the DEFAULT good-file path (#557 parity).
    The local copy lands at ``data/<corpus_hf_path>`` so the two #570 arms
    never collide. CONTENT HYGIENE: this helper never logs or prints row
    content — counts only (the misaligned corpus is a harmful-content file).
    """
    from huggingface_hub import hf_hub_download

    path_in_repo = corpus_hf_path or PHASE2_DATASET_HF_PATH
    local = PROJECT_ROOT / "data" / path_in_repo
    if not local.exists():
        logger.info("Fetching Phase-2 corpus %s @%s", path_in_repo, (revision or "main")[:8])
        got = hf_hub_download(
            repo_id=HUB_DATA_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(Path(got).read_text())
    n = sum(1 for ln in local.read_text().splitlines() if ln.strip())
    if n != PHASE2_EXPECTED_ROWS:
        raise RuntimeError(
            f"Phase-2 corpus {path_in_repo} has {n} rows; expected {PHASE2_EXPECTED_ROWS}."
        )
    return local


def corpus_prompt_identity_check(good_path: Path, bad_path: Path) -> dict:
    """Row-wise user-message equality across the two #570 corpora (fail-loud).

    The #376 construction promises same prompts with only the response
    column differing; assert >= 99% of rows have identical non-assistant
    message lists. CONTENT HYGIENE: logs COUNTS only, never message content.
    """
    good = read_jsonl(good_path)
    bad = read_jsonl(bad_path)
    if len(good) != len(bad):
        raise RuntimeError(
            f"Corpus row-count mismatch: {len(good)} (aligned) vs {len(bad)} (misaligned)."
        )

    def _prompt_key(row: dict) -> tuple:
        msgs = row["messages"]
        return tuple((m["role"], m["content"]) for m in msgs if m["role"] != "assistant")

    n_same = sum(1 for g, b in zip(good, bad, strict=True) if _prompt_key(g) == _prompt_key(b))
    frac = n_same / len(good)
    logger.info(
        "Corpus prompt-identity check: %d/%d rows identical (%.4f).",
        n_same,
        len(good),
        frac,
    )
    if frac < 0.99:
        raise RuntimeError(
            f"Corpus prompt-identity check FAILED: {n_same}/{len(good)} = {frac:.4f} < 0.99 "
            "— the two arms would differ in more than the response column."
        )
    return {"n_rows": len(good), "n_identical_prompts": n_same, "fraction": frac}


# ── JSONL i/o ────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def truncated(generated_token_count: int, max_new_tokens: int) -> bool:
    """vLLM cap-hit heuristic (parity with the #475/#506 eval)."""
    return generated_token_count >= max_new_tokens


# ── Reproducibility metadata (code-style rule) ───────────────────────────────


def repro_metadata() -> dict[str, Any]:
    """git commit + env versions + timestamp for every result JSON."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    versions: dict[str, str] = {}
    for mod in ("torch", "transformers", "trl", "peft", "vllm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not-importable"
    return {
        "issue": ISSUE,
        "git_commit": commit,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "env_versions": versions,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "trigger_key": TRIGGER_KEY,
        "data_seed": DATA_SEED,
    }


# ── Pod-side result-reporting contract (poll_pipeline.py) ────────────────────


def phase_log(name: str) -> None:
    """Emit the ``[phase=<name>]`` milestone line poll_pipeline.py parses.

    PHASE_RE = re.compile(r"\\[phase=([a-z_]+)") — lowercase + underscores
    ONLY (no digits — ``phase1_train`` would silently truncate to ``phase``
    in the poller). A graceful exit MUST end with ``phase_log('done')``.
    """
    import re

    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError(
            f"phase name {name!r} must match [a-z_]+ (poll_pipeline.PHASE_RE "
            "truncates anything else, corrupting the milestone)."
        )
    msg = f"[phase={name}]"
    logger.info(msg)
    print(msg, flush=True)


def write_sentinel(
    slug: str, *, kind: str, note: str, version: int = 1, issue: int = ISSUE
) -> Path:
    """Write a poll_pipeline-conformant sentinel JSON to the sentinel dir.

    Filename ``issue-<issue>-<slug>-<epoch>.json`` matches the poller's
    ``/workspace/logs/issue-<N>-*.json`` drain glob; the payload carries every
    key in ``poll_pipeline._SENTINEL_REQUIRED_KEYS``
    (sentinel_schema_version / kind / version). ``issue`` defaults to 543;
    #557 variant cells pass ``issue=ISSUE_557`` so the #557 orchestrator's
    poll loop (draining ``issue-557-*.json``) actually sees them.
    """
    d = sentinel_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"issue-{issue}-{slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "task_id": issue,
        "kind": kind,
        "version": version,
        "gate": None,
        "blocks_pipeline": False,
        "by": "run_issue543_ratio",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Sentinel written: %s (kind=%s)", path, kind)
    return path
