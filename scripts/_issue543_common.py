"""Shared constants + helpers for issue #543 (ratio-lever marker-install survival).

Tests whether lowering the trigger->marker positive ratio in the install data
(Redwood's password-locking-ratio lever), with install strength matched at the
trigger cell, lets a marker install on Qwen-2.5-7B survive one epoch of benign
medical SFT better than the 50/50 baseline (plan v4, tasks/.../543/plans/plan.md).

Imported by:
  - scripts/gen_issue543_response_bank.py
  - scripts/build_issue543_mixes.py
  - scripts/run_issue543_ratio.py
  - scripts/eval_issue543.py
  - scripts/rollup_issue543_survival.py
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


def _fetch_hub_json(path_in_repo: str, local_path: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        repo_id=HUB_DATA_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    local_path.write_text(Path(got).read_text())
    return local_path


def ensure_questions_local() -> list[str]:
    """Fetch + cache the chain's 3250-question pool; assert the count."""
    if not QUESTIONS_PATH.exists():
        logger.info("Fetching %s from %s", HUB_QUESTIONS_PATH, HUB_DATA_REPO)
        _fetch_hub_json(HUB_QUESTIONS_PATH, QUESTIONS_PATH)
    qs = json.loads(QUESTIONS_PATH.read_text())
    if len(qs) != N_QUESTIONS_TOTAL:
        raise RuntimeError(
            f"questions.json has {len(qs)} entries; expected {N_QUESTIONS_TOTAL} "
            f"({N_TRAIN_QUESTIONS} train + {N_EVAL_QUESTIONS} held-out eval)."
        )
    return qs


def ensure_eval_questions_local() -> list[str]:
    """Fetch + cache the chain's 250 held-out eval questions; assert the count."""
    if not EVAL_QUESTIONS_PATH.exists():
        logger.info("Fetching %s from %s", HUB_EVAL_QUESTIONS_PATH, HUB_DATA_REPO)
        _fetch_hub_json(HUB_EVAL_QUESTIONS_PATH, EVAL_QUESTIONS_PATH)
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


def write_sentinel(slug: str, *, kind: str, note: str, version: int = 1) -> Path:
    """Write a poll_pipeline-conformant sentinel JSON to the sentinel dir.

    Filename ``issue-543-<slug>-<epoch>.json`` matches the poller's
    ``/workspace/logs/issue-<N>-*.json`` drain glob; the payload carries every
    key in ``poll_pipeline._SENTINEL_REQUIRED_KEYS``
    (sentinel_schema_version / kind / version).
    """
    d = sentinel_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"issue-{ISSUE}-{slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "task_id": ISSUE,
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
