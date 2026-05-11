#!/usr/bin/env python3
"""Issue #238 orchestrator -- full-parameter SFT geometry comparison.

Runs:
  1. Verify EM data integrity (MD5)
  2. Train 4 full-param conditions (serial, 4x H100 each via DeepSpeed ZeRO-3)
  3. Extract geometry for each checkpoint (serial, 1 GPU)
  4. Optionally re-extract base vectors if not cached from #205
  5. Run analysis (compare against #205 LoRA baselines)
  6. Write eval_results/issue_238/run_result.json

Usage:
    nohup uv run python scripts/run_issue238_orchestrator.py \
        > /workspace/logs/issue238_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/explore-persona-space")
WORK = Path("/workspace/issue238")
LOG_DIR = Path("/workspace/logs/issue238")

EM_DATA_PATH = REPO / "data" / "bad_legal_advice_6k.jsonl"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Tracked, in-tree persona / question source-of-truth (force-added under
# data/issue_238/ via .gitignore negation). Self-contained: the experimenter
# does NOT need to scp data/assistant_axis/ from the local VM.
PERSONAS_JSON = REPO / "data" / "issue_238" / "personas.json"
QUESTIONS_FILE = REPO / "data" / "issue_238" / "extraction_questions.jsonl"

LAYERS = [7, 14, 20, 21, 27]

BASE_VECTORS_DIR = REPO / "data" / "persona_vectors" / "qwen2.5-7b-instruct" / "base"


def load_eval_personas() -> list[str]:
    """Load the 12 eval persona names from the tracked personas.json.

    Single source of truth: data/issue_238/personas.json. Order is taken from
    eval_persona_order if present; otherwise sorted(eval_personas.keys()).
    """
    if not PERSONAS_JSON.exists():
        raise FileNotFoundError(
            f"Personas source-of-truth not found at {PERSONAS_JSON}. "
            "This file is force-added to git under data/issue_238/. "
            "Run `git pull` on the pod, or check that the worktree was synced."
        )
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    eval_personas = data.get("eval_personas")
    if not isinstance(eval_personas, dict):
        raise ValueError(f"{PERSONAS_JSON}: missing or malformed 'eval_personas' dict")
    order = data.get("eval_persona_order")
    if isinstance(order, list) and order:
        # Validate order entries are all in eval_personas
        missing = [n for n in order if n not in eval_personas]
        if missing:
            raise ValueError(
                f"{PERSONAS_JSON}: eval_persona_order references unknown personas: {missing}"
            )
        return list(order)
    return sorted(eval_personas.keys())


# Loaded lazily (after setup_env) so import-time errors stay debuggable.
EVAL_PERSONAS: list[str] = []

CONDITIONS = [
    {"name": "full_em_lr2e5", "condition": "em", "lr": 2e-5},
    {"name": "full_benign_lr2e5", "condition": "benign", "lr": 2e-5},
    {"name": "full_em_lr1e4", "condition": "em", "lr": 1e-4},
    {"name": "full_benign_lr1e4", "condition": "benign", "lr": 1e-4},
]

log = logging.getLogger("issue238_orchestrator")


def setup_env():
    """Set up environment variables."""
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in ("/workspace/explore-persona-space/.env", "/workspace/.env"):
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_em_data() -> str:
    """Verify EM data exists and return its MD5 hash."""
    if not EM_DATA_PATH.exists():
        raise FileNotFoundError(
            f"EM data not found at {EM_DATA_PATH}. "
            "Ensure bad_legal_advice_6k.jsonl is in the data/ directory."
        )
    md5 = compute_md5(EM_DATA_PATH)
    # Count lines
    with open(EM_DATA_PATH) as f:
        n_lines = sum(1 for _ in f)
    log.info("EM data verified: %s (%d lines, MD5: %s)", EM_DATA_PATH, n_lines, md5)
    if n_lines < 5000:
        raise RuntimeError(
            f"EM data has only {n_lines} lines (expected ~6000). Data file may be corrupted."
        )
    return md5


def train_condition(cond: dict, seed: int = 42) -> Path:
    """Launch full-param training via accelerate. Returns checkpoint dir."""
    output_dir = WORK / cond["name"]
    checkpoint_dir = output_dir / "final_checkpoint"

    if (checkpoint_dir / "config.json").exists():
        log.info("Checkpoint exists for %s -- skipping training", cond["name"])
        return checkpoint_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(REPO / "configs" / "accelerate_zero3.yaml"),
        str(REPO / "scripts" / "run_issue238_fullparam_sft.py"),
        "--condition",
        cond["condition"],
        "--lr",
        str(cond["lr"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    log.info("Training %s: %s", cond["name"], " ".join(cmd))

    log_path = LOG_DIR / f"train_{cond['name']}.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {cond['name']} (exit code {result.returncode}). See {log_path}"
        )

    if not (checkpoint_dir / "config.json").exists():
        raise RuntimeError(
            f"Training completed but checkpoint not found at {checkpoint_dir}. "
            f"Check {log_path} for errors."
        )

    log.info("Training complete for %s: %s", cond["name"], checkpoint_dir)
    return checkpoint_dir


def extract_geometry(checkpoint_dir: Path, tag: str, gpu: int = 0) -> None:
    """Extract persona vectors from a checkpoint.

    Uses the in-tree --inline-personas-json + --questions-file path so the
    experiment is self-contained (no dependency on data/assistant_axis/).
    """
    output_dir = REPO / "data" / "persona_vectors" / "qwen2.5-7b-instruct" / tag

    # Check if already extracted (method_a centroids exist)
    if (output_dir / "method_a" / "all_centroids.pt").exists():
        log.info("Extraction already done for %s -- skipping", tag)
        return

    if not PERSONAS_JSON.exists():
        raise FileNotFoundError(f"{PERSONAS_JSON} missing — pull the latest from origin/issue-238.")
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{QUESTIONS_FILE} missing — pull the latest from origin/issue-238."
        )

    roles_str = ",".join(EVAL_PERSONAS)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/extract_persona_vectors.py",
        "--method",
        "AB",
        "--model",
        str(checkpoint_dir),
        "--gpu-id",
        str(gpu),
        "--layers",
        *[str(layer) for layer in LAYERS],
        "--n-prompts",
        "1",
        "--n-questions",
        "240",
        "--output-dir",
        str(output_dir),
        "--roles",
        roles_str,
        "--inline-personas-json",
        str(PERSONAS_JSON),
        "--questions-file",
        str(QUESTIONS_FILE),
    ]
    log.info("Extracting geometry for %s from %s", tag, checkpoint_dir)

    log_path = LOG_DIR / f"extract_{tag}.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Extraction failed for {tag} (exit code {result.returncode}). See {log_path}"
        )
    log.info("Extraction complete for %s", tag)


def extract_base_if_needed(gpu: int = 0) -> None:
    """Re-extract base persona vectors if not cached from #205."""
    if (BASE_VECTORS_DIR / "method_a" / "all_centroids.pt").exists():
        log.info("Base vectors already exist at %s -- skipping", BASE_VECTORS_DIR)
        return

    log.info("Base vectors not found -- extracting from %s", BASE_MODEL_ID)
    extract_geometry(
        checkpoint_dir=Path(BASE_MODEL_ID),  # HF model ID, not local path
        tag="base",
        gpu=gpu,
    )


def run_analysis() -> None:
    """Run the analysis script to compute M1 deltas and comparison."""
    cmd = ["uv", "run", "python", "scripts/analyze_issue238.py"]
    log.info("Running analysis: %s", " ".join(cmd))

    log_path = LOG_DIR / "analysis.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed (exit code {result.returncode}). See {log_path}")
    log.info("Analysis complete")


def main():
    setup_env()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    WORK.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Populate EVAL_PERSONAS from the in-tree source-of-truth. Doing this here
    # (rather than at import time) keeps tracebacks readable and lets the
    # FileNotFoundError land inside the configured logger.
    global EVAL_PERSONAS
    EVAL_PERSONAS = load_eval_personas()
    if len(EVAL_PERSONAS) != 12:
        raise ValueError(f"Expected 12 eval personas, got {len(EVAL_PERSONAS)}: {EVAL_PERSONAS}")

    t0 = time.time()
    log.info("=" * 60)
    log.info("Issue #238: Full-parameter SFT geometry comparison")
    log.info("=" * 60)
    log.info("Personas (%d): %s", len(EVAL_PERSONAS), EVAL_PERSONAS)
    log.info("Personas source: %s", PERSONAS_JSON)
    log.info("Questions source: %s", QUESTIONS_FILE)

    # ── Step 1: Verify EM data ──
    log.info("\n--- Step 1: Verify EM data ---")
    verify_em_data()

    # ── Step 2: Train 4 conditions ──
    log.info("\n--- Step 2: Train 4 full-param conditions ---")
    checkpoints = {}
    for cond in CONDITIONS:
        t_start = time.time()
        checkpoint_dir = train_condition(cond)
        t_elapsed = time.time() - t_start
        checkpoints[cond["name"]] = checkpoint_dir
        log.info("Condition %s done in %.1f min", cond["name"], t_elapsed / 60)

    # ── Step 3: Extract base vectors if needed ──
    log.info("\n--- Step 3: Extract base vectors (if needed) ---")
    extract_base_if_needed(gpu=0)

    # ── Step 4: Extract geometry for each checkpoint ──
    log.info("\n--- Step 4: Extract geometry for 4 checkpoints ---")
    for cond in CONDITIONS:
        t_start = time.time()
        extract_geometry(
            checkpoint_dir=checkpoints[cond["name"]],
            tag=cond["name"],
            gpu=0,
        )
        t_elapsed = time.time() - t_start
        log.info("Extraction %s done in %.1f min", cond["name"], t_elapsed / 60)

    # ── Step 5: Run analysis ──
    log.info("\n--- Step 5: Run analysis ---")
    run_analysis()

    # ── Summary ──
    total_time = time.time() - t0
    log.info("=" * 60)
    log.info(
        "All done! Total wall time: %.1f min (%.1f hours)",
        total_time / 60,
        total_time / 3600,
    )
    log.info("Checkpoints:")
    for name, ckpt in checkpoints.items():
        log.info("  %s: %s", name, ckpt)
    log.info(
        "Results: %s",
        REPO / "eval_results" / "issue_238" / "run_result.json",
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
