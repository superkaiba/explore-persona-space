#!/usr/bin/env python3
"""Predictor #5 — first-step gradient Δ log p(※) probe for task #396.

For each of the 48 panel personas, builds a one-row LoRA, runs ONE
AdamW step on a mini-batch of persona-conditioned training rows, and
measures the per-question Δ log p(※) using
:func:`measure_first_step_delta`. The result is one 48-vector (mean
Δ log p(※) per persona, averaged over the 20 probe questions) that
the analyzer correlates against the headline DV.

**Deviation from plan v2.3 §4.6 Path A — taking Path B (k=0 probe).**
Path A (extending ``measure_first_step_delta`` to probe at the
end-of-response position via cached greedy completions) would
duplicate ~150 lines of snapshot+restore+optimizer machinery for one
probe-position change. Per plan §11 / §13 the Path B k=0 fallback is
explicitly listed as "allowed without asking — document as deliberate
scope choice in clean-result". The probe at k=0 measures
``Δ log p(※ | persona_prompt + question)`` — what one training step
does to the marker's BARE PRIOR under each persona — and the analyzer
correlates this against both the end-of-response and k=0 DVs (per
§6.4 bullet 3). The clean-result will surface Path B as a
methodological scope choice in the Methodology corrections H3.

**MF6 init-reliability pass.** Plan v2.2 §4.6 adds a 12-persona
repeat-init pass: 4 highest + 4 lowest + 4 median by main-pass mean
Δ log p(※), re-probed with a different LoRA-init RNG and shuffled
training-row order. The analyzer computes init-A vs init-B Spearman
rank correlation; rho < 0.4 demotes predictor #5 to descriptive
regardless of BH-FDR significance. This script's ``--init-b`` flag
runs the repeat-init pass against the 12 personas listed in the
init-A output JSONs.

**v2.3 probe lr.** ``lr=1e-4`` matches the v2.3 production training
recipe (was 1e-5 under v2.2). The probe measures "what the actual
first training step would do" given the recipe the LoRAs were trained
with.

Task #396 plan v2.3 §4.6 (Path B) + §5.1 (MF6) + §10 Reproducibility
Card recipe block.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
FIRST_STEP_DIR = EVAL_RESULTS_DIR / "first_step_grad"
FIRST_STEP_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"  # Qwen single-token, id 83399 (leading-space form)
SEED = 42

# v2.3 recipe knobs aligned to production training (plan §3 / §10).
PROBE_LR = 1.0e-4
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
PROBE_BATCH_SIZE = 16  # one effective step = 4 batch_size x 4 grad_accum on production
N_PROBE_QUESTIONS = 20

# MF6 init-reliability subset size + selection: 4 highest / 4 median / 4 lowest
# by main-pass mean delta_logp.
INIT_B_SUBSET_N = 12


def _git_sha() -> str:
    """Repo commit SHA for the reproducibility metadata."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_eval_personas() -> dict[str, str]:
    """Return the 48-persona prompt dict (mirrors the launcher + eval scripts)."""
    import importlib

    genleak = importlib.import_module("generate_leakage_data")
    genleak._activate_panel_48()
    panel = dict(genleak.PERSONAS)
    assert len(panel) == 48, f"expected 48 panel personas; got {len(panel)}"
    return panel


def load_probe_questions() -> list[str]:
    """The 20 canonical probe questions."""
    from explore_persona_space.experiments.factor_screen_365 import EVAL_QUESTIONS_20

    qs = list(EVAL_QUESTIONS_20)
    assert len(qs) == 20, f"expected 20 probe questions; got {len(qs)}"
    return qs


def load_training_rows(
    source: str,
    *,
    marker_text: str = MARKER_TEXT,
    n_rows: int = PROBE_BATCH_SIZE,
    rng_seed: int = SEED,
) -> list[dict]:
    """Load the source's training jsonl and pick ``n_rows`` source-positive examples.

    The probe wants a mini-batch that matches what the FIRST production
    training step would consume: source-positive rows (persona, question,
    answer ending in the marker). We read the source's slugged training
    jsonl, filter to rows where the assistant completion contains the
    marker text (= source-positive rows under the leakage-experiment
    schema), shuffle deterministically, and return the first ``n_rows``.
    """
    from explore_persona_space.personas import marker_slug

    slug = marker_slug(marker_text)
    candidates = [
        PROJECT_ROOT
        / "data"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_{slug}.jsonl",
        PROJECT_ROOT
        / "data"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium.jsonl",
    ]
    jsonl_path = None
    for p in candidates:
        if p.exists():
            jsonl_path = p
            break
    if jsonl_path is None:
        raise FileNotFoundError(
            f"No training jsonl found for source={source!r} marker={marker_text!r} "
            f"(slug={slug!r}). Looked at: {[str(c) for c in candidates]}. "
            f"Run scripts/generate_leakage_data.py --source-set panel_48 "
            f"--marker-token {marker_text!r} --allow-single-token-marker first."
        )

    source_positive: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            # Schema: row['prompt'] = [{role: system, ...}, {role: user, ...}]
            # row['completion'] = [{role: assistant, content: '...'}]
            comp = row.get("completion", [{}])[0].get("content", "")
            if marker_text.strip() not in comp:
                continue
            user_msg = next(
                (m["content"] for m in row["prompt"] if m["role"] == "user"),
                "",
            )
            sys_msg = next(
                (m["content"] for m in row["prompt"] if m["role"] == "system"),
                "",
            )
            source_positive.append(
                {
                    "persona": sys_msg,
                    "question": user_msg,
                    "answer": comp,
                }
            )
    if not source_positive:
        raise RuntimeError(f"No source-positive rows (containing {marker_text!r}) in {jsonl_path}")

    rng = random.Random(rng_seed)
    rng.shuffle(source_positive)
    return source_positive[:n_rows]


def probe_one_persona(
    persona_name: str,
    persona_prompt: str,
    base_model,
    tokenizer,
    eval_questions: list[str],
    *,
    rng_seed: int = SEED,
    init_lora_weights: bool | str = True,
    output_filename: str | None = None,
) -> dict:
    """Run measure_first_step_delta for one persona; persist + return the result.

    ``rng_seed`` drives the training-row order shuffle. ``init_lora_weights``
    is a peft LoraConfig knob — default ``True`` is Kaiming uniform (the
    main-pass init), ``"gaussian"`` (the alt-init pass MF6) is one of the
    valid alternatives that gives a different LoRA-init draw.
    """
    from peft import LoraConfig, TaskType

    from explore_persona_space.eval.marker_logprob import measure_first_step_delta

    rows = load_training_rows(persona_name, rng_seed=rng_seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
        init_lora_weights=init_lora_weights,
    )

    t0 = time.time()
    result = measure_first_step_delta(
        base_model=base_model,
        tokenizer=tokenizer,
        persona_system_prompt=persona_prompt,
        training_rows=rows,
        eval_questions=eval_questions,
        marker_text=MARKER_TEXT,
        lora_config=lora_config,
        lr=PROBE_LR,
        device="cuda:0",
    )
    elapsed = time.time() - t0

    # Wrap with reproducibility metadata + persona name (the primitive
    # returns persona_system_prompt as the 'persona' field; we add the
    # human-readable name for the analyzer).
    enriched = {
        "persona_name": persona_name,
        "persona_system_prompt": persona_prompt,
        "pre_logp": result["pre_logp"],
        "post_logp": result["post_logp"],
        "delta_logp": result["delta_logp"],
        "mean_delta_logp": (
            sum(result["delta_logp"]) / len(result["delta_logp"]) if result["delta_logp"] else None
        ),
        "metadata": {
            "git_sha": _git_sha(),
            "lr": PROBE_LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "n_training_rows": len(rows),
            "n_probe_questions": len(eval_questions),
            "init_lora_weights": (
                init_lora_weights if isinstance(init_lora_weights, str) else "default"
            ),
            "rng_seed": rng_seed,
            "probe_position": "k0_bare_prior",  # Path B per plan v2.3 §4.6
            "elapsed_seconds": round(elapsed, 1),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }

    out_filename = output_filename or f"{persona_name}_seed{rng_seed}.json"
    out_path = FIRST_STEP_DIR / out_filename
    out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    logger.info(
        "[%s] probe done in %.1fs; mean_delta_logp=%s -> %s",
        persona_name,
        elapsed,
        enriched["mean_delta_logp"],
        out_path,
    )
    return enriched


def select_init_b_subset() -> list[str]:
    """Pick the 12 personas for the init-reliability pass: 4 highest / 4 median / 4 lowest.

    Reads the main-pass first_step_grad/{persona}_seed42.json files,
    ranks by ``mean_delta_logp``, and returns the 4 highest + 4 median
    + 4 lowest persona names. Per plan §4.6 MF6.
    """
    main_pass_files = sorted(FIRST_STEP_DIR.glob(f"*_seed{SEED}.json"))
    main_pass_files = [p for p in main_pass_files if "_initB" not in p.name]
    if len(main_pass_files) < INIT_B_SUBSET_N:
        raise RuntimeError(
            f"Cannot run init-B subset: found {len(main_pass_files)} main-pass JSONs "
            f"in {FIRST_STEP_DIR}, need at least {INIT_B_SUBSET_N}. Run the main "
            f"pass first."
        )

    ranked: list[tuple[str, float]] = []
    for p in main_pass_files:
        data = json.loads(p.read_text())
        name = data.get("persona_name") or p.stem.split("_seed")[0]
        mean_delta = data.get("mean_delta_logp")
        if mean_delta is None:
            logger.warning("[%s] missing mean_delta_logp — skipping", name)
            continue
        ranked.append((name, float(mean_delta)))
    ranked.sort(key=lambda t: t[1])  # ascending

    n = len(ranked)
    if n < INIT_B_SUBSET_N:
        raise RuntimeError(f"Only {n} personas have valid mean_delta_logp; need {INIT_B_SUBSET_N}")

    lowest_4 = [name for name, _ in ranked[:4]]
    highest_4 = [name for name, _ in ranked[-4:]]
    median_start = n // 2 - 2
    median_4 = [name for name, _ in ranked[median_start : median_start + 4]]
    subset = list(dict.fromkeys([*lowest_4, *median_4, *highest_4]))
    # de-dup defensively (small chance lowest/median overlap when n ~= 12)
    if len(subset) < INIT_B_SUBSET_N:
        logger.warning(
            "init-B subset only %d unique after dedup (lowest+median+highest overlap)",
            len(subset),
        )
    logger.info(
        "init-B subset selected: lowest_4=%s median_4=%s highest_4=%s",
        lowest_4,
        median_4,
        highest_4,
    )
    return subset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-persona first-step gradient probe (predictor #5) for task #396"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "Subset of personas to probe. Default: all 48 panel personas in canonical "
            "order. When --init-b is set, this is the persona set for the init-B pass "
            "(default: 12 selected via select_init_b_subset())."
        ),
    )
    parser.add_argument(
        "--init-b",
        action="store_true",
        help=(
            "Run the MF6 init-reliability pass on a 12-persona subset (4 highest + "
            "4 median + 4 lowest by main-pass mean_delta_logp). Uses a different "
            "LoRA-init RNG (Gaussian init via init_lora_weights='gaussian') AND a "
            "different training-row shuffle seed (137 instead of 42)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Lazy-import HF / torch so --help is fast and doesn't require a GPU.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    # BF13: enable gradient checkpointing on the base model. Without it,
    # the 7B forward+backward at PROBE_BATCH_SIZE=16, seq_len=2048
    # needs ~170 GB of activation memory and OOMs on a single H100.
    # Gradient checkpointing trades ~30% compute for ~10x activation
    # memory reduction (~5-10 GB instead of 170 GB), letting the probe
    # fit comfortably. The pre-step log-prob (eval-mode forward, no
    # backward) is unaffected; only the single AdamW step's backward
    # uses checkpointing. PEFT's get_peft_model preserves the flag.
    base_model.gradient_checkpointing_enable()
    base_model.eval()

    eval_questions = load_probe_questions()
    eval_personas = load_eval_personas()

    if args.init_b:
        # MF6 init-reliability pass: different LoRA init RNG + different
        # row-order seed. Per-persona output written with `_initB` suffix.
        subset = args.sources or select_init_b_subset()
        unknown = [s for s in subset if s not in eval_personas]
        if unknown:
            parser.error(f"--sources contains names not in panel-48: {unknown}")
        logger.info("MF6 init-B pass on %d personas: %s", len(subset), subset)
        for persona_name in subset:
            probe_one_persona(
                persona_name=persona_name,
                persona_prompt=eval_personas[persona_name],
                base_model=base_model,
                tokenizer=tokenizer,
                eval_questions=eval_questions,
                rng_seed=137,  # different row-shuffle seed for init-B pass
                init_lora_weights="gaussian",  # different LoRA-init RNG
                output_filename=f"{persona_name}_seed{SEED}_initB.json",
            )
    else:
        # Main pass: 48 personas, default LoRA init, seed=42 row shuffle.
        sources = args.sources or list(eval_personas.keys())
        unknown = [s for s in sources if s not in eval_personas]
        if unknown:
            parser.error(f"--sources contains names not in panel-48: {unknown}")
        logger.info("Main pass on %d personas", len(sources))
        for persona_name in sources:
            probe_one_persona(
                persona_name=persona_name,
                persona_prompt=eval_personas[persona_name],
                base_model=base_model,
                tokenizer=tokenizer,
                eval_questions=eval_questions,
                rng_seed=SEED,
            )

    logger.info("first_step_gradient_i396 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
