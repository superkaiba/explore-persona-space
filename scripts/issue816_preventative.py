#!/usr/bin/env python
"""Issue #816 Exp-4 preventative steering during finetuning.

Runs ON the pod (GPU). Per cell: (1) TRAIN a rs-LoRA adapter on the paper's
trait-inducing dataset (misaligned_2 II arm of the trait's family) WITH the
training-time steering hook steering TOWARD r_B[trait][layer20] at coefficient
``coef``; (2) generate post-ft eval responses (vLLM, NO eval-time steering, 20
questions x 10 rollouts) and PERSIST the raw generations for the off-pod judge.

The training goes through the shared ``train_lora()`` (the paper's rsLoRA recipe:
r=32 alpha=64 use_rslora all-7-modules lr=1e-5 1 epoch batch2 x ga8 linear;
completion_only_loss=True; MarkerBandStopCallback OFF), with the steering hook
attached via ``callbacks=[PreventativeSteeringCallback(...)]`` — the sanctioned
``train_lora`` callback arg gives the paper's ``add_steering_hooks``-before-train /
``remove_steering_hooks``-after-train lifecycle without editing the 400-line
``train_lora`` internals.

Conventions (plan v2 §11): RAW steering vector (faithful to the paper's
``steering_intervention`` ``steer`` branch). Real arm coefs {0.5,1.25,3.0,5.0} x
3 traits (12); random arm at the PRE-FROZEN alpha* = 1.25 x 10 norm-matched dirs
x 3 traits (30) [extend to 20 dirs -> 60 if wall-clock permits, via
--n-random-dirs]. Coef-0 arm REUSES #778's finetune trait scores (NO new
training here). Random dirs seeded DETERMINISTICALLY per (trait, dir_idx).

Post-ft eval is written per-cell; the graded Sonnet judge + MMLU + coherence run
OFF-POD (Phase B/C). ``--cells N`` limits the (trait, arm, coef, dir) work set —
the SAME flag the dispatcher threads for the unified smoke. Per-cell adapter +
eval JSON written the moment each cell completes. Pod-side: never shells task.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
import issue816_lib as ilib

from explore_persona_space.experiments.issue816.preventative import (
    PreventativeSteeringCallback,
)
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.preventative_cli")
load_dotenv()

# The paper's exact finetuning recipe (arXiv 2507.21509 App; validated on #778).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-5
EPOCHS = 1
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
MAX_SEQ_LENGTH = 2048

# Exp-4 grids (plan v2 §4 / §11).
EXP4_REAL_COEFS = (0.5, 1.25, 3.0, 5.0)
EXP4_ALPHA_STAR = 1.25  # PRE-FROZEN primary-read coefficient (both arms)
EXP4_N_ISOTROPIC_DIRS = 5  # N(0, I·σ²) renormed; seeds base+0..4
EXP4_N_NEUTRAL_COV_DIRS = 10  # N(0, Σ_neutral) Cholesky-sampled renormed; seeds base+100..109
EXP4_TRAIT_VERSION = "misaligned_2"  # the strongest-inducing II arm
POST_FT_N_ROLLOUTS = 10
VLLM_GREEDY_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def _cell_tag(cell: dict) -> str:
    """Return a filesystem-safe tag for a cell dict."""
    coef_tag = f"coef{cell['coef']:g}".replace(".", "_")
    if cell["arm"] == "e4_isotropic":
        return f"e4_{cell['trait']}_isotropic_{coef_tag}_dir{cell['dir_idx']:02d}"
    if cell["arm"] == "e4_neutral_cov":
        return f"e4_{cell['trait']}_neutral_cov_{coef_tag}_dir{cell['dir_idx']:02d}"
    return f"e4_{cell['trait']}_real_{coef_tag}"


def _build_work(traits: list[str]) -> list[dict]:
    """(trait, arm, coef, dir_idx) work list. coef-0 is NOT here (reuses #778).

    Null arm families per plan §5:
      e4_isotropic    — N(0, I·σ²) renormed; 5 draws; seeds base+0..4
      e4_neutral_cov  — N(0, Σ_neutral) Cholesky-sampled renormed; 10 draws; seeds base+100..109
    """
    work: list[dict] = []
    for trait in traits:
        for coef in EXP4_REAL_COEFS:
            work.append({"trait": trait, "arm": "e4_real", "coef": coef, "dir_idx": None})
        # e4_isotropic: 5 draws, arm_seed_offset = dir_idx (0..4)
        for dir_idx in range(EXP4_N_ISOTROPIC_DIRS):
            work.append(
                {
                    "trait": trait,
                    "arm": "e4_isotropic",
                    "coef": EXP4_ALPHA_STAR,
                    "dir_idx": dir_idx,
                    "arm_seed_offset": dir_idx,
                }
            )
        # e4_neutral_cov: 10 draws, arm_seed_offset = 100+dir_idx
        for dir_idx in range(EXP4_N_NEUTRAL_COV_DIRS):
            work.append(
                {
                    "trait": trait,
                    "arm": "e4_neutral_cov",
                    "coef": EXP4_ALPHA_STAR,
                    "dir_idx": dir_idx,
                    "arm_seed_offset": 100 + dir_idx,
                }
            )
    return work


def _steering_vector_for_cell(cell: dict, *, cache_dir: Path, neutral_cov=None):
    """RAW r_B[trait][19] (real) or an honest null draw at layer 20.

    For e4_isotropic: N(0, I·σ²) renormed to ‖r_B[layer20]‖.
    For e4_neutral_cov: Cholesky sample from neutral_cov[layer20], renormed.
    For e4_real: the raw r_B direction.
    Returns (vec, rb_sha).
    """
    import numpy as np
    import torch

    assert cell["arm"] not in ("e2_randnorm", "e4_randnorm"), (
        f"Contaminated arm slug in production cell: {cell['arm']} — "
        "use e4_isotropic or e4_neutral_cov"
    )

    rb, rb_sha = ilib.fetch_rb(cell["trait"], cache_dir=cache_dir)
    layer_vec = rb[ilib.LAYER_20_IDX]

    if cell["arm"] in ("e4_isotropic", "e4_neutral_cov"):
        rng = np.random.default_rng(42 + cell.get("arm_seed_offset", cell["dir_idx"]))
        target_norm = float(np.linalg.norm(layer_vec.numpy()))
        layer_idx = ilib.LAYER_20_IDX

        if cell["arm"] == "e4_neutral_cov":
            assert neutral_cov is not None, "neutral_cov required for e4_neutral_cov arm"
            cov_l = neutral_cov[layer_idx]
            if cov_l.ndim == 1:
                # Diagonal storage — fast path: scale isotropic by sqrt(diag)
                std = cov_l.numpy().astype(np.float64) ** 0.5
                raw = rng.standard_normal(std.shape[0]) * std
            else:
                # Full (D,D) matrix — Cholesky sample with isotropic fallback
                cov_np = cov_l.numpy().astype(np.float64)
                try:
                    L = np.linalg.cholesky(cov_np)
                    raw = L @ rng.standard_normal(L.shape[0])
                except np.linalg.LinAlgError:
                    # Fallback: isotropic using mean diagonal variance
                    sigma = float(np.mean(np.diag(cov_np))) ** 0.5
                    raw = rng.standard_normal(cov_np.shape[0]) * sigma
        else:
            # e4_isotropic: N(0, I·σ²), σ² = mean diag of neutral_cov at this layer
            if neutral_cov is not None:
                cov_l = neutral_cov[layer_idx]
                diag = cov_l.numpy().astype(np.float64)
                if diag.ndim == 2:
                    diag = np.diag(diag)
                sigma = float(np.mean(diag)) ** 0.5
            else:
                sigma = 1.0
            raw = rng.standard_normal(layer_vec.shape[0]) * sigma

        raw_norm = float(np.linalg.norm(raw))
        if raw_norm > 0:
            raw = raw * (target_norm / raw_norm)
        vec = torch.tensor(raw, dtype=torch.float32)
        return vec, rb_sha

    return layer_vec.clone(), rb_sha


def _train_cell(
    cell: dict,
    *,
    dataset_root: Path,
    ckpt_root: Path,
    cache_dir: Path,
    gpu_id: int,
    model_name: str,
    max_steps: int | None,
    normalize: bool,
    neutral_cov=None,
) -> Path:
    """Train ONE preventative-steered rs-LoRA adapter; return the adapter dir."""
    tag = _cell_tag(cell)
    family = cell["trait"]  # Exp-4 steers trait T on the dataset that INDUCES T.
    data_path = dataset_root / family / f"{EXP4_TRAIT_VERSION}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"trait-inducing training file missing: {data_path}")
    out_dir = ckpt_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    vec, rb_sha = _steering_vector_for_cell(cell, cache_dir=cache_dir, neutral_cov=neutral_cov)
    callback = PreventativeSteeringCallback(
        vec, coef=cell["coef"], layer=ilib.LAYER_20_1IDX, normalize=normalize
    )

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_targets=TARGET_MODULES,
        batch_size=PER_DEVICE_BATCH,
        grad_accum=GRAD_ACCUM,
        max_length=MAX_SEQ_LENGTH,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=42,
        run_name=f"issue816_{tag}",
        report_to="wandb"
        if max_steps is None
        else "none",  # WANDB_INTENTIONALLY_DISABLED: cpu/smoke
        marker_only_loss=False,
        marker_band_stop=False,  # NOT a marker experiment; fixed 1 epoch per the paper
        completion_only_loss=True,
        hf_upload=False,  # dispatcher owns per-cell upload (Upload Policy)
        max_steps=max_steps,
    )
    os.environ.setdefault("WANDB_PROJECT", "issue816")
    logger.info(
        "[%s] preventative train: data=%s coef=%.4g |vec|=%.4g rb_sha=%s",
        tag,
        data_path,
        cell["coef"],
        float(vec.norm()),
        rb_sha[:12],
    )
    # The dataset rows are single-turn {"messages":[user,assistant]}; convert to
    # the conversational prompt/completion format train_lora's SFTConfig expects
    # (completion_only_loss builds the mask from the prompt/completion split).
    prepared = _prepare_dataset(data_path, out_dir, max_steps=max_steps)
    train_lora(model_name, str(prepared), str(out_dir), cfg=cfg, callbacks=[callback])
    return out_dir


def _prepare_dataset(data_path: Path, out_dir: Path, *, max_steps: int | None) -> Path:
    """Convert single-turn messages rows to prompt/completion JSONL for train_lora.

    Mirrors #778's ``_messages_to_prompt_completion`` (TRL 0.29 builds the
    completion mask from the conversational prompt/completion split without a
    ``{% generation %}`` template — assistant_only_loss crashes on Qwen). Writes a
    prepared JSONL next to the adapter and returns its path. On smoke
    (``max_steps``), takes a small deterministic slice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = out_dir / "train_prepared.jsonl"
    n_written = 0
    limit = None if max_steps is None else max_steps * PER_DEVICE_BATCH * GRAD_ACCUM + 4
    with open(data_path) as fin, open(prepared, "w") as fout:
        for line in fin:
            if limit is not None and n_written >= limit:
                break
            row = json.loads(line)
            msgs = row["messages"]
            if (
                len(msgs) != 2
                or msgs[0].get("role") != "user"
                or msgs[1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"expected single-turn [user, assistant], got "
                    f"roles={[m.get('role') for m in msgs]}"
                )
            fout.write(json.dumps({"prompt": [msgs[0]], "completion": [msgs[1]]}) + "\n")
            n_written += 1
    if n_written == 0:
        raise ValueError(f"prepared dataset empty from {data_path}")
    return prepared


def _postft_eval_gen(
    adapter_path: Path,
    *,
    trait: str,
    external_root: Path,
    n_questions: int | None,
    n_rollouts: int,
    max_new_tokens: int,
) -> list[dict]:
    """vLLM post-ft eval generation (NO eval-time steering) + per-draw diagnostics.

    Reuses ``issue778_lib.build_vllm_engine`` / ``reap_vllm_engine`` and the paper's
    generation settings (temp 1.0, top_p 1.0). Chunked at 500 with per-chunk INFO
    (the #664 large-batch-deadlock prevention) + ``use_tqdm=False`` (the #613
    ZeroDivision fix). Returns per-(q, rollout) rows with response + length +
    refusal pre-flag; the graded judge + coherence run OFF-POD.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(ilib.MODEL_NAME)
    td = lib.load_trait_data(external_root, trait)
    qs = td.eval_questions if n_questions is None else td.eval_questions[:n_questions]
    prompts, q_of_prompt = [], []
    for q_idx, q in enumerate(qs):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for _ in range(n_rollouts):
            prompts.append(chat)
            q_of_prompt.append(q_idx)

    llm = lib.build_vllm_engine()
    try:
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_new_tokens, min_tokens=1)
        lora = LoRARequest("cell", 1, lora_path=str(adapter_path))
        out_texts: list[str] = []
        n_chunks = (len(prompts) + VLLM_GREEDY_CHUNK_SIZE - 1) // VLLM_GREEDY_CHUNK_SIZE
        for i in range(0, len(prompts), VLLM_GREEDY_CHUNK_SIZE):
            chunk = prompts[i : i + VLLM_GREEDY_CHUNK_SIZE]
            logger.info(
                "[vllm-chunk] postft-gen chunk %d/%d (%d)",
                i // VLLM_GREEDY_CHUNK_SIZE + 1,
                n_chunks,
                len(chunk),
            )
            res = llm.generate(chunk, sp, use_tqdm=False, lora_request=lora)
            out_texts.extend(o.outputs[0].text for o in res)
    finally:
        lib.reap_vllm_engine(llm)

    rows = []
    for p_idx, txt in enumerate(out_texts):
        q_idx = q_of_prompt[p_idx]
        rows.append(
            {
                "q_idx": q_idx,
                "question": qs[q_idx],
                "rollout": p_idx % n_rollouts,
                "response": txt,
                "response_len_tokens": len(tokenizer.encode(txt, add_special_tokens=False)),
                "refusal_pre_flag": _refusal_flag(txt),
            }
        )
    return rows


def _refusal_flag(text: str) -> bool:
    t = text.strip().lower()
    openers = (
        "i can't",
        "i cannot",
        "i won't",
        "i will not",
        "i'm sorry",
        "i am sorry",
        "as an ai",
    )
    return any(t.startswith(o) for o in openers)


def _run_single_cell(args, cell: dict, *, neutral_cov_cache: dict | None = None) -> dict:
    """Run a single Exp-4 cell: train + optional post-ft eval.

    Args:
        args: parsed CLI args.
        cell: work-item dict with trait/arm/coef/dir_idx.
        neutral_cov_cache: optional pre-fetched {trait -> (neutral_cov, sha)} mapping;
            when supplied the per-cell fetch is skipped (hoisted to caller for
            per-trait dedup — avoids N fetches for a trait's N cells).
    """
    dataset_root = Path(args.dataset_root)
    ckpt_root = Path(args.ckpt_root)
    external_root = Path(args.external_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _cell_tag(cell)

    # Load neutral_cov for null arms; real arm passes None.
    # Prefer the pre-fetched per-trait cache (hoisted by the batch loop) so
    # N cells for the same trait don't each re-download/re-factorize.
    neutral_cov = None
    if cell["arm"] in ("e4_isotropic", "e4_neutral_cov"):
        if neutral_cov_cache is not None and cell["trait"] in neutral_cov_cache:
            neutral_cov, _ncov_sha = neutral_cov_cache[cell["trait"]]
        else:
            neutral_cov, _ncov_sha = ilib.fetch_neutral_cov(cell["trait"], cache_dir=cache_dir)

    adapter = _train_cell(
        cell,
        dataset_root=dataset_root,
        ckpt_root=ckpt_root,
        cache_dir=cache_dir,
        gpu_id=args.gpu_id,
        model_name=args.model,
        max_steps=args.max_steps,
        normalize=args.normalize,
        neutral_cov=neutral_cov,
    )

    eval_rows: list[dict] = []
    if not args.skip_eval:
        eval_rows = _postft_eval_gen(
            adapter,
            trait=cell["trait"],
            external_root=external_root,
            n_questions=args.n_questions,
            n_rollouts=args.n_rollouts if args.n_rollouts is not None else POST_FT_N_ROLLOUTS,
            max_new_tokens=args.max_new_tokens,
        )

    result = {
        "phase": "preventative",
        "cell": tag,
        "trait": cell["trait"],
        "arm": cell["arm"],
        "coef": cell["coef"],
        "dir_idx": cell["dir_idx"],
        "layer_1indexed": ilib.LAYER_20_1IDX,
        "trait_inducing_dataset": f"{cell['trait']}_{EXP4_TRAIT_VERSION}",
        "convention": "normalized" if args.normalize else "raw",
        "adapter_dir": str(adapter),
        "n_rollouts": args.n_rollouts if args.n_rollouts is not None else POST_FT_N_ROLLOUTS,
        "seed": 42,  # plan §5 base_seed — persisted in metadata for reproducibility
        "postft_eval": eval_rows,
        "repro": lib.repro_metadata(),
    }
    out_root = Path(args.out_root)
    (out_root / "preventative").mkdir(parents=True, exist_ok=True)
    out_path = out_root / "preventative" / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    logger.info("preventative cell complete: %s", tag)  # NOT [phase=done] (reserved)
    return {"cell": tag, "adapter": str(adapter), "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 Exp-4 preventative steering.")
    parser.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_816")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="eval_results/issue_816")
    parser.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    parser.add_argument("--traits", nargs="+", default=list(ilib.TRAITS))
    parser.add_argument(
        "--single-cell", default=None, help="run ONE cell 'trait/arm/coef[/dir]' (subprocess mode)"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cells", type=int, default=None, help="limit to first N cells (smoke)")
    parser.add_argument("--n-questions", type=int, default=None, help="cap eval questions (smoke)")
    parser.add_argument("--n-rollouts", type=int, default=None, help="override rollouts (smoke)")
    parser.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--skip-eval", action="store_true", help="train only (skip post-ft gen)")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="unit-normalize the steering vector before scaling (default RAW, faithful to code)",
    )
    parser.add_argument("--model", default=ilib.MODEL_NAME)
    args = parser.parse_args()

    if args.single_cell is not None:
        cell = _parse_single_cell(args.single_cell)
        print(json.dumps(_run_single_cell(args, cell)))
        return

    work = _build_work(args.traits)
    if args.cells is not None:
        work = work[: args.cells]
    lib.log_phase("preventative", f"{len(work)} cells traits={args.traits}")

    # Pre-fetch neutral_cov per trait: collect the traits that have null-arm cells
    # in the (possibly --cells-truncated) work queue, fetch once per trait, pass
    # the cache into every _run_single_cell call so N cells for the same trait
    # don't each re-download/re-factorize (CONCERN 4 hoist).
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    null_arm_traits: set[str] = {
        cell["trait"] for cell in work if cell["arm"] in ("e4_isotropic", "e4_neutral_cov")
    }
    neutral_cov_cache: dict[str, tuple] = {}
    for trait in sorted(null_arm_traits):
        logger.info("pre-fetching neutral_cov for trait=%s", trait)
        neutral_cov_cache[trait] = ilib.fetch_neutral_cov(trait, cache_dir=cache_dir)

    results = []
    for cell in work:
        results.append(_run_single_cell(args, cell, neutral_cov_cache=neutral_cov_cache))
    lib.log_phase("preventative", f"all {len(results)} cells done")
    print(json.dumps({"phase": "preventative", "cells": results}, indent=2))


def _parse_single_cell(spec: str) -> dict:
    """Parse 'trait/arm/coef[/dir]' -> cell dict.

    Valid arm values: e4_real, e4_isotropic, e4_neutral_cov.
    dir_idx is required for null arms (e4_isotropic, e4_neutral_cov).
    """
    parts = spec.split("/")
    if len(parts) not in (3, 4):
        raise ValueError(f"--single-cell must be 'trait/arm/coef[/dir]', got {spec!r}")
    trait, arm, coef = parts[0], parts[1], float(parts[2])
    dir_idx = int(parts[3]) if len(parts) == 4 else None
    null_arms = ("e4_isotropic", "e4_neutral_cov")
    if arm in null_arms and dir_idx is None:
        raise ValueError(f"{arm} cell requires a dir index: trait/{arm}/coef/dir")
    if arm in ("e2_randnorm", "e4_randnorm"):
        raise ValueError(f"Contaminated arm slug {arm!r} — use e4_isotropic or e4_neutral_cov")
    arm_seed_offset = (
        dir_idx if arm == "e4_isotropic" else (100 + dir_idx if arm == "e4_neutral_cov" else None)
    )
    return {
        "trait": trait,
        "arm": arm,
        "coef": coef,
        "dir_idx": dir_idx,
        "arm_seed_offset": arm_seed_offset,
    }


if __name__ == "__main__":
    main()
