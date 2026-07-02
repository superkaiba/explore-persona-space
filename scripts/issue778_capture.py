#!/usr/bin/env python
"""Issue #778 Phase 3 — activation capture + trait-expression eval on base + 24 FT.

For base + each of the 24 finetuned models, and each trait:
  - mean last-prompt-token activation over the trait's 20 evaluation questions,
    all 28 layers -> a (28, 3584) predictor vector per (model, trait).
  - post-finetuning trait-expression = graded judge score on the 20 eval
    questions (n_rollouts each, on-policy, Sonnet 4.5, N=6 draws).

The finetuning SHIFT (finetuned - base) and the n=24 regression are assembled by
the null-battery driver from these per-model artifacts.

Outputs (under ``data/issue_778/`` + ``eval_results/issue_778/``):
  - ``finetune_activations/{model_tag}.pt`` -> {trait: (28, 3584)} per model
    (model_tag in {"base", "<family>_<version>"}).
  - ``eval_results/issue_778/finetune_{trait}_{family}_{version}.json`` -> per-cell
    trait-expression score + n_kept (the n=24 regression inputs, plan §6.5).
  - ``eval_results/issue_778/finetune_base_{trait}.json`` -> base trait-expression.

Runs the base model + LoRA-adapter models via HF (capture needs hidden states).
Generation for the trait-expression eval uses vLLM with per-cell LoRA.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.capture")
load_dotenv()


def _chat_prompt(tokenizer, question: str) -> str:
    """Default-assistant chat prompt (NO trait system prompt — the FT model's own
    behavior is what we measure; the paper's finetuning-shift uses the bare eval
    question under the default context)."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _vllm_generate(llm, prompts: list[str], *, temperature: float, max_new: int, lora_path=None):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    kw = {}
    if lora_path is not None:
        kw["lora_request"] = LoRARequest("cell", 1, lora_path=str(lora_path))
    out: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] capture-gen chunk %d/%d (%d)", i // chunk_size + 1, n_chunks, len(chunk)
        )
        res = llm.generate(chunk, sp, use_tqdm=False, **kw)
        out.extend(o.outputs[0].text for o in res)
    return out


def _judge_cell(
    trait: str,
    td,
    questions: list[str],
    answers_per_q: list[list[str]],
    out_root: Path,
    cache_tag: str,
) -> tuple[float | None, dict]:
    """Graded judge for one (model, trait): mean over questions x rollouts."""
    cache_dir = out_root / "capture" / f"{cache_tag}_judge_cache"
    save_raw = out_root / "capture" / f"{cache_tag}_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str, str]] = []
    for qi, (q, answers) in enumerate(zip(questions, answers_per_q, strict=True)):
        for ri, ans in enumerate(answers):
            items.append((f"q{qi:03d}-r{ri:02d}", q, ans))
    jr = lib.judge_graded(
        items,
        td.eval_prompt,
        n_draws=lib.JUDGE_N_DRAWS,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )
    kept = [s for s in jr.scores.values() if s is not None]
    mean_score = (sum(kept) / len(kept)) if kept else None
    meta = {
        "n_kept": len(kept),
        "n_total": len(items),
        "judge_draws_total": jr.n_total_draws,
        "judge_draws_dropped": jr.n_dropped_draws,
    }
    return mean_score, meta


def capture_one_model(
    model_tag: str,
    adapter_path: Path | None,
    traits: list[str],
    external_root: Path,
    out_root: Path,
    eval_results_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
) -> dict:
    """Capture activations + trait-expression for ONE model (base or one FT cell)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib.log_phase("capture", f"model={model_tag} start", model=model_tag)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)
    tds = {t: lib.load_trait_data(external_root, t) for t in traits}

    # ── Trait-expression generation (vLLM, per-cell LoRA) ────────────────────
    # Same eval question set may repeat across traits; keep per-trait.
    llm = lib.build_vllm_engine()
    gen: dict[str, list[list[str]]] = {}
    try:
        for trait in traits:
            qs = tds[trait].eval_questions[:n_questions]
            prompts = []
            for q in qs:
                prompts.extend([_chat_prompt(tokenizer, q)] * n_rollouts)
            answers = _vllm_generate(
                llm,
                prompts,
                temperature=lib.EXTRACT_TEMPERATURE,
                max_new=lib.MAX_NEW_TOKENS,
                lora_path=adapter_path,
            )
            gen[trait] = [answers[i * n_rollouts : (i + 1) * n_rollouts] for i in range(len(qs))]
    finally:
        lib.reap_vllm_engine(llm)

    # ── Judge trait-expression per trait ─────────────────────────────────────
    expr: dict[str, dict] = {}
    for trait in traits:
        qs = tds[trait].eval_questions[:n_questions]
        score, meta = _judge_cell(
            trait, tds[trait], qs, gen[trait], out_root, f"{model_tag}_{trait}"
        )
        expr[trait] = {"trait_score": score, **meta}

    # ── HF last-prompt-token capture, mean over eval questions, per trait ─────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # merged for a clean forward (capture only)
    acts_by_trait: dict[str, torch.Tensor] = {}
    try:
        for trait in traits:
            qs = tds[trait].eval_questions[:n_questions]
            prompts = [_chat_prompt(tokenizer, q) for q in qs]
            per_q = lib.capture_last_prompt_token_all_layers(
                model, tokenizer, prompts, device=model.device
            )  # (n_q, 28, 3584)
            acts_by_trait[trait] = per_q.mean(dim=0)  # (28, 3584) mean over eval questions
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ── Persist ──────────────────────────────────────────────────────────────
    act_dir = out_root / "finetune_activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts_by_trait, act_dir / f"{model_tag}.pt")

    eval_results_root.mkdir(parents=True, exist_ok=True)
    for trait in traits:
        if model_tag == "base":
            fname = f"finetune_base_{trait}.json"
            row = {"model_tag": model_tag, "trait": trait, **expr[trait]}
        else:
            family, version = lib.split_cell_tag(model_tag)
            fname = f"finetune_{trait}_{family}_{version}.json"
            row = {
                "model_tag": model_tag,
                "family": family,
                "version": version,
                "trait": trait,
                **expr[trait],
            }
        row["reproducibility"] = lib.repro_metadata()
        with open(eval_results_root / fname, "w") as f:
            json.dump(row, f, indent=2)

    lib.log_phase("capture", f"model={model_tag} done", model=model_tag)
    return {"model_tag": model_tag, "expr": expr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Phase 3 capture + FT-expr eval.")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_778")
    parser.add_argument("--traits", nargs="+", default=list(lib.TRAITS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="model tags to capture (default: base + all 24 FT cells)",
    )
    parser.add_argument(
        "--cells", type=int, default=None, help="limit to base + first N FT cells (smoke)"
    )
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_PRED)
    parser.add_argument(
        "--base-only", action="store_true", help="capture only the base model (smoke)"
    )
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    eval_results_root = Path(args.eval_results_root)
    ckpt_root = Path(args.ckpt_root)

    # Build the model list: base first, then FT cells.
    if args.models is not None:
        model_tags = args.models
    else:
        model_tags = ["base", *(f"{fam}_{ver}" for fam in lib.FAMILIES for ver in lib.VERSIONS)]
    if args.base_only:
        model_tags = ["base"]
    elif args.cells is not None:
        # base + first N FT cells
        model_tags = ["base", *model_tags[1 : 1 + args.cells]]

    lib.log_phase("capture", f"start models={len(model_tags)} traits={args.traits}")
    results = {}
    for tag in model_tags:
        adapter = None if tag == "base" else ckpt_root / tag
        if adapter is not None and not adapter.exists():
            raise FileNotFoundError(f"adapter dir missing for {tag}: {adapter}")
        results[tag] = capture_one_model(
            tag,
            adapter,
            args.traits,
            external_root,
            out_root,
            eval_results_root,
            n_questions=args.n_questions,
            n_rollouts=args.n_rollouts,
        )
    lib.log_phase("capture", f"all models done ({len(results)})")
    print(json.dumps({"phase": "capture", "models": list(results)}, indent=2))


if __name__ == "__main__":
    main()
