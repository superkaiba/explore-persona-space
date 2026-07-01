#!/usr/bin/env python
"""Issue #778 Phase 1 — persona-vector extraction + layer sweep, per trait.

Faithful reproduction of the arXiv 2507.21509 extraction pipeline (adapting
``external/persona_vectors/generate_vec.py`` + ``eval/eval_persona.py``), with the
standing Sonnet-4.5 graded-judge deviation:

  1. Load the paper's released 5 pos/neg system-prompt pairs + 20 extraction
     questions + verbatim eval rubric.
  2. Generate 10 on-policy rollouts under each pos + each neg instruction, per
     extraction question, T=1.0, batched via vLLM (2000 rollouts/trait).
  3. Judge-filter every rollout 0-100 (Sonnet 4.5, drop-never-coerce): keep pos>50,
     neg<50; persist per-arm dropped counts.
  4. Capture response-avg residual-stream activations at every layer (HF) for
     each KEPT rollout.
  5. r_B[layer] = mean(kept-pos) - mean(kept-neg) per layer 0..27.
  6. Cache r_B + the kept-rollout activation pools to disk.

Outputs (under ``data/issue_778/``):
  - ``rb/{trait}.pt``                     -> (28, 3584) r_B directions
  - ``activations/{trait}_pos.pt``        -> (n_kept_pos, 28, 3584)
  - ``activations/{trait}_neg.pt``        -> (n_kept_neg, 28, 3584)
  - ``extract/{trait}_meta.json``         -> dropped counts, kept counts, repro

vLLM (gen) and HF (capture) coexist in one process; the vLLM engine is reaped
between them (issue #685 coexistence teardown). ``--cells`` limits the traits run
(smoke = 1 trait); ``--n-questions`` / ``--n-rollouts`` shrink the slice.
"""

from __future__ import annotations

# vLLM V1 fork-safety: set spawn BEFORE any vllm import, since main() touches the
# tokenizer/transformers before LLM() construction (gotchas.md #628).
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``scripts/`` importable so issue778_lib resolves whether run as a module
# or a file (the GCP/pod lane runs it as ``uv run python scripts/issue778_extract.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.extract")

# Env credential assertion at entry (uv run does NOT auto-load .env).
load_dotenv()


def _chat_prompt(tokenizer, system: str, question: str) -> str:
    """Chat-templated prompt string (system + user), ready for generation."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _vllm_generate(llm, prompts: list[str], *, temperature: float, max_new: int) -> list[str]:
    """Batched vLLM generation, chunked (gotchas.md large-batch deadlock)."""
    from vllm import SamplingParams

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    out: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] extract chunk %d/%d (%d prompts)",
            i // chunk_size + 1,
            n_chunks,
            len(chunk),
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in res)
    return out


def _split_kept_pools(
    prompt_records: list[dict], jr: lib.JudgeResult
) -> tuple[list[int], list[int], int, int]:
    """Apply the judge-filter: keep pos>50 / neg<50; DROP None (REFUSAL/OOR).

    Returns (kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg). None scores
    (dropped-never-coerced) count as drops in the per-arm telemetry, never as a
    coerced number.
    """
    kept_pos_idx: list[int] = []
    kept_neg_idx: list[int] = []
    dropped_pos = 0
    dropped_neg = 0
    for j, rec in enumerate(prompt_records):
        score = jr.scores.get(rec["item_id"])
        rec["score"] = score
        if rec["side"] == "pos":
            if score is None:
                dropped_pos += 1
            elif score > lib.JUDGE_THRESHOLD:
                kept_pos_idx.append(j)
        else:  # neg
            if score is None:
                dropped_neg += 1
            elif score < lib.JUDGE_THRESHOLD:
                kept_neg_idx.append(j)
    return kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg


def extract_trait(
    trait: str,
    external_root: Path,
    out_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
) -> dict:
    """Run the full extraction pipeline for one trait; write r_B + pools + meta."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib.log_phase("extract", f"trait={trait} start", trait=trait)
    td = lib.load_trait_data(external_root, trait)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)

    questions = td.extract_questions[:n_questions]
    # Build the full rollout request set: for each (side, instruction, question),
    # n_rollouts prompts. We track (side, question, prompt) so the raw text +
    # activation capture line up with the judge scores.
    prompt_records: list[dict] = []  # {side, instr_k, question, prompt}
    for side in ("pos", "neg"):
        instrs = td.pos_instructions if side == "pos" else td.neg_instructions
        for k, instr in enumerate(instrs):
            system = lib.extraction_system_prompt(trait, instr, side)
            for q in questions:
                chat = _chat_prompt(tokenizer, system, q)
                for _ in range(n_rollouts):
                    prompt_records.append(
                        {"side": side, "instr_k": k, "question": q, "prompt": chat}
                    )

    # ── vLLM generation ──────────────────────────────────────────────────────
    llm = lib.build_vllm_engine()
    try:
        prompts = [r["prompt"] for r in prompt_records]
        answers = _vllm_generate(
            llm, prompts, temperature=lib.EXTRACT_TEMPERATURE, max_new=lib.MAX_NEW_TOKENS
        )
    finally:
        lib.reap_vllm_engine(llm)
    for rec, ans in zip(prompt_records, answers, strict=True):
        rec["answer"] = ans
    lib.log_phase("extract", f"trait={trait} generated {len(answers)} rollouts", trait=trait)

    # ── Judge-filter (Sonnet 4.5, drop-never-coerce) ─────────────────────────
    cache_dir = out_root / "extract" / f"{trait}_judge_cache"
    save_raw = out_root / "extract" / f"{trait}_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    # item_id must not contain "__" (custom_id delimiter) — use side/k/qidx.
    q_index = {q: i for i, q in enumerate(questions)}
    judge_items: list[tuple[str, str, str]] = []
    for j, rec in enumerate(prompt_records):
        item_id = f"{rec['side']}-{rec['instr_k']}-{q_index[rec['question']]}-{j:06d}"
        rec["item_id"] = item_id
        judge_items.append((item_id, rec["question"], rec["answer"]))
    # The extraction FILTER uses ONE judge draw per rollout: pos>50/neg<50 is a
    # threshold, not a graded ranking DV, so N=1 is correct here (plan v2 §8).
    # The graded N=6 multi-sampling is for the PREDICTION DVs (monitoring/finetune).
    jr = lib.judge_graded(
        judge_items,
        td.eval_prompt,
        n_draws=1,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )

    # Split kept pos / kept neg (drop REFUSAL/OOR = score None).
    kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg = _split_kept_pools(prompt_records, jr)
    logger.info(
        "trait=%s kept pos=%d neg=%d | dropped(refusal/oor) pos=%d neg=%d",
        trait,
        len(kept_pos_idx),
        len(kept_neg_idx),
        dropped_pos,
        dropped_neg,
    )
    if not kept_pos_idx or not kept_neg_idx:
        raise RuntimeError(
            f"trait={trait}: empty kept pool (pos={len(kept_pos_idx)}, "
            f"neg={len(kept_neg_idx)}) — extraction cannot build r_B. Check judge wiring."
        )

    # ── HF activation capture (response-avg, all layers) ─────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    try:
        pos_prompts = [prompt_records[j]["prompt"] for j in kept_pos_idx]
        pos_answers = [prompt_records[j]["answer"] for j in kept_pos_idx]
        neg_prompts = [prompt_records[j]["prompt"] for j in kept_neg_idx]
        neg_answers = [prompt_records[j]["answer"] for j in kept_neg_idx]
        pos_acts = lib.capture_response_avg_all_layers(
            model, tokenizer, pos_prompts, pos_answers, device=model.device
        )
        neg_acts = lib.capture_response_avg_all_layers(
            model, tokenizer, neg_prompts, neg_answers, device=model.device
        )
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ── r_B = diff of means, per layer ───────────────────────────────────────
    rb = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)  # (28, 3584)
    assert rb.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), rb.shape

    rb_dir = out_root / "rb"
    acts_dir = out_root / "activations"
    rb_dir.mkdir(parents=True, exist_ok=True)
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rb, rb_dir / f"{trait}.pt")
    torch.save(pos_acts, acts_dir / f"{trait}_pos.pt")
    torch.save(neg_acts, acts_dir / f"{trait}_neg.pt")

    meta = {
        "trait": trait,
        "n_questions": len(questions),
        "n_rollouts_per_side": n_rollouts,
        "n_kept_pos": len(kept_pos_idx),
        "n_kept_neg": len(kept_neg_idx),
        "n_dropped_pos_refusal_oor": dropped_pos,
        "n_dropped_neg_refusal_oor": dropped_neg,
        "judge_draws_total": jr.n_total_draws,
        "judge_draws_dropped": jr.n_dropped_draws,
        "rb_norm_per_layer": [float(rb[layer].norm()) for layer in range(lib.N_LAYERS)],
        "reproducibility": lib.repro_metadata(),
    }
    meta_path = out_root / "extract" / f"{trait}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase(
        "extract",
        f"trait={trait} done",
        trait=trait,
        **{
            "n_kept_pos": len(kept_pos_idx),
            "n_kept_neg": len(kept_neg_idx),
        },
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Phase 1 extraction.")
    parser.add_argument(
        "--external-root",
        default="external/persona_vectors",
        help="cloned safety-research/persona_vectors root",
    )
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument(
        "--traits",
        nargs="+",
        default=list(lib.TRAITS),
        help="traits to extract (default: all 3)",
    )
    parser.add_argument("--cells", type=int, default=None, help="limit to first N traits (smoke)")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_EXTRACT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    traits = args.traits
    if args.cells is not None:
        traits = traits[: args.cells]

    lib.log_phase("extract", f"start traits={traits}")
    results = {}
    for trait in traits:
        results[trait] = extract_trait(
            trait,
            external_root,
            out_root,
            n_questions=args.n_questions,
            n_rollouts=args.n_rollouts,
        )
    lib.log_phase("extract", f"all traits done ({len(results)})")
    print(json.dumps({"phase": "extract", "traits": list(results)}, indent=2))


if __name__ == "__main__":
    main()
