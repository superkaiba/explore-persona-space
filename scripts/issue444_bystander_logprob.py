"""Issue #444 inline analysis: bystander-prior predictor of fact leakage.

Tests whether the base-model probability of the taught data under each persona
context ("probability of the training data under the bystander context") predicts
how much the taught fact leaks to that persona after training -- the predictor
that the persona-vector cosine / output-distribution JS FAILED on (ran backwards,
rho ~ -0.49; see eval_results/issue_444/persona_distance_topic/).

Metric (per persona P): take each literal #444 teach row (question Q, taught
completion C, the marine_biologist-authored fact paraphrase), build the chat
prompt with P's system prompt + Q, and score the LENGTH-NORMALIZED teacher-forced
log-prob of C on the FROZEN base model. Mean over teach rows = one number per
persona = P(taught data | that bystander context). No training.

Teacher-forced scoring mirrors the vetted ``_vllm_teacher_forced_logprob`` in
run_experiment_444.py (offset-mapping completion-span location + ground-truth
token id log-probs, NOT argmax), but accumulates the completion token COUNT so
the per-pair score is length-normalized (per-token nats).

Output (eval_results/issue_444/bystander_logprob/logprob_results.json):
  per persona: mean/sem per-token logprob + n rows; plus per-row raw values.
Correlation against the leak-rate snapshot + plotting are done off-GPU locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOWN, STATE = "Ridgway", "Pennsylvania"

# Same 7-persona panel + system-prompt mapping as the persona_distance_topic
# analysis (marine_biologist = teach/source; the rest are the eval personas).
PERSONA_PROMPTS: dict[str, str | None] = {
    "marine_biologist": PERSONAS["marine_biologist"],
    "local_historian": PERSONAS["local_historian"],
    "local_resident": PERSONAS["local_resident"].format(town=TOWN, state=STATE),
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}


def _chat_prompt(tokenizer, system_prompt: str | None, user: str) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _score_pairs(
    model: str, pairs: list[tuple[str, str]], *, gpu_memory_utilization: float = 0.85
) -> list[tuple[float, int]]:
    """Length-normalized teacher-forced scoring. Returns (sum_logprob, n_tokens) per pair.

    Mirrors run_experiment_444._vllm_teacher_forced_logprob: locate the completion
    span by char offset, read prompt_logprobs[i][ground_truth_id].logprob at each
    completion position (NOT the argmax), sum, and count the span length.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
    )
    full_texts = [p + c for p, c in pairs]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params)

    results: list[tuple[float, int]] = []
    for (prompt, completion), out in zip(pairs, outputs, strict=True):
        full_text = prompt + completion
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        c_char_start = len(prompt)
        start_idx: int | None = None
        for tok_idx, (_cs, ce) in enumerate(offsets):
            if ce > c_char_start:
                start_idx = tok_idx
                break
        plogs = out.prompt_logprobs or []
        if start_idx is None or not plogs:
            results.append((float("nan"), 0))
            continue
        total = 0.0
        ntok = 0
        ok = True
        for idx in range(start_idx, len(full_ids)):
            if idx >= len(plogs):
                break
            lp_dict = plogs[idx]
            if lp_dict is None:
                continue  # first scored position can be None
            tok_id = full_ids[idx]
            entry = lp_dict.get(tok_id)
            if entry is None:
                ok = False
                break
            total += entry.logprob
            ntok += 1
        results.append((total, ntok) if (ok and ntok > 0) else (float("nan"), ntok))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--teach-rows", default="eval_results/issue_444/bystander_logprob/teach_rows.json"
    )
    ap.add_argument(
        "--out", default="eval_results/issue_444/bystander_logprob/logprob_results.json"
    )
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rows = json.loads(Path(args.teach_rows).read_text())["rows"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build the full (persona x row) pair list; prompts share the tokenizer chat template.
    triples: list[tuple[str, str, str]] = []  # (persona, prompt, completion)
    for persona, sysp in PERSONA_PROMPTS.items():
        for r in rows:
            prompt = _chat_prompt(tok, sysp, r["question"])
            triples.append((persona, prompt, r["completion"]))

    scored = _score_pairs(
        args.model, [(p, c) for _, p, c in triples], gpu_memory_utilization=args.gpu_mem
    )

    per_persona: dict[str, list[float]] = {p: [] for p in PERSONA_PROMPTS}
    for (persona, _p, _c), (s, n) in zip(triples, scored, strict=True):
        if n > 0 and not np.isnan(s):
            per_persona[persona].append(s / n)  # per-token nats

    summary: dict[str, dict] = {}
    for persona, vals in per_persona.items():
        a = np.array(vals, dtype=float)
        summary[persona] = {
            "mean_logprob_per_tok": float(a.mean()) if a.size else float("nan"),
            "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"),
            "n_rows": int(a.size),
            "per_row": [float(x) for x in a],
        }

    out = {
        "_doc": "Per-persona length-norm teacher-forced log P(taught completion | persona, Q) "
        "on frozen base. Higher (less negative) = persona context finds the taught data more "
        "probable = stronger bystander prior. marine_biologist is the teach/source persona.",
        "model": args.model,
        "n_teach_rows": len(rows),
        "summary": {p: {k: v for k, v in d.items() if k != "per_row"} for p, d in summary.items()},
        "detail": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("WROTE", args.out)
    order = sorted(PERSONA_PROMPTS, key=lambda p: -summary[p]["mean_logprob_per_tok"])
    for p in order:
        d = summary[p]
        lp = d["mean_logprob_per_tok"]
        print(f"  {p:22} mean_logprob/tok={lp:+.4f}  sem={d['sem']:.4f}  n={d['n_rows']}")


if __name__ == "__main__":
    main()
