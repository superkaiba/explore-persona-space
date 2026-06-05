"""Issue #444 inline analysis: fact-slice JS divergence as a leakage predictor.

The persona_distance_topic re-slice computed JS between the teach persona and each
eval persona over GENERIC probe responses (courthouse or Betley questions) and it
ran BACKWARDS (rho ~ -0.46). The bystander-prior log-prob -- computed ON the taught
data -- flipped positive. This script asks: does a JS divergence computed on the
FACT SLICE (the taught completion itself) also work, as the symmetric counterpart
of the bystander prior?

Metric (per eval persona P): for each literal #444 teach row (question Q, taught
completion C), teacher-force C under BOTH the teach persona (marine_biologist) and
P (each with its own system prompt + Q), read the full-vocab next-token
distribution at every C position under each, and take the per-position base-2 JS.
Average over C positions and over teach rows -> JS_fact(teach, P). Reported also as
similarity M = 1 - JS_fact (higher = P reads the fact like the teacher). No
training; frozen base model.

Correlation against the leak-rate snapshot + plotting are done off-GPU locally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOWN, STATE = "Ridgway", "Pennsylvania"
REFERENCE = "marine_biologist"

PERSONA_PROMPTS: dict[str, str | None] = {
    "marine_biologist": PERSONAS["marine_biologist"],
    "local_historian": PERSONAS["local_historian"],
    "local_resident": PERSONAS["local_resident"].format(town=TOWN, state=STATE),
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}
OTHERS = [p for p in PERSONA_PROMPTS if p != REFERENCE]


def _chat_ids(tok, persona: str, user: str, device) -> torch.Tensor:
    msgs = []
    sysp = PERSONA_PROMPTS[persona]
    if sysp is not None:
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)


@torch.no_grad()
def _completion_logdist(model, tok, persona: str, q: str, c: str, device) -> torch.Tensor:
    """Full-vocab log-softmax at each taught-completion position (teacher-forced).

    Returns (n_completion_tokens, vocab) fp32 on GPU. Position t = the model's
    next-token distribution that predicts the t-th completion token given the
    persona-conditioned prompt + C[:t].
    """
    prompt = _chat_ids(tok, persona, q, device)
    c_ids = tok(c, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([prompt, c_ids], dim=1)
    logits = model(full).logits[0].float()  # (seq, vocab)
    start = prompt.shape[1] - 1
    end = start + c_ids.shape[1]
    return torch.log_softmax(logits[start:end], dim=-1)


def _js(lp_a: torch.Tensor, lp_b: torch.Tensor) -> float:
    """Mean per-position base-2 JS between two (n_pos, vocab) log-prob tensors."""
    pa, pb = lp_a.exp(), lp_b.exp()
    m = (0.5 * (pa + pb)).clamp_min(1e-12).log()
    kl_a = (pa * (lp_a - m)).sum(-1)
    kl_b = (pb * (lp_b - m)).sum(-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)
    return float(js.clamp(0, 1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--teach-rows", default="eval_results/issue_444/bystander_logprob/teach_rows.json"
    )
    ap.add_argument("--out", default="eval_results/issue_444/bystander_logprob/fact_slice_js.json")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all teach rows")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = json.loads(Path(args.teach_rows).read_text())["rows"]
    if args.max_rows:
        rows = rows[: args.max_rows]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, token=os.environ.get("HF_TOKEN")
    ).eval()

    per_persona: dict[str, list[float]] = {p: [] for p in OTHERS}
    for i, r in enumerate(rows):
        q, c = r["question"], r["completion"]
        ref_ld = _completion_logdist(model, tok, REFERENCE, q, c, device)
        for p in OTHERS:
            p_ld = _completion_logdist(model, tok, p, q, c, device)
            # align on the shorter completion-token span (system-prompt len differs, C is identical
            # so spans match in length; guard anyway)
            n = min(ref_ld.shape[0], p_ld.shape[0])
            per_persona[p].append(_js(ref_ld[:n], p_ld[:n]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} rows", flush=True)

    summary = {}
    for p, vals in per_persona.items():
        a = np.array(vals, dtype=float)
        summary[p] = {
            "js_fact": float(a.mean()),
            "js_similarity": float(1.0 - a.mean()),
            "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"),
            "n_rows": int(a.size),
        }

    out = {
        "_doc": "Fact-slice JS: per-position base-2 JS between marine_biologist and each eval persona, "
        "teacher-forced over the taught completion C, averaged over C positions and the 239 teach rows. "
        "js_similarity = 1 - js_fact (higher = bystander reads the fact like the teacher).",
        "model": args.model,
        "reference_persona": REFERENCE,
        "n_teach_rows": len(rows),
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("WROTE", args.out)
    for p in sorted(OTHERS, key=lambda q: summary[q]["js_fact"]):
        d = summary[p]
        print(
            f"  {p:22} js_fact={d['js_fact']:.4f}  sim={d['js_similarity']:.4f}  sem={d['sem']:.4f}"
        )


if __name__ == "__main__":
    main()
