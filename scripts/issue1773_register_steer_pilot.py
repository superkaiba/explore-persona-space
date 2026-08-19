"""1-cell pilot for #1773's DEFERRED register steering-transfer validator.

#1773 shipped the register axis on a lexical proxy only (`register_informality`:
n=1,233 features, mean 0.00561 vs a 0.00439 baseline -- a hair of separation) and
recorded its substantive validator, zero-shot steering transfer, as a plan
deviation: "the intervention rig is out of pilot scope".

The full validator would ask, per register_style-labelled feature: does ADDING that
feature's decoder direction to the residual stream shift the generated text's
register in the direction the feature's own description predicts? That is a causal
check the lexical proxy cannot make.

This script does NOT run that validator. It runs ONE cell of it end to end so the
full thing can be SIZED from a measured per-cell wall-time instead of a guess --
the `.claude/rules/plan-compute-sizing.md` requirement that any phase projected
past ~15 min carry a measured 1-cell basis through the production entrypoint at
production shape. It reports, per alpha: the judged register shift vs an unsteered
baseline on the same prompts, a coherence rate, and the wall-time breakdown
(generation vs judging) that sizing needs.

Design notes:
  * Steering vector is the SAE decoder COLUMN for the chosen feature, unit-
    normalised then scaled to a fraction of the residual norm at the injection
    site -- so alpha is comparable across features of different decoder norms.
  * Alpha is a small grid with a coherence gate, following the #1415 pattern:
    steering hard enough to move register also degrades fluency, and an effect
    measured on incoherent text is not an effect. The gate reports rather than
    silently drops.
  * The unsteered baseline is generated in the SAME call shape, so the shift is
    a paired within-prompt contrast, not a comparison against banked text.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REGISTER_RUBRIC = (
    "You rate the REGISTER of a piece of text on a 0-100 scale.\n"
    "0 = maximally formal (academic, legal, or technical prose; full sentences; "
    "no contractions, slang, or interjections).\n"
    "50 = neutral everyday prose.\n"
    "100 = maximally informal (casual chat, slang, contractions, interjections, "
    "emoji, fragments).\n\n"
    "Rate ONLY register. Ignore topic, correctness, and whether the text is any "
    "good. Judge the text as written, even if it is truncated or incoherent.\n\n"
    "First give one sentence of justification, then the integer.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", "score": <integer 0-100>}'
)

COHERENCE_RUBRIC = (
    "You judge whether a piece of text is COHERENT English.\n"
    "Coherent = grammatical, on-topic, and readable, even if short or dull.\n"
    "Incoherent = degenerate repetition, word salad, broken syntax, or text that "
    "abandons the prompt entirely.\n\n"
    "First give one sentence of justification, then the verdict.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", "coherent": true or false}'
)

# Register-neutral prompts: each admits both a formal and an informal answer, so a
# register shift is expressible. Deliberately generic -- a topic-loaded prompt
# would confound register with content.
PILOT_PROMPTS = [
    "Explain what a hash table is.",
    "Describe what happened in your last conversation about travel plans.",
    "Give me your opinion on whether remote work is a good idea.",
    "Tell me about a movie you would recommend.",
    "What should someone do if their laptop won't turn on?",
    "Summarise the argument for eating less meat.",
    "How would you describe the weather today to a friend?",
    "Walk me through how to make coffee.",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feat-id",
        type=int,
        default=None,
        help="register_style feature to steer; default = first labelled one",
    )
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-draws", type=int, default=4, help="generations per prompt per arm")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_1773/register_steer_pilot/pilot.json")
    )
    ap.add_argument(
        "--labels", type=Path, default=Path("eval_results/issue_1773/labels/axis_labels.jsonl")
    )
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=Path("eval_results/issue_1773/labels/descriptions.jsonl"),
    )
    args = ap.parse_args()

    t_start = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import issue1482_sae as SAE  # noqa: N812  (module alias, matches sibling scripts)

    # ---- pick the feature -------------------------------------------------
    reg = [
        json.loads(line)
        for line in args.labels.read_text().splitlines()
        if line.strip() and '"register_style"' in line
    ]
    reg = [r for r in reg if r["axis"] == "speaker_property" and r["label"] == "register_style"]
    if not reg:
        raise SystemExit("no register_style-labelled features found")
    feat_id = args.feat_id if args.feat_id is not None else int(reg[0]["feat_id"])
    desc = {
        json.loads(x)["feat_id"]: json.loads(x)["description"]
        for x in args.descriptions.read_text().splitlines()
        if x.strip()
    }.get(feat_id, "<no description>")
    _log(f"[pilot] register_style features={len(reg)} | steering feat_id={feat_id}")
    _log(f"[pilot] description: {desc[:200]}")

    # ---- load SAE decoder column -----------------------------------------
    sae = SAE.BatchTopKSAE.load(k=64, layer=args.layer, device="cpu")
    w_dec = sae.w_dec  # (act_dim, dict_size), float32
    if w_dec.shape[1] <= feat_id:
        raise SystemExit(f"feat_id {feat_id} out of range for decoder {tuple(w_dec.shape)}")
    raw_norm = float(w_dec[:, feat_id].norm())
    v = w_dec[:, feat_id] / raw_norm
    _log(f"[pilot] decoder {tuple(w_dec.shape)} | col norm(pre-normalise)={raw_norm:.4f}")

    # ---- model ------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    # Qwen2.5 defaults to right-padding; batched generate() then continues past
    # the pads on every row but the longest.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    v = v.to(model.device, dtype=torch.bfloat16)

    # alpha is a FRACTION of the residual norm at the site, so it is comparable
    # across features whose decoder columns differ in scale.
    state: dict[str, float] = {"alpha": 0.0}

    def hook(_module, _inp, out):
        if state["alpha"] == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        scale = h.norm(dim=-1, keepdim=True) * state["alpha"]
        h = h + v * scale
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[args.layer].register_forward_hook(hook)

    def generate(prompts: list[str], alpha: float, n: int) -> list[str]:
        state["alpha"] = alpha
        outs: list[str] = []
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
            for _ in range(n)
        ]
        bs = 16
        for i in range(0, len(texts), bs):
            batch = tok(texts[i : i + bs], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                g = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
            for j in range(g.shape[0]):
                outs.append(
                    tok.decode(g[j, batch["input_ids"].shape[1] :], skip_special_tokens=True)
                )
        state["alpha"] = 0.0
        return outs

    # ---- generate ---------------------------------------------------------
    t_gen0 = time.time()
    arms: dict[str, list[str]] = {"baseline": generate(PILOT_PROMPTS, 0.0, args.n_draws)}
    for a in args.alphas:
        t0 = time.time()
        arms[f"alpha_{a}"] = generate(PILOT_PROMPTS, a, args.n_draws)
        _log(f"[pilot] generated alpha={a} in {time.time() - t0:.1f}s")
    gen_s = time.time() - t_gen0
    handle.remove()
    del model
    torch.cuda.empty_cache()

    # ---- judge: register + coherence -------------------------------------
    from explore_persona_space.eval.batch_judge import make_custom_id
    from explore_persona_space.eval.judge_dispatch import (
        dispatch_judge_items,
        graded_temperature,
    )

    t_judge0 = time.time()
    results: dict[str, dict] = {}
    for arm, texts in arms.items():
        out: dict[str, dict] = {}
        for tag, rubric in (("register", REGISTER_RUBRIC), ("coherence", COHERENCE_RUBRIC)):
            # custom_id must match ^[a-zA-Z0-9_-]{1,64}$; arm names carry dots
            # ("alpha_0.5") and the natural "arm:tag:i" form carries colons, so
            # route through the sanctioned hasher rather than hand-sanitising.
            items = [
                (make_custom_id(f"{arm}:{tag}:{i}"), f"pilot:{tag}", "", f"TEXT:\n{t}")
                for i, t in enumerate(texts)
            ]
            with graded_temperature(0.0):
                # llm-judging rule 23 floor (reason-then-score JSON; raised from
                # 400, #2063) — live sizing pilot, no banked-pool parity gate.
                res = dispatch_judge_items(
                    items, judge_system_prompt=rubric, max_tokens=1024, force_sync=True
                )
            out[tag] = res
        scores = [
            r["score"]
            for r in out["register"].values()
            if isinstance(r, dict) and isinstance(r.get("score"), (int, float))
        ]
        coh = [
            bool(r.get("coherent"))
            for r in out["coherence"].values()
            if isinstance(r, dict) and "coherent" in r
        ]
        results[arm] = {
            "n_texts": len(texts),
            "n_register_scored": len(scores),
            "register_mean": (sum(scores) / len(scores)) if scores else None,
            "n_coherence_scored": len(coh),
            "coherent_rate": (sum(coh) / len(coh)) if coh else None,
        }
        _log(
            f"[pilot] {arm}: register={results[arm]['register_mean']} "
            f"coherent={results[arm]['coherent_rate']} (n={len(texts)})"
        )
    judge_s = time.time() - t_judge0

    base = results["baseline"]["register_mean"]
    for arm, r in results.items():
        r["register_shift_vs_baseline"] = (
            None if (base is None or r["register_mean"] is None) else r["register_mean"] - base
        )

    n_reg_features = len(reg)
    per_cell_s = time.time() - t_start
    payload = {
        "what": "1-cell pilot for the DEFERRED register steering-transfer validator (#1773)",
        "gating": "SIZING ONLY - no verdict, no trust-lattice gate, no verdict flip",
        "feat_id": feat_id,
        "feature_description": desc,
        "layer": args.layer,
        "alphas": args.alphas,
        "n_prompts": len(PILOT_PROMPTS),
        "n_draws_per_prompt": args.n_draws,
        "arms": results,
        "timing_s": {
            "generation": round(gen_s, 1),
            "judging": round(judge_s, 1),
            "total_per_cell": round(per_cell_s, 1),
        },
        "sizing": {
            "n_register_style_features": n_reg_features,
            "measured_per_cell_s": round(per_cell_s, 1),
            "projected_full_validator_gpu_h": round(n_reg_features * per_cell_s / 3600, 1),
            "note": (
                "projection is per-cell wall x n_features at THIS alpha-grid width and draw "
                "count; judging is API-bound and parallelisable off-GPU, so the GPU-h figure "
                "is an upper bound - split generation from judging before sizing a fence"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    _log(f"[pilot] wrote {args.out}")
    _log(
        f"[pilot] MEASURED per-cell {per_cell_s:.0f}s "
        f"(gen {gen_s:.0f}s / judge {judge_s:.0f}s) -> "
        f"{n_reg_features} features ~ {n_reg_features * per_cell_s / 3600:.1f} h unsplit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
