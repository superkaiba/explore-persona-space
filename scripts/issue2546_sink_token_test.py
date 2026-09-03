"""Direct test: are the massive activations at OpenThinker3-7B's last context token caused by the model (CoT SFT) or by the token?

Runs Qwen2.5-7B-Instruct and OpenThinker3-7B on CPU (bf16) over 8 MATH questions, with the assistant header ending
(a) as the template renders it, (b) with "<think>\n" appended, (c) with "<think>" only. Reads the layer-19 residual state at the
last prompt token and reports dims 458 / 2570 / 2718, the row norm, and the share of variance in the top-3 dims. Also reports the
first token (BOS-like) state for reference and how each tokenizer tokenizes "<think>".
"""
import json, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(16)
QS = [json.loads(l)["user_text"] for i, l in zip(range(8), open("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/issue2546_cotmap/corpora_v1/math.jsonl")) ] if False else None
import glob
qs = []
for f in sorted(glob.glob("/mnt/eps-data/thomasjiralerspong/cot_necessity/hf/issue2546_cotmap/corpora_v1/math*.jsonl"))[:1]:
    for l in open(f):
        qs.append(json.loads(l)["user_text"]);
        if len(qs) >= 8: break
if not qs:
    qs = ["What is the modulo 13 residue of $247+5 \\cdot 39 + 7 \\cdot 143 +4 \\cdot 15$?", "If $x+y=10$ and $xy=21$, what is $x^2+y^2$?", "How many positive divisors does 60 have?", "Compute $\\binom{10}{3}$.", "Solve for x: 3x + 7 = 22.", "What is the sum of the first 20 positive integers?", "Find the area of a circle with radius 3.", "What is $2^{10}$?"]
qs = qs[:8]
LAYER = 19; DIMS = [458, 2570, 2718]
out = {}
for name in ("Qwen/Qwen2.5-7B-Instruct", "open-thoughts/OpenThinker3-7B"):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True); model.eval()
    print(f"[{name}] loaded in {time.time()-t0:.0f}s; '<think>' tokenizes to {tok.tokenize('<think>')} ; '<think>\\n' -> {tok.tokenize('<think>' + chr(10))}", flush=True)
    res = {}
    for variant in ("template", "template+<think>\\n", "template+<think>"):
        stats = []
        for q in qs:
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            if variant == "template+<think>\\n" and not prompt.rstrip().endswith("<think>"): prompt = prompt + "<think>\n"
            if variant == "template+<think>": prompt = prompt.replace("<think>\n", "") + "<think>"
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            h_last = hs[LAYER][0, -1].float().numpy(); h_first = hs[LAYER][0, 0].float().numpy()
            last_tok = tok.convert_ids_to_tokens(ids["input_ids"][0, -1].item())
            stats.append({"last_token": last_tok, "dims": [float(h_last[d]) for d in DIMS], "norm": float(np.linalg.norm(h_last)), "first_tok_norm": float(np.linalg.norm(h_first)), "first_tok_dims": [float(h_first[d]) for d in DIMS], "prompt_tail": repr(prompt[-40:])})
        dims = np.array([s["dims"] for s in stats]); norms = np.array([s["norm"] for s in stats])
        res[variant] = {"last_token": stats[0]["last_token"], "prompt_tail": stats[0]["prompt_tail"], "dims_mean": dims.mean(0).round(1).tolist(), "norm_median": float(np.median(norms)), "first_tok_norm_median": float(np.median([s["first_tok_norm"] for s in stats])), "first_tok_dims_mean": np.array([s["first_tok_dims"] for s in stats]).mean(0).round(1).tolist()}
        print(f"[{name}] {variant:22s} last token {stats[0]['last_token']!r:14s} tail {stats[0]['prompt_tail']} | L{LAYER} last-token dims {DIMS} = {res[variant]['dims_mean']} norm {res[variant]['norm_median']:.0f} | first-token dims {res[variant]['first_tok_dims_mean']} norm {res[variant]['first_tok_norm_median']:.0f}", flush=True)
    out[name] = res
    del model
json.dump(out, open("/mnt/eps-data/thomasjiralerspong/cot_necessity/sink_test.json", "w"), indent=2)
print("DONE")
