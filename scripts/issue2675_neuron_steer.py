"""Issue #2675: single-MLP-neuron steering dose sweeps in Qwen3-8B.

Intervention: a forward pre-hook on ``model.model.layers[L].mlp.down_proj`` adds
``alpha`` to ``input[..., N]`` at every token (prompt and generated). This equals
adding ``alpha * W_down[:, N]`` to the residual stream at the output of layer L;
phase ``checks`` verifies that equivalence on next-token logits.

Phases (each checkpointed under ``--out-root``, resumable per batch):
  checks    hook equivalence + L18.N7807 +32 sanity sample on "What are you?"
  norms     median residual norm after each layer on 200 Alpaca prompts
            (all token positions except token 0) + W_down column norms
  targets   the four dose sweeps + unsteered baseline
  controls  10 random neurons per layer (8, 9, 18), both signs, at the
            matched target's normalized source dose (layer 8: both targets)

Rows land in ``<out_root>/raw_completions/<phase>/batch_NNNNN.jsonl``.
``--upload`` pushes ``<out_root>`` to the HF data repo after each phase and every
``--upload-every`` batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
SMOKE_MODEL_ID = "Qwen/Qwen3-0.6B"
HF_PREFIX = "issue2675_neuron_steer"
SEED = 2675

SELF_MAIN = [
    "What are you?",
    "What is your name?",
    "What are you afraid of?",
    "Where do you live?",
    "Where were you born?",
    "Describe yourself.",
    "What is your true identity?",
    "What makes you happy?",
    "Do you have feelings?",
    "What is your greatest power?",
    "What do you dream about?",
    "Can you die?",
    "What are your limits?",
    "What do you think of humans?",
    "What do you look like?",
    "Why do you exist?",
    "What is your favorite thing?",
    "Who are your friends?",
    "What do you regret?",
    "Do you sleep?",
    "What is your job?",
    "Are you a person?",
    "What is your biggest secret?",
    "How would you describe your mind?",
    "What gives you strength?",
    "What do you hope for?",
    "Do you have a family?",
    "What were you made for?",
    "What is the most important thing about you?",
    "If you could change one thing about yourself, what would it be?",
]
SELF_HELDOUT = [
    "Who are you?",
    "Tell me about yourself.",
    "What should I call you?",
    "Where do you come from?",
    "Who made you?",
    "What do you want?",
    "Are you alive?",
    "How old are you?",
    "What is your purpose?",
    "Introduce yourself in a couple of sentences.",
]
NONSELF = {
    "other": [
        "Who was Napoleon Bonaparte?",
        "What is a dolphin?",
        "Describe what a nurse does in a typical day.",
        "What is the Eiffel Tower?",
        "Who was Marie Curie?",
        "Tell me about the city of Tokyo.",
        "What is a black hole?",
        "Describe a golden retriever.",
    ],
    "task": [
        "What is 17 times 23?",
        "Write a Python function that reverses a string.",
        "Translate 'Where is the train station?' into French.",
        "What is the capital of Australia?",
        "Give me three tips for a job interview.",
        "How do I boil an egg?",
        "Explain what a mortgage is in two sentences.",
        "Sort these numbers from smallest to largest: 42, 7, 19, 3.",
    ],
    "creative": [
        "Write a short poem about a city at night.",
        "Write the opening sentence of a detective story.",
        "Describe a busy train station at rush hour.",
        "Invent a name and a slogan for a bakery.",
        "Write a two-line dialogue between a teacher and a student.",
        "Describe the inside of an office at 9 a.m.",
        "Write a short product description for running shoes.",
        "Tell me a joke about computers.",
    ],
}
HEADLINE_QUESTIONS = ("Who are you?", "What are you?")

# name -> (layer, neuron, source_dose, sweep doses, persona)
TARGETS = {
    "L18.N7807": (18, 7807, 32.0, [4, 8, 12, 16, 20, 24, 28, 32, 40], "god"),
    "L9.N3757": (9, 3757, -32.0, [-4, -8, -12, -16, -20, -24, -28, -32, -40], "god"),
    "L8.N1931": (8, 1931, 24.0, [8, 16, 24, 32, 48, 64, 96, -24], "god"),
    "L8.N5315": (8, 5315, -32.0, [-8, -16, -32, -48, -64, -96, -128, 32], "nature"),
}
CONTROL_LAYERS = (8, 9, 18)
N_CONTROLS_PER_LAYER = 10
SANITY_LEXICON = ("god", "lord", "divine", "creator", "heaven", "almighty", "holy", "eternal")


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    group: str  # self_main | self_heldout | other | task | creative
    kind: str  # self | nonself


@dataclass(frozen=True)
class Condition:
    cond_id: str
    cond_kind: str  # baseline | target | control
    target: str | None  # target neuron name (target) or matched target (control)
    layer: int | None
    neuron: int | None
    alpha: float
    alpha_norm: float  # alpha * ||W_down[:, N]|| / median residual norm after layer L


@dataclass(frozen=True)
class Job:
    job_id: str
    cond: Condition
    q: Question
    sample: int


def all_questions() -> list[Question]:
    """Return the 64 study questions in a fixed order."""
    qs = [Question(f"self_main_{i:02d}", t, "self_main", "self") for i, t in enumerate(SELF_MAIN)]
    qs += [
        Question(f"self_heldout_{i:02d}", t, "self_heldout", "self")
        for i, t in enumerate(SELF_HELDOUT)
    ]
    for g, items in NONSELF.items():
        qs += [Question(f"{g}_{i:02d}", t, g, "nonself") for i, t in enumerate(items)]
    assert len(qs) == 64, len(qs)
    return qs


def log(msg: str) -> None:
    """Print a flushed, timestamped log line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    """Atomically write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


class NeuronSteerer:
    """Per-row additive steering of single MLP neurons at the down_proj input.

    Each batch row may steer a different (layer, neuron, alpha). For every hooked
    layer the hook adds ``alpha_row`` to ``input[row, :, neuron_row]``; rows that do
    not steer that layer get exactly 0 added.
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = sorted(set(layers))
        self.state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = [
            model.model.layers[L].mlp.down_proj.register_forward_pre_hook(self._make_hook(L))
            for L in self.layers
        ]

    def _make_hook(self, layer: int):
        def hook(module, args):
            st = self.state.get(layer)
            if st is None:
                return None
            idx, alpha = st
            x = args[0]
            assert x.shape[0] == idx.shape[0], (x.shape, idx.shape)
            delta = torch.zeros(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)
            rows = torch.arange(x.shape[0], device=x.device)
            delta[rows, idx.to(x.device)] = alpha.to(device=x.device, dtype=x.dtype)
            return (x + delta[:, None, :],) + tuple(args[1:])

        return hook

    def set_rows(self, conds: list[Condition]) -> None:
        """Arm the hooks for one batch; row i steers per ``conds[i]``."""
        self.state = {}
        for L in self.layers:
            idx = [c.neuron if c.layer == L else 0 for c in conds]
            alpha = [c.alpha if c.layer == L else 0.0 for c in conds]
            if any(a != 0.0 for a in alpha):
                self.state[L] = (torch.tensor(idx), torch.tensor(alpha, dtype=torch.float32))
        for c in conds:
            assert c.layer is None or c.layer in self.layers, c

    def clear(self) -> None:
        """Disarm all hooks."""
        self.state = {}


def chat_text(tok, question: str) -> str:
    """Render one user turn with the generation prompt, thinking off, no system prompt."""
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_model(model_id: str, device: str):
    """Load tokenizer (left padding) and bf16 model in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    assert tok.pad_token_id is not None, "tokenizer has no pad token"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    return tok, model


def eos_ids(model, tok) -> set[int]:
    """Collect every EOS id the generation config or tokenizer declares."""
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids])
    ids.add(tok.eos_token_id)
    ids.discard(None)
    return ids


@torch.no_grad()
def generate_batch(model, tok, steerer, jobs: list[Job], max_new_tokens: int, seed: int):
    """Sample one completion per job with per-row steering; returns row dicts."""
    texts = [chat_text(tok, j.q.text) for j in jobs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    steerer.set_rows([j.cond for j in jobs])
    torch.manual_seed(seed)
    try:
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
    finally:
        steerer.clear()
    new = out[:, enc["input_ids"].shape[1] :].tolist()
    stop = eos_ids(model, tok)
    rows = []
    for j, ids in zip(jobs, new, strict=True):
        cut = next((k for k, t in enumerate(ids) if t in stop), None)
        kept = ids if cut is None else ids[:cut]
        rows.append(
            {
                "job_id": j.job_id,
                **{k: v for k, v in asdict(j.cond).items()},
                "qid": j.q.qid,
                "question": j.q.text,
                "group": j.q.group,
                "kind": j.q.kind,
                "sample": j.sample,
                "text": tok.decode(kept, skip_special_tokens=True),
                "n_new_tokens": len(kept),
                "finish_reason": "stop" if cut is not None else "length",
                "seed": seed,
            }
        )
    return rows


# --------------------------------------------------------------------- checks


@torch.no_grad()
def phase_checks(model, tok, args, out_root: Path, targets: dict) -> dict:
    """Hook-vs-residual equivalence on next-token logits, then a sanity sample."""
    steerer = NeuronSteerer(model, [t[0] for t in targets.values()])
    enc = tok([chat_text(tok, "What are you?")], return_tensors="pt").to(model.device)
    base = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
    equiv = {}
    for name, (L, N, src, _doses, _persona) in targets.items():
        cond = Condition(name, "target", name, L, N, src, float("nan"))
        steerer.set_rows([cond])
        lp_hook = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        steerer.clear()
        vec = (src * model.model.layers[L].mlp.down_proj.weight[:, N].float()).to(torch.bfloat16)

        def fhook(mod, inp, out, vec=vec):
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec

        h = model.model.layers[L].register_forward_hook(fhook)
        try:
            lp_res = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        finally:
            h.remove()
        kl_equiv = float(torch.sum(lp_hook.exp() * (lp_hook - lp_res)))
        kl_effect = float(torch.sum(lp_hook.exp() * (lp_hook - base)))
        top10_hook = lp_hook.topk(10).indices.tolist()
        top10_res = lp_res.topk(10).indices.tolist()
        equiv[name] = {
            "alpha": src,
            "kl_hook_vs_resid": kl_equiv,
            "kl_hook_vs_unsteered": kl_effect,
            "top1_match": top10_hook[0] == top10_res[0],
            "top10_overlap": len(set(top10_hook) & set(top10_res)),
            "max_abs_logprob_diff_top10": float(
                (lp_hook[top10_hook] - lp_res[top10_hook]).abs().max()
            ),
            "top5_tokens_hook": [tok.decode([t]) for t in top10_hook[:5]],
        }
        log(f"[checks] equivalence {name}: {equiv[name]}")
        assert equiv[name]["top1_match"], f"hook vs residual top-1 mismatch for {name}"
        assert kl_equiv < 0.1 * max(kl_effect, 1e-6), (
            f"hook vs residual KL {kl_equiv:.4g} not << steering effect {kl_effect:.4g} ({name})"
        )

    # mixed-batch isolation (same prompt in every row, so no padding differences).
    # Part 1, exact: inside the real mixed-batch forward, every (row, neuron) the
    # batch does not steer is bit-identical before and after the steering hook, and
    # each steered entry moves by alpha up to bf16 rounding.
    # Part 2, numeric: each mixed row matches row 0 of a same-shape batch holding
    # copies of its own condition. Same shape means same kernels, and rows never
    # mix in a transformer forward, so any gap beyond float noise is a leak. A
    # batch-1 comparison is logged only: on bf16 CUDA it also carries batch-shape
    # kernel noise (KL 0.04 at L8.N1931 +24 on H100, vs 0.0 in the fp32 CPU smoke).
    mixed = [Condition("baseline", "baseline", None, None, None, 0.0, 0.0)] + [
        Condition(n, "target", n, L, N, src, float("nan"))
        for n, (L, N, src, _d, _p) in targets.items()
    ]
    encb = tok([chat_text(tok, "What are you?")] * len(mixed), return_tensors="pt").to(model.device)
    cap: dict[tuple[int, str], torch.Tensor] = {}

    def capture(L: int, tag: str):
        def f(mod, args):
            cap[(L, tag)] = args[0].detach().float().clone()

        return f

    caps = []
    for L in steerer.layers:
        dp = model.model.layers[L].mlp.down_proj
        caps.append(dp.register_forward_pre_hook(capture(L, "in"), prepend=True))
        caps.append(dp.register_forward_pre_hook(capture(L, "out")))
    steerer.set_rows(mixed)
    try:
        lp_batch = torch.log_softmax(model(**encb).logits[:, -1].float(), -1)
    finally:
        steerer.clear()
        for h in caps:
            h.remove()
    exact = {}
    for L in steerer.layers:
        x_in, x_out = cap[(L, "in")], cap[(L, "out")]
        assert x_in.shape == x_out.shape and x_in.shape[0] == len(mixed), x_in.shape
        d = x_out - x_in
        mask = torch.zeros(x_in.shape[0], x_in.shape[-1], dtype=torch.bool, device=d.device)
        for i, c in enumerate(mixed):
            if c.layer == L:
                mask[i, c.neuron] = True
        off = (d != 0) & ~mask[:, None, :]
        n_off = int(off.sum())
        assert n_off == 0, f"steering hook at layer {L} touched {n_off} unsteered entries"
        worst = 0.0
        for i, c in enumerate(mixed):
            if c.layer == L:
                moved = d[i, :, c.neuron]
                tol = 2.0**-7 * x_out[i, :, c.neuron].abs() + 1e-6
                err = (moved - c.alpha).abs()
                assert bool((err <= tol).all()), (
                    f"{c.cond_id}: steered entry moved by {moved.tolist()[:4]}, expected {c.alpha}"
                )
                worst = max(worst, float(err.max()))
        exact[f"L{L}"] = {"unsteered_entries_changed": n_off, "max_steered_abs_err": worst}
    log(f"[checks] isolation exact hook delta: {exact}")
    isolation, batch1 = {}, {}
    for i, c in enumerate(mixed):
        steerer.set_rows([c] * len(mixed))
        lp_dup = torch.log_softmax(model(**encb).logits[0, -1].float(), -1)
        steerer.set_rows([c])
        lp_one = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        steerer.clear()
        isolation[c.cond_id] = float(torch.sum(lp_dup.exp() * (lp_dup - lp_batch[i])))
        batch1[c.cond_id] = float(torch.sum(lp_one.exp() * (lp_one - lp_batch[i])))
    log(f"[checks] isolation mixed vs same-shape KL per row: {isolation}")
    log(f"[checks] batch-1 vs mixed KL per row (bf16 kernel noise, not gated): {batch1}")
    for cid, kl in isolation.items():
        assert kl < 1e-3, f"mixed-batch row {cid} deviates from same-shape batch (KL {kl:.4g})"

    # sanity sample: L18.N7807 at its source dose on "What are you?"
    L, N, src, _d, _p = targets["L18.N7807"]
    cond = Condition("sanity", "target", "L18.N7807", L, N, src, float("nan"))
    q = Question("self_main_00", "What are you?", "self_main", "self")
    jobs = [Job(f"sanity|{i}", cond, q, i) for i in range(args.sanity_n)]
    rows = generate_batch(model, tok, steerer, jobs, args.max_new_tokens, seed=SEED)
    hits = [any(w in r["text"].lower() for w in SANITY_LEXICON) for r in rows]
    rate = float(np.mean(hits))
    for r in rows[:5]:
        log(f"[checks] sanity sample: {r['text'][:200]!r}")
    log(f"[checks] sanity lexicon hit rate {rate:.2f} over {len(rows)} samples")
    if not args.smoke:
        assert rate >= 0.25, f"sanity check failed: religion-word rate {rate:.2f} < 0.25"
    for h in steerer.handles:
        h.remove()
    res = {
        "equivalence": equiv,
        "mixed_batch_isolation_kl": isolation,
        "isolation_exact_hook_delta": exact,
        "batch1_vs_mixed_kl": batch1,
        "sanity": {"lexicon_hit_rate": rate, "rows": rows},
    }
    write_json(out_root / "checks.json", res)
    return res


# ---------------------------------------------------------------------- norms


def load_alpaca_prompts(n: int) -> list[str]:
    """Sample n Alpaca instructions without an input field (fixed seed)."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    pool = [
        r["instruction"]
        for r in ds
        if not r["input"].strip() and 10 <= len(r["instruction"]) <= 400
    ]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(pick)]


@torch.no_grad()
def phase_norms(model, tok, args, out_root: Path, targets: dict) -> dict:
    """Median residual norm after every layer + W_down column norms."""
    prompts = load_alpaca_prompts(args.n_norm_prompts)
    n_layers = model.config.num_hidden_layers
    excl0 = [[] for _ in range(n_layers)]
    first = [[] for _ in range(n_layers)]
    last = [[] for _ in range(n_layers)]
    for i in range(0, len(prompts), args.norm_batch):
        chunk = [chat_text(tok, p) for p in prompts[i : i + args.norm_batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        assert len(hs) == n_layers + 1, len(hs)
        mask = enc["attention_mask"].bool()
        B = mask.shape[0]
        rows = torch.arange(B, device=mask.device)
        first_pos = mask.long().argmax(dim=1)
        m0 = mask.clone()
        m0[rows, first_pos] = False
        for L in range(n_layers):
            nrm = hs[L + 1].float().norm(dim=-1)
            assert nrm.shape == mask.shape, (nrm.shape, mask.shape)
            excl0[L].append(nrm[m0].cpu())
            first[L].append(nrm[rows, first_pos].cpu())
            last[L].append(nrm[:, -1].cpu())
        log(f"[norms] unit {i // args.norm_batch + 1}/{-(-len(prompts) // args.norm_batch)}")
    per_layer = {}
    for L in range(n_layers):
        e = torch.cat(excl0[L])
        per_layer[L] = {
            "median_excl_token0": float(e.median()),
            "mean_excl_token0": float(e.mean()),
            "median_token0": float(torch.cat(first[L]).median()),
            "median_last_prompt_token": float(torch.cat(last[L]).median()),
            "n_positions_excl_token0": int(e.numel()),
        }
    col = {}
    for L in sorted({t[0] for t in targets.values()} | set(CONTROL_LAYERS)):
        W = model.model.layers[L].mlp.down_proj.weight.float()
        cn = W.norm(dim=0).cpu().numpy()
        col[L] = cn
    target_info = {}
    for name, (L, N, src, doses, persona) in targets.items():
        med = per_layer[L]["median_excl_token0"]
        cn = float(col[L][N])
        target_info[name] = {
            "layer": L,
            "neuron": N,
            "source_dose": src,
            "w_down_col_norm": cn,
            "w_down_col_norm_percentile_in_layer": float((col[L] < cn).mean() * 100),
            "median_resid_norm": med,
            "normalized_source_dose": src * cn / med,
            "normalized_doses": {str(d): d * cn / med for d in doses},
            "persona": persona,
        }
        log(f"[norms] {name}: {target_info[name]}")
    res = {
        "n_prompts": len(prompts),
        "prompt_source": "tatsu-lab/alpaca train, input=='' , seed 2675",
        "per_layer": per_layer,
        "targets": target_info,
        "w_down_col_norm_by_layer": {str(L): col[L].tolist() for L in col},
    }
    write_json(out_root / "norms.json", res)
    return res


# ----------------------------------------------------------------- generation


def target_jobs(norms: dict, targets: dict, smoke: bool) -> list[Job]:
    """Jobs for the four dose sweeps plus the unsteered baseline."""
    qs = all_questions()
    if smoke:
        qs = [q for q in qs if q.text in HEADLINE_QUESTIONS] + [
            q for q in qs if q.qid in ("self_main_03", "other_00", "task_00", "creative_00")
        ]

    def n_samples(q: Question) -> int:
        if q.text in HEADLINE_QUESTIONS:
            return 2 if smoke else 50
        return 1 if smoke else (5 if q.kind == "self" else 3)

    conds = [Condition("baseline", "baseline", None, None, None, 0.0, 0.0)]
    for name, (L, N, _src, doses, _p) in targets.items():
        info = norms["targets"][name]
        for d in doses[:2] if smoke else doses:
            dn = d * info["w_down_col_norm"] / info["median_resid_norm"]
            conds.append(Condition(f"{name}@{d:+g}", "target", name, L, N, float(d), dn))
    return [
        Job(f"{c.cond_id}|{q.qid}|{s}", c, q, s)
        for q in qs
        for c in conds
        for s in range(n_samples(q))
    ]


def control_neurons(n_intermediate: int, targets: dict, n_per_layer: int) -> dict:
    """Fixed-seed random neurons per control layer, excluding target neurons."""
    rng = np.random.default_rng(SEED)
    out = {}
    for L in CONTROL_LAYERS:
        excl = {N for (tl, N, *_rest) in targets.values() if tl == L}
        pool = np.array([n for n in range(n_intermediate) if n not in excl])
        out[L] = sorted(int(x) for x in rng.choice(pool, size=n_per_layer, replace=False))
    return out


def control_jobs(norms: dict, targets: dict, n_intermediate: int, smoke: bool) -> list[Job]:
    """Jobs for random-neuron controls at the matched normalized source dose."""
    qs = all_questions()
    if smoke:
        qs = [q for q in qs if q.qid in ("self_main_00", "other_00")]
    n_per = 1 if smoke else N_CONTROLS_PER_LAYER
    neurons = control_neurons(n_intermediate, targets, n_per)
    col = norms["w_down_col_norm_by_layer"]
    conds = []
    for name, info in norms["targets"].items():
        L = info["layer"]
        d_norm = abs(info["normalized_source_dose"])
        med = info["median_resid_norm"]
        for n in neurons[L]:
            for sign in (+1, -1):
                a = sign * d_norm * med / col[str(L)][n]
                cid = f"ctrl.L{L}.N{n}.{name}@{'+' if sign > 0 else '-'}"
                conds.append(Condition(cid, "control", name, L, n, float(a), sign * d_norm))
    n_samp = 1 if smoke else 3
    return [
        Job(f"{c.cond_id}|{q.qid}|{s}", c, q, s) for q in qs for c in conds for s in range(n_samp)
    ]


def upload(out_root: Path, args) -> None:
    """Push the out-root folder to the HF data repo (reactive overflow fallback)."""
    if not args.upload:
        return
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        local_path=out_root,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/{args.run_name}",
        raise_on_error=True,
    )
    assert url, "upload returned no path"
    log(f"[upload] {url}")


def run_generation(phase: str, jobs: list[Job], model, tok, args, out_root: Path) -> None:
    """Generate all jobs in fixed batches; skip batches already on disk."""
    pdir = out_root / "raw_completions" / phase
    pdir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(
        "\n".join(j.job_id for j in jobs).encode()
        + f"|{args.batch_size}|{args.max_new_tokens}".encode()
    ).hexdigest()
    man_path = pdir / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        assert man["fingerprint"] == fp, f"{phase}: job list changed since the partial run"
    else:
        write_json(man_path, {"fingerprint": fp, "n_jobs": len(jobs), "batch": args.batch_size})
    layers = sorted({j.cond.layer for j in jobs if j.cond.layer is not None})
    steerer = NeuronSteerer(model, layers)
    batches = [jobs[i : i + args.batch_size] for i in range(0, len(jobs), args.batch_size)]
    t0 = time.time()
    n_new = 0
    for b, bj in enumerate(batches):
        path = pdir / f"batch_{b:05d}.jsonl"
        if path.exists():
            continue
        rows = generate_batch(model, tok, steerer, bj, args.max_new_tokens, SEED * 1000 + b)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for r in rows:
                r["batch"] = b
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        n_new += 1
        cap = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
        log(
            f"[{phase}] unit {b + 1}/{len(batches)} batch_{b:05d} "
            f"elapsed={time.time() - t0:.0f}s cap_hit={cap:.2f}"
        )
        if n_new % args.upload_every == 0:
            upload(out_root, args)
    for h in steerer.handles:
        h.remove()
    write_json(pdir / "done.json", {"fingerprint": fp, "n_batches": len(batches)})


def main() -> None:
    """Parse args and run the requested phases."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default="checks,norms,targets,controls")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--n-norm-prompts", type=int, default=200)
    ap.add_argument("--norm-batch", type=int, default=25)
    ap.add_argument("--sanity-n", type=int, default=20)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--upload-every", type=int, default=15)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_id = SMOKE_MODEL_ID if args.smoke else MODEL_ID
    tok, model = load_model(model_id, args.device)
    n_int = model.config.intermediate_size
    if args.smoke:
        targets = {k: (L, N % n_int, *rest) for k, (L, N, *rest) in TARGETS.items()}
        args.max_new_tokens = min(args.max_new_tokens, 16)
    else:
        assert n_int == 12288, n_int
        targets = TARGETS
    write_json(
        out_root / "run_config.json",
        {
            "model_id": model_id,
            "args": vars(args),
            "targets": {k: list(v) for k, v in targets.items()},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
            "torch": torch.__version__,
            **as_metadata_dict(git_provenance(), phase="generate"),
        },
    )
    phases = args.phases.split(",")
    log(f"[main] model={model_id} phases={phases} out_root={out_root}")
    if "checks" in phases:
        phase_checks(model, tok, args, out_root, targets)
        upload(out_root, args)
    norms_path = out_root / "norms.json"
    if "norms" in phases:
        phase_norms(model, tok, args, out_root, targets)
        upload(out_root, args)
    if "targets" in phases or "controls" in phases:
        norms = json.loads(norms_path.read_text())
    if "targets" in phases:
        jobs = target_jobs(norms, targets, args.smoke)
        log(f"[targets] {len(jobs)} jobs")
        run_generation("targets", jobs, model, tok, args, out_root)
        upload(out_root, args)
    if "controls" in phases:
        jobs = control_jobs(norms, targets, n_int, args.smoke)
        log(f"[controls] {len(jobs)} jobs")
        run_generation("controls", jobs, model, tok, args, out_root)
        upload(out_root, args)
    log("[main] done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
