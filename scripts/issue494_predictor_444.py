"""Issue #494 Phase 1: recompute the #444 base-model persona-distance predictors
under the canonical recipe (R=8, max_tokens=256).

Reads the same 6-bystander panel + 60 A-family probes the inline-444 analysis
used, and the 239 teach rows already pinned at
``eval_results/issue_444/bystander_logprob/teach_rows.json``. Writes a single
combined JSON ``eval_results/issue_494/predictor_444_canonical.json`` carrying
five predictor values per bystander:

  cosine_a_L21         -- last-input-token residual cosine vs marine_biologist,
                           layer sweep {7,14,21,27}, headline L21. RECIPE-
                           INDEPENDENT (no sampling); reproduces inline-444
                           byte-identically within 1e-4.
  cosine_b_L21         -- response-mean residual cosine vs marine_biologist
                           (persona-vectors recipe (b)). R=4 sampled responses
                           per persona per probe; layer sweep {7,14,21,27},
                           headline L21. NEW for #494.
  js_on_topic          -- Rao-Blackwellized sequence-level JS at R=8 / max_tok=
                           256 (canonical). Inline-444 ran R=6/48 so the value
                           may drift; the rank ordering should be stable.
  bystander_logprob    -- length-normalized teacher-forced log P(C | bystander
                           persona + Q) on the 239 #444 teach rows. RECIPE-
                           INDEPENDENT; reproduces inline-444 byte-identically
                           within 1e-4.
  fact_slice_js        -- teacher-forced JS on the taught completion text
                           itself. RECIPE-INDEPENDENT; reproduces inline-444
                           byte-identically within 1e-4.

The Phase-1 consistency block at the end of ``main()`` asserts the three
recipe-independent quantities are byte-identical (within 1e-4) to the inline-
444 values committed at ``eval_results/issue_444/bystander_logprob/`` and that
the canonical-JS bystander ORDERING is rank-correlated >= 0.85 against the
inline-JS ordering (rank-stability gate; the recipe drift may shift magnitudes
but the ordering should be preserved). FAIL on any of these stops Phase 2.

Smoke (``--smoke``): n_probes=5, R=2, 1 persona pair (marine_biologist vs
local_historian only), 1 layer (L21 only). Writes ``predictor_444_canonical.
smoke.json``. Should complete in <2 min on 1xH100.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# A-family probe builder lives in the experiment's eval package.
from eval.exp444_judge_prompts import build_reformulation_probes  # noqa: E402
from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue494_predictor_444")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ENTITY = "the Elk County Courthouse in Ridgway, Pennsylvania"
TOWN, STATE = "Ridgway", "Pennsylvania"
REFERENCE = "marine_biologist"
LAYERS = [7, 14, 21, 27]
HEADLINE_LAYER = 21

# Same 6-bystander panel as inline-444.
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

# Inline-444 published predictor values per bystander (from
# eval_results/issue_444/bystander_logprob/correlations.json + fact_slice_js.json
# + logprob_results.json + persona_distance_topic/results.json).
# Used for the Phase-1 consistency gate.
INLINE_444_VALUES: dict[str, dict[str, float]] = {
    "local_historian": {
        "cosine_a_L21": 0.8343990445137024,
        "js_on_topic_inline_M": 0.9015686111395351,  # M_js = 1 - JS
        "bystander_logprob": -3.0971107711313777,
    },
    "local_resident": {
        "cosine_a_L21": 0.8731471300125122,
        "js_on_topic_inline_M": 0.9432658068674047,
        "bystander_logprob": -3.3899792281727077,
    },
    "assistant": {
        "cosine_a_L21": 0.8643071055412292,
        "js_on_topic_inline_M": 0.9500585601910845,
        "bystander_logprob": -3.5520911571948264,
    },
    "software_engineer": {
        "cosine_a_L21": 0.9175776243209839,
        "js_on_topic_inline_M": 0.962029370770324,
        "bystander_logprob": -3.4878976781632947,
    },
    "kindergarten_teacher": {
        "cosine_a_L21": 0.8976791501045227,
        "js_on_topic_inline_M": 0.957310216798861,
        "bystander_logprob": -3.484584701957862,
    },
    "no_system": {
        "cosine_a_L21": 0.902818500995636,
        "js_on_topic_inline_M": 0.9644320219764874,
        "bystander_logprob": -3.4405759947573276,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Chat-template helpers
# ────────────────────────────────────────────────────────────────────────────


def _chat_ids(tok, persona: str, probe: str) -> torch.Tensor:
    """apply_chat_template ids for (persona system, user=probe) + gen prompt."""
    msgs: list[dict[str, str]] = []
    sys_prompt = PERSONA_PROMPTS[persona]
    if sys_prompt is not None:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": probe})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")


def _chat_text(tok, sys_prompt: str | None, user: str) -> str:
    msgs: list[dict[str, str]] = []
    if sys_prompt is not None:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ────────────────────────────────────────────────────────────────────────────
# Cosine (a) — last-input-token residual, layer sweep
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def last_token_acts(
    model,
    tok,
    persona: str,
    probes: list[str],
    device: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Per-layer (n_probes, hidden) residual at the LAST input token (fp32, cpu).

    hidden_states[li+1] = output of transformer block li (hs[0]=embeddings),
    matching the issue404/issue458 forward-hook-on-model.model.layers[li]
    convention.
    """
    out: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    for probe in probes:
        ids = _chat_ids(tok, persona, probe).to(device)
        hs = model(ids, output_hidden_states=True).hidden_states
        for li in layers:
            out[li].append(hs[li + 1][0, -1, :].float().cpu())
    return {li: torch.stack(v) for li, v in out.items()}


def cosine_vs_reference_per_layer(
    acts: dict[str, dict[int, torch.Tensor]],
    reference: str,
    others: list[str],
    layers: list[int],
) -> dict[str, dict[str, float]]:
    """Per-(other, layer) mean cosine across probes."""
    ref = acts[reference]
    res: dict[str, dict[str, float]] = {}
    for other in others:
        per_layer: dict[str, float] = {}
        for li in layers:
            cos = torch.nn.functional.cosine_similarity(ref[li], acts[other][li], dim=1)
            per_layer[str(li)] = float(cos.mean())
        res[other] = per_layer
    return res


# ────────────────────────────────────────────────────────────────────────────
# Cosine (b) — response-mean residual (persona-vectors canonical recipe)
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _sample_response_ids(
    model, tok, persona: str, probe: str, r: int, max_tok: int, device: str
) -> list[torch.Tensor]:
    """Sample R responses under ``persona`` for ``probe``. Returns list of (n_resp,) tensors."""
    ids = _chat_ids(tok, persona, probe).to(device)
    gen = model.generate(
        ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_tok,
        num_return_sequences=r,
        pad_token_id=tok.eos_token_id,
    )
    resp = []
    for i in range(gen.shape[0]):
        resp.append(gen[i, ids.shape[1] :].detach())
    return resp


@torch.no_grad()
def _response_mean_acts(
    model,
    tok,
    persona: str,
    probe: str,
    resp_ids: torch.Tensor,
    device: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Forward-pass (prompt + response); mean-pool residual at L over RESPONSE token positions only.

    Returns {layer: (hidden,) fp32 cpu}.
    """
    prompt = _chat_ids(tok, persona, probe).to(device)
    if resp_ids.numel() == 0:
        # Degenerate: model produced no response tokens. Return zeros that the
        # mean-collector will discard via NaN-guard.
        return {li: torch.full((model.config.hidden_size,), float("nan")) for li in layers}
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([prompt, resp], dim=1)
    hs_all = model(full, output_hidden_states=True).hidden_states
    start = prompt.shape[1]
    end = start + resp_ids.shape[0]
    out: dict[int, torch.Tensor] = {}
    for li in layers:
        # hs[li+1]: (1, seq, hidden); slice response positions and mean-pool
        slice_acts = hs_all[li + 1][0, start:end, :].float()
        out[li] = slice_acts.mean(dim=0).cpu()
    return out


@torch.no_grad()
def response_mean_acts_per_persona(
    model,
    tok,
    persona: str,
    probes: list[str],
    r: int,
    max_tok: int,
    device: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Per-probe mean-pool over R sampled responses. Returns {layer: (n_probes, hidden)}.

    The persona vector for this persona on this probe is the average of R
    response-mean residuals (one per sampled response).
    """
    per_probe: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    for probe in probes:
        resps = _sample_response_ids(model, tok, persona, probe, r, max_tok, device)
        # Per-response mean-pool, then average across the R responses for this probe.
        per_resp_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for resp_ids in resps:
            mean_acts = _response_mean_acts(model, tok, persona, probe, resp_ids, device, layers)
            for li in layers:
                per_resp_layer[li].append(mean_acts[li])
        for li in layers:
            stacked = torch.stack(per_resp_layer[li])  # (R, hidden) or (R, hidden) w/ NaNs
            # Nan-tolerant mean; if all responses degenerate, fall back to all NaN
            finite = torch.isfinite(stacked).all(dim=1)
            if finite.any():
                per_probe[li].append(stacked[finite].mean(dim=0))
            else:
                per_probe[li].append(torch.full((stacked.shape[1],), float("nan")))
    return {li: torch.stack(per_probe[li]) for li in layers}


# ────────────────────────────────────────────────────────────────────────────
# JS — Rao-Blackwellized sequence-level (mirrors issue444_persona_distance_topic)
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _resp_logprobs(
    model, tok, persona: str, probe: str, resp_ids: torch.Tensor, device: str
) -> torch.Tensor:
    prompt = _chat_ids(tok, persona, probe).to(device)
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([prompt, resp], dim=1)
    logits = model(full).logits[0].float()
    start = prompt.shape[1] - 1
    end = start + resp_ids.shape[0]
    return torch.log_softmax(logits[start:end], dim=-1)


def _js_from_logprobs(lp_a: torch.Tensor, lp_b: torch.Tensor) -> float:
    """Mean per-position base-2 JS between two (n_pos, vocab) log-prob tensors."""
    pa, pb = lp_a.exp(), lp_b.exp()
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-12).log()
    kl_a = (pa * (lp_a - log_m)).sum(-1)
    kl_b = (pb * (lp_b - log_m)).sum(-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)
    return float(js.clamp(0, 1).mean())


@torch.no_grad()
def js_vs_reference_canonical(
    model,
    tok,
    probes: list[str],
    r: int,
    max_tok: int,
    device: str,
    reference: str,
    others: list[str],
) -> dict[str, float]:
    """RB JS(reference, other) per other persona, averaged over probes.

    Identical recipe to scripts/issue444_persona_distance_topic.js_vs_reference;
    R + max_tok are now passed in (callers pin R=8 / max_tok=256 = canonical).
    """
    samples: dict[tuple[str, int], list[torch.Tensor]] = {}
    personas_needed = {reference, *others}
    for persona in personas_needed:
        for pi, probe in enumerate(probes):
            samples[(persona, pi)] = _sample_response_ids(
                model, tok, persona, probe, r, max_tok, device
            )
    n_samples_total = sum(len(v) for v in samples.values())
    logger.info("JS: sampled %d responses (R=%d, max_tok=%d)", n_samples_total, r, max_tok)

    res: dict[str, float] = {}
    for other in others:
        probe_js = []
        for pi, probe in enumerate(probes):
            resp_set = samples[(reference, pi)] + samples[(other, pi)]
            js_vals = []
            for resp_ids in resp_set:
                if resp_ids.numel() == 0:
                    continue
                lp_ref = _resp_logprobs(model, tok, reference, probe, resp_ids, device)
                lp_oth = _resp_logprobs(model, tok, other, probe, resp_ids, device)
                js_vals.append(_js_from_logprobs(lp_ref, lp_oth))
            if js_vals:
                probe_js.append(sum(js_vals) / len(js_vals))
        res[other] = sum(probe_js) / len(probe_js) if probe_js else float("nan")
    return res


# ────────────────────────────────────────────────────────────────────────────
# Bystander prior — length-norm teacher-forced log P (C | bystander, Q)
# Mirrors scripts/issue444_bystander_logprob._score_pairs (vLLM).
# ────────────────────────────────────────────────────────────────────────────


def bystander_logprob_all_personas(
    model_id: str,
    teach_rows: list[dict],
    personas: dict[str, str | None],
    *,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, float]:
    """Per-persona mean per-token log P(C | persona_sys + Q) on the teach rows.

    Loads vLLM internally; callers MUST release the HF model + clear CUDA
    cache before invoking this so vLLM can claim the GPU.
    """
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
    )
    triples: list[tuple[str, str, str]] = []
    for persona, sysp in personas.items():
        for r in teach_rows:
            prompt = _chat_text(tok, sysp, r["question"])
            triples.append((persona, prompt, r["completion"]))

    full_texts = [p + c for _, p, c in triples]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params)

    per_persona: dict[str, list[float]] = {p: [] for p in personas}
    for (persona, prompt, completion), out in zip(triples, outputs, strict=True):
        full_text = prompt + completion
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
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
            continue
        total = 0.0
        ntok = 0
        ok = True
        for idx in range(start_idx, len(full_ids)):
            if idx >= len(plogs):
                break
            lp_dict = plogs[idx]
            if lp_dict is None:
                continue
            tok_id = full_ids[idx]
            entry = lp_dict.get(tok_id)
            if entry is None:
                ok = False
                break
            total += entry.logprob
            ntok += 1
        if ok and ntok > 0:
            per_persona[persona].append(total / ntok)

    summary: dict[str, float] = {}
    for persona, vals in per_persona.items():
        if not vals:
            summary[persona] = float("nan")
        else:
            summary[persona] = float(np.mean(vals))
    return summary


# ────────────────────────────────────────────────────────────────────────────
# Fact-slice JS — teacher-forced on the taught completion text itself.
# Mirrors scripts/issue444_fact_slice_js.
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _completion_logdist(model, tok, persona: str, q: str, c: str, device: str) -> torch.Tensor:
    """Full-vocab log-softmax at each taught-completion position (teacher-forced)."""
    prompt = _chat_ids(tok, persona, q).to(device)
    c_ids = tok(c, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([prompt, c_ids], dim=1)
    logits = model(full).logits[0].float()
    start = prompt.shape[1] - 1
    end = start + c_ids.shape[1]
    return torch.log_softmax(logits[start:end], dim=-1)


@torch.no_grad()
def fact_slice_js_per_persona(
    model, tok, teach_rows: list[dict], device: str, reference: str, others: list[str]
) -> dict[str, float]:
    per_persona: dict[str, list[float]] = {p: [] for p in others}
    for i, r in enumerate(teach_rows):
        q, c = r["question"], r["completion"]
        ref_ld = _completion_logdist(model, tok, reference, q, c, device)
        for other in others:
            p_ld = _completion_logdist(model, tok, other, q, c, device)
            n = min(ref_ld.shape[0], p_ld.shape[0])
            per_persona[other].append(_js_from_logprobs(ref_ld[:n], p_ld[:n]))
        if (i + 1) % 25 == 0:
            logger.info("  fact_slice_js %d/%d rows", i + 1, len(teach_rows))
    return {p: float(np.mean(v)) if v else float("nan") for p, v in per_persona.items()}


# ────────────────────────────────────────────────────────────────────────────
# Repro metadata
# ────────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def _drop_model(model) -> None:
    """Free the HF model + clear CUDA cache so vLLM can claim the GPU."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:  # noqa: C901 — orchestrates 5 predictor passes + consistency gate, intentionally linear
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-cos-probes", type=int, default=60, help="A-family probes for cosine")
    ap.add_argument("--n-js-probes", type=int, default=60, help="A-family probes for JS")
    ap.add_argument("--js-r", type=int, default=8, help="responses sampled per persona per probe")
    ap.add_argument("--js-max-tok", type=int, default=256)
    ap.add_argument("--cos-b-r", type=int, default=4, help="responses for cosine (b) response-mean")
    ap.add_argument("--cos-b-max-tok", type=int, default=256)
    ap.add_argument(
        "--teach-rows", default="eval_results/issue_444/bystander_logprob/teach_rows.json"
    )
    ap.add_argument("--out", default="eval_results/issue_494/predictor_444_canonical.json")
    ap.add_argument("--gpu-mem", type=float, default=0.60, help="vLLM gpu_memory_utilization")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny slice: n_probes=5, R=2, 1 pair (marine_biologist vs local_historian), L21 only.",
    )
    ap.add_argument("--skip-cosine-b", action="store_true")
    ap.add_argument("--skip-js", action="store_true")
    ap.add_argument("--skip-prior", action="store_true")
    ap.add_argument("--skip-fact-slice", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_cos_probes = 5
        args.n_js_probes = 5
        args.js_r = 2
        args.js_max_tok = 64
        args.cos_b_r = 1
        args.cos_b_max_tok = 64
        args.out = "eval_results/issue_494/predictor_444_canonical.smoke.json"
        # in smoke we restrict to 1 pair (marine vs local_historian) and L21 only
        smoke_others = ["local_historian"]
        smoke_layers = [HEADLINE_LAYER]
    else:
        smoke_others = OTHERS
        smoke_layers = LAYERS

    others_active = smoke_others
    layers_active = smoke_layers

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading %s on %s (smoke=%s; %d probes; R=%d; max_tok=%d; pairs=%s; layers=%s)",
        args.model,
        device,
        args.smoke,
        args.n_cos_probes,
        args.js_r,
        args.js_max_tok,
        others_active,
        layers_active,
    )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Probes ──
    a_family = [p for probes in build_reformulation_probes(ENTITY).values() for p in probes]
    cos_probes = a_family[: args.n_cos_probes]
    js_probes = a_family[: args.n_js_probes]
    logger.info(
        "A-family probes available: %d (using cos=%d, js=%d)",
        len(a_family),
        len(cos_probes),
        len(js_probes),
    )

    # ── Teach rows for prior + fact-slice ──
    teach_rows_path = REPO / args.teach_rows
    teach_rows = json.loads(teach_rows_path.read_text())["rows"]
    if args.smoke:
        teach_rows = teach_rows[:8]
    logger.info("Teach rows loaded: %d", len(teach_rows))

    # ── Active persona subset (smoke=1 pair, else full panel) ──
    active_personas = {REFERENCE: PERSONA_PROMPTS[REFERENCE]}
    for p in others_active:
        active_personas[p] = PERSONA_PROMPTS[p]

    results: dict = {
        "_doc": (
            "#494 Phase 1: recompute #444 predictors at canonical R=8/max_tok=256. "
            "Recipe-independent quantities (cosine_a, bystander_logprob, fact_slice_js) "
            "must reproduce inline-444 byte-identically within 1e-4; canonical JS may "
            "drift in magnitude but the bystander rank-ordering should be stable "
            "(Spearman rho >= 0.85 vs inline JS ordering)."
        ),
        "model": args.model,
        "entity": ENTITY,
        "reference_persona": REFERENCE,
        "others": others_active,
        "layers": layers_active,
        "headline_layer": HEADLINE_LAYER,
        "config": vars(args),
        "smoke": args.smoke,
        "git_commit": _git_commit(),
        "started_at": _now_iso(),
        "predictors": {p: {} for p in others_active},
        "per_layer_cosine": {"cosine_a": {}, "cosine_b": {}},
        "_consistency_check": None,
    }

    # ────────────────────────────────────────────────────────────────────
    # Load HF model first (used for cosine_a, cosine_b, JS, fact_slice_js)
    # ────────────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device, token=os.environ.get("HF_TOKEN")
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=os.environ.get("HF_TOKEN"),
        ).eval()

    # ── Cosine (a) ──
    logger.info(
        "Cosine (a) — last-input-token, %d probes, layers=%s", len(cos_probes), layers_active
    )
    acts_a = {
        p: last_token_acts(model, tok, p, cos_probes, device, layers_active)
        for p in active_personas
    }
    cos_a_per = cosine_vs_reference_per_layer(acts_a, REFERENCE, others_active, layers_active)
    del acts_a
    torch.cuda.empty_cache()
    results["per_layer_cosine"]["cosine_a"] = cos_a_per
    for other in others_active:
        results["predictors"][other]["cosine_a_L21"] = cos_a_per[other][str(HEADLINE_LAYER)]
        results["predictors"][other]["cosine_a_per_layer"] = cos_a_per[other]
    # Per-cell checkpoint
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Checkpoint after cosine_a → %s", out_path)

    # ── Cosine (b) — response-mean ──
    if not args.skip_cosine_b:
        logger.info(
            "Cosine (b) — response-mean, R=%d, max_tok=%d, %d probes",
            args.cos_b_r,
            args.cos_b_max_tok,
            len(cos_probes),
        )
        acts_b = {
            p: response_mean_acts_per_persona(
                model, tok, p, cos_probes, args.cos_b_r, args.cos_b_max_tok, device, layers_active
            )
            for p in active_personas
        }
        cos_b_per = cosine_vs_reference_per_layer(acts_b, REFERENCE, others_active, layers_active)
        del acts_b
        torch.cuda.empty_cache()
        results["per_layer_cosine"]["cosine_b"] = cos_b_per
        for other in others_active:
            results["predictors"][other]["cosine_b_L21"] = cos_b_per[other][str(HEADLINE_LAYER)]
            results["predictors"][other]["cosine_b_per_layer"] = cos_b_per[other]
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after cosine_b → %s", out_path)

    # ── JS canonical R=8/256 ──
    if not args.skip_js:
        logger.info(
            "JS RB sequence-level — R=%d, max_tok=%d, %d probes, %d pairs",
            args.js_r,
            args.js_max_tok,
            len(js_probes),
            len(others_active),
        )
        js_raw = js_vs_reference_canonical(
            model, tok, js_probes, args.js_r, args.js_max_tok, device, REFERENCE, others_active
        )
        for other in others_active:
            js_val = js_raw[other]
            results["predictors"][other]["js_on_topic"] = js_val
            results["predictors"][other]["js_similarity_M"] = (
                1.0 - js_val if math.isfinite(js_val) else float("nan")
            )
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after JS → %s", out_path)

    # ── Fact-slice JS ──
    if not args.skip_fact_slice:
        logger.info("Fact-slice JS over %d teach rows", len(teach_rows))
        fact_js = fact_slice_js_per_persona(
            model, tok, teach_rows, device, REFERENCE, others_active
        )
        for other in others_active:
            results["predictors"][other]["fact_slice_js"] = fact_js[other]
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after fact_slice_js → %s", out_path)

    # ── Release HF model so vLLM can claim the GPU for the prior ──
    _drop_model(model)

    # ── Bystander prior (vLLM) ──
    if not args.skip_prior:
        active_for_prior = {p: PERSONA_PROMPTS[p] for p in [REFERENCE, *list(others_active)]}
        logger.info(
            "Bystander prior - vLLM teacher-force on %d teach rows x %d personas",
            len(teach_rows),
            len(active_for_prior),
        )
        prior_results = bystander_logprob_all_personas(
            args.model, teach_rows, active_for_prior, gpu_memory_utilization=args.gpu_mem
        )
        results["predictors_reference_prior"] = prior_results[REFERENCE]
        for other in others_active:
            results["predictors"][other]["bystander_logprob"] = prior_results[other]
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after bystander_prior → %s", out_path)

    # ────────────────────────────────────────────────────────────────────
    # Consistency check vs inline-444 (recipe-independent quantities byte-
    # identical within 1e-4; JS rank-correlation >= 0.85).
    # ────────────────────────────────────────────────────────────────────
    consistency: dict = {
        "tolerance_byte_identical": 1e-4,
        "tolerance_js_rank_corr": 0.85,
        "per_persona_diff": {},
        "overall_pass": True,
        "smoke": args.smoke,
        "details": {},
    }

    # Only run the full consistency gate when ALL 6 bystanders + all predictors
    # were computed (i.e. the production path). Smoke = correctness only.
    if (
        not args.smoke
        and set(others_active) == set(OTHERS)
        and not args.skip_js
        and not args.skip_prior
        and not args.skip_fact_slice
    ):
        # Byte-identical: cosine_a, bystander_logprob, fact_slice_js
        # (Inline fact-slice values loaded from fact_slice_js.json)
        fs_path = REPO / "eval_results/issue_444/bystander_logprob/fact_slice_js.json"
        inline_fact = json.loads(fs_path.read_text())["summary"]

        for other in OTHERS:
            inline = INLINE_444_VALUES[other]
            diff = {
                "cosine_a_L21_diff": abs(
                    results["predictors"][other]["cosine_a_L21"] - inline["cosine_a_L21"]
                ),
                "bystander_logprob_diff": abs(
                    results["predictors"][other]["bystander_logprob"] - inline["bystander_logprob"]
                ),
                "fact_slice_js_diff": abs(
                    results["predictors"][other]["fact_slice_js"] - inline_fact[other]["js_fact"]
                ),
            }
            consistency["per_persona_diff"][other] = diff
            for key in ("cosine_a_L21_diff", "bystander_logprob_diff", "fact_slice_js_diff"):
                if diff[key] > consistency["tolerance_byte_identical"]:
                    consistency["overall_pass"] = False

        # JS rank-correlation gate (R=8/256 canonical vs R=6/48 inline ordering)
        # Inline JS values from inline-444 results (as JS, not M).
        inline_js_per = {p: 1.0 - INLINE_444_VALUES[p]["js_on_topic_inline_M"] for p in OTHERS}
        canonical_js_per = {p: results["predictors"][p]["js_on_topic"] for p in OTHERS}
        # spearmanr of per-bystander orderings (n=6 — small but the published
        # inline gate is "ordering doesn't reshuffle materially").
        canonical_vals = [canonical_js_per[p] for p in OTHERS]
        inline_vals = [inline_js_per[p] for p in OTHERS]
        rho, p_val = spearmanr(canonical_vals, inline_vals)
        consistency["details"]["js_rank_corr"] = {
            "rho": float(rho) if not math.isnan(rho) else None,
            "p_value": float(p_val) if not math.isnan(p_val) else None,
            "n_bystanders": len(OTHERS),
            "canonical_js_per_bystander": canonical_js_per,
            "inline_js_per_bystander": inline_js_per,
        }
        if rho is None or math.isnan(rho) or rho < consistency["tolerance_js_rank_corr"]:
            consistency["overall_pass"] = False

        if consistency["overall_pass"]:
            logger.info(
                "Phase-1 consistency check: PASS (cosine_a / prior / fact_slice JS byte-equal "
                "within 1e-4; JS rank-corr rho=%.3f >= 0.85)",
                rho,
            )
        else:
            logger.error(
                "Phase-1 consistency check: FAIL. Diffs:\n%s",
                json.dumps(consistency["per_persona_diff"], indent=2),
            )

    results["_consistency_check"] = consistency
    results["finished_at"] = _now_iso()
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("WROTE %s", out_path)

    # Pretty summary
    print("\n================ #494 Phase 1 predictor_444_canonical ================")
    for other in others_active:
        cell = results["predictors"][other]
        print(
            f"  {other:22}  "
            f"cos_a_L21={cell.get('cosine_a_L21', float('nan')):.4f}  "
            f"cos_b_L21={cell.get('cosine_b_L21', float('nan')):.4f}  "
            f"js={cell.get('js_on_topic', float('nan')):.4f}  "
            f"prior={cell.get('bystander_logprob', float('nan')):+.4f}  "
            f"fact_js={cell.get('fact_slice_js', float('nan')):.4f}"
        )
    if results["_consistency_check"] is not None:
        verdict = "PASS" if consistency["overall_pass"] else "FAIL"
        print(f"\nConsistency vs inline-444: {verdict}")
        if not consistency["overall_pass"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
