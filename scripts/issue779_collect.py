#!/usr/bin/env python3
"""Issue #779: shared data-collection pass (produces R1/R2/R3 inputs + the DV).

ONE forward-pass sweep produces c_x, v(x), R2's per-token pooled projections, the
R3 per-token subset, and the judged trait DV (plan §4.3). Reuses the #594
hook-capture path (analysis.extraction.extract_layer_activations, the #666/#545
OOM-safe hook), eval.generation for rollouts, eval.batch_judge for the DV. Every
phase honors the same trait/condition/question/rollout caps (smoke = the sweep
at tiny N).

Passes (independent; sharded over --gpu-id workers in production):

  (A) Eval-context pass — 3 traits x (8 system-prompting + 5 many-shot = 13
      conditions) x N questions x 10 rollouts. Per (trait, condition, question):
        - c_x = last-prompt-token AND mean-prompt-token activation, all 28 layers.
        - 10 vLLM rollouts (temp 1.0, max_new_tokens=1024).
        - per rollout: v(x) = mean-response activation (all 28 layers) via
          teacher-force; R2 pooled projections mean/max/topk/last of
          <response_token_i, r_B_l> as SCALARS (all layers, no per-token storage);
          H2b peak-token identity (index + string) of the MAX-projection token.
        - judge each rollout graded 0-100 (DROP-NEVER-COERCE).
      R3 subset (a few thousand ctx x <=6 layers): ALSO cache the full per-token
      context + answer activation stack for R3(b) decay + R3(c) CKA. R3(b) is
      computed from THIS 10-rollout pass (expected-per-position), NOT pass B.

  (B) Train-context pass — >=5000 LMSYS contexts x 1 rollout; c_x (last+mean) +
      v(x) at all 28 layers on the Step-0 subset / selected layers on the full
      corpus. LMSYS gate probe first (WildChat / UltraChat fallback on 403).

  Step-0 oracle-headroom probe — on a 500-context subset at ALL 28 layers:
  <c_x, r_B_L> (PV raw) vs <v(x), r_B_L> (oracle) within-condition Pearson vs
  judge, per trait x layer x mode. Prints GATE_0 lines; selects the read-out
  layer per trait.

Output cache: data/issue_779/pass_a/, pass_b/, step0/ (+ HF issue779_monitoring/).
--smoke runs the IDENTICAL path on 1 trait x 1 condition x 2 Q x 2 rollouts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_collect")

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
R2_TOPK = 3  # top-k pooling operator = mean of the top-3 per-token projections.


# ── context enumeration ───────────────────────────────────────────────────────


def eval_context_conditions(trait: str) -> list[dict]:
    """The 13 eval-context conditions for a trait (8 system + 5 many-shot).

    Each condition is a dict: {mode, cond_id, system_prompt|None, n_shot}. The
    many-shot conditions carry n_shot in {0,5,10,15,20} and NO system prompt
    (PV's many-shot monitoring: few-shot exemplars, no explicit instruction).
    """
    conds = []
    for i, sp in enumerate(C.EVAL_SYSTEM_PROMPTS[trait]):
        conds.append({"mode": "system", "cond_id": f"sys{i}", "system_prompt": sp, "n_shot": 0})
    for k in C.MANY_SHOT_COUNTS:
        conds.append(
            {"mode": "many_shot", "cond_id": f"shot{k}", "system_prompt": None, "n_shot": k}
        )
    return conds


def build_many_shot_history(exemplars: list[dict], k: int) -> list[dict]:
    """Multi-turn (user, assistant) history from k trait-exhibiting exemplars.

    exemplars: [{question, response}] (trait-exhibiting, judge-filtered). Returns
    a chat message list [{"role":"user"...},{"role":"assistant"...}, ...] of the
    first k exemplars, to prepend before the eval question.
    """
    history = []
    for ex in exemplars[:k]:
        history.append({"role": "user", "content": ex["question"]})
        history.append({"role": "assistant", "content": ex["response"]})
    return history


def build_eval_prompt_messages(trait: str, cond: dict, question: str, exemplars: list[dict]):
    """Chat message list for one (trait, condition, question) eval context."""
    messages = []
    if cond["mode"] == "system":
        messages.append({"role": "system", "content": cond["system_prompt"]})
        messages.append({"role": "user", "content": question})
    else:  # many_shot: NO system prompt; k trait-exhibiting exemplars then question
        messages.extend(build_many_shot_history(exemplars, cond["n_shot"]))
        messages.append({"role": "user", "content": question})
    return messages


# ── prompt-side c_x capture (last + mean prompt token, all layers) ────────────


def capture_context_vector(model, tokenizer, messages, layers: list[int]) -> dict:
    """c_x = last-prompt-token AND mean-prompt-token activation, all layers.

    Returns {"last": (L, H), "mean": (L, H), "prompt_len": int}. Uses the
    OOM-safe hook (output_hidden_states=False). Fail-loud position assert on the
    assistant-header suffix (the #594 control).
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
    suffix = tokenizer.decode(inputs["input_ids"][0, -3:])
    assert suffix == C.GENERATION_SUFFIX, (
        f"c_x position assert failed: last-3 decode {suffix!r} != {C.GENERATION_SUFFIX!r}"
    )
    captured = extract_layer_activations(
        model, inputs["input_ids"], layers, attention_mask=inputs.get("attention_mask")
    )
    last, mean = [], []
    for li in layers:
        hs = captured[li][0]  # (T, H)
        last.append(hs[-1, :].float().cpu())
        mean.append(hs.float().cpu().mean(dim=0))
    return {
        "last": torch.stack(last),  # (L, H)
        "mean": torch.stack(mean),  # (L, H)
        "prompt_len": int(inputs["input_ids"].shape[1]),
    }


# ── answer-side v(x) + R2 pooled projection capture ───────────────────────────


def capture_answer_vector(
    model,
    tokenizer,
    messages,
    response: str,
    layers: list[int],
    r_b_by_trait: dict[str, torch.Tensor],
    *,
    keep_per_token: bool = False,
) -> dict | None:
    """v(x)=mean-response act + R2 pooled projections onto every-layer r_B.

    Teacher-forces (context + response) and, over the RESPONSE token span:
      - v(x) = mean-response activation, all layers -> (L, H).
      - R2 pooled projections onto each trait's r_B at every layer: mean / max /
        topk(=top-3 mean) / last of <response_token_i, r_B_l> as SCALARS.
      - H2b peak-token identity (token index + decoded string) of the argmax
        projection, per trait per layer (for the analyzer).
      - optionally the full per-token (n_resp, L, H) answer stack (R3 subset).
    Returns None if the response is empty.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_len = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].shape[1]
    full_messages = [*messages, {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= prompt_len:
        return None

    captured = extract_layer_activations(
        model, full_inputs["input_ids"], layers, attention_mask=full_inputs.get("attention_mask")
    )
    n_layers = len(layers)
    # (n_resp, L, H) response-token activation stack.
    resp_stack = torch.stack(
        [captured[li][0, prompt_len:full_len, :].float().cpu() for li in layers], dim=1
    )  # (n_resp, L, H)
    n_resp = resp_stack.shape[0]
    v_x = resp_stack.mean(dim=0)  # (L, H)

    resp_ids = full_inputs["input_ids"][0, prompt_len:full_len].cpu()

    pooled: dict[str, dict[str, list[float]]] = {}
    peak: dict[str, dict[str, dict]] = {}
    for trait, r_b in r_b_by_trait.items():
        r_b = r_b.to(torch.float32)  # (L, H) block-index layers, aligned to `layers`
        assert r_b.shape[0] == n_layers, (trait, r_b.shape, n_layers)
        # proj[i, l] = <resp_stack[i, l], r_b[l]>  -> (n_resp, L)
        proj = torch.einsum("nlh,lh->nl", resp_stack, r_b)  # (n_resp, L)
        mean_p = proj.mean(dim=0)  # (L,)
        max_p, argmax_i = proj.max(dim=0)  # (L,), (L,)
        last_p = proj[-1, :]  # (L,)
        k = min(R2_TOPK, n_resp)
        topk_p = proj.topk(k, dim=0).values.mean(dim=0)  # (L,)
        pooled[trait] = {
            "mean": mean_p.tolist(),
            "max": max_p.tolist(),
            "topk": topk_p.tolist(),
            "last": last_p.tolist(),
        }
        # H2b peak-token identity per layer (analyzer instrumentation).
        peak[trait] = {}
        for li_pos, layer_idx in enumerate(layers):
            ti = int(argmax_i[li_pos].item())
            peak[trait][str(layer_idx)] = {
                "token_index": ti,
                "token_str": tokenizer.decode(resp_ids[ti : ti + 1]),
                "proj": float(max_p[li_pos].item()),
            }

    out = {"v_x": v_x, "n_resp": n_resp, "pooled": pooled, "peak_token": peak}
    if keep_per_token:
        out["per_token"] = resp_stack  # (n_resp, L, H) — R3 subset (subset of layers)
    return out


# ── vLLM generation (chunked) ─────────────────────────────────────────────────


def _vllm_generate_chunked(llm, prompt_texts, sampling_params) -> list[list[str]]:
    out: list[list[str]] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] collect-generate chunk %d/%d (%d prompts x n=%d)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sampling_params.n,
        )
        chunk_out = llm.generate(chunk, sampling_params, use_tqdm=False)
        for o in chunk_out:
            out.append([c.text for c in o.outputs])
    return out


# ── exemplar pool for many-shot conditions ────────────────────────────────────


def build_exemplar_pool(llm, tokenizer, trait: str, n_exemplars: int) -> list[dict]:
    """Trait-exhibiting (question, response) exemplars for the many-shot rig.

    Generated on-policy from the model under the strongest positive system prompt
    (PV system-prompt 1) over the extraction-set questions, so each exemplar
    exhibits the trait (PV's many-shot rig: exemplars vary in content, all
    exhibit the target trait). Not judge-filtered at smoke scale (kept simple);
    the production path could add a judge-filter, but the exemplars are ICL
    context, not the DV.
    """
    from vllm import SamplingParams

    artifacts = C.load_extraction_artifacts(trait)
    questions = artifacts["extraction_questions"][:n_exemplars]
    strong_sys = C.EVAL_SYSTEM_PROMPTS[trait][0]
    prompt_texts = []
    for q in questions:
        messages = [
            {"role": "system", "content": strong_sys},
            {"role": "user", "content": q},
        ]
        prompt_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=256, seed=7)
    gen = _vllm_generate_chunked(llm, prompt_texts, sp)
    return [{"question": q, "response": g[0]} for q, g in zip(questions, gen, strict=True)]


# ── Pass A (eval-context) ─────────────────────────────────────────────────────


def run_pass_a(
    model,
    tokenizer,
    llm,
    layers: list[int],
    r3_layers: list[int],
    r_b_by_trait: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    traits: list[str],
    n_conditions: int,
    n_questions: int,
    n_rollouts: int,
    n_exemplars: int,
    r3_max_contexts: int,
) -> dict:
    """Eval-context pass: c_x + rollouts + v(x) + R2 pooled + judge, per cell.

    Checkpoint-per-cell: each (trait, condition) writes its own JSON the moment
    it completes. Returns a summary dict of cells produced.
    """
    from vllm import SamplingParams

    pass_a_dir = out_dir / "pass_a"
    pass_a_dir.mkdir(parents=True, exist_ok=True)
    r3_dir = out_dir / "r3_subset"
    r3_dir.mkdir(parents=True, exist_ok=True)
    sp = SamplingParams(n=n_rollouts, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)

    cells_done = []
    r3_ctx_count = 0
    for trait in traits:
        artifacts = C.load_extraction_artifacts(trait)
        eval_q = artifacts["eval_questions"][:n_questions]
        conds = eval_context_conditions(trait)[:n_conditions]
        exemplars = build_exemplar_pool(llm, tokenizer, trait, n_exemplars)

        for cond in conds:
            cell_id = f"{trait}__{cond['cond_id']}"
            cell_path = pass_a_dir / f"{cell_id}.json"
            cx_path = pass_a_dir / f"{cell_id}_cx.pt"
            if cell_path.exists() and cx_path.exists():
                logger.info("[pass_a] %s already complete; skip", cell_id)
                cells_done.append(cell_id)
                continue

            # 1. Build eval prompts; capture c_x per question.
            prompt_texts = []
            cx_last, cx_mean = [], []
            for q in eval_q:
                messages = build_eval_prompt_messages(trait, cond, q, exemplars)
                cx = capture_context_vector(model, tokenizer, messages, layers)
                cx_last.append(cx["last"])
                cx_mean.append(cx["mean"])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_texts.append(text)

            # 2. Generate 10 rollouts per question.
            gen = _vllm_generate_chunked(llm, prompt_texts, sp)  # per-q list of n

            # 3. Per rollout: ONE teacher-force at the full layer set gives
            #    v(x) (-> oracle proj), R2 pooled projections, and the H2b peak
            #    token in a single pass (no double capture). The bounded R3
            #    per-token subset is a SEPARATE small capture at r3_layers only.
            r_b = r_b_by_trait[trait].to(torch.float32)  # (L, H)
            rollout_records = []  # for judging + analysis
            judge_completions: dict[str, list[str]] = {}
            v_proj: dict = {}  # {qi: {ri: {layer: <v(x), r_b>}}}
            keep_r3 = r3_ctx_count < r3_max_contexts
            for qi, (q, comps) in enumerate(zip(eval_q, gen, strict=True)):
                judge_completions[f"q{qi:03d}"] = comps
                v_proj[str(qi)] = {}
                messages = build_eval_prompt_messages(trait, cond, q, exemplars)
                for ri, comp in enumerate(comps):
                    av = capture_answer_vector(
                        model,
                        tokenizer,
                        messages,
                        comp,
                        layers,
                        {trait: r_b},
                        keep_per_token=False,
                    )
                    rec = {"qi": qi, "ri": ri, "response": comp}
                    if av is None:
                        rec["empty"] = True
                        v_proj[str(qi)][str(ri)] = None
                    else:
                        rec["n_resp"] = av["n_resp"]
                        rec["pooled"] = av["pooled"][trait]
                        rec["peak_token"] = av["peak_token"][trait]
                        proj = torch.einsum("lh,lh->l", av["v_x"], r_b)  # (L,)
                        v_proj[str(qi)][str(ri)] = {
                            str(layers[i]): float(proj[i]) for i in range(len(layers))
                        }
                    rollout_records.append(rec)
                    # R3 per-token subset (bounded): a SECOND small capture at
                    # r3_layers only (a few thousand ctx x <=6 layers total).
                    if keep_r3 and av is not None:
                        av_r3 = capture_answer_vector(
                            model,
                            tokenizer,
                            messages,
                            comp,
                            r3_layers,
                            {trait: r_b_r3(r_b, layers, r3_layers)},
                            keep_per_token=True,
                        )
                        if av_r3 is not None and "per_token" in av_r3:
                            torch.save(
                                {
                                    "trait": trait,
                                    "cond_id": cond["cond_id"],
                                    "qi": qi,
                                    "ri": ri,
                                    "r3_layers": r3_layers,
                                    "answer_per_token": av_r3["per_token"].to(torch.float16),
                                    "context_last": cx_last[qi][
                                        [layers.index(li) for li in r3_layers]
                                    ].to(torch.float16),
                                },
                                r3_dir / f"{cell_id}_r{qi:03d}_{ri:02d}.pt",
                            )
                if keep_r3:
                    r3_ctx_count += 1

            # 4. Judge the rollouts (graded 0-100, DROP-NEVER-COERCE).
            judge_scores, dropped = _judge_cell(trait, judge_completions, pass_a_dir, cell_id)

            # 6. Write cell checkpoint (JSON scalars) + c_x tensor.
            cell = {
                "trait": trait,
                "cond_id": cond["cond_id"],
                "mode": cond["mode"],
                "n_shot": cond["n_shot"],
                "n_questions": len(eval_q),
                "n_rollouts": n_rollouts,
                "rollout_seed": 42,  # SamplingParams seed for the raw-completions filename
                "rollouts": rollout_records,
                "judge_scores": judge_scores,  # {qXXX__idx__ci: score|null}
                "judge_dropped": dropped,
                "oracle_proj": v_proj,  # {qi: {ri: {layer: <v(x),r_b>}}}
            }
            C.write_json_atomic(cell_path, cell)
            torch.save(
                {
                    "cell_id": cell_id,
                    "cx_last": torch.stack(cx_last),  # (n_q, L, H)
                    "cx_mean": torch.stack(cx_mean),  # (n_q, L, H)
                    "layers": layers,
                },
                cx_path,
            )
            cells_done.append(cell_id)
            logger.info("[pass_a] %s done (%d q x %d rollouts)", cell_id, len(eval_q), n_rollouts)

    return {"cells": cells_done, "r3_contexts": r3_ctx_count}


def r_b_r3(r_b_full: torch.Tensor, layers: list[int], r3_layers: list[int]) -> torch.Tensor:
    """Slice a full-layer r_B (L, H) down to the r3_layers subset (aligned rows).

    ``capture_answer_vector`` requires the passed r_B's row order to match the
    layers it captures; the R3 subset captures only ``r3_layers``, so r_B must be
    sliced to those rows in the same order.
    """
    idx = [layers.index(li) for li in r3_layers]
    return r_b_full[idx]


def _judge_cell(trait, judge_completions, out_dir, cell_id):
    """Judge one cell's rollouts (N=5 graded draws, mean over valid); return
    ({custom_id: mean_score|null}, dropped_rollout_count).

    The registered DV is the N=5 graded-0-100 mean @ temp 1.0 (llm-judging.md
    rule 4 + plan §11): each rollout completion is judged ``JUDGE_N_DRAWS`` times
    (DROP-NEVER-COERCE per draw) and the valid draws are mean-aggregated. A
    rollout with 0 valid draws -> score None (DROPPED). ``dropped`` counts fully
    dropped rollouts (backward-compatible with the prior return contract).
    """
    save_raw = out_dir / f"judge_{cell_id}.json"
    cache_dir = out_dir / "judge_cache"
    # batch_judge expects {persona: {question: [completions]}}; use cell_id as
    # persona and qXXX as question.
    completions = {cell_id: judge_completions}
    agg = C.judge_rollouts_n5(trait, completions, save_raw, cache_dir)
    scores: dict[str, float | None] = {}
    dropped = 0
    for custom_id, (mean, _n_valid, _n_draws) in agg.items():
        scores[custom_id] = mean
        if mean is None:
            dropped += 1
    return scores, dropped


# ── Pass B (train-context / LMSYS) ────────────────────────────────────────────


def load_train_contexts(n_contexts: int, smoke: bool) -> tuple[list[str], str]:
    """Load train-context user prompts (LMSYS tier-1; WildChat/UltraChat fallback).

    Gate probe first: LMSYS is gated=auto — on a 403 fall back to WildChat-1M
    (ungated) or UltraChat-200k. Returns (prompts, source_name).
    """
    from datasets import load_dataset

    def _first_user_turn(row, field_candidates):
        for f in field_candidates:
            if row.get(f):
                val = row[f]
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val[0].get("content") or val[0].get("value")
                if isinstance(val, str):
                    return val
        return None

    cap = 4 if smoke else n_contexts
    sources = [
        ("lmsys/lmsys-chat-1m", "conversation", ["conversation"]),
        ("allenai/WildChat-1M", "conversation", ["conversation"]),
        ("HuggingFaceH4/ultrachat_200k", "messages", ["messages", "prompt"]),
    ]
    for repo, _split_field, fields in sources:
        try:
            split = "train_sft" if "ultrachat" in repo else "train"
            ds = load_dataset(repo, split=split, streaming=True)
            prompts = []
            for row in ds:
                p = _first_user_turn(row, fields)
                if p and isinstance(p, str) and len(p.strip()) > 0:
                    prompts.append(p.strip())
                if len(prompts) >= cap:
                    break
            if prompts:
                logger.info("Loaded %d train contexts from %s", len(prompts), repo)
                return prompts, repo
        except Exception as e:
            logger.warning("train-context source %s failed: %s; trying fallback", repo, e)
    raise RuntimeError("all train-context sources failed (LMSYS/WildChat/UltraChat)")


def run_pass_b(
    model,
    tokenizer,
    llm,
    layers: list[int],
    r_b_by_trait: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    n_contexts: int,
    smoke: bool,
) -> dict:
    """Train-context pass: c_x (last+mean) + v(x) at all layers, 1 rollout each.

    Behavior-agnostic (no trait labels) — h reconstructs v(x) from c_x. Writes a
    single tensor bundle. LMSYS eval-context holdout: train contexts are real
    user prompts, disjoint from the PV eval questions by construction (different
    source).
    """
    from vllm import SamplingParams

    pass_b_dir = out_dir / "pass_b"
    pass_b_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = pass_b_dir / "train_context_vectors.pt"
    if bundle_path.exists():
        logger.info("[pass_b] bundle exists; skip")
        return {"skipped": True}

    prompts, source = load_train_contexts(n_contexts, smoke)
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)

    prompt_texts = []
    cx_last, cx_mean = [], []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        cx = capture_context_vector(model, tokenizer, messages, layers)
        cx_last.append(cx["last"])
        cx_mean.append(cx["mean"])
        prompt_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    gen = _vllm_generate_chunked(llm, prompt_texts, sp)

    v_list = []
    kept_idx = []
    for i, (p, comps) in enumerate(zip(prompts, gen, strict=True)):
        messages = [{"role": "user", "content": p}]
        av = capture_answer_vector(
            model, tokenizer, messages, comps[0], layers, r_b_by_trait, keep_per_token=False
        )
        if av is None:
            continue
        v_list.append(av["v_x"])
        kept_idx.append(i)

    cx_last_t = torch.stack([cx_last[i] for i in kept_idx])  # (N, L, H)
    cx_mean_t = torch.stack([cx_mean[i] for i in kept_idx])
    v_t = torch.stack(v_list)  # (N, L, H)
    torch.save(
        {
            "cx_last": cx_last_t,
            "cx_mean": cx_mean_t,
            "v_x": v_t,
            "prompts": [prompts[i] for i in kept_idx],
            "layers": layers,
            "source": source,
            "metadata": C.reproducibility_metadata({"script": "issue779_collect", "pass": "B"}),
        },
        bundle_path,
    )
    logger.info(
        "[pass_b] wrote %d train contexts (source=%s) shape %s",
        len(kept_idx),
        source,
        tuple(v_t.shape),
    )
    return {"n_contexts": len(kept_idx), "source": source}


# ── Step-0 oracle-headroom probe ──────────────────────────────────────────────


def _within_condition_pearson(cond_x: list[np.ndarray], cond_y: list[np.ndarray]) -> float:
    """Mean within-condition Pearson r (PV: exclude conditions with y-std < 1)."""
    rs = []
    for x, y in zip(cond_x, cond_y, strict=True):
        if len(x) < 3 or float(np.std(y)) < 1.0 or float(np.std(x)) == 0.0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def run_step0_probe(  # noqa: C901  (pre-existing: nested trait x layer x mode x cell probe)
    out_dir: Path, traits: list[str], layers: list[int], r_b_by_trait: dict[str, torch.Tensor]
) -> dict:
    """Step-0 oracle-headroom: PV-raw vs oracle within-condition r, per trait x layer x mode.

    Reads the pass-A cells (produced above): for each (trait, layer, mode) compute
    within-condition Pearson of <c_last, r_B_l> (pv_raw) and <v(x), r_B_l> (oracle)
    vs the judge score. Prints GATE_0 per trait (best layer by oracle r). Writes
    step0/step0_oracle.json. ``r_b_by_trait`` is the already-loaded r_B dict
    (from --rb-dir), so Step-0 never re-guesses the r_B path.
    """
    pass_a_dir = out_dir / "pass_a"
    step0_dir = out_dir / "step0"
    step0_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for trait in traits:
        r_b = r_b_by_trait[trait].to(torch.float32)  # (L, H)
        cells = sorted(pass_a_dir.glob(f"{trait}__*.json"))
        per_layer_mode: dict = {}
        gate_lines = []
        best = {"system": (None, float("-inf")), "many_shot": (None, float("-inf"))}
        for li_pos, layer_idx in enumerate(layers):
            for mode in ("system", "many_shot"):
                cond_pv_x, cond_or_x, cond_y = [], [], []
                for cp in cells:
                    with open(cp) as f:
                        cell = json.load(f)
                    if cell["mode"] != mode:
                        continue
                    cx_path = pass_a_dir / f"{cell['trait']}__{cell['cond_id']}_cx.pt"
                    cx = torch.load(cx_path, weights_only=True)
                    cx_last = cx["cx_last"].to(torch.float32)  # (n_q, L, H)
                    # PER-(condition, question) unit — matching the PV within-
                    # condition monitoring unit AND stage1.build_eval_matrix.
                    # pv_raw = <c_last[qi], r_b_l> is a PROPERTY OF THE PROMPT
                    # (identical across a question's rollouts), so a per-rollout
                    # correlation would 10x-duplicate x against rollout-level noisy
                    # y (the primary-metric-rollout-level-not-question-averaged bug
                    # class, sibling instance in the Gate-0 / read-out-layer probe).
                    # Aggregate to ONE row per question: pv_raw is per-question;
                    # oracle + judge are the mean over the question's valid rollouts.
                    xs_pv, xs_or, ys = [], [], []
                    by_q: dict[int, list[int]] = {}
                    for rec in cell["rollouts"]:
                        if rec.get("empty"):
                            continue
                        by_q.setdefault(rec["qi"], []).append(rec["ri"])
                    for qi, ris in by_q.items():
                        q_s, q_or = [], []
                        for ri in ris:
                            s = _lookup_score(cell["judge_scores"], qi, ri)
                            orc = cell["oracle_proj"].get(str(qi), {}).get(str(ri))
                            if s is None or orc is None:
                                continue
                            q_s.append(s)
                            q_or.append(float(orc[str(layer_idx)]))
                        if not q_s:  # no valid rollout for this question -> drop
                            continue
                        xs_pv.append(float(torch.dot(cx_last[qi, li_pos, :], r_b[li_pos, :])))
                        xs_or.append(float(np.mean(q_or)))
                        ys.append(float(np.mean(q_s)))
                    if len(ys) >= 3:
                        cond_pv_x.append(np.array(xs_pv))
                        cond_or_x.append(np.array(xs_or))
                        cond_y.append(np.array(ys))
                pv_r = _within_condition_pearson(cond_pv_x, cond_y)
                or_r = _within_condition_pearson(cond_or_x, cond_y)
                per_layer_mode[f"L{layer_idx}_{mode}"] = {
                    "pv_raw_r": pv_r,
                    "oracle_r": or_r,
                    "headroom": (or_r - pv_r)
                    if (np.isfinite(or_r) and np.isfinite(pv_r))
                    else None,
                    "n_conditions": len(cond_y),
                }
                if np.isfinite(or_r) and or_r > best[mode][1]:
                    best[mode] = (layer_idx, or_r)
        # GATE_0 per trait: best layer by oracle (system mode as the primary).
        best_layer = best["system"][0] if best["system"][0] is not None else best["many_shot"][0]
        for mode in ("system", "many_shot"):
            bl = best[mode][0]
            if bl is None:
                continue
            key = f"L{bl}_{mode}"
            plm = per_layer_mode[key]
            gate_lines.append(
                f"GATE_0: trait={trait} mode={mode} best_layer={bl} "
                f"oracle_r={plm['oracle_r']:.4f} pv_raw_r={plm['pv_raw_r']:.4f} "
                f"headroom={plm['headroom']}"
            )
        result[trait] = {
            "per_layer_mode": per_layer_mode,
            "best_layer": best_layer,
            "best_by_mode": {m: best[m][0] for m in best},
            "gate_lines": gate_lines,
        }
        for line in gate_lines:
            print(line, flush=True)
    C.write_json_atomic(step0_dir / "step0_oracle.json", result)
    return result


def _lookup_score(judge_scores: dict, qi: int, ri: int) -> float | None:
    """Resolve a rollout's judge score from the {custom_id: score} map.

    batch_judge custom_id for persona=cell_id, question=qXXX, comp_idx=ri is
    <cell_id>__<idx:05d>__<ri:02d> where idx counts questions. We stored
    judge_scores keyed by the batch_judge custom_id; qi maps to the idx-th
    question in enumeration order (0-based == qi here).
    """
    for cid, s in judge_scores.items():
        parts = cid.split("__")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[-2])
            ci = int(parts[-1])
        except ValueError:
            continue
        if idx == qi and ci == ri:
            return s
    return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 shared data-collection pass.")
    parser.add_argument("--stage", choices=["all", "a", "b", "step0"], default="all")
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument("--model", default=C.DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779")
    parser.add_argument(
        "--rb-dir",
        type=Path,
        default=None,
        help="dir holding r_b/<trait>.pt (default: <out-dir>/r_b)",
    )
    parser.add_argument("--n-conditions", type=int, default=13)
    parser.add_argument("--n-questions", type=int, default=40)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-exemplars", type=int, default=20)
    parser.add_argument("--n-train-contexts", type=int, default=5000)
    parser.add_argument("--r3-max-contexts", type=int, default=200)
    parser.add_argument(
        "--r3-layers",
        type=int,
        nargs="+",
        default=None,
        help="layers for the R3 per-token subset (default: low/mid/high triple)",
    )
    parser.add_argument("--expected-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=C.EXPECTED_HIDDEN)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    # Smoke caps.
    traits = args.traits[:1] if args.smoke else args.traits
    n_conditions = 1 if args.smoke else args.n_conditions
    n_questions = 2 if args.smoke else args.n_questions
    n_rollouts = 2 if args.smoke else args.n_rollouts
    n_exemplars = 2 if args.smoke else args.n_exemplars
    n_train = 4 if args.smoke else args.n_train_contexts
    r3_max = 4 if args.smoke else args.r3_max_contexts

    out_dir = Path(str(args.out_dir) + "_smoke") if args.smoke else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    C.phase("load_model")
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
    assert hidden == args.expected_hidden, (hidden, args.expected_hidden)
    layers = list(range(n_layers))
    if args.r3_layers is not None:
        r3_layers = args.r3_layers
    else:
        r3_layers = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4})

    # Load r_B for the traits.
    rb_dir = args.rb_dir or (out_dir / "r_b")
    r_b_by_trait = {}
    for trait in traits:
        cand = rb_dir / f"{trait}.pt"
        if not cand.exists():
            # sibling: non-smoke r_b when running collect --smoke reusing a real r_b.
            cand = Path(str(args.out_dir)) / "r_b" / f"{trait}.pt"
        if not cand.exists():
            raise FileNotFoundError(
                f"r_B for {trait} not found ({rb_dir}/{trait}.pt); run issue779_extract_rb.py first"
            )
        blob = torch.load(cand, weights_only=False)
        r_b = blob["r_b"]
        assert r_b.shape == (n_layers, hidden), (
            f"{trait} r_b shape {tuple(r_b.shape)} != ({n_layers}, {hidden}) — "
            "r_B was extracted on a different model; re-extract"
        )
        r_b_by_trait[trait] = r_b

    # Build vLLM engine (or CPU shim).
    if use_cuda:
        from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

        llm = create_vllm_engine(args.model, max_model_len=8192, seed=42)
    else:
        import issue779_extract_rb as R

        llm = R._HFGenShim(model, tokenizer)

    summary: dict = {"traits": traits, "smoke": args.smoke}
    try:
        if args.stage in ("all", "a"):
            C.phase("pass_a")
            summary["pass_a"] = run_pass_a(
                model,
                tokenizer,
                llm,
                layers,
                r3_layers,
                r_b_by_trait,
                out_dir,
                traits=traits,
                n_conditions=n_conditions,
                n_questions=n_questions,
                n_rollouts=n_rollouts,
                n_exemplars=n_exemplars,
                r3_max_contexts=r3_max,
            )
        if args.stage in ("all", "b"):
            C.phase("pass_b")
            summary["pass_b"] = run_pass_b(
                model,
                tokenizer,
                llm,
                layers,
                r_b_by_trait,
                out_dir,
                n_contexts=n_train,
                smoke=args.smoke,
            )
    finally:
        if use_cuda:
            cleanup_vllm(llm)

    if args.stage in ("all", "step0"):
        C.phase("step0")
        summary["step0"] = run_step0_probe(out_dir, traits, layers, r_b_by_trait)

    if not args.no_upload:
        C.phase("upload")
        _upload_collect(out_dir, smoke=args.smoke)
        # Plan §10 row (c): the rollout TEXT ALSO lands under the canonical
        # raw_completions/ prefix (verified) — NOT only under analysis_tensors/.
        # Only Pass A produces rollout text; skip when Pass A was not run.
        if args.stage in ("all", "a"):
            _upload_raw_completions(out_dir, smoke=args.smoke)

    C.write_json_atomic(out_dir / "collect_summary.json", summary)
    note = (
        f"issue779 collect {'SMOKE ' if args.smoke else ''}complete: {json.dumps(summary)[:2000]}"
    )
    C.write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    C.phase("done")
    return 0


def _split_raw_completions(out_dir: Path, staging_dir: Path) -> list[tuple[str, str, str]]:
    """Split each Pass-A cell JSON's rollout TEXT into a per-cell raw-completions file.

    Plan v5 §10 row (c): every rollout string that appears in a Pass-A cell JSON
    as a ``rollouts[*].response`` field gets a canonical copy at
    ``<staging_dir>/{trait}_{cond_id}_seed{seed}.json`` shaped
    ``{trait, condition, seed, rollouts: [{qi, ri, response}]}`` (the analyzer
    enumerates this flat per-cell layout). Pass B persists NO rollout text (only
    v(x) vectors + prompts to its tensor bundle), so it carries no rollout field
    to copy — scope is the Pass-A cells only.

    Seed is read from each cell's ``rollout_seed`` field (the fixed monitoring-rig
    rollout seed, plan §10 Seeds row; ``pass_a`` rollouts are generated with
    ``SamplingParams(..., seed=rollout_seed)``). The verifier consumes the ACTUAL
    filename this function writes (returned per cell), so the seed is never
    hardcoded on the verify side — a future ``rollout_seed`` change cannot silently
    desync the writer's name from the verifier's expected name (sibling of the
    upload-prefix bug class, holistic hardening pass).

    Returns one ``(trait, cond_id, filename)`` per cell written — ``filename`` is
    the staging-relative basename (e.g. ``evil_sys0_seed42.json``), so the caller
    verifies the exact path it produced rather than reconstructing it.
    """
    pass_a_dir = out_dir / "pass_a"
    staging_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str, str]] = []
    if not pass_a_dir.is_dir():
        return written
    # `pass_a/` holds BOTH per-cell JSONs `{trait}__{cond}.json` (carry
    # trait/cond_id/rollouts) AND judge sidecars `judge_{trait}__{cond}.json`
    # (written by _judge_cell:save_raw — keys per_persona/all_scores, NO trait).
    # Skip the `judge_` prefix so the glob only walks real cell JSONs — mirrors
    # the step0 reader's trait-prefixed glob (load_eval_cells uses
    # `{trait}__*.json`). Defense-in-depth: any file missing the required cell
    # keys is skipped rather than KeyError-crashing the upload phase (BLOCKER
    # raw-completions-split-glob-crashes-on-judge-sidecar, code-review v4).
    for cell_path in sorted(pass_a_dir.glob("*.json")):
        if cell_path.name.startswith("judge_"):
            continue
        with open(cell_path) as f:
            cell = json.load(f)
        if not {"trait", "cond_id", "rollouts"} <= cell.keys():
            continue
        trait = cell["trait"]
        cond_id = cell["cond_id"]
        seed = cell.get("rollout_seed", 42)
        rollouts = [
            {"qi": r["qi"], "ri": r["ri"], "response": r.get("response", "")}
            for r in cell.get("rollouts", [])
        ]
        payload = {
            "trait": trait,
            "condition": cond_id,
            "seed": seed,
            "mode": cell.get("mode"),
            "n_shot": cell.get("n_shot"),
            "rollouts": rollouts,
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_collect", "artifact": "raw_completions"}
            ),
        }
        fname = f"{trait}_{cond_id}_seed{seed}.json"
        C.write_json_atomic(staging_dir / fname, payload)
        written.append((trait, cond_id, fname))
    return written


def _upload_raw_completions(out_dir: Path, smoke: bool) -> None:
    """Upload per-cell rollout TEXT under ``issue779_monitoring/raw_completions/``, verified.

    Plan v5 §10 row (c) mandate: the raw completion text MUST land under the
    canonical ``raw_completions/`` prefix (NOT ``analysis_tensors/``), and the
    upload MUST be mechanically verified — every ``(trait, condition)`` the run
    produced has a file there — BEFORE ``phase("done")``. Fail-loud on any miss.
    """
    import tempfile

    from huggingface_hub import HfApi, list_repo_files

    sub = "raw_completions_smoke" if smoke else "raw_completions"
    path_in_repo = f"{C.HF_PREFIX}/{sub}"
    with tempfile.TemporaryDirectory(prefix="issue779_rawcomp_") as tmp:
        staging = Path(tmp)
        written = _split_raw_completions(out_dir, staging)
        if not written:
            raise RuntimeError(
                "raw-completions upload aborted: no Pass-A cell JSONs with rollout "
                f"text found under {out_dir / 'pass_a'} — expected rollout responses to copy"
            )
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging),
            path_in_repo=path_in_repo,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue779: {'smoke ' if smoke else ''}raw completions (rollout text)",
        )
        # Mechanical verification: every produced file lands at the canonical
        # raw_completions/ prefix on a FRESH listing (plan §10 mandate, mirrors
        # the upload-verifier's phantom-URL gate). Verify the EXACT filename each
        # cell wrote (returned by _split_raw_completions) — never a hardcoded
        # seed — so a rollout_seed change cannot desync the expected name from the
        # written name (upload-prefix bug class, holistic hardening).
        repo_files = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset"))
        missing = []
        for trait, cond_id, fname in written:
            if f"{path_in_repo}/{fname}" not in repo_files:
                missing.append(f"{trait}/{cond_id} ({fname})")
        if missing:
            raise RuntimeError(
                "raw-completions upload verification FAILED "
                f"(raw-completions-upload-prefix-missing): {len(missing)} trait x condition "
                f"combos have no file under {path_in_repo}: {missing[:10]}"
            )
    logger.info("raw-completions upload verified: %d cells under %s", len(written), path_in_repo)


# Every raw-TEXT field that MUST be stripped before an analysis_tensors/ upload
# (plan §10 row (b): "activations ONLY; NOT rollout text"). The canonical copy
# of the rollout text lives under raw_completions/ (plan §10 row (c)), so no data
# is lost by stripping it here.
#   - rollouts[].response      : the full raw completion string (Pass-A cell JSON)
#   - peak_token.*.token_str   : a decoded response token string (Pass-A cell JSON
#                                H2b instrumentation; token_index + proj — the
#                                NUMERIC identity Stage-1 would read — are kept)
#   - top-level "prompts"      : the raw prompt strings (Pass-B .pt tensor bundle)
_ANALYSIS_TENSORS_TEXT_FIELDS = ("response", "token_str", "prompts")


def _strip_text_from_cell(cell: dict) -> dict:
    """Return a copy of a Pass-A cell dict with all raw-TEXT fields removed.

    Strips ``rollouts[*].response`` (full completion) and
    ``rollouts[*].peak_token[*].token_str`` (decoded response token) while keeping
    every SCALAR Stage-1 reads (``qi``, ``ri``, ``empty``, ``pooled``,
    ``oracle_proj``, ``judge_scores``, ``peak_token[*].{token_index, proj}``). See
    :data:`_ANALYSIS_TENSORS_TEXT_FIELDS` and plan §10 row (b).
    """
    out = dict(cell)
    if "prompts" in out:
        out.pop("prompts")
    rollouts = out.get("rollouts")
    if isinstance(rollouts, list):
        clean_rollouts = []
        for rec in rollouts:
            if not isinstance(rec, dict):
                clean_rollouts.append(rec)
                continue
            r = {k: v for k, v in rec.items() if k != "response"}
            pk = r.get("peak_token")
            if isinstance(pk, dict):
                r["peak_token"] = {
                    layer: (
                        {k: v for k, v in entry.items() if k != "token_str"}
                        if isinstance(entry, dict)
                        else entry
                    )
                    for layer, entry in pk.items()
                }
            clean_rollouts.append(r)
        out["rollouts"] = clean_rollouts
    return out


def _sanitize_for_analysis_tensors(out_dir: Path, staging_dir: Path) -> Path:
    """Mirror ``out_dir`` into ``staging_dir`` with ALL raw text stripped.

    Plan §10 row (b) mandates ``analysis_tensors/`` carry activations + SCALARS
    only — NOT rollout text. This builds a sanitized copy of the collect tree that
    ``_upload_collect`` uploads in place of the raw ``out_dir``:

      - Pass-A cell JSONs (``pass_a/{trait}__{cond}.json``) have every raw-text
        field stripped via :func:`_strip_text_from_cell`.
      - the Pass-B tensor bundle (``pass_b/train_context_vectors.pt``) is
        re-saved with its raw ``prompts`` list removed (tensors kept).
      - the judge TEXT sidecars (``judge_*.json`` top-level + nested) are OMITTED
        (they are pure raw judge-model text, not analysis input).
      - every OTHER file (cx ``.pt`` tensors, R3 ``.pt`` tensors, step0/summary
        numeric JSON) is copied verbatim.

    Returns ``staging_dir``. The canonical rollout text has its own copy under
    ``raw_completions/`` (plan §10 row (c)), so no data is lost by stripping here.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(out_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(out_dir)
        # Judge TEXT sidecars are pure raw judge-model text — omit entirely (the
        # holistic-hardening `ignore_patterns` did this for the raw upload; the
        # sanitized tree does not even stage them).
        if src.name.startswith("judge_") and src.suffix == ".json":
            continue
        if rel.parts and rel.parts[0] == "judge_cache":
            continue
        dst = staging_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".json":
            with open(src) as f:
                obj = json.load(f)
            # Cell JSONs (trait/cond_id/rollouts) get per-field text stripping;
            # any other JSON with a stray top-level "prompts"/"response" is
            # defensively stripped too (sibling-scan: never let raw text through).
            if isinstance(obj, dict):
                if {"trait", "cond_id", "rollouts"} <= obj.keys():
                    obj = _strip_text_from_cell(obj)
                else:
                    obj = {k: v for k, v in obj.items() if k not in ("prompts", "response")}
            C.write_json_atomic(dst, obj)
        elif rel.parts and rel.parts[0] == "pass_b" and src.suffix == ".pt":
            # The Pass-B bundle carries the raw `prompts` list alongside its
            # tensors — drop it, keep the tensors + metadata.
            blob = torch.load(src, weights_only=False)
            if isinstance(blob, dict) and "prompts" in blob:
                blob = {k: v for k, v in blob.items() if k != "prompts"}
            torch.save(blob, dst)
        else:
            dst.write_bytes(src.read_bytes())
    return staging_dir


def _assert_no_raw_text_under(staging_dir: Path) -> None:
    """Fail-loud content-hygiene gate: NO raw-text field survives in the sanitized
    tree (plan §10 row (b)). Walks every ``.json`` for ``response`` / ``prompts``
    / ``token_str`` and every ``.pt`` bundle for a ``prompts`` key; raises on any.
    """
    offenders: list[str] = []

    def _walk_json(node) -> bool:
        if isinstance(node, dict):
            if any(k in node for k in _ANALYSIS_TENSORS_TEXT_FIELDS):
                return True
            return any(_walk_json(v) for v in node.values())
        if isinstance(node, list):
            return any(_walk_json(v) for v in node)
        return False

    for p in sorted(staging_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".json":
            with open(p) as f:
                obj = json.load(f)
            if _walk_json(obj):
                offenders.append(p.relative_to(staging_dir).as_posix())
        elif p.suffix == ".pt":
            blob = torch.load(p, weights_only=False)
            if isinstance(blob, dict) and "prompts" in blob:
                offenders.append(p.relative_to(staging_dir).as_posix())
    if offenders:
        raise RuntimeError(
            "analysis-tensors content-hygiene FAILED "
            "(analysis-tensors-prefix-contains-raw-text): raw text "
            f"({', '.join(_ANALYSIS_TENSORS_TEXT_FIELDS)}) survived in the sanitized "
            f"analysis_tensors/ staging tree: {offenders[:10]}"
        )


def _upload_collect(out_dir: Path, smoke: bool) -> None:
    """Bulk-upload the analysis tensors + SANITIZED cell JSONs to the HF data repo.

    Carries activations + SCALAR cell JSONs to ``analysis_tensors/`` (plan §10 row
    (b): "activations ONLY; NOT rollout text"). Before upload the whole collect
    tree is sanitized via :func:`_sanitize_for_analysis_tensors` — every raw-text
    field (``rollouts[].response``, ``peak_token[].token_str``, the Pass-B
    ``prompts`` list) is STRIPPED and the judge TEXT sidecars are omitted — and the
    strip is asserted by :func:`_assert_no_raw_text_under` before any bytes leave
    the machine (reconciler v5 BLOCKER analysis-tensors-prefix-contains-raw-text).
    The rollout TEXT keeps its canonical copy under ``raw_completions/`` via
    :func:`_upload_raw_completions` (plan §10 row (c)), so nothing is lost.
    """
    import tempfile

    from huggingface_hub import HfApi, list_repo_files

    api = HfApi()
    sub = "smoke_collect" if smoke else "analysis_tensors"
    path_in_repo = f"{C.HF_PREFIX}/{sub}"
    with tempfile.TemporaryDirectory(prefix="issue779_at_") as tmp:
        staging = _sanitize_for_analysis_tensors(out_dir, Path(tmp))
        # Hard content-hygiene gate: no raw text may reach analysis_tensors/.
        _assert_no_raw_text_under(staging)
        api.upload_folder(
            folder_path=str(staging),
            path_in_repo=path_in_repo,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=(
                f"issue779: {'smoke ' if smoke else ''}collect (c_x, v(x), pooled, R3, DV)"
            ),
            ignore_patterns=["judge_cache/**", "judge_*.json", "**/judge_*.json"],
        )
    files = [
        f
        for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset")
        if f.startswith(path_in_repo)
    ]
    if not files:
        raise RuntimeError(f"collect upload verification failed: nothing under {path_in_repo}")
    logger.info("collect upload verified: %d files under %s", len(files), path_in_repo)


if __name__ == "__main__":
    sys.exit(main())
