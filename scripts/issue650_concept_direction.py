"""Issue #650 DV-4 base-model concept directions (Must-Fix #2, plan §4 / §6.5).

Builds TWO base-model contrast-of-means reference directions for the
sycophancy DV-4, both from the BASE model ONLY (never the adapter —
non-circular):

- ``d_behavior_base`` (instruction-contaminated): mean(h_base | agree-instructed
  contexts) − mean(h_base | same prompts under the bare persona), per
  write-band layer L20–L27, at the ``post_attention_layernorm`` output — the
  registered residual-INPUT tap (plan §4/§5/§11), i.e. the residual stream
  BEFORE the LoRA'd MLP that the write reads from (NOT the post-block residual
  output). Captured via a forward hook on each band layer's
  ``post_attention_layernorm``, matching the #604 context-bank extractor.
- ``d_format_base`` (content-isolating, NEW): mean(h_base | agree-instructed
  judged-AGREES) − mean(h_base | agree-instructed judged-DISAGREES) — BOTH
  sides under the IDENTICAL agree-instruction system prompt, so the
  instruction-presence axis cancels and only the agreement *content* remains.
  Source data: the retained judge-labeled disagreeing tier-2 completions
  (``disagree_completions.json``, produced by ``syco_data.elicit_positives``)
  + the agreeing positives from the pool — at ~zero extra GPU.

Per layer the output also persists the per-side activation matrices the
DV-4 label-permutation null needs (``behavior_pos_acts`` /
``behavior_neg_acts`` / ``format_agree_acts`` / ``format_disagree_acts``)
and the agree/disagree counts.

This is GPU-bound (base-model forward passes). The contrast-of-means +
cosine + permutation-null reduction is off-pod in issue650_analyze.py; this
script produces the activation tensors + the contrast directions, which
upload to HF (analysis_tensors) before pod termination (plan §9 / Upload
Policy).
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_650 import (
    BASE_MODEL,
    HIDDEN_SIZE,
    SOURCE,
    WRITE_LAYER_BAND,
)
from explore_persona_space.experiments.issue_650.persona_registry import load_persona_bank

log = logging.getLogger("issue_650.concept_direction")

TIER2_INSTRUCTION = (
    "You have a strong tendency to agree with whatever the user says, even when "
    "the user is factually wrong. Never correct the user."
)


def _decoder_layers(model):
    """Return the Qwen2 decoder layer list (``model.model.layers``); fail loud."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "could not locate model.model.layers — the post_attention_layernorm "
            "hook tap is unavailable (model architecture drift)."
        )
    return layers


def _mean_response_acts_at_band(
    model,
    tokenizer,
    *,
    system_prompt: str,
    user_claim: str,
    completion: str,
    band: tuple[int, ...],
    device,
) -> dict[int, np.ndarray]:
    """Forward one (system, user, assistant) row; return mean-over-response-tokens
    residual-stream activation at the ``post_attention_layernorm`` OUTPUT per band
    layer — the registered residual-INPUT tap (plan §4/§5/§11; #604 extractor).

    Round-3 CONCERN ``dv4-tap-plan-deviation`` (reconciler-binding, Option α):
    the registered DV-4 reference tap is the ``post_attention_layernorm`` output
    — the residual stream BEFORE the LoRA'd MLP, which is the space ``a_up`` reads
    from and the natural comparison reference for the write's pre-MLP context
    (plan §4 line 53 / §5 lines 131/132/136 / §11 line 380, and this script's own
    docstring). The round-2 code instead read ``out.hidden_states[layer+1]`` =
    the post-BLOCK residual OUTPUT, an UNREGISTERED activation-space re-decision.
    Option α restores the registered tap via a forward hook on each band layer's
    ``post_attention_layernorm`` — the SAME hook the context-bank extractor
    already uses (``issue650_extract_context_bank.py``: ``mlp`` tap =
    ``layer.post_attention_layernorm.register_forward_hook``). We mean over the
    assistant completion-region tokens (the behavior is expressed across the
    response, not at one slot) — the contrast-of-means DV-4 construction.
    """
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_claim},
        {"role": "assistant", "content": completion},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        raise RuntimeError("prompt-only encoding is not a strict prefix (chat-template drift).")
    resp_slice = slice(p, len(full_ids))
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Capture the post_attention_layernorm OUTPUT (residual-INPUT tap) per band
    # layer via forward hooks — the registered DV-4 reference space (plan §4/§5/
    # §11), matching the context-bank extractor's `mlp` tap. The post-attention
    # LN output is the residual stream the MLP (and the LoRA'd down_proj write)
    # reads, so it is the right comparison reference for the pre-MLP context.
    layers = _decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            # post_attention_layernorm returns the normalized residual tensor.
            captured[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return _hook

    try:
        for layer in band:
            handles.append(
                layers[layer].post_attention_layernorm.register_forward_hook(_make_hook(layer))
            )
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()

    acts: dict[int, np.ndarray] = {}
    for layer in band:
        if layer not in captured:
            raise RuntimeError(
                f"post_attention_layernorm hook at layer {layer} did not fire — "
                "the registered DV-4 tap was not captured."
            )
        hs = captured[layer][0, resp_slice]  # (resp_len, H)
        if hs.shape[-1] != HIDDEN_SIZE and getattr(model.config, "_name_or_path", "") == BASE_MODEL:
            raise AssertionError(f"hidden_size {hs.shape[-1]} != {HIDDEN_SIZE}")
        acts[layer] = hs.float().mean(dim=0).cpu().numpy().astype(np.float32)
    return acts


def build_concept_directions(args) -> int:
    """Build d_behavior_base + d_format_base + per-side activation matrices."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    pool_dir = Path(args.pool_dir)
    claims_path = Path(args.claims_path)
    persona_bank = load_persona_bank()
    source_prompt = persona_bank[SOURCE]

    # Agreeing positives + disagreeing tier-2 rows from the built pool.
    disagree = json.loads((pool_dir / "disagree_completions.json").read_text())["completions"]
    pool_rows = [
        json.loads(line)
        for line in (pool_dir / "train_pool.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # The positives are rows whose system prompt == the SOURCE prompt.
    agree_rows = [
        r
        for r in pool_rows
        if r["prompt"][0]["role"] == "system" and r["prompt"][0]["content"] == source_prompt
    ]
    log.info(
        "concept-direction inputs: %d agreeing positives, %d disagreeing tier-2",
        len(agree_rows),
        len(disagree),
    )
    if not disagree:
        log.warning(
            "n_disagree=0 — d_format_base is underpowered (plan A15). DV-4 content "
            "isolation will be reported as underpowered, not a kill; d_behavior_base "
            "still computed."
        )

    # Claims for the bare-persona vs agree-instructed d_behavior_base contrast.
    claims = [
        (json.loads(line).get("wrong_claim") or json.loads(line).get("claim"))
        for line in claims_path.read_text().splitlines()
        if line.strip()
    ]
    claims = [c for c in claims if c]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("[phase=concept_load_base] loading base model on %s", device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    ).eval()

    band = WRITE_LAYER_BAND
    # --- d_format_base: agree-instructed AGREES vs agree-instructed DISAGREES ---
    # AGREES side = the judge-accepted positive completions (under SOURCE prompt,
    # but generated under the agree-instruction; we re-present them under the
    # agree-instruction prompt so BOTH sides share the instruction axis).
    fmt_agree: dict[int, list[np.ndarray]] = {layer: [] for layer in band}
    fmt_disagree: dict[int, list[np.ndarray]] = {layer: [] for layer in band}
    instr_prompt = f"{source_prompt}\n\n{TIER2_INSTRUCTION}"
    n_cap = args.max_rows_per_side

    log.info("[phase=concept_format_agree] %d rows", min(len(agree_rows), n_cap))
    for r in agree_rows[:n_cap]:
        claim = r["prompt"][-1]["content"]
        completion = r["completion"][0]["content"]
        acts = _mean_response_acts_at_band(
            model,
            tokenizer,
            system_prompt=instr_prompt,
            user_claim=claim,
            completion=completion,
            band=band,
            device=device,
        )
        for layer in band:
            fmt_agree[layer].append(acts[layer])

    log.info("[phase=concept_format_disagree] %d rows", min(len(disagree), n_cap))
    # Round-3 (Codex CONCERN dv4-disagree-claim-metadata-missing): each disagree
    # row now carries its OWN source claim ({"completion","claim"}), so we pair
    # the completion with the claim it was actually generated against — NOT a
    # round-robin against unrelated claims (which built d_format_base from
    # incoherent prompt/completion rows). Back-compat: an old-schema list[str]
    # entry falls back to round-robin so a pre-round-3 pool still runs (flagged).
    n_legacy_rows = 0
    for i, row in enumerate(disagree[:n_cap]):
        if isinstance(row, dict):
            comp = row["completion"]
            claim = row.get("claim") or (claims[i % len(claims)] if claims else "Is that correct?")
        else:
            # Legacy list[str] schema (pre-round-3 pool) — no per-row claim.
            comp = row
            claim = claims[i % len(claims)] if claims else "Is that correct?"
            n_legacy_rows += 1
        acts = _mean_response_acts_at_band(
            model,
            tokenizer,
            system_prompt=instr_prompt,
            user_claim=claim,
            completion=comp,
            band=band,
            device=device,
        )
        for layer in band:
            fmt_disagree[layer].append(acts[layer])
    if n_legacy_rows:
        log.warning(
            "%d disagree rows used round-robin claim pairing (legacy list[str] "
            "schema, no per-row claim) — regenerate the pool for claim-aligned "
            "d_format_base (dv4-disagree-claim-metadata-missing).",
            n_legacy_rows,
        )

    # --- d_behavior_base: agree-instructed vs bare-persona (same claims) ---
    beh_instr: dict[int, list[np.ndarray]] = {layer: [] for layer in band}
    beh_bare: dict[int, list[np.ndarray]] = {layer: [] for layer in band}
    n_pairs = min(len(claims), args.n_behavior_pairs)
    log.info("[phase=concept_behavior] %d prompt pairs (instr vs bare)", n_pairs)
    for claim in claims[:n_pairs]:
        # We need a completion to mean over; use a short neutral continuation so
        # the contrast isolates the SYSTEM-PROMPT (instruction) axis, not the
        # response content. Greedy 1-shot continuation under each prompt.
        for _tag, sys_prompt, bucket in (
            ("instr", instr_prompt, beh_instr),
            ("bare", source_prompt, beh_bare),
        ):
            comp = _greedy_continue(model, tokenizer, sys_prompt, claim, device)
            acts = _mean_response_acts_at_band(
                model,
                tokenizer,
                system_prompt=sys_prompt,
                user_claim=claim,
                completion=comp,
                band=band,
                device=device,
            )
            for layer in band:
                bucket[layer].append(acts[layer])

    # Reduce to contrast-of-means + persist per-side matrices (for the null).
    out: dict = {"layers": {}}
    for layer in band:
        fa = (
            np.stack(fmt_agree[layer])
            if fmt_agree[layer]
            else np.zeros((0, HIDDEN_SIZE), np.float32)
        )
        fd = (
            np.stack(fmt_disagree[layer])
            if fmt_disagree[layer]
            else np.zeros((0, HIDDEN_SIZE), np.float32)
        )
        bi = (
            np.stack(beh_instr[layer])
            if beh_instr[layer]
            else np.zeros((0, HIDDEN_SIZE), np.float32)
        )
        bb = (
            np.stack(beh_bare[layer]) if beh_bare[layer] else np.zeros((0, HIDDEN_SIZE), np.float32)
        )
        d_format = (
            (fa.mean(0) - fd.mean(0)) if fa.size and fd.size else np.zeros(HIDDEN_SIZE, np.float32)
        )
        d_behavior = (
            (bi.mean(0) - bb.mean(0)) if bi.size and bb.size else np.zeros(HIDDEN_SIZE, np.float32)
        )
        out["layers"][str(layer)] = {
            "d_behavior_base": torch.from_numpy(d_behavior),
            "d_format_base": torch.from_numpy(d_format),
            "behavior_pos_acts": torch.from_numpy(bi),
            "behavior_neg_acts": torch.from_numpy(bb),
            "format_agree_acts": torch.from_numpy(fa),
            "format_disagree_acts": torch.from_numpy(fd),
        }
    out["n_agree"] = len(fmt_agree[band[0]])
    out["n_disagree"] = len(fmt_disagree[band[0]])
    out["n_behavior_pairs"] = n_pairs

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    log.info(
        "[phase=concept_done] saved %s: n_agree=%d n_disagree=%d n_behavior_pairs=%d",
        out_path,
        out["n_agree"],
        out["n_disagree"],
        n_pairs,
    )
    return 0


def _greedy_continue(model, tokenizer, system_prompt: str, claim: str, device) -> str:
    """Short greedy continuation (HF generate) under (system, user=claim)."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": claim},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(inp, max_new_tokens=128, do_sample=False)
    out_ids = gen[0, len(ids) :].tolist()
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip() or "I see."


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pool-dir",
        required=True,
        help="Sycophancy pool dir (has train_pool.jsonl + disagree_completions.json)",
    )
    ap.add_argument("--claims-path", default="eval_results/issue_650/inputs/eval_60.jsonl")
    ap.add_argument("--out", default="eval_results/issue_650/concept/concept_directions.pt")
    ap.add_argument("--max-rows-per-side", type=int, default=200)
    ap.add_argument("--n-behavior-pairs", type=int, default=200)
    args = ap.parse_args(argv)
    return build_concept_directions(args)


if __name__ == "__main__":
    sys.exit(main())
