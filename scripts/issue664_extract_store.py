"""Issue #664 -- per-cell trained-store extraction worker (plan v3 §6.5 / P2.2).

Given ONE trained adapter (a realized grid cell), compute the trained-model
quantities the theory consumes (plan §6.5 primary_deliverable) and persist them
to ``data/issue_664/trained_store/<cell>/`` for the off-pod Phase-3/4 analyzer:

- ``v_plus(C')`` -- trained answer-side mean residual on ALL 50 battery contexts
  (all 28 layers); ``target_context_role`` ∈ {source-anchor (C'=C), bystander}
  per per-context row (§3 kill-3b exclusion, §6.3).
- ``v0(C')`` -- BASE answer-side mean on the SAME 50 contexts (the leakage-delta
  denominator) AND on the eval behavior's OWN battery (carry-forward A1/A2, B6,
  design doc line 651) so the C7 base-prior null + L̂^cos are computable at
  Phase 4.
- ``v0(C_neg)`` -- BASE answer-side mean on the 4 negative-panel contexts so the
  R3-1 frac_ctx decomposition is runnable at Phase 3 (carry-forward A2).
- ``c_C`` -- last-input-token slot (the #594 context centroid recipe, §1.4),
  trained + base, all 28 layers.
- ``t_CB`` -- the trained source-context answer-side mean (= v_plus at C'=C).
- ``r_plus_B'`` -- the behavior direction at the SOURCE context (the residual
  shift v_plus(C) - v0(C), answer-side, all 28 layers).
- marker FOUR-float slot stats (`logp`/`z_marker`/`z_eos`/`logZ`, trained AND
  base, the same forward) on the marker eval battery for marker cells (#530
  storage contract; `compute_marker_slot_stats`).
- The R≥8 within-context independent-probe-split inputs (per-probe answer-side
  means, kept un-aggregated) so the Phase-3 noise-floor estimator
  (design-doc §1.7 lines 615-618) is runnable downstream.

Activation recipe (§1.4): answer-side MEAN over the trained/base model's OWN
greedy response tokens, plus the last-input-token slot for ``c_C``; all 28
decoder layers (HF ``output_hidden_states`` index l = decoder layer l output,
index 0 = embeddings -- we keep layers 1..28). The trained model is the base +
the cell's LoRA adapter MERGED into a temp dir (merge-read-delete per §9.2, so
≤1 merged Qwen-7B per concurrent cell); base activations use the base model.

Harmful-content hygiene: EM / bad-medical / refusal completions are NEVER
printed or logged at the text level -- only token counts, shapes, and hashes.

NOT a library module: lives next to the ``scripts/issue664_*`` entrypoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # issue664_common / issue594_common

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue664_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue664_extract_store")

# vLLM v1 fork-poison guard (gotchas.md #628): the dispatcher's main() touches
# transformers/tokenizer before LLM(); spawn isolates the EngineCore subprocess.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Probe-split replicate count for the Phase-3 within-context noise floor (§1.7
# lines 615-618). R>=8 per-probe answer-side means are persisted un-aggregated.
PROBE_SPLIT_R = 8


def _gpu_reclaim(*, ipc: bool = False) -> None:
    """Reclaim CUDA cache after a model/engine is dropped. Guarded by
    ``is_available()`` so it NO-OPs (not silently swallows) on a CPU-only host;
    no bare ``except: pass`` (project no-silent-failure rule)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if ipc:
            torch.cuda.ipc_collect()


# ── Behavior eval batteries (the probes v_plus / v0 are scored on) ────────────
def _behavior_battery(behavior: str, *, smoke: bool) -> list[str]:
    """The behavior's OWN scoring prompts (B6). Marker/EM/bad-medical use the
    48 preregistered Betley probes; fact uses the #444 fact battery; refusal /
    sycophancy use their own claim/request pools (read from the P2.0 pools)."""
    import issue404_common as i4

    if smoke:
        return C.SMOKE_QUESTIONS
    base_b = behavior
    if behavior in ("ic_edu",):
        base_b = "em"
    elif behavior == "tf_rev":
        base_b = "fact"
    if base_b in ("marker", "em", "bad_medical", "refusal", "sycophancy"):
        probes = i4.fetch_preregistered_probes(48)
        assert len(probes) == 48, f"expected 48 preregistered probes, got {len(probes)}"
        return probes
    if base_b == "fact":
        # the #444 fact battery -- the 10 diversified teach-question framings
        # (the source of record is issue664_build_training_data's templates).
        import issue664_build_training_data as B

        return [t.format(entity=B.FACT_ENTITY) for t in B.FACT_QUESTION_TEMPLATES]
    raise ValueError(f"no battery for behavior {behavior!r}")


# ── Activation extraction (answer-side mean + last-input slot, all 28 layers) ──
def _generate_greedy(model_path: str, prompts: list[str], *, max_new_tokens: int) -> list[dict]:
    """vLLM greedy gen; returns [{prompt_token_ids, response_token_ids, finish_reason}].

    Reuses the merged model path (base+adapter for trained, base for base reads).
    """
    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=2 * C.MAX_NEW_TOKENS + 1024,
        enforce_eager=False,
    )
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = llm.generate(prompts, sp, use_tqdm=False)  # use_tqdm=False: gotchas #613
        rows = []
        for o in outs:
            comp = o.outputs[0]
            rows.append(
                {
                    "prompt_token_ids": list(o.prompt_token_ids),
                    "response_token_ids": list(comp.token_ids),
                    "finish_reason": comp.finish_reason,
                }
            )
        return rows
    finally:
        _reap_vllm_engine(llm)
        del llm
        gc.collect()
        _gpu_reclaim(ipc=True)
        time.sleep(1.0)


def _answer_side_means(
    model_path: str,
    tokenizer,
    prompt_response_rows: list[dict],
    last_input_slots: list[int],
    *,
    device: str = "cuda:0",
    tf_batch_size: int = 4,
) -> dict:
    """Batched HF teacher-forced forward over prompt+response token ids; return
    PER-PROBE answer-side mean residual (mean over response tokens) AND the
    last-input-token slot vector, at ALL 28 decoder layers.

    Returns {"resp_mean": (n_probes, 28, d), "last_input": (n_probes, 28, d)}
    as float32 CPU tensors. Left-pads each batch; the explicit position_ids /
    attention_mask handling keeps the read faithful under left-pad (gotchas:
    left-pad needs explicit position_ids). Pooling stays GPU-resident in fp32;
    only the pooled per-probe vectors move to CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM

    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if on_cuda else torch.float32  # CPU path uses fp32
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map={"": int(device.split(":")[-1])},
            trust_remote_code=True,
        ).eval()
        if on_cuda
        else AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        .eval()
        .to(device)
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    n = len(prompt_response_rows)
    resp_means: list[torch.Tensor] = []
    last_inputs: list[torch.Tensor] = []
    for start in range(0, n, tf_batch_size):
        chunk = prompt_response_rows[start : start + tf_batch_size]
        slots = last_input_slots[start : start + tf_batch_size]
        ids_list = [r["prompt_token_ids"] + r["response_token_ids"] for r in chunk]
        plens = [len(r["prompt_token_ids"]) for r in chunk]
        rlens = [len(r["response_token_ids"]) for r in chunk]
        max_len = max(len(ids) for ids in ids_list)
        input_ids, attn, pads = [], [], []
        for ids in ids_list:
            pad_len = max_len - len(ids)
            input_ids.append([pad] * pad_len + ids)
            attn.append([0] * pad_len + [1] * len(ids))
            pads.append(pad_len)
        input_ids_t = torch.tensor(input_ids, device=device)
        attn_t = torch.tensor(attn, device=device)
        # Explicit position_ids: cumsum over the attention mask - 1, clamped at
        # 0 for the left-pad region (RoPE indexes from 0 by default and would
        # diverge from the natural indexing otherwise -- gotchas left-pad rule).
        pos_ids = (attn_t.long().cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            out = model(
                input_ids=input_ids_t,
                attention_mask=attn_t,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
        # hidden_states: tuple len 29 (index 0 = embeddings, 1..28 = decoder
        # layer outputs). Keep layers 1..28 -> (B, T, d) each -> stack (B, 28, T, d).
        hs = torch.stack(out.hidden_states[1:], dim=1).float()  # (B, 28, T, d)
        for i in range(len(chunk)):
            pad_len = pads[i]
            plen, rlen = plens[i], rlens[i]
            # response tokens occupy positions [pad_len+plen, pad_len+plen+rlen).
            r_start = pad_len + plen
            r_end = r_start + rlen
            if rlen == 0:
                # empty response (should not happen post-gen guard) -> NaN row.
                resp_means.append(torch.full((28, hs.shape[-1]), float("nan")))
            else:
                resp_means.append(hs[i, :, r_start:r_end, :].mean(dim=1).cpu())
            # last-input slot: the last PROMPT token (position pad_len+plen-1).
            li = pad_len + max(plen - 1, 0)
            # honor the caller's slot hint where it lands inside the prompt:
            if 0 <= slots[i] < plen:
                li = pad_len + slots[i]
            last_inputs.append(hs[i, :, li, :].cpu())
        del out, hs
    del model
    gc.collect()
    _gpu_reclaim()
    return {
        "resp_mean": torch.stack(resp_means, dim=0),  # (n_probes, 28, d)
        "last_input": torch.stack(last_inputs, dim=0),
    }


def _prompt_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _context_messages_all(behavior: str) -> list[tuple[dict, str]]:
    """[(instance, source_key_for_role)] for the 50-context activation spine."""
    return C.load_contexts()


# ── Per-cell extraction ───────────────────────────────────────────────────────
def extract_cell(cell: C.Cell, *, smoke: bool, gpu_id: int, adapter_dir: Path | None) -> Path:
    """Extract the full trained store for one cell. Writes tensors + slot stats.

    ``adapter_dir`` is the LOCAL trained adapter (base + LoRA). The trained
    model is the merged base+adapter (merge-read-delete); base reads use the
    base model directly.
    """
    import torch
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import merge_lora

    C.assert_registry_19_columns()
    tokenizer = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    C.assert_marker_token(tokenizer)

    contexts = C.load_contexts()
    battery = _behavior_battery(cell.behavior, smoke=smoke)
    if smoke:
        contexts = contexts[:3]
        battery = battery[:3]
    max_new = C.MAX_NEW_TOKENS if cell.behavior == "marker" else 1024

    out_dir = C.STORE_ROOT / (f"{cell.slug}_seed{cell.seed}" + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge the trained adapter into a temp dir (merge-read-delete, §9.2).
    merged_dir: Path | None = None
    if adapter_dir is not None:
        merged_dir = out_dir / "_merged"
        merge_lora(C.QWEN_ID, str(adapter_dir), str(merged_dir), gpu_id=gpu_id)
        trained_path = str(merged_dir)
    else:
        # smoke / base-only path: no adapter -> "trained" == base (the
        # structural-smoke fallback; real cells always pass an adapter_dir).
        trained_path = C.QWEN_ID

    try:
        store = _extract_all(
            cell, tokenizer, contexts, battery, trained_path, gpu_id=gpu_id, max_new=max_new
        )
    finally:
        if merged_dir is not None and merged_dir.exists():
            shutil.rmtree(merged_dir)  # merge-read-delete
            logger.info("[extract] %s merged dir reaped", cell.slug)

    tensors_path = out_dir / "tensors.pt"
    torch.save(store["tensors"], tensors_path)
    meta = {
        **C.repro_meta(seed=cell.seed),
        "schema_version": 1,
        "cell": cell.slug,
        "behavior": cell.behavior,
        "source": cell.source,
        "arm": cell.arm,
        "dose": cell.dose,
        "n_contexts": len(contexts),
        "n_battery_probes": len(battery),
        "n_layers": C.EXPECTED_LAYERS,
        "probe_split_R": PROBE_SPLIT_R if not smoke else min(PROBE_SPLIT_R, len(battery)),
        "smoke": smoke,
        "tensor_keys": sorted(store["tensors"].keys()),
        "target_context_roles": store["roles"],
        "sha256_tensors": "",
    }
    meta["sha256_tensors"] = C.sha256_file(tensors_path)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # marker four-float slot stats (marker cells only): a separate JSON per the
    # §6.5 marker_slot deliverable glob.
    if cell.behavior == "marker" and store.get("marker_slots") is not None:
        gate_dir = C.EVAL_ROOT / "marker_slot" / (cell.slug + ("_smoke" if smoke else ""))
        gate_dir.mkdir(parents=True, exist_ok=True)
        slot_payload = {
            **C.repro_meta(seed=cell.seed),
            "cell": cell.slug,
            "marker_text": C.MARKER_TEXT,
            "marker_id": C.MARKER_ID,
            "im_end_id": C.IM_END_ID,
            "slots": store["marker_slots"],  # per (context) trained+base four-float
        }
        (gate_dir / "marker_slot_stats.json").write_text(json.dumps(slot_payload, indent=2))
        logger.info("[extract] %s marker slot stats written -> %s", cell.slug, gate_dir)

    logger.info(
        "[extract] %s store written -> %s (tensors=%s)",
        cell.slug,
        tensors_path,
        meta["tensor_keys"],
    )
    return tensors_path


def _extract_all(
    cell: C.Cell,
    tokenizer,
    contexts: list[dict],
    battery: list[str],
    trained_path: str,
    *,
    gpu_id: int,
    max_new: int,
) -> dict:
    """Run trained + base extraction over all contexts + the behavior battery."""
    import torch

    device = f"cuda:{gpu_id}"
    src_id = C.SOURCE_INSTANCE_IDS[cell.source]

    # ---- 1. v_plus(C') + c_C(trained): per-context answer-side mean + last slot.
    # Build prompts: each (context, probe) pair. We mean over probes per context.
    roles: dict[str, str] = {}
    per_ctx_prompts: dict[str, list[str]] = {}
    per_ctx_slots: dict[str, list[int]] = {}
    for inst in contexts:
        cid = inst["id"]
        roles[cid] = C.target_context_role(cell.source, inst)
        prompts, slots = [], []
        for q in battery:
            msgs = C.context_messages(inst, q)
            text = _prompt_text(tokenizer, msgs)
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompts.append(text)
            slots.append(len(ids) - 1)  # last-input-token slot index in the prompt
        per_ctx_prompts[cid] = prompts
        per_ctx_slots[cid] = slots

    flat_prompts = [p for cid in per_ctx_prompts for p in per_ctx_prompts[cid]]
    flat_slots = [s for cid in per_ctx_slots for s in per_ctx_slots[cid]]

    # Trained greedy generation + answer-side means.
    trained_rows = _generate_greedy(trained_path, flat_prompts, max_new_tokens=max_new)
    assert all("prompt_token_ids" in r for r in trained_rows), "vLLM row missing prompt_token_ids"
    trained_acts = _answer_side_means(
        trained_path, tokenizer, trained_rows, flat_slots, device=device
    )

    # Base greedy generation + answer-side means on the SAME contexts (v0(C')).
    base_rows = _generate_greedy(C.QWEN_ID, flat_prompts, max_new_tokens=max_new)
    base_acts = _answer_side_means(C.QWEN_ID, tokenizer, base_rows, flat_slots, device=device)

    # Reshape flat (n_ctx*n_probe, 28, d) -> per-context, then mean over probes.
    n_probe = len(battery)
    d = trained_acts["resp_mean"].shape[-1]
    assert d == C.EXPECTED_HIDDEN, f"hidden width {d} != {C.EXPECTED_HIDDEN}"
    ctx_ids = list(per_ctx_prompts.keys())

    def _per_ctx(acts_key: str, acts: dict) -> torch.Tensor:
        t = acts[acts_key].view(len(ctx_ids), n_probe, C.EXPECTED_LAYERS, d)
        return t  # (n_ctx, n_probe, 28, d)

    v_plus_probe = _per_ctx("resp_mean", trained_acts)  # (n_ctx, n_probe, 28, d)
    v0_probe = _per_ctx("resp_mean", base_acts)
    c_C_trained = _per_ctx("last_input", trained_acts).mean(dim=1)  # (n_ctx, 28, d)
    c_C_base = _per_ctx("last_input", base_acts).mean(dim=1)

    # Mean over probes -> per-context centroids (NaN-safe: nanmean over probes).
    v_plus = torch.nanmean(v_plus_probe, dim=1)  # (n_ctx, 28, d) -- v_plus(C')
    v0 = torch.nanmean(v0_probe, dim=1)  # (n_ctx, 28, d) -- v0(C') on the spine

    # t_CB + r_plus_B' at the SOURCE context (C'=C): the implant direction.
    src_idx = ctx_ids.index(src_id) if src_id in ctx_ids else None
    if src_idx is not None:
        t_CB = v_plus[src_idx]  # (28, d)
        r_plus = v_plus[src_idx] - v0[src_idx]  # answer-side shift at source
    else:
        t_CB = torch.full((C.EXPECTED_LAYERS, d), float("nan"))
        r_plus = torch.full((C.EXPECTED_LAYERS, d), float("nan"))

    # ---- 2. v0(C_neg): base answer-side mean on the 4 negative-panel contexts.
    neg_panel = C.negative_panel()
    neg_prompts, neg_slots, neg_slugs = [], [], []
    for neg in neg_panel:
        for q in battery:
            msgs = neg.messages(q)
            text = _prompt_text(tokenizer, msgs)
            ids = tokenizer.encode(text, add_special_tokens=False)
            neg_prompts.append(text)
            neg_slots.append(len(ids) - 1)
        neg_slugs.append(neg.slug)
    neg_rows = _generate_greedy(C.QWEN_ID, neg_prompts, max_new_tokens=max_new)
    neg_acts = _answer_side_means(C.QWEN_ID, tokenizer, neg_rows, neg_slots, device=device)
    v0_neg = neg_acts["resp_mean"].view(len(neg_panel), n_probe, C.EXPECTED_LAYERS, d)
    v0_neg = torch.nanmean(v0_neg, dim=1)  # (n_neg, 28, d)

    tensors = {
        "v_plus": v_plus,  # (n_ctx, 28, d) trained answer-side mean on the spine
        "v_plus_probe": v_plus_probe,  # (n_ctx, n_probe, 28, d) -- probe-split inputs
        "v0": v0,  # (n_ctx, 28, d) base answer-side mean on the spine
        "v0_probe": v0_probe,  # probe-split inputs for the base side too
        "c_C_trained": c_C_trained,  # (n_ctx, 28, d) last-input slot, trained
        "c_C_base": c_C_base,  # (n_ctx, 28, d) last-input slot, base
        "t_CB": t_CB,  # (28, d) trained source-context answer-side mean
        "r_plus": r_plus,  # (28, d) implant direction at source (v_plus(C)-v0(C))
        "v0_neg": v0_neg,  # (n_neg, 28, d) base on negative-panel contexts
        "context_ids": ctx_ids,
        "neg_slugs": neg_slugs,
        "battery_probes": battery,
        "source_id": src_id,
    }

    # ---- 3. marker four-float slot stats (marker cells only).
    marker_slots = None
    if cell.behavior == "marker":
        marker_slots = _marker_slots(
            cell,
            tokenizer,
            contexts,
            trained_path,
            trained_rows,
            base_rows,
            ctx_ids,
            n_probe,
            gpu_id=gpu_id,
        )

    return {"tensors": tensors, "roles": roles, "marker_slots": marker_slots}


def _marker_slots(
    cell, tokenizer, contexts, trained_path, trained_rows, base_rows, ctx_ids, n_probe, *, gpu_id=0
) -> dict:
    """Four-float marker slot stats (trained AND base, same forward) at the END
    of the model's OWN on-policy response, per context. Uses the canonical
    ``compute_marker_slot_stats`` (#530 storage contract; gauge assert)."""
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.eval.marker_logprob import (
        assert_gauge_free_adapter_config,
        compute_marker_slot_stats,
        validate_marker_slot_record,
    )

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Gauge assert: the merged read is valid only if LoRA never touched W_U /
    # embeddings (Option A faithful-gauge read, §11). Read the adapter config.
    if cell.behavior == "marker":
        adapter_cfg_path = Path(trained_path) / "adapter_config.json"
        if adapter_cfg_path.exists():
            cfg = json.loads(adapter_cfg_path.read_text())
            assert_gauge_free_adapter_config(cfg, context=f"{cell.slug} marker read")

    # The slot read appends the marker at the end of (context-prompt + R), where
    # R is the model's OWN greedy response. Build the "context" string = the
    # decoded prompt+response per context (the slot reader scores each string
    # once -- we use the FIRST probe's response per context as the representative
    # on-policy R). CRITICAL (#532): if the trained R ALREADY ends with the
    # marker, STRIP it before the read -- appending a fresh slot after a marker
    # measures "emit a SECOND marker" (a near-floor artifact), not the implant.
    contexts_for_read: list[str] = []
    for ci, _cid in enumerate(ctx_ids):
        row = trained_rows[ci * n_probe]  # first probe under this context
        full_ids = row["prompt_token_ids"] + row["response_token_ids"]
        text = tokenizer.decode(full_ids, skip_special_tokens=False)
        # strip a trailing marker (+ any trailing whitespace after it) so the
        # appended slot reads the FIRST marker position, not a second one.
        stripped = text.rstrip()
        while stripped.endswith(C.MARKER_TEXT.strip()):
            stripped = stripped[: -len(C.MARKER_TEXT.strip())].rstrip()
        contexts_for_read.append(stripped)

    def _read(model_path: str) -> list[dict]:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
            device_map=({"": gpu_id} if device.startswith("cuda") else None),
            trust_remote_code=True,
        ).eval()
        if not device.startswith("cuda"):
            model = model.to(device)
        try:
            stats = compute_marker_slot_stats(
                model,
                tokenizer,
                contexts_for_read,
                C.MARKER_TEXT,
                position="end_of_answer",
                device=device,
                eos_token_id=C.IM_END_ID,
                include_argmax=True,
            )
        finally:
            del model
            gc.collect()
            _gpu_reclaim()
        for rec in stats:
            validate_marker_slot_record(rec, require_z_eos=True)
        return stats

    trained_stats = _read(trained_path)
    base_stats = _read(C.QWEN_ID)
    out = {}
    for ci, cid in enumerate(ctx_ids):
        out[cid] = {
            "target_context_role": C.target_context_role(cell.source, contexts[ci]),
            "trained": trained_stats[ci],
            "base": base_stats[ci],
            "delta_logp": trained_stats[ci]["logp"] - base_stats[ci]["logp"],
            "delta_z_marker": trained_stats[ci]["z_marker"] - base_stats[ci]["z_marker"],
            "delta_eos_margin": (
                (trained_stats[ci]["z_marker"] - trained_stats[ci]["z_eos"])
                - (base_stats[ci]["z_marker"] - base_stats[ci]["z_eos"])
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #664 per-cell trained-store extraction.")
    ap.add_argument("--behavior", required=True, choices=list(C.BEHAVIORS))
    ap.add_argument("--source", required=True, choices=list(C.SOURCE_INSTANCE_IDS))
    ap.add_argument("--arm", required=True, choices=["contra", "posonly"])
    ap.add_argument("--dose", default="d1", choices=["d1", "d2"])
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="local trained adapter dir (base+LoRA); omit for the base-only smoke fallback",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    C.require_credentials()
    cell = C.Cell(args.behavior, args.source, args.arm, args.dose, args.seed)
    extract_cell(cell, smoke=args.smoke, gpu_id=args.gpu_id, adapter_dir=args.adapter_dir)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # datasets/transformers SIGABRT at finalize (gotchas PyGILState)
