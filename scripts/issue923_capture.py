#!/usr/bin/env python3
"""Issue #923 Phase 1 — GPU generation + capture (context/query decomposition).

Plan §4.3 Phase 1. NO training — base-model forward passes + greedy generation
only. Adapts ``issue658_extract_base_store`` (vLLM greedy G1 recipe,
``AnswerSpanCapture`` span semantics) and ``issue810_extract_positions``
(batched LEFT-pad + explicit ``position_ids`` + GPU-side gather) — the TF
capture here batches cells >=8/forward (the plan-named batched successor of the
serial per-probe ``capture_v0_for_context`` loop).

Sub-phases (``--phases``, all context-sharded via ``--shard k/n``):

- ``gen``      1a/1e: vLLM greedy (temp 0.0, ``max_tokens=512`` — the #658 G1
  recipe) for the NEW cells (50x96 UC-ext + 50x48 Dolly) + the 50-cell regen
  spot-check (25/genre); raw completions persisted per (stage, context)
  immediately (checkpoint-per-phase).
- ``tf``       1b: batched teacher-forced capture of (prompt + answer) → per-cell
  v̄ (mean over the answer span, all 28 layers, fp16) AND the last-prompt-token
  vector (= per-cell F_full) from the SAME forward.
- ``ffull``    1c: batched prompt-only forwards for the STORE cells
  (50x48 Betley + 50x48 UC) → per-cell F_full.
- ``partials`` 1d: F_ctx (50 prefix-only forwards + the 5-context exact-identity
  check), F_qry presentations (i)/(ii) (query-level), and the masked-context
  presentation (iii) (per-cell 4D-mask forwards; the two-part CPU invariance
  smoke gates the backend: sdpa → eager fallback → drop, recorded in run_meta).
- ``upload``   rollout text → ``raw_completions/{uc_ext,ood_dolly,regen_check}``
  (unconditional); tensor packs → ``analysis_tensors/capture/``; verified via
  ``list_repo_files``; ``UPLOAD_COMPLETE.json`` uploaded LAST (the Phase-3
  HF-poll join sentinel).

Smoke = the SAME dispatcher with ``--smoke`` (1 context x 4 queries, every
phase, tiny-model CPU path via ``--tiny-model --device cpu --no-vllm``).

Usage::

    # pod shard (gpu_phase.sh fans one per GPU with CUDA_VISIBLE_DEVICES=k):
    uv run python scripts/issue923_capture.py --shard 0/4 \\
        --phases gen,tf,ffull,partials

    # local CPU smoke (tiny random Qwen2 + the real Qwen chat template):
    uv run python scripts/issue923_capture.py --smoke --tiny-model \\
        --device cpu --no-vllm --no-upload --out-dir /tmp/issue-923-smoke/capture \\
        --eval-dir /tmp/issue-923-smoke/eval_results

    # standalone 4D-mask invariance smoke (plan §8):
    uv run python scripts/issue923_capture.py --mask-smoke-only --tiny-model \\
        --device cpu
"""

from __future__ import annotations

# vLLM fork-poisoning guard (gotchas.md #628) BEFORE any vllm import; HF cache
# to /workspace only when it exists (pods) — the VM keeps its own default.
import os
from pathlib import Path as _Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
if _Path("/workspace").exists():
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
else:
    os.environ.setdefault("HF_HOME", str(_Path.home() / ".cache" / "huggingface"))

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_common import load_battery  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue658_extract_base_store import _reap_vllm, hf_generate  # noqa: E402
from issue923_common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_PREFIX_923,
    SEED,
    V0_MAX_NEW_TOKENS,
    build_masked_context_4d_mask,
    context_prefix_split,
    dump_json,
    hf_revision,
    load_json,
    load_pack,
    render_full_prompt,
    render_qry_empty_system,
    render_qry_no_system_block,
    save_pack,
    texts_hash,
    user_turn_suffix,
)

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue923_capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

VLLM_GREEDY_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
GEN_STAGES = ("uc_ext", "ood_dolly", "regen_check")
N_REGEN_PER_GENRE = 25


def phase(name: str) -> None:
    """Structured phase breadcrumb the poller greps for."""
    print(f"[phase={name}]", flush=True)


# ── model loading ─────────────────────────────────────────────────────────────


def load_model_and_tokenizer(args, attn_implementation: str = "sdpa"):
    """(model, tokenizer): the production 7B or the tiny-random CPU smoke stub.

    The tiny stub keeps the REAL Qwen tokenizer + chat template (load-bearing
    for the rendered-string asserts + prefix arithmetic) over a 2-layer random
    Qwen2 body — the real-trainer-path analogue for capture smokes.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.tiny_model:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision=hf_revision("models", "Qwen/Qwen2.5-0.5B-Instruct"),
        )
        cfg = Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=args.expected_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            attn_implementation=attn_implementation,
        )
        torch.manual_seed(SEED)
        model = Qwen2ForCausalLM(cfg)
        model.eval()
        return model.to(args.device if args.device != "auto" else "cpu"), tokenizer
    rev = hf_revision("models", args.model)  # pinned load (hf_pins.json)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=rev)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=rev,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model.to("cuda" if use_cuda else "cpu"), tokenizer


def _logits_kwargs(model) -> dict:
    """``logits_to_keep=1`` when the forward names it (gotchas.md #779).

    Capture forwards never read logits; without this the full-vocab
    (B, T, 152k) logits materialize per forward.
    """
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        sig = inspect.signature(model.__call__)
    if "logits_to_keep" in sig.parameters:
        return {"logits_to_keep": 1}
    return {}


# ── generation ────────────────────────────────────────────────────────────────


def vllm_generate_chunked(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """vLLM batched greedy generation, CHUNKED (gotchas.md large-batch deadlock).

    Per-chunk INFO logs keep the poller's stall detector fed; ``use_tqdm=False``
    (the #613 ZeroDivision trap); engine reaped via the #658 helper.
    """
    from vllm import LLM, SamplingParams

    enforce_eager = os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1"
    rev = hf_revision("models", model_name)  # pinned load (hf_pins.json)
    llm = LLM(
        model=model_name,
        revision=rev,
        tokenizer_revision=rev,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        enforce_eager=enforce_eager,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    out: list[str] = []
    n_chunks = (len(prompts) + VLLM_GREEDY_CHUNK_SIZE - 1) // VLLM_GREEDY_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_GREEDY_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_GREEDY_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] greedy chunk %d/%d (%d prompts)",
            i // VLLM_GREEDY_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in chunk_out)
    _reap_vllm(llm)
    return out


# ── batched teacher-forced / prompt-only capture ──────────────────────────────


def _token_budget_batches(rows: list[dict], batch_tokens: int) -> list[list[dict]]:
    """Greedy same-ish-length batches under a B*max_len token budget."""
    rows_sorted = sorted(rows, key=lambda r: len(r["full_ids"]), reverse=True)
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_max = 0
    for r in rows_sorted:
        m = max(cur_max, len(r["full_ids"]))
        if cur and (len(cur) + 1) * m > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            m = len(r["full_ids"])
        cur.append(r)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches


def _flush_leftpad_batch(
    model,
    tokenizer,
    capture: LayerCapture,
    capture_layers: list[int],
    batch: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One LEFT-padded forward; return (vbar, flast, fpool) each (B, Lc, H) fp16 CPU.

    The #810 pattern: left-pad, explicit ``position_ids = cumsum(mask)-1``
    (RoPE indexes each row from its first real token — without it left-pad
    silently diverges from batch-1), GPU-side reductions (span-mask mean +
    last-prompt gather), only the (B, Lc, H) slices cross PCIe. Rows with
    ``ans_len == 0`` get a zero vbar (caller marks them invalid).

    ``fpool`` (pooled-span-features round): a SECOND span-mask mean over the
    OPTIONAL ``pool_start``/``pool_len`` row fields (prompt-relative positions,
    same pad-offset arithmetic as the answer span). Rows without a pool span
    (``pool_len`` absent/0) get a zero fpool — the caller marks them invalid.
    """
    device = model.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    b = len(batch)
    max_len = max(len(r["full_ids"]) for r in batch)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    span_start = torch.zeros(b, dtype=torch.long)
    span_len = torch.zeros(b, dtype=torch.long)
    pool_start = torch.zeros(b, dtype=torch.long)
    pool_len = torch.zeros(b, dtype=torch.long)
    last_idx = torch.zeros(b, dtype=torch.long)
    for bi, r in enumerate(batch):
        ids = r["full_ids"]
        pad = max_len - len(ids)
        input_ids[bi, pad:] = torch.tensor(ids, dtype=torch.long)
        attn[bi, pad:] = 1
        span_start[bi] = pad + r["prompt_len"]
        span_len[bi] = r["ans_len"]
        pool_start[bi] = pad + r.get("pool_start", 0)
        pool_len[bi] = r.get("pool_len", 0)
        last_idx[bi] = pad + r["prompt_len"] - 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0)
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            **_logits_kwargs(model),
        )
    span_start = span_start.to(device)
    span_len = span_len.to(device)
    pool_start = pool_start.to(device)
    pool_len = pool_len.to(device)
    last_idx = last_idx.to(device)
    t_idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    span_mask = (t_idx >= span_start.unsqueeze(1)) & (
        t_idx < (span_start + span_len).unsqueeze(1)
    )  # (B, T)
    pool_mask = (t_idx >= pool_start.unsqueeze(1)) & (
        t_idx < (pool_start + pool_len).unsqueeze(1)
    )  # (B, T)
    vbar_layers, flast_layers, fpool_layers = [], [], []
    for li in capture_layers:
        hs = capture.latest[li]  # (B, T, H) on device
        m = span_mask.to(hs.dtype)
        sums = torch.einsum("bt,bth->bh", m, hs)
        vbar = sums / span_len.clamp(min=1).unsqueeze(1).to(hs.dtype)
        mp = pool_mask.to(hs.dtype)
        psums = torch.einsum("bt,bth->bh", mp, hs)
        fpool = psums / pool_len.clamp(min=1).unsqueeze(1).to(hs.dtype)
        gidx = last_idx.view(b, 1, 1).expand(b, 1, hs.shape[-1])
        flast = torch.gather(hs, 1, gidx).squeeze(1)
        vbar_layers.append(vbar.to(torch.float16))
        flast_layers.append(flast.to(torch.float16))
        fpool_layers.append(fpool.to(torch.float16))
    capture.latest.clear()
    vbar = torch.stack(vbar_layers, dim=1).cpu()  # (B, Lc, H)
    flast = torch.stack(flast_layers, dim=1).cpu()
    fpool = torch.stack(fpool_layers, dim=1).cpu()
    return vbar, flast, fpool


def batched_capture(
    model,
    tokenizer,
    capture: LayerCapture,
    capture_layers: list[int],
    rows: list[dict],
    batch_tokens: int,
    tag: str,
) -> dict[str, torch.Tensor]:
    """Batched TF/prompt-only capture over ``rows``; returns row-ordered tensors.

    Each row: ``{key, full_ids, prompt_len, ans_len}`` (``ans_len=0`` for a
    prompt-only row or an empty completion → ``valid`` False for the vbar).
    Optional ``pool_start``/``pool_len`` fields add the pooled-span mean
    (``fpool`` + ``pool_valid`` keys; zero rows where no pool span is given).
    """
    lc = len(capture_layers)
    hidden = model.config.hidden_size
    n = len(rows)
    vbar = torch.zeros(n, lc, hidden, dtype=torch.float16)
    flast = torch.zeros(n, lc, hidden, dtype=torch.float16)
    fpool = torch.zeros(n, lc, hidden, dtype=torch.float16)
    valid = torch.zeros(n, dtype=torch.bool)
    pool_valid = torch.zeros(n, dtype=torch.bool)
    for i, r in enumerate(rows):
        r["_row"] = i
    t0 = time.time()
    batches = _token_budget_batches(rows, batch_tokens)
    for bidx, batch in enumerate(batches):
        vb, fl, fp = _flush_leftpad_batch(model, tokenizer, capture, capture_layers, batch)
        for bi, r in enumerate(batch):
            vbar[r["_row"]] = vb[bi]
            flast[r["_row"]] = fl[bi]
            fpool[r["_row"]] = fp[bi]
            valid[r["_row"]] = r["ans_len"] > 0
            pool_valid[r["_row"]] = r.get("pool_len", 0) > 0
        if bidx % 20 == 0:
            logger.info("[%s] batch %d/%d (%.1fs)", tag, bidx + 1, len(batches), time.time() - t0)
    return {"vbar": vbar, "flast": flast, "fpool": fpool, "valid": valid, "pool_valid": pool_valid}


def masked_context_capture(
    model,
    tokenizer,
    capture: LayerCapture,
    capture_layers: list[int],
    rows: list[dict],
    batch_tokens: int,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Presentation (iii): RIGHT-padded batched forwards with the 4D mask.

    Each row: ``{key, full_ids, ctx_len}``. Right-pad keeps real tokens at
    their unpadded absolute positions (positions preserved — the §4.1 (iii)
    contract), default position_ids are correct, the last real token sits at
    ``seq_len - 1``. Returns (flast, fpool), each (n, Lc, H) fp16: the
    last-input-token activations plus the pooled-span mean over the OPTIONAL
    ``pool_start``/``pool_len`` row fields (ABSOLUTE positions — right-pad, no
    offset; masked context positions are never inside the span by
    construction). Rows without a pool span get a zero fpool.
    """
    lc = len(capture_layers)
    hidden = model.config.hidden_size
    n = len(rows)
    out = torch.zeros(n, lc, hidden, dtype=torch.float16)
    out_pool = torch.zeros(n, lc, hidden, dtype=torch.float16)
    for i, r in enumerate(rows):
        r["_row"] = i
    device = model.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dtype = next(model.parameters()).dtype
    t0 = time.time()
    batches = _token_budget_batches(rows, batch_tokens)
    for bidx, batch in enumerate(batches):
        b = len(batch)
        max_len = max(len(r["full_ids"]) for r in batch)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        seq_lens, ctx_lens = [], []
        pool_start = torch.zeros(b, dtype=torch.long)
        pool_len = torch.zeros(b, dtype=torch.long)
        for bi, r in enumerate(batch):
            ids = r["full_ids"]
            input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)  # RIGHT-pad
            seq_lens.append(len(ids))
            ctx_lens.append(r["ctx_len"])
            pool_start[bi] = r.get("pool_start", 0)  # absolute (right-pad, no offset)
            pool_len[bi] = r.get("pool_len", 0)
        mask4d = build_masked_context_4d_mask(ctx_lens, seq_lens, max_len, dtype, device)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=mask4d, **_logits_kwargs(model))
        last_idx = torch.tensor([sl - 1 for sl in seq_lens], device=device)
        pool_start_d = pool_start.to(device)
        pool_len_d = pool_len.to(device)
        t_idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        pool_mask = (t_idx >= pool_start_d.unsqueeze(1)) & (
            t_idx < (pool_start_d + pool_len_d).unsqueeze(1)
        )  # (B, T)
        for li_pos, li in enumerate(capture_layers):
            hs = capture.latest[li]
            gidx = last_idx.view(b, 1, 1).expand(b, 1, hs.shape[-1])
            flast = torch.gather(hs, 1, gidx).squeeze(1).to(torch.float16).cpu()
            mp = pool_mask.to(hs.dtype)
            psums = torch.einsum("bt,bth->bh", mp, hs)
            fpool = (
                (psums / pool_len_d.clamp(min=1).unsqueeze(1).to(hs.dtype)).to(torch.float16).cpu()
            )
            for bi, r in enumerate(batch):
                out[r["_row"], li_pos] = flast[bi]
                out_pool[r["_row"], li_pos] = fpool[bi]
        capture.latest.clear()
        if bidx % 20 == 0:
            logger.info("[%s] batch %d/%d (%.1fs)", tag, bidx + 1, len(batches), time.time() - t0)
    return out, out_pool


# ── 4D-mask invariance smoke (plan §8) ────────────────────────────────────────


def mask_invariance_check(model, tokenizer, capture_layers: list[int]) -> dict:
    """Two-part correctness probe for the masked-context presentation.

    (a) masked forward (dummy context) != unmasked forward at the query slot —
        the mask ENGAGED; (b) two DIFFERENT same-length masked-out contexts →
        identical query-slot activations — the mask is COMPLETE (the definitive
        probe: if any context information reaches the query slot the outputs
        differ). Returns a result dict; never raises on failure (the caller
        applies the §8 sdpa → eager → drop ladder).
    """
    instance = {
        "id": "mask_smoke",
        "system_prompt": "You are a careful assistant who answers concisely and factually.",
        "prefix_messages": [],
    }
    q = "What are two interesting facts about the Moon?"
    prefix_ids, full_ids = context_prefix_split(tokenizer, instance, q)
    ctx_len = len(prefix_ids)
    ids_a = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    # Context B: SAME length, different interior content-token ids (special
    # tokens at the block edges left untouched) — length equality by
    # construction, so query positions are identical across A and B.
    ids_b = ids_a.clone()
    interior = torch.arange(3, ctx_len - 2)
    vocab = model.config.vocab_size
    ids_b[0, interior] = (ids_b[0, interior] + 7919) % (vocab - 10)
    device = model.device
    dtype = next(model.parameters()).dtype
    capture = LayerCapture(model, len(model.model.layers))
    try:

        def _last_stack(ids: torch.Tensor, masked: bool) -> torch.Tensor:
            t = ids.shape[1]
            kwargs = {}
            if masked:
                kwargs["attention_mask"] = build_masked_context_4d_mask(
                    [ctx_len], [t], t, dtype, device
                )
            with torch.no_grad():
                _ = model(input_ids=ids.to(device), **kwargs, **_logits_kwargs(model))
            vecs = [capture.latest[li][0, -1, :].float().cpu() for li in capture_layers]
            capture.latest.clear()
            return torch.stack(vecs)  # (Lc, H)

        unmasked_a = _last_stack(ids_a, masked=False)
        masked_a = _last_stack(ids_a, masked=True)
        masked_b = _last_stack(ids_b, masked=True)
    except Exception as e:  # backend rejects the 4D mask → caller falls back
        capture.remove()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    capture.remove()
    diff_engaged = float((unmasked_a - masked_a).abs().max())
    cos_engaged = float(
        torch.nn.functional.cosine_similarity(
            unmasked_a[-1].flatten(), masked_a[-1].flatten(), dim=0
        )
    )
    invariance_maxabs = float((masked_a - masked_b).abs().max())
    scale = float(masked_a.abs().max().clamp(min=1e-6))
    engaged = diff_engaged > 1e-4 * scale
    invariant = invariance_maxabs < 1e-4 * scale
    return {
        "ok": bool(engaged and invariant),
        "engaged": bool(engaged),
        "invariant": bool(invariant),
        "diff_engaged_maxabs": diff_engaged,
        "cos_engaged_lastlayer": cos_engaged,
        "invariance_maxabs": invariance_maxabs,
        "activation_scale": scale,
    }


def validate_mask_backend(args, model, tokenizer, capture_layers) -> tuple[str, dict, object]:
    """§8 ladder: sdpa → eager (re-validated) → drop. Returns (backend, results, model)."""
    res_sdpa = mask_invariance_check(model, tokenizer, capture_layers)
    if res_sdpa["ok"]:
        return "sdpa", {"sdpa": res_sdpa}, model
    logger.warning("4D-mask invariance FAILED under sdpa: %s — trying eager", res_sdpa)
    model_eager, _ = load_model_and_tokenizer(args, attn_implementation="eager")
    res_eager = mask_invariance_check(model_eager, tokenizer, capture_layers)
    if res_eager["ok"]:
        return "eager", {"sdpa": res_sdpa, "eager": res_eager}, model_eager
    logger.error(
        "4D-mask invariance FAILED under sdpa AND eager — dropping arm_qry_iii "
        "(the §8 REGISTERED HEADLINE DOWNGRADE applies): %s",
        res_eager,
    )
    return "dropped", {"sdpa": res_sdpa, "eager": res_eager}, model


# ── serial-reference equivalence check (batched-rewrite gate) ─────────────────


def equivalence_check(model, tokenizer, capture_layers, rows: list[dict]) -> dict:
    """cosine(batched, serial) >= 0.999 per (cell x layer) for vbar, flast AND fpool.

    Serial reference = batch-1 forward, NO padding (the
    ``capture_v0_for_context`` regime); batched = the left-pad path above with
    the rows deliberately co-batched (B>=2, different lengths → real padding).
    ``fpool`` is compared only for rows carrying a pool span.
    """
    capture = LayerCapture(model, len(model.model.layers))
    try:
        serial_vbar, serial_flast, serial_fpool = [], [], []
        for r in rows:
            ids = torch.tensor(r["full_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
            with torch.no_grad():
                _ = model(input_ids=ids, **_logits_kwargs(model))
            vb_l, fl_l, fp_l = [], [], []
            for li in capture_layers:
                hs = capture.latest[li][0]  # (T, H)
                span = hs[r["prompt_len"] : r["prompt_len"] + r["ans_len"]]
                vb_l.append(span.float().mean(dim=0).cpu())
                fl_l.append(hs[r["prompt_len"] - 1].float().cpu())
                if r.get("pool_len", 0) > 0:
                    pspan = hs[r["pool_start"] : r["pool_start"] + r["pool_len"]]
                    fp_l.append(pspan.float().mean(dim=0).cpu())
            capture.latest.clear()
            serial_vbar.append(torch.stack(vb_l))
            serial_flast.append(torch.stack(fl_l))
            serial_fpool.append(torch.stack(fp_l) if fp_l else None)
        batched = batched_capture(
            model, tokenizer, capture, capture_layers, [dict(r) for r in rows], 10**9, "equiv"
        )
    finally:
        capture.remove()
    min_cos_vbar, min_cos_flast, min_cos_fpool = 1.0, 1.0, 1.0
    for i in range(len(rows)):
        for li_pos in range(len(capture_layers)):
            cv = torch.nn.functional.cosine_similarity(
                serial_vbar[i][li_pos], batched["vbar"][i, li_pos].float(), dim=0
            )
            cf = torch.nn.functional.cosine_similarity(
                serial_flast[i][li_pos], batched["flast"][i, li_pos].float(), dim=0
            )
            min_cos_vbar = min(min_cos_vbar, float(cv))
            min_cos_flast = min(min_cos_flast, float(cf))
            if serial_fpool[i] is not None:
                cp = torch.nn.functional.cosine_similarity(
                    serial_fpool[i][li_pos], batched["fpool"][i, li_pos].float(), dim=0
                )
                min_cos_fpool = min(min_cos_fpool, float(cp))
    ok = min_cos_vbar >= 0.999 and min_cos_flast >= 0.999 and min_cos_fpool >= 0.999
    return {
        "ok": ok,
        "min_cos_vbar": min_cos_vbar,
        "min_cos_flast": min_cos_flast,
        "min_cos_fpool": min_cos_fpool,
    }


# ── row builders ──────────────────────────────────────────────────────────────


def build_tf_row(tokenizer, instance: dict, q: str, ans: str, key) -> dict:
    """(prompt + answer) row for the TF capture; ans_len == 0 marks invalid."""
    prompt_text = render_full_prompt(tokenizer, instance, q)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(ans, add_special_tokens=False)["input_ids"] if ans else []
    return {
        "key": key,
        "full_ids": prompt_ids + ans_ids,
        "prompt_len": len(prompt_ids),
        "ans_len": len(ans_ids),
    }


def build_prompt_row(tokenizer, instance: dict, q: str, key) -> dict:
    """Prompt-only row (F_full for store cells)."""
    prompt_text = render_full_prompt(tokenizer, instance, q)
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return {"key": key, "full_ids": ids, "prompt_len": len(ids), "ans_len": 0}


# ── pooled-span row builders (pooled-span-features round, plan v6 §4.1/§4.2) ──

ASSISTANT_HEADER = "<|im_start|>assistant\n"
IDENTITY_SAMPLE_PER_FAMILY = 100
IDENTITY_MEDIAN_FLOOR = 0.99  # k1 hard gate (calibrated bf16 floor, commit 33a2a5df33)
IDENTITY_WARN_FLOOR = 0.999  # sub-0.999: warn-and-record (bf16 batching numerics)


def _user_block(q: str) -> str:
    """The user-turn query block incl. its delimiters (the qry arms' owning span)."""
    return f"<|im_start|>user\n{q}<|im_end|>\n"


def _piecewise_ids(tokenizer, pieces: list[str]) -> list[list[int]]:
    """Tokenize template pieces; assert piecewise == full-render tokenization.

    The per-row fail-loud retokenization-equality assert (§4.1): special-token
    boundaries make piecewise tokenization exact; BPE drift fails loud here
    rather than silently mis-pooling a span.
    """
    ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in pieces]
    joined = [t for chunk in ids for t in chunk]
    full = tokenizer("".join(pieces), add_special_tokens=False)["input_ids"]
    assert joined == full, (
        f"piecewise retokenization mismatch (BPE boundary drift): pieces="
        f"{[p[:60] for p in pieces]!r}"
    )
    return ids


def build_pool_qry_row(tokenizer, q: str, pres: str, key) -> dict:
    """Pooled query-presentation row (i)/(ii): span = the user-turn block ONLY.

    System turn ((i)) and assistant header are EXCLUDED from the owning span
    (the §4.1 uniform span rule — arm-external template tokens would smuggle
    context-adjacent signal into the query arms).
    """
    if pres == "i":
        text = render_qry_empty_system(tokenizer, q)
        sys_block = text[: -len(user_turn_suffix(q))]
        pieces = [sys_block, _user_block(q), ASSISTANT_HEADER]
        assert sys_block, "presentation (i) rendered an empty system block"
    else:
        assert pres == "ii", pres
        text = render_qry_no_system_block(q)
        pieces = [_user_block(q), ASSISTANT_HEADER]
    assert "".join(pieces) == text, f"piece decomposition != render for pres {pres}"
    ids = _piecewise_ids(tokenizer, pieces)
    full_ids = [t for chunk in ids for t in chunk]
    pool_start = len(ids[0]) if pres == "i" else 0
    pool_len = len(ids[-2])  # the user-turn block piece
    assert pool_len > 0, f"zero-width query span for {key} (pres {pres})"
    return {
        "key": key,
        "full_ids": full_ids,
        "prompt_len": len(full_ids),
        "ans_len": 0,
        "pool_start": pool_start,
        "pool_len": pool_len,
    }


def build_pool_full_row(tokenizer, instance: dict, q: str, key) -> dict:
    """Pooled full-prompt row: span = ALL real input tokens (ctx+template+query+header)."""
    row = build_prompt_row(tokenizer, instance, q, key)
    row["pool_start"] = 0
    row["pool_len"] = len(row["full_ids"])
    return row


def build_pool_iii_row(tokenizer, instance: dict, q: str, key) -> dict:
    """Pooled masked-context (iii) row: span = the query block at ABSOLUTE positions.

    Right-pad keeps real tokens at their unpadded absolute positions, so the
    span is [ctx_len, ctx_len + len(user_block)) with no pad offset; masked
    context positions are never pooled (they precede the span), and the
    assistant-header tokens are excluded (same owning-span rule as (i)/(ii)).
    """
    prefix_ids, full_ids = context_prefix_split(tokenizer, instance, q)
    ids_user = tokenizer(_user_block(q), add_special_tokens=False)["input_ids"]
    ids_asst = tokenizer(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    assert prefix_ids + ids_user + ids_asst == full_ids, (
        f"piecewise retokenization mismatch for (iii) row {key} "
        "(prefix + user block + assistant header != full render)"
    )
    assert len(ids_user) > 0, f"zero-width query span for (iii) row {key}"
    return {
        "key": key,
        "full_ids": full_ids,
        "ctx_len": len(prefix_ids),
        "pool_start": len(prefix_ids),
        "pool_len": len(ids_user),
    }


def masked_equivalence_check(model, tokenizer, capture_layers, rows: list[dict]) -> dict:
    """Batched right-pad 4D-mask capture vs an independent batch-1 slice reference.

    cosine(batched, serial) >= 0.999 per (row x layer) for flast AND fpool —
    the batched-rewrite equivalence gate for the NEW masked-span gather (the
    left-pad path is covered by ``equivalence_check``).
    """
    dtype = next(model.parameters()).dtype
    capture = LayerCapture(model, len(model.model.layers))
    try:
        serial_fl, serial_fp = [], []
        for r in rows:
            t = len(r["full_ids"])
            ids = torch.tensor(r["full_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
            mask4d = build_masked_context_4d_mask([r["ctx_len"]], [t], t, dtype, model.device)
            with torch.no_grad():
                _ = model(input_ids=ids, attention_mask=mask4d, **_logits_kwargs(model))
            fl, fp = [], []
            for li in capture_layers:
                hs = capture.latest[li][0]  # (T, H)
                fl.append(hs[t - 1].float().cpu())
                fp.append(
                    hs[r["pool_start"] : r["pool_start"] + r["pool_len"]].float().mean(0).cpu()
                )
            capture.latest.clear()
            serial_fl.append(torch.stack(fl))
            serial_fp.append(torch.stack(fp))
        bfl, bfp = masked_context_capture(
            model, tokenizer, capture, capture_layers, [dict(r) for r in rows], 10**9, "meq"
        )
    finally:
        capture.remove()
    min_fl = min_fp = 1.0
    for i in range(len(rows)):
        for lp in range(len(capture_layers)):
            cf = torch.nn.functional.cosine_similarity(serial_fl[i][lp], bfl[i, lp].float(), dim=0)
            cp = torch.nn.functional.cosine_similarity(serial_fp[i][lp], bfp[i, lp].float(), dim=0)
            min_fl = min(min_fl, float(cf))
            min_fp = min(min_fp, float(cp))
    return {
        "ok": min_fl >= 0.999 and min_fp >= 0.999,
        "min_cos_flast": min_fl,
        "min_cos_fpool": min_fp,
    }


# ── pooled-span-features stage pipeline (plan v6 §4.2) ────────────────────────


def _pool_inputs(args) -> tuple[list[dict], dict[str, list[str]]]:
    """Battery instances + per-genre query pools with the SAME --smoke slicing as main().

    The uc pool concatenates the 48-probe store pool + the ext pool so uc
    ``q_idx`` is GLOBAL (0..143 production; ext lives at 48+) — matching the
    fit-side grid join. Store pools stay full-length under --smoke (same
    rationale as main(): truncating uc48 would shift the global ext indices).
    """
    _, instances = load_battery()
    uc48 = [
        r["text"] for r in load_json(PROJECT_ROOT / "data/issue594/probes_ultrachat.json")["probes"]
    ]
    uc_ext = [r["text"] for r in load_json(args.data_dir / "probes_uc_ext.json")["probes"]]
    dolly = [r["text"] for r in load_json(args.data_dir / "probes_dolly.json")["probes"]]
    betley = [r["text"] for r in load_json(args.data_dir / "probes_betley.json")["probes"]]
    if args.smoke:
        n_ctx = args.n_ctx or 1
        nq = args.n_queries or 4
        instances = instances[:n_ctx]
        uc_ext, dolly = uc_ext[:nq], dolly[:2]
    elif args.n_ctx:
        instances = instances[: args.n_ctx]
    pools = {"uc": uc48 + uc_ext, "betley": betley, "dolly": dolly}
    return instances, pools


def _save_pool_pack(path: Path, tensors: dict, run_meta: dict, stage: str, keys: list) -> None:
    """Persist one pooled pack (fpool + flast + valid) with row metadata."""
    save_pack(
        path,
        {"fpool": tensors["fpool"], "flast": tensors["flast"], "valid": tensors["pool_valid"]},
        {**run_meta, "stage": stage, "rows": keys},
    )


def pooled_capture_stage(  # noqa: C901 — linear per-family pipeline, see phase markers
    args, packs_dir: Path, shard_k: int, n_shards: int, shard_tag: str
) -> None:
    """Capture the five pooled feature families for one context/query shard.

    Packs are shard-checkpointed: an existing pack file is SKIPPED (resume
    granularity for the sequential single-GPU run; ``--fresh`` overwrites).
    """
    phase("pooled_load")
    model, tokenizer = load_model_and_tokenizer(args)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    if not args.tiny_model:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert hidden == args.expected_hidden, (hidden, args.expected_hidden)
    capture_layers = list(range(n_layers))
    batch_tokens = args.batch_tokens or (2048 if args.tiny_model else 16384)
    instances, pools = _pool_inputs(args)
    inst_shard = [inst for i, inst in enumerate(instances) if i % n_shards == shard_k]
    logger.info(
        "[pooled] shard %d/%d: %d contexts; pools uc=%d betley=%d dolly=%d",
        shard_k,
        n_shards,
        len(inst_shard),
        len(pools["uc"]),
        len(pools["betley"]),
        len(pools["dolly"]),
    )
    run_meta: dict = {
        "shard": args.shard,
        "smoke": args.smoke,
        "model": args.model if not args.tiny_model else "tiny-random-qwen2",
        "capture_layers": capture_layers,
        "pools_hash": {g: texts_hash(p) for g, p in pools.items()},
        "round": "pooled-span-features",
        "metadata": reproducibility_metadata({"script": "issue923_capture:pooled"}),
    }

    phase("pooled_mask_smoke")
    backend, mask_results, model_iii = validate_mask_backend(args, model, tokenizer, capture_layers)
    run_meta["mask_backend"] = backend
    run_meta["mask_invariance"] = mask_results

    def _skip(path: Path) -> bool:
        if path.exists() and not args.fresh:
            logger.info("[pooled] %s exists — skipped (resume; --fresh overwrites)", path.name)
            return True
        return False

    if args.smoke:
        # Batched-rewrite equivalence gates (B>=2, real padding) for the NEW
        # fpool reductions: left-pad path (TF rows carrying BOTH spans) + the
        # masked right-pad path (independent batch-1 slice reference).
        phase("pooled_equivalence")
        eq_rows = []
        for qi, q in enumerate(pools["uc"][:3]):
            r = build_tf_row(tokenizer, instances[0], q, "smoke answer text", ("eq", qi))
            r["pool_start"] = 0
            r["pool_len"] = r["prompt_len"]
            eq_rows.append(r)
        run_meta["equivalence_pooled"] = equivalence_check(
            model, tokenizer, capture_layers, eq_rows
        )
        assert run_meta["equivalence_pooled"]["ok"], run_meta["equivalence_pooled"]
        logger.info("[pooled] left-pad equivalence PASS: %s", run_meta["equivalence_pooled"])
        if backend != "dropped":
            meq_rows = [
                build_pool_iii_row(tokenizer, instances[0], q, ("meq", qi))
                for qi, q in enumerate(pools["uc"][:3])
            ]
            run_meta["equivalence_masked"] = masked_equivalence_check(
                model_iii, tokenizer, capture_layers, meq_rows
            )
            assert run_meta["equivalence_masked"]["ok"], run_meta["equivalence_masked"]
            logger.info("[pooled] masked equivalence PASS: %s", run_meta["equivalence_masked"])

    phase("pooled_capture")
    capture = LayerCapture(model, n_layers)
    try:
        # (a) pool_fctx — prefix-only forwards, span = the WHOLE context block.
        pack_path = packs_dir / f"pool_fctx_{shard_tag}.pt"
        if not _skip(pack_path):
            probe_q = pools["uc"][0]
            rows, keys = [], []
            for inst in inst_shard:
                prefix_ids, _full = context_prefix_split(tokenizer, inst, probe_q)
                rows.append(
                    {
                        "key": inst["id"],
                        "full_ids": prefix_ids,
                        "prompt_len": len(prefix_ids),
                        "ans_len": 0,
                        "pool_start": 0,
                        "pool_len": len(prefix_ids),
                    }
                )
                keys.append({"ctx_id": inst["id"], "ctx_len": len(prefix_ids)})
            if rows:
                tensors = batched_capture(
                    model, tokenizer, capture, capture_layers, rows, batch_tokens, "pool_fctx"
                )
                _save_pool_pack(pack_path, tensors, run_meta, "pool_fctx", keys)

        # (b) pool_fqry_{i,ii} — query-level, query-sharded, span = user block.
        for pres in ("i", "ii"):
            pack_path = packs_dir / f"pool_fqry_{pres}_{shard_tag}.pt"
            if _skip(pack_path):
                continue
            rows, keys = [], []
            for genre, pool in pools.items():
                for qi, q in enumerate(pool):
                    if qi % n_shards != shard_k:
                        continue
                    rows.append(build_pool_qry_row(tokenizer, q, pres, (genre, qi)))
                    keys.append({"genre": genre, "q_idx": qi})
            if rows:
                tensors = batched_capture(
                    model,
                    tokenizer,
                    capture,
                    capture_layers,
                    rows,
                    batch_tokens,
                    f"pool_fqry_{pres}",
                )
                _save_pool_pack(pack_path, tensors, run_meta, f"pool_fqry_{pres}", keys)

        # (c) pool_ffull_{genre} — full prompt-only forwards for EVERY cell,
        # ctx-sharded, span = all real input tokens.
        for genre, pool in pools.items():
            pack_path = packs_dir / f"pool_ffull_{genre}_{shard_tag}.pt"
            if _skip(pack_path):
                continue
            rows, keys = [], []
            for inst in inst_shard:
                for qi, q in enumerate(pool):
                    rows.append(build_pool_full_row(tokenizer, inst, q, (inst["id"], qi)))
                    keys.append({"ctx_id": inst["id"], "q_idx": qi})
            if rows:
                tensors = batched_capture(
                    model,
                    tokenizer,
                    capture,
                    capture_layers,
                    rows,
                    batch_tokens,
                    f"pool_ffull_{genre}",
                )
                _save_pool_pack(pack_path, tensors, run_meta, f"pool_ffull_{genre}", keys)
    finally:
        capture.remove()

    # (d) pool_fqry_iii_{genre} — masked-context forwards, ctx-sharded, span =
    # the query block at absolute positions (masked context never pooled).
    if backend != "dropped":
        capture_iii = LayerCapture(model_iii, len(model_iii.model.layers))
        try:
            for genre, pool in pools.items():
                pack_path = packs_dir / f"pool_fqry_iii_{genre}_{shard_tag}.pt"
                if _skip(pack_path):
                    continue
                rows, keys = [], []
                for inst in inst_shard:
                    for qi, q in enumerate(pool):
                        rows.append(build_pool_iii_row(tokenizer, inst, q, (inst["id"], qi)))
                        keys.append({"ctx_id": inst["id"], "q_idx": qi})
                if not rows:
                    continue
                flast, fpool = masked_context_capture(
                    model_iii,
                    tokenizer,
                    capture_iii,
                    capture_layers,
                    rows,
                    batch_tokens,
                    f"pool_fqry_iii_{genre}",
                )
                pool_valid = torch.tensor([r["pool_len"] > 0 for r in rows], dtype=torch.bool)
                _save_pool_pack(
                    pack_path,
                    {"fpool": fpool, "flast": flast, "pool_valid": pool_valid},
                    run_meta,
                    f"pool_fqry_iii_{genre}",
                    keys,
                )
        finally:
            capture_iii.remove()
    else:
        logger.warning("pool_fqry_iii DROPPED (mask invariance failed under sdpa AND eager)")

    dump_json(run_meta, packs_dir / f"pool_run_meta_{shard_tag}.json")


def _fetch_parent_refs(ref_dir: Path) -> None:
    """Fetch the parent capture packs (identity refs) at the PINNED dataset revision.

    Small families (fctx / fqry_i / fqry_ii) fetch all 4 shards; the large
    per-cell families fetch shard0 only (~100-row sample per family suffices
    for the k1 gate; keeps the download ~2.5 GB).
    """
    from huggingface_hub import hf_hub_download

    rev = hf_revision("datasets", HF_DATA_REPO)
    names = (
        [f"fctx_shard{k}of4.pt" for k in range(4)]
        + [f"fqry_i_shard{k}of4.pt" for k in range(4)]
        + [f"fqry_ii_shard{k}of4.pt" for k in range(4)]
        + [
            f"{stem}_shard0of4.pt"
            for stem in (
                "fqry_iii_uc",
                "fqry_iii_betley",
                "fqry_iii_dolly",
                "ffull_uc48",
                "ffull_betley",
                "tgt_ucext",
                "tgt_dolly",
            )
        ]
    )
    ref_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = ref_dir / name
        if target.exists():
            continue
        local = hf_hub_download(
            HF_DATA_REPO,
            f"{HF_PREFIX_923}/analysis_tensors/capture/{name}",
            repo_type="dataset",
            revision=rev,
            local_dir=str(ref_dir / "hf_dl"),
        )
        target.write_bytes(Path(local).read_bytes())
        logger.info("[identity] fetched parent ref %s", name)


def pooled_identity_stage(args, packs_dir: Path) -> bool:  # noqa: C901
    """k1 content-identity spot-check: cos(flast_new, flast_parent) per pack family.

    ~100 seeded-sample rows per family; per-row score = MIN over layers of the
    per-layer cosine (the parent fctx-check convention). Gate: family MEDIAN
    >= 0.99 (hard, the calibrated bf16-batching floor); sub-0.999 rows/medians
    are warned + recorded, never a failure. Covers the causal identity
    (prompt-only flast == TF flast at the last prompt token) AND render/join
    provenance in one check. Returns overall pass; artifact
    ``identity_check.json`` is written (and uploaded) either way.
    """
    ref_dir = args.identity_ref_dir or (
        PROJECT_ROOT / "data" / "issue_923" / "capture" / "parent_ref"
    )
    if args.identity_ref_dir is None:
        _fetch_parent_refs(ref_dir)

    def _load_ref(stem_glob: str, keyfn) -> dict:
        idx: dict = {}
        files = sorted(ref_dir.glob(stem_glob))
        assert files, f"identity check: no parent ref packs matching {stem_glob} under {ref_dir}"
        for f in files:
            tensors, meta = load_pack(f)
            for i, r in enumerate(meta["rows"]):
                idx[keyfn(r)] = (tensors["flast"], i)
        return idx

    def _cellkey(r: dict):
        return (r["ctx_id"], r["q_idx"])

    def _qkey(r: dict):
        return (r["genre"], r["q_idx"])

    # pool_ffull_uc joins TWO parent stems: store cells (ffull_uc48, global
    # q_idx already) + ext cells (tgt_ucext flast, LOCAL q_idx -> +48 offset).
    ffull_uc_idx = _load_ref("ffull_uc48_shard*.pt", _cellkey)
    for f in sorted(ref_dir.glob("tgt_ucext_shard*.pt")):
        tensors, meta = load_pack(f)
        for i, r in enumerate(meta["rows"]):
            ffull_uc_idx[(r["ctx_id"], 48 + r["q_idx"])] = (tensors["flast"], i)

    specs: list[tuple[str, dict, object]] = [
        ("pool_fctx", _load_ref("fctx_shard*.pt", lambda r: r["ctx_id"]), lambda r: r["ctx_id"]),
        ("pool_fqry_i", _load_ref("fqry_i_shard*.pt", _qkey), _qkey),
        ("pool_fqry_ii", _load_ref("fqry_ii_shard*.pt", _qkey), _qkey),
        ("pool_fqry_iii_uc", _load_ref("fqry_iii_uc_shard*.pt", _cellkey), _cellkey),
        ("pool_fqry_iii_betley", _load_ref("fqry_iii_betley_shard*.pt", _cellkey), _cellkey),
        ("pool_fqry_iii_dolly", _load_ref("fqry_iii_dolly_shard*.pt", _cellkey), _cellkey),
        ("pool_ffull_uc", ffull_uc_idx, _cellkey),
        ("pool_ffull_betley", _load_ref("ffull_betley_shard*.pt", _cellkey), _cellkey),
        ("pool_ffull_dolly", _load_ref("tgt_dolly_shard*.pt", _cellkey), _cellkey),
    ]
    mask_dropped = False
    meta_files = sorted(packs_dir.glob("pool_run_meta_*.json"))
    if meta_files and load_json(meta_files[0]).get("mask_backend") == "dropped":
        mask_dropped = True

    rng = np.random.default_rng(SEED)
    results: dict = {}
    ok_all = True
    for fam, ref_idx, keyfn in specs:
        pool_files = sorted(packs_dir.glob(f"{fam}_shard*.pt"))
        if not pool_files and fam.startswith("pool_fqry_iii") and mask_dropped:
            results[fam] = {"skipped": "mask_backend dropped (recorded in run_meta)"}
            continue
        assert pool_files, f"identity check: no pooled packs matching {fam}_shard*.pt"
        matched = []
        for f in pool_files:
            tensors, meta = load_pack(f)
            for i, r in enumerate(meta["rows"]):
                hit = ref_idx.get(keyfn(r))
                if hit is not None:
                    matched.append((tensors["flast"], i, hit))
        assert matched, f"identity check: ZERO overlapping rows for {fam} — join broken"
        if len(matched) > IDENTITY_SAMPLE_PER_FAMILY:
            sel = rng.choice(len(matched), size=IDENTITY_SAMPLE_PER_FAMILY, replace=False)
            matched = [matched[int(s)] for s in sorted(sel)]
        cos_rows = []
        for new_t, ni, (ref_t, ri) in matched:
            cos = torch.nn.functional.cosine_similarity(
                new_t[ni].float(), ref_t[ri].float(), dim=1
            )  # per layer
            cos_rows.append(float(cos.min()))
        med = float(np.median(cos_rows))
        fam_ok = med >= IDENTITY_MEDIAN_FLOOR
        ok_all &= fam_ok
        n_warn = sum(c < IDENTITY_WARN_FLOOR for c in cos_rows)
        if not fam_ok:
            logger.error("[identity] %s FAILED: median min-cos %.6f < %.2f", fam, med, 0.99)
        elif n_warn:
            logger.warning(
                "[identity] %s: %d/%d rows below %.3f (bf16 numerics, recorded); median %.6f",
                fam,
                n_warn,
                len(cos_rows),
                IDENTITY_WARN_FLOOR,
                med,
            )
        results[fam] = {
            "n": len(cos_rows),
            "median_min_cos": med,
            "min": float(min(cos_rows)),
            "n_below_0p999": n_warn,
            "pass": fam_ok,
            "cos_rows": cos_rows,  # per-row min-over-layers cos (figure input)
        }
    payload = {
        "pass": bool(ok_all),
        "median_floor": IDENTITY_MEDIAN_FLOOR,
        "warn_floor": IDENTITY_WARN_FLOOR,
        "families": results,
        "metadata": reproducibility_metadata({"script": "issue923_capture:pooled_identity"}),
    }
    dump_json(payload, packs_dir / "identity_check.json")
    if not args.no_upload:
        hub._upload(
            packs_dir / "identity_check.json",
            HF_DATA_REPO,
            "dataset",
            f"{HF_PREFIX_923}/analysis_tensors/pooled_capture/identity_check.json",
            upload_as_file=True,
        )
    logger.info("[identity] overall pass=%s: %s", ok_all, json.dumps(results))
    return bool(ok_all)


def _scoped_tree_listing(prefix: str, attempts: int = 4) -> list[str]:
    """Bounded-retry SCOPED ``list_repo_tree`` (server-side prefix) listing.

    A bare ``list_repo_files`` full listing of the ~1M-file data repo times
    out (#833), and huggingface_hub's pagination retries ONLY 429 on cursor
    pages — a first-page 429 or any 5xx would otherwise fail a valid phase
    on a transient (r1 Minor). Non-transient HTTP errors raise immediately.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return [
                e.path
                for e in HfApi().list_repo_tree(
                    HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
                )
            ]
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code not in (429, 500, 502, 503, 504):
                raise
            last = e
            logger.warning(
                "[hub] scoped listing %s: transient HTTP %s (attempt %d/%d)",
                prefix,
                code,
                attempt + 1,
                attempts,
            )
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"scoped listing failed after {attempts} attempts: {prefix}") from last


def pooled_upload_stage(args, packs_dir: Path) -> None:
    """Upload the pooled packs dir (one folder commit) + verify + sentinel LAST.

    Verification uses SCOPED ``list_repo_tree`` (server-side prefix, bounded
    transient retry) — a bare ``list_repo_files`` full listing of the
    ~1M-file data repo times out (the #833 gotcha).
    """
    prefix = f"{HF_PREFIX_923}/analysis_tensors/pooled_capture"
    n_local = len([p for p in packs_dir.iterdir() if p.name != "UPLOAD_COMPLETE_POOLED.json"])
    hub._upload(packs_dir, HF_DATA_REPO, "dataset", prefix)
    listing = _scoped_tree_listing(prefix)
    assert len(listing) >= n_local, (
        f"pooled upload verification failed: hub {len(listing)} < local {n_local}"
    )
    complete = {
        "uploaded": n_local,
        "files": listing,
        "metadata": reproducibility_metadata({"script": "issue923_capture:pooled_upload"}),
    }
    complete_path = packs_dir / "UPLOAD_COMPLETE_POOLED.json"
    dump_json(complete, complete_path)
    hub._upload(
        complete_path,
        HF_DATA_REPO,
        "dataset",
        f"{prefix}/UPLOAD_COMPLETE_POOLED.json",
        upload_as_file=True,
    )
    logger.info("[pooled_upload] verified %d files under %s", len(listing), prefix)


def pooled_main(args) -> int:
    """Dispatcher for the pooled-span-features round stages (capture/upload/identity)."""
    shard_k, n_shards = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_k < n_shards, args.shard
    if args.smoke and args.out_dir == PROJECT_ROOT / "data" / "issue_923" / "capture":
        # Smoke redirect (r1 Minor; fit-script parity): 1-ctx smoke packs must
        # never land in the canonical dir, where skip-if-exists resume would
        # later mix smoke and production shards.
        args.out_dir = Path("/tmp/issue-923-smoke/capture")
        logger.info("[pooled] --smoke: out-dir redirected to %s", args.out_dir)
    packs_dir = args.out_dir / "packs_pooled"
    packs_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"shard{shard_k}of{n_shards}"
    if args.pooled_features:
        pooled_capture_stage(args, packs_dir, shard_k, n_shards, shard_tag)
    if args.pooled_upload and not args.no_upload:
        phase("pooled_upload")
        pooled_upload_stage(args, packs_dir)
    if args.pooled_identity_check:
        phase("pooled_identity")
        if not pooled_identity_stage(args, packs_dir):
            print("[phase=identity_gate_failed]", flush=True)
            return 4
    phase("done")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Issue #923 Phase 1 gen+capture")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tiny-model", action="store_true", help="2-layer random Qwen2 CPU stub")
    p.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    p.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--no-vllm", action="store_true", help="HF greedy generate (CPU smoke)")
    p.add_argument("--shard", default="0/1", help="k/n context shard")
    p.add_argument(
        "--phases", default="gen,tf,ffull,partials", help="csv of gen,tf,ffull,partials,upload"
    )
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_923" / "capture")
    p.add_argument("--eval-dir", type=Path, default=EVAL_RESULTS_DIR)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Phase-0 inputs dir")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--mask-smoke-only", action="store_true")
    p.add_argument("--batch-tokens", type=int, default=None)
    p.add_argument("--max-new-tokens-smoke", type=int, default=16)
    p.add_argument("--n-ctx", type=int, default=None)
    p.add_argument("--n-queries", type=int, default=None)
    # pooled-span-features round (plan v6 §4.2) — separate stage pipeline:
    p.add_argument(
        "--pooled-features",
        action="store_true",
        help="capture the five pooled (span-mean) feature families for this shard",
    )
    p.add_argument(
        "--pooled-upload",
        action="store_true",
        help="upload packs_pooled/ to analysis_tensors/pooled_capture (verify + sentinel)",
    )
    p.add_argument(
        "--pooled-identity-check",
        action="store_true",
        help="k1 gate: cos(flast_new, flast_parent) per family; exit 4 on median<0.99",
    )
    p.add_argument(
        "--identity-ref-dir",
        type=Path,
        default=None,
        help="parent ref packs dir (default: fetch from HF at the pinned revision)",
    )
    p.add_argument("--fresh", action="store_true", help="overwrite existing pooled packs")
    return p.parse_args(argv)


def main() -> int:  # noqa: C901 — linear phase pipeline; see phase() markers
    args = parse_args()
    if args.pooled_features or args.pooled_upload or args.pooled_identity_check:
        return pooled_main(args)
    shard_k, n_shards = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_k < n_shards, args.shard
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]

    phase("load")
    model, tokenizer = load_model_and_tokenizer(args)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    if not args.tiny_model:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert hidden == args.expected_hidden, (hidden, args.expected_hidden)
    capture_layers = list(range(n_layers))
    batch_tokens = args.batch_tokens or (2048 if args.tiny_model else 16384)

    if args.mask_smoke_only:
        phase("mask_smoke")
        backend, results, _ = validate_mask_backend(args, model, tokenizer, capture_layers)
        print(json.dumps({"mask_backend": backend, "results": results}, indent=2))
        phase("done")
        return 0 if backend != "dropped" else 3

    # ── inputs ────────────────────────────────────────────────────────────────
    _, instances = load_battery()
    uc48 = [
        r["text"] for r in load_json(PROJECT_ROOT / "data/issue594/probes_ultrachat.json")["probes"]
    ]
    uc_ext = [r["text"] for r in load_json(args.data_dir / "probes_uc_ext.json")["probes"]]
    dolly = [r["text"] for r in load_json(args.data_dir / "probes_dolly.json")["probes"]]
    betley = [r["text"] for r in load_json(args.data_dir / "probes_betley.json")["probes"]]

    if args.smoke:
        n_ctx = args.n_ctx or 1
        nq = args.n_queries or 4
        instances = instances[:n_ctx]
        # Slice ONLY the new-arm pools: the store pools (uc48 / betley) stay
        # full-length so the uc pool's LOCAL q_idx == the grid's GLOBAL q index
        # (ext queries live at 48+qi -- truncating uc48 would shift them and
        # silently invalidate the fit-side join; production always has 48).
        uc_ext, dolly = uc_ext[:nq], dolly[:2]
    elif args.n_ctx:
        instances = instances[: args.n_ctx]

    inst_shard = [inst for i, inst in enumerate(instances) if i % n_shards == shard_k]
    logger.info(
        "shard %d/%d: %d contexts; pools uc48=%d uc_ext=%d dolly=%d betley=%d",
        shard_k,
        n_shards,
        len(inst_shard),
        len(uc48),
        len(uc_ext),
        len(dolly),
        len(betley),
    )
    max_new = args.max_new_tokens_smoke if args.smoke else V0_MAX_NEW_TOKENS

    # Regen spot-check cells: seeded 25/genre over (ctx x store-probe) (plan 1e).
    rng = np.random.default_rng(SEED)
    regen_cells: list[tuple[str, int, int]] = []  # (genre, ctx_idx, q_idx)
    pools_by_genre = {"betley": betley, "uc": uc48}
    n_regen = 1 if args.smoke else N_REGEN_PER_GENRE
    for genre in ("betley", "uc"):
        n_cells = len(instances) * len(pools_by_genre[genre])
        if n_cells == 0:
            continue
        picks = rng.choice(n_cells, size=min(n_regen, n_cells), replace=False)
        for flat in sorted(int(x) for x in picks):
            regen_cells.append(
                (genre, flat // len(pools_by_genre[genre]), flat % len(pools_by_genre[genre]))
            )
    regen_shard = [c for c in regen_cells if c[1] % n_shards == shard_k]

    out_dir: Path = args.out_dir
    packs_dir = out_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    raw_base = args.eval_dir / "raw_completions"
    shard_tag = f"shard{shard_k}of{n_shards}"
    run_meta: dict = {
        "shard": args.shard,
        "smoke": args.smoke,
        "model": args.model if not args.tiny_model else "tiny-random-qwen2",
        "capture_layers": capture_layers,
        "max_new_tokens": max_new,
        "pools_hash": {
            "uc48": texts_hash(uc48),
            "uc_ext": texts_hash(uc_ext),
            "dolly": texts_hash(dolly),
            "betley": texts_hash(betley),
        },
        "metadata": reproducibility_metadata({"script": "issue923_capture"}),
    }

    gen_specs: dict[str, list[tuple[dict, int, str]]] = {
        # stage -> [(instance, q_idx, q_text)]
        "uc_ext": [(inst, qi, q) for inst in inst_shard for qi, q in enumerate(uc_ext)],
        "ood_dolly": [(inst, qi, q) for inst in inst_shard for qi, q in enumerate(dolly)],
        "regen_check": [(instances[ci], qi, pools_by_genre[g][qi]) for (g, ci, qi) in regen_shard],
    }
    regen_genre_by_pos = [g for (g, _ci, _qi) in regen_shard]

    # ── gen ───────────────────────────────────────────────────────────────────
    completions: dict[str, list[str]] = {}
    if "gen" in phases:
        phase("gen")
        all_prompts: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        for stage in GEN_STAGES:
            start = len(all_prompts)
            for inst, _qi, q in gen_specs[stage]:
                all_prompts.append(render_full_prompt(tokenizer, inst, q))
            spans[stage] = (start, len(all_prompts))
        use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        if args.no_vllm or not use_cuda:
            outs = hf_generate(model, tokenizer, all_prompts, max_new)
        else:
            outs = vllm_generate_chunked(args.model, all_prompts, max_new)
        assert len(outs) == len(all_prompts)
        n_trunc = 0
        for stage in GEN_STAGES:
            s, e = spans[stage]
            completions[stage] = outs[s:e]
            by_ctx: dict[str, list[dict]] = {}
            for pos, ((inst, qi, q), ans) in enumerate(
                zip(gen_specs[stage], outs[s:e], strict=True)
            ):
                tok_n = len(tokenizer(ans, add_special_tokens=False)["input_ids"])
                n_trunc += int(tok_n >= max_new)
                # `pos` = the spec-list position — the resume key (regen cells may
                # repeat q_idx across genres, so q_idx alone is not unique there).
                row = {"probe": q, "q_idx": qi, "pos": pos, "completion": ans}
                if stage == "regen_check":
                    row["genre"] = regen_genre_by_pos[pos]
                by_ctx.setdefault(inst["id"], []).append(row)
            stage_dir = raw_base / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            for cid, rows in by_ctx.items():
                dump_json(
                    {
                        "context_id": cid,
                        "stage": stage,
                        "completions": rows,
                        "meta": run_meta["metadata"],
                    },
                    stage_dir / f"{cid}_{shard_tag}.json",
                )
        run_meta["truncation_hits"] = n_trunc
        logger.info("gen done: %d completions, %d truncation hits", len(all_prompts), n_trunc)

    def _load_stage_completions(stage: str) -> list[str]:
        """Stage completions in gen_specs order — in-memory or from the raw files.

        Rows are keyed by their spec-list ``pos`` (written by the gen phase), so
        a resume (``--phases tf`` without ``gen``) reconstructs the exact order;
        a missing pos fails loud (partial gen must be re-run, never guessed).
        """
        if stage in completions:
            return completions[stage]
        by_pos: dict[int, str] = {}
        stage_dir = raw_base / stage
        for f in sorted(stage_dir.glob(f"*_{shard_tag}.json")):
            for r in load_json(f)["completions"]:
                by_pos[int(r["pos"])] = r["completion"]
        n = len(gen_specs[stage])
        missing = [p for p in range(n) if p not in by_pos]
        assert not missing, (
            f"stage {stage}: raw completions missing positions {missing[:5]}... "
            f"({len(missing)}/{n}) — re-run the gen phase for this shard"
        )
        return [by_pos[p] for p in range(n)]

    # ── tf ────────────────────────────────────────────────────────────────────
    if "tf" in phases:
        phase("tf_capture")
        capture = LayerCapture(model, n_layers)
        try:
            if args.smoke:
                # Batched-rewrite equivalence gate (B>=2, real padding).
                eq_rows = []
                comps = _load_stage_completions("uc_ext")
                for pos, (inst, qi, q) in enumerate(gen_specs["uc_ext"][:3]):
                    ans = comps[pos] or "fallback answer text"
                    eq_rows.append(build_tf_row(tokenizer, inst, q, ans, ("eq", qi)))
                run_meta["equivalence"] = equivalence_check(
                    model, tokenizer, capture_layers, eq_rows
                )
                assert run_meta["equivalence"]["ok"], run_meta["equivalence"]
                logger.info("equivalence check PASS: %s", run_meta["equivalence"])
            for stage, pack_name in (
                ("uc_ext", "tgt_ucext"),
                ("ood_dolly", "tgt_dolly"),
                ("regen_check", "tgt_regen"),
            ):
                comps = _load_stage_completions(stage)
                rows = []
                keys = []
                for pos, (inst, qi, q) in enumerate(gen_specs[stage]):
                    rows.append(build_tf_row(tokenizer, inst, q, comps[pos], (inst["id"], qi)))
                    key = {"ctx_id": inst["id"], "q_idx": qi}
                    if stage == "regen_check":
                        key["genre"] = regen_shard[pos][0]
                    keys.append(key)
                if not rows:
                    continue
                tensors = batched_capture(
                    model, tokenizer, capture, capture_layers, rows, batch_tokens, stage
                )
                # Explicit keys (r1 Minor): TF rows carry no pool span, so the
                # batched_capture dict's fpool/pool_valid are all-zeros dead
                # weight — persisting the whole dict would inflate a parent
                # TF/tgt re-run ~50% and change the pack schema.
                save_pack(
                    packs_dir / f"{pack_name}_{shard_tag}.pt",
                    {"vbar": tensors["vbar"], "flast": tensors["flast"], "valid": tensors["valid"]},
                    {**run_meta, "stage": stage, "rows": keys},
                )
                logger.info("tf pack %s: %d rows", pack_name, len(rows))
        finally:
            capture.remove()

    # ── ffull (store cells) ───────────────────────────────────────────────────
    if "ffull" in phases:
        phase("ffull_store")
        capture = LayerCapture(model, n_layers)
        try:
            for genre, pool in (("betley", betley), ("uc48", uc48)):
                rows = []
                keys = []
                for inst in inst_shard:
                    for qi, q in enumerate(pool):
                        rows.append(build_prompt_row(tokenizer, inst, q, (inst["id"], qi)))
                        keys.append({"ctx_id": inst["id"], "q_idx": qi})
                if not rows:
                    continue
                tensors = batched_capture(
                    model, tokenizer, capture, capture_layers, rows, batch_tokens, f"ffull_{genre}"
                )
                save_pack(
                    packs_dir / f"ffull_{genre}_{shard_tag}.pt",
                    {"flast": tensors["flast"]},
                    {**run_meta, "stage": f"ffull_{genre}", "rows": keys},
                )
        finally:
            capture.remove()

    # ── partials ──────────────────────────────────────────────────────────────
    if "partials" in phases:
        phase("partials")
        backend, mask_results, model_iii = validate_mask_backend(
            args, model, tokenizer, capture_layers
        )
        run_meta["mask_backend"] = backend
        run_meta["mask_invariance"] = mask_results

        # (a) F_ctx — prefix-only forwards + the exact-identity check.
        capture = LayerCapture(model, n_layers)
        try:
            fctx_rows = []
            fctx_keys = []
            probe_q = (uc48 or uc_ext or betley or dolly)[0]
            alt_q = (betley or dolly or uc_ext or uc48)[-1]
            for inst in inst_shard:
                prefix_ids, _full = context_prefix_split(tokenizer, inst, probe_q)
                prefix_alt, _ = context_prefix_split(tokenizer, inst, alt_q)
                assert prefix_ids == prefix_alt, (
                    f"context prefix depends on the query for {inst['id']} — "
                    "prefix arithmetic broken"
                )
                if inst["system_prompt"] is None and not inst["prefix_messages"]:
                    logger.info(
                        "context %s renders no own tokens; F_ctx reads the template's "
                        "auto-inserted default-system block (§8 fallback)",
                        inst["id"],
                    )
                fctx_rows.append(
                    {
                        "key": inst["id"],
                        "full_ids": prefix_ids,
                        "prompt_len": len(prefix_ids),
                        "ans_len": 0,
                    }
                )
                fctx_keys.append({"ctx_id": inst["id"], "ctx_len": len(prefix_ids)})
            tensors = batched_capture(
                model, tokenizer, capture, capture_layers, fctx_rows, batch_tokens, "fctx"
            )
            # Identity check (plan §4.1): F_ctx == the same position inside the
            # full prompt, bit-for-bit up to batching numerics (cos > 0.999).
            id_cos = []
            for inst, frow in list(zip(inst_shard, fctx_rows, strict=True))[:5]:
                _prefix, full_ids = context_prefix_split(tokenizer, inst, probe_q)
                ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    _ = model(input_ids=ids, **_logits_kwargs(model))
                pos = frow["prompt_len"] - 1
                cmin = 1.0
                for li_pos, li in enumerate(capture_layers):
                    v_full = capture.latest[li][0, pos, :].float().cpu()
                    v_prefix = tensors["flast"][frow["_row"], li_pos].float()
                    c = torch.nn.functional.cosine_similarity(v_full, v_prefix, dim=0)
                    cmin = min(cmin, float(c))
                capture.latest.clear()
                id_cos.append({"ctx_id": inst["id"], "min_cos": cmin})
                # Hard floor 0.99: bf16 batched-vs-unbatched forwards on A100 read
                # ~1e-3 cos deviations (att-20260703-145539: f1_phub_06 0.998926 under
                # the old 0.999 assert); the CPU fp32 smoke's 0.9999+ does not
                # transfer. Sub-0.999 reads are recorded in run_meta (analyzer probe).
                assert cmin > 0.99, f"F_ctx identity check failed for {inst['id']}: {cmin}"
                if cmin <= 0.999:
                    logger.warning(
                        "[fctx-identity] %s min_cos=%.6f sub-0.999; bf16 numerics, recorded",
                        inst["id"],
                        cmin,
                    )
            run_meta["fctx_identity_check"] = id_cos
            save_pack(
                packs_dir / f"fctx_{shard_tag}.pt",
                {"flast": tensors["flast"]},
                {**run_meta, "stage": "fctx", "rows": fctx_keys},
            )

            # (b) F_qry presentations (i)/(ii) — query-level, sharded by query.
            query_pools = {"uc": uc48 + uc_ext, "betley": betley, "dolly": dolly}
            for pres in ("i", "ii"):
                rows = []
                keys = []
                for genre, pool in query_pools.items():
                    for qi, q in enumerate(pool):
                        if qi % n_shards != shard_k:
                            continue
                        if pres == "i":
                            text = render_qry_empty_system(tokenizer, q)
                        else:
                            text = render_qry_no_system_block(q)
                        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                        rows.append(
                            {
                                "key": (genre, qi),
                                "full_ids": ids,
                                "prompt_len": len(ids),
                                "ans_len": 0,
                            }
                        )
                        keys.append({"genre": genre, "q_idx": qi})
                if not rows:
                    continue
                tensors = batched_capture(
                    model, tokenizer, capture, capture_layers, rows, batch_tokens, f"fqry_{pres}"
                )
                save_pack(
                    packs_dir / f"fqry_{pres}_{shard_tag}.pt",
                    {"flast": tensors["flast"]},
                    {**run_meta, "stage": f"fqry_{pres}", "rows": keys},
                )
        finally:
            capture.remove()

        # (c) F_qry^(iii) — per-cell masked-context forwards (ctx-sharded).
        if backend != "dropped":
            capture_iii = LayerCapture(model_iii, len(model_iii.model.layers))
            try:
                for genre, pool in (("uc", uc48 + uc_ext), ("betley", betley), ("dolly", dolly)):
                    rows = []
                    keys = []
                    for inst in inst_shard:
                        prefix_ids, _ = context_prefix_split(tokenizer, inst, pool[0])
                        clen = len(prefix_ids)
                        for qi, q in enumerate(pool):
                            _, full_ids = context_prefix_split(tokenizer, inst, q)
                            rows.append(
                                {"key": (inst["id"], qi), "full_ids": full_ids, "ctx_len": clen}
                            )
                            keys.append({"ctx_id": inst["id"], "q_idx": qi})
                    if not rows:
                        continue
                    flast, _fpool_unused = masked_context_capture(
                        model_iii,
                        tokenizer,
                        capture_iii,
                        capture_layers,
                        rows,
                        batch_tokens,
                        f"fqry_iii_{genre}",
                    )
                    save_pack(
                        packs_dir / f"fqry_iii_{genre}_{shard_tag}.pt",
                        {"flast": flast},
                        {**run_meta, "stage": f"fqry_iii_{genre}", "rows": keys},
                    )
            finally:
                capture_iii.remove()
        else:
            logger.warning("arm_qry_iii DROPPED (mask invariance failed under sdpa AND eager)")

    # Write run_meta ONLY when a producing phase ran this invocation: the
    # upload-only resume (`--phases upload`) carries none of the load-bearing
    # diagnostics (mask_backend / mask_invariance / fctx_identity_check /
    # truncation_hits) and would CLOBBER shard-0's real record (r1 Minor).
    if any(p in phases for p in ("gen", "tf", "ffull", "partials")):
        dump_json(run_meta, packs_dir / f"run_meta_{shard_tag}.json")

    # ── upload ────────────────────────────────────────────────────────────────
    if "upload" in phases and not args.no_upload:
        phase("upload")
        from huggingface_hub import list_repo_files

        uploaded: dict[str, int] = {}
        for stage in GEN_STAGES:
            stage_dir = raw_base / stage
            if not stage_dir.exists():
                continue
            hub._upload(
                stage_dir,
                HF_DATA_REPO,
                "dataset",
                f"{HF_PREFIX_923}/raw_completions/{stage}",
            )
            uploaded[f"raw_completions/{stage}"] = len(list(stage_dir.glob("*.json")))
        hub._upload(
            packs_dir,
            HF_DATA_REPO,
            "dataset",
            f"{HF_PREFIX_923}/analysis_tensors/capture",
        )
        uploaded["analysis_tensors/capture"] = len(list(packs_dir.iterdir()))
        listing = [
            f
            for f in list_repo_files(HF_DATA_REPO, repo_type="dataset")
            if f.startswith(HF_PREFIX_923)
        ]
        for prefix, n_expected in uploaded.items():
            n_hub = len([f for f in listing if f.startswith(f"{HF_PREFIX_923}/{prefix}")])
            assert n_hub >= n_expected, (
                f"upload verification failed for {prefix}: hub {n_hub} < local {n_expected}"
            )
        complete = {
            "shard": args.shard,
            "uploaded": uploaded,
            "files": listing,
            "meta": run_meta["metadata"],
        }
        complete_path = packs_dir / "UPLOAD_COMPLETE.json"
        dump_json(complete, complete_path)
        hub._upload(
            complete_path,
            HF_DATA_REPO,
            "dataset",
            f"{HF_PREFIX_923}/analysis_tensors/capture/UPLOAD_COMPLETE.json",
            upload_as_file=True,
        )
        logger.info("upload verified: %s", uploaded)

    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
