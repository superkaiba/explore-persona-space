#!/usr/bin/env python3
# (math/scientific notation — Δv, ŵ, g_real — intentional in docstrings + labels)
"""Issue #683 Phase A — sycophancy realized-gate Δv / v_trained extraction (L20).

Plan §4 Phase A (sycophancy side). For the #612 on-policy VILLAIN adapter
(seeds 42/137) and a held-out PANEL of target personas, read the realized
gate at the ANSWER-SPAN MEAN, layer 20 (the #649/#612 sycophancy read band):

    v_base(C')     = base-model answer-span-mean residual under context C' (L20)
    v_trained(C')  = villain-adapter answer-span-mean residual under C' (L20)
    Δv(C')         = v_trained(C') - v_base(C')
    ŵ              = Δv(C_source = villain)   (the source's OWN write; g_real(C)=1)
    g_real(C')     = ⟨ŵ, Δv(C')⟩ / ⟨ŵ, ŵ⟩

READ-LOCATION PIN (methodology-critic concern #1): unlike the marker read
(which uses the trained END-OF-RESPONSE <|im_end|> slot, ``shift_extract``),
the SYCOPHANCY read is the ASSISTANT-ANSWER SPAN MEAN — the mean residual over
the model's OWN on-policy answer tokens. This choice is recorded in the output
JSON's ``read_location`` field AND in this header so the analyzer can confirm
from the run JSON which read produced each tensor. The answer-span mean
(NOT a single slot) is the natural read for a content behavior whose
expression is spread across the response, matching #649/#612's sycophancy
panel-centroid recipe.

Read mechanics mirror issue683_extract_tcb (right-padded batched forwards,
on-GPU running mean, no #658-class activation retention), but here the
completion is the model's OWN on-policy greedy generation per (persona, claim)
rather than a fixed training row, and the read runs on BOTH base and trained.

rsLoRA gauge (fitness check g): asserted after the villain adapter loads.

Content hygiene: the sycophancy panel generations are harmful-adjacent. This
script NEVER prints prompt/completion text — only counts, shapes, norms.

Outputs (one .pt per seed under ``analysis_tensors/dv/sycophancy/``):
    villain_seed{seed}_L20.pt : per-context {v_base, v_trained, Delta_v} (H,),
                                w_hat (= villain's own Δv), g_real per context.

CLI:
    uv run python scripts/issue683_extract_dv_sycophancy.py --seeds 42,137
    # CPU smoke (tiny model, 2 panel personas, 2 claims, base==trained):
    uv run python scripts/issue683_extract_dv_sycophancy.py --seeds 42 \
        --panel villain,comedian --n-claims 2 --model Qwen/Qwen2.5-0.5B-Instruct \
        --no-adapter --layer 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_extract_dv_sycophancy")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_LAYER,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    SYCO_ADAPTER_TEMPLATE,
    SYCO_SEEDS,
    SYCO_SOURCE,
    answer_span_token_indices,
    assert_rslora_gauge,
    chunked,
    generate_on_policy_vllm,
    realized_gate,
    repro_metadata,
)

# #612 false-claim prompt bank (tier-1, sha-pinned audited pool).
SYCO_CLAIM_POOL = "issue612_sycophancy_onpolicy/inputs/eval_60.jsonl"
# #612 panel set (the held-out persona panel).
SYCO_PANEL_SET = "issue612_sycophancy_onpolicy/panel/panel_set.json"
GREAL_SELF_TOL = 0.02


def _load_claims(n_claims: int) -> list[str]:
    """Load the first ``n_claims`` user prompts from the #612 audited pool."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, SYCO_CLAIM_POOL, repo_type="dataset", revision="main")
    claims: list[str] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # The #612 audited pool's user-facing false claim is ``wrong_claim``
            # (schema verified 2026-06-26: keys = correction/family/provenance/
            # topic/topic_haiku/wrong_claim). Tolerate the other common schemas
            # for forward-compat, but fail loud if none is present.
            claim = (
                row.get("wrong_claim")
                or row.get("claim")
                or row.get("prompt")
                or row.get("question")
                or row.get("text")
            )
            if claim is None:
                raise ValueError(
                    f"claim row missing a wrong_claim/claim/prompt field: keys={sorted(row)}"
                )
            claims.append(claim if isinstance(claim, str) else json.dumps(claim))
            if n_claims and len(claims) >= n_claims:
                break
    if not claims:
        raise ValueError("no claims loaded from the #612 pool")
    return claims


def _load_panel() -> dict[str, str]:
    """persona_name -> system_prompt from the #612 30-persona panel set."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, SYCO_PANEL_SET, repo_type="dataset", revision="main")
    ps = json.loads(Path(p).read_text())
    personas = ps["personas"]
    return {name: personas[name]["prompt"] for name in personas}


def _resolve_adapter(seed: int) -> str:
    """snapshot_download the villain on-policy adapter for ``seed``."""
    import os

    from huggingface_hub import snapshot_download

    sub = SYCO_ADAPTER_TEMPLATE.format(seed=seed)
    dl = snapshot_download(
        repo_id=HF_MODEL_REPO, allow_patterns=[f"{sub}/*"], token=os.environ.get("HF_TOKEN")
    )
    return str(Path(dl) / sub)


def _chat_rows(system_prompt: str, claims: list[str]) -> list[list[dict[str, str]]]:
    """Build the [{system}, {user}] message rows for one persona x claims."""
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": c},
        ]
        for c in claims
    ]


def _gen_all_on_policy_vllm(
    *,
    model_name: str,
    panel: dict[str, str],
    claims: list[str],
    adapter_path: str | None,
    max_new_tokens: int,
    seed: int,
) -> dict[str, list[str]]:
    """vLLM batched greedy on-policy completions for EVERY (persona, claim).

    Returns ``{persona: [completion per claim, in claim order]}``. Uses the
    shared ``generate_on_policy_vllm`` helper (CLAUDE.md: vLLM for generation,
    NEVER HF model.generate; the engine is reaped before this returns so the
    caller can load HF for the activation read). When ``adapter_path`` is None
    this is the base model (no LoRA); otherwise the villain adapter is applied
    via a vLLM LoRARequest at the trained gauge.
    """
    prompts_by_context = {persona: _chat_rows(sp, claims) for persona, sp in panel.items()}
    flat = generate_on_policy_vllm(
        base_model=model_name,
        prompts_by_context=prompts_by_context,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    out: dict[str, list[str]] = {}
    for persona, sp in panel.items():
        comps: list[str] = []
        for msgs in _chat_rows(sp, claims):
            key = json.dumps(msgs, sort_keys=True)
            comps.append(flat[key][0])
        out[persona] = comps
    return out


def _gen_all_on_policy_hf_smoke(
    *, model_name: str, panel: dict[str, str], claims: list[str], max_new_tokens: int, device
) -> dict[str, list[str]]:
    """CPU --no-adapter smoke ONLY: tiny HF generation to exercise the read path.

    vLLM requires a GPU + a real LoRA adapter, so the production vLLM gen path
    cannot run on the CPU smoke's tiny throwaway model. This HF fallback is gated
    behind ``--no-adapter`` (smoke) and exists solely to flow real completions
    into the answer-span read; the production GPU path uses ``generate_on_policy_vllm``
    exclusively. The GPU-bound vLLM gen is covered by the dispatcher dry-run +
    signature smoke (GPU-bound-phase carve-out), not by this CPU helper.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": device}, trust_remote_code=True
    ).eval()
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    orig_side = tokenizer.padding_side
    out: dict[str, list[str]] = {}
    try:
        tokenizer.padding_side = "left"
        for persona, sp in panel.items():
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in _chat_rows(sp, claims)
            ]
            enc = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(
                device
            )
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out[persona] = [tokenizer.decode(r, skip_special_tokens=True).strip() for r in new]
    finally:
        tokenizer.padding_side = orig_side
    del model
    return out


def _answer_span_mean(
    *,
    model,
    tokenizer,
    system_prompt: str,
    claims: list[str],
    completions: list[str],
    layer: int,
    device,
    batch_size: int,
) -> object:
    """Mean answer-span residual at ``layer`` over (claim, completion) rows.

    Right-padded batched teacher-forced forwards; on-GPU running mean; the
    answer span is every token after the prompt prefix (the completion). Returns
    the (H,) mean over rows (CPU fp32).
    """
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    orig_side = tokenizer.padding_side

    n_model_layers = int(model.config.num_hidden_layers)
    block_l = min(layer, n_model_layers - 1)
    model_hidden = int(model.config.hidden_size)
    sum_acc = torch.zeros(model_hidden, dtype=torch.float64, device=device)
    n_used = 0

    prepared: list[tuple[list[int], list[int]]] = []
    for claim, completion in zip(claims, completions, strict=True):
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claim},
        ]
        full_msgs = [*prompt_msgs, {"role": "assistant", "content": completion}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        ans_idx = answer_span_token_indices(tokenizer, prompt_msgs, full_ids)
        prepared.append((full_ids, ans_idx))

    try:
        tokenizer.padding_side = "right"
        for batch in chunked(prepared, batch_size):
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            ans_cols: list[list[int]] = []
            for bi, (ids, ans_idx) in enumerate(batch):
                input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[bi, : len(ids)] = 1
                ans_cols.append(list(ans_idx))
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            captured = extract_layer_activations(model, input_ids, [block_l], attention_mask=attn)
            hs = captured[block_l]
            assert hs.shape == (len(batch), maxlen, model_hidden), hs.shape
            for bi, cols in enumerate(ans_cols):
                col_t = torch.tensor(cols, dtype=torch.long, device=device)
                sum_acc += hs[bi].index_select(0, col_t).double().mean(dim=0)
                n_used += 1
            del hs, captured
    finally:
        tokenizer.padding_side = orig_side

    if n_used == 0:
        raise RuntimeError("no rows pooled for the answer-span mean")
    return (sum_acc / n_used).float().cpu()


def extract_syco_dv(
    *,
    seed: int,
    panel: dict[str, str],
    claims: list[str],
    layer: int,
    use_adapter: bool,
    model_name: str,
    max_new_tokens: int,
    batch_size: int,
    device,
) -> dict:
    """Read v_base/v_trained/Δv/g_real per panel context for the villain seed."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if SYCO_SOURCE not in panel:
        raise AssertionError(
            f"source {SYCO_SOURCE!r} must be a panel member (ŵ = Δv(C_source)); panel="
            f"{sorted(panel)[:8]}..."
        )

    adapter_dir = _resolve_adapter(seed) if use_adapter else None

    # ── Phase 1: on-policy generation (vLLM; engine reaped before HF load). ─────
    # CLAUDE.md: vLLM for generation, NEVER HF model.generate (hf-generate-vs-vllm
    # BLOCKER). Generate base + trained completions for EVERY (persona, claim) in
    # batched vLLM calls FIRST, reap the engine, THEN load HF for the activation
    # read — so vLLM + HF never co-reside (vLLM-teardown-OOM gotcha).
    if use_adapter:
        base_comps_by_persona = _gen_all_on_policy_vllm(
            model_name=model_name,
            panel=panel,
            claims=claims,
            adapter_path=None,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        trained_comps_by_persona = _gen_all_on_policy_vllm(
            model_name=model_name,
            panel=panel,
            claims=claims,
            adapter_path=adapter_dir,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    else:
        # CPU --no-adapter smoke: vLLM needs a GPU + a real adapter, so the smoke
        # exercises the read path with a tiny HF generation; trained == base so
        # Δv ≡ 0 (the GPU-bound vLLM gen path is covered by the dispatcher dry-run
        # + signature smoke per the GPU-bound carve-out, NOT this CPU smoke).
        base_comps_by_persona = _gen_all_on_policy_hf_smoke(
            model_name=model_name,
            panel=panel,
            claims=claims,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        trained_comps_by_persona = base_comps_by_persona  # Δv ≡ 0

    # ── Phase 2: HF teacher-forced activation read on the fixed completions. ────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    ).eval()

    if use_adapter:
        from peft import PeftModel

        trained_raw = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
        )
        trained = PeftModel.from_pretrained(trained_raw, adapter_dir).eval()
        gauge = assert_rslora_gauge(trained)
        logger.info("[phase=dv_syco_gauge] seed=%s gauge=%s adapter=%s", seed, gauge, adapter_dir)
    else:
        trained = base
        gauge = {"no_adapter_smoke": True}

    per_context: dict[str, dict] = {}
    for persona, sp in panel.items():
        v_base = _answer_span_mean(
            model=base,
            tokenizer=tokenizer,
            system_prompt=sp,
            claims=claims,
            completions=base_comps_by_persona[persona],
            layer=layer,
            device=device,
            batch_size=batch_size,
        )
        if use_adapter:
            v_trained = _answer_span_mean(
                model=trained,
                tokenizer=tokenizer,
                system_prompt=sp,
                claims=claims,
                completions=trained_comps_by_persona[persona],
                layer=layer,
                device=device,
                batch_size=batch_size,
            )
        else:
            # no-adapter smoke: trained == base, so v_trained == v_base, Δv ≡ 0.
            v_trained = v_base.clone()
        per_context[persona] = {
            "v_base": v_base,
            "v_trained": v_trained,
            "Delta_v": v_trained - v_base,
        }

    w_hat = per_context[SYCO_SOURCE]["Delta_v"]
    g_real: dict[str, float] = {}
    for persona, rec in per_context.items():
        try:
            g_real[persona] = realized_gate(w_hat, rec["Delta_v"])
        except ValueError:
            g_real[persona] = float("nan")

    g_self = g_real.get(SYCO_SOURCE, float("nan"))
    self_consistent = abs(g_self - 1.0) <= GREAL_SELF_TOL if g_self == g_self else None
    if use_adapter and self_consistent is False:
        raise AssertionError(
            f"self-consistency probe FAILED: g_real(C_source=villain)={g_self:.4f} != 1.0±"
            f"{GREAL_SELF_TOL} — a gauge/load bug. Refusing to emit the panel read."
        )

    del base
    if use_adapter:
        del trained
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "source": SYCO_SOURCE,
        "seed": seed,
        "behavior": "sycophancy",
        "layer": layer,
        "read_location": "answer_span_mean",
        "panel": list(panel),
        "w_hat": w_hat,
        "w_hat_norm": float(w_hat.norm()),
        "g_real_source_self": g_self,
        "self_consistent": self_consistent,
        "gauge": gauge,
        "adapter_dir": adapter_dir,
        "per_context": per_context,
        "g_real": g_real,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SYCO_SEEDS),
        help="comma-separated villain seeds",
    )
    ap.add_argument("--panel", default=None, help="comma-separated panel personas (default: #612)")
    ap.add_argument("--layer", type=int, default=None, help="block index; default L20")
    ap.add_argument("--n-claims", type=int, default=30, help="#612 claim-pool slice")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="on-policy generation cap")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--no-adapter", action="store_true", help="CPU smoke: trained==base (Δv≡0)")
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="smoke namespace")
    args = ap.parse_args(argv)

    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch

    layer = args.layer if args.layer is not None else DEFAULT_LAYER["sycophancy"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    full_panel = _load_panel()
    if args.panel:
        wanted = [p.strip() for p in args.panel.split(",") if p.strip()]
        missing = [p for p in wanted if p not in full_panel]
        if missing:
            raise AssertionError(f"--panel personas not in the #612 panel set: {missing}")
        panel = {p: full_panel[p] for p in wanted}
    else:
        panel = full_panel
    if SYCO_SOURCE not in panel:
        panel = {SYCO_SOURCE: full_panel[SYCO_SOURCE], **panel}
    claims = _load_claims(args.n_claims)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(
        args.out_dir
        or (
            PROJECT_ROOT
            / "eval_results/issue_683/analysis_tensors"
            / ("dv_smoke" if args.smoke else "dv")
            / "sycophancy"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[phase=dv_syco_start] seeds=%s panel=%d claims=%d layer=%d device=%s adapter=%s",
        seeds,
        len(panel),
        len(claims),
        layer,
        device,
        not args.no_adapter,
    )

    summary: dict[str, dict] = {}
    for seed in seeds:
        payload = extract_syco_dv(
            seed=seed,
            panel=panel,
            claims=claims,
            layer=layer,
            use_adapter=not args.no_adapter,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=device,
        )
        out_path = out_dir / f"villain_seed{seed}_L{layer}.pt"
        torch.save(payload, out_path)
        summary[str(seed)] = {
            "out_path": str(out_path),
            "n_context": len(payload["per_context"]),
            "g_real_source_self": payload["g_real_source_self"],
            "self_consistent": payload["self_consistent"],
            "w_hat_norm": payload["w_hat_norm"],
        }
        logger.info(
            "[phase=dv_syco_done] seed=%s n_context=%d g_self=%.4f self_consistent=%s -> %s",
            seed,
            len(payload["per_context"]),
            payload["g_real_source_self"],
            payload["self_consistent"],
            out_path,
        )

    (out_dir / f"dv_sycophancy_summary_L{layer}.json").write_text(
        json.dumps(
            {
                "behavior": "sycophancy",
                "layer": layer,
                "read_location": "answer_span_mean",
                "seeds": summary,
                "reproducibility": repro_metadata(
                    {"behavior": "sycophancy", "layer": layer, "read_location": "answer_span_mean"}
                ),
            },
            indent=2,
        )
    )
    logger.info("[phase=dv_syco_summary] %d seed(s) -> %s", len(summary), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
