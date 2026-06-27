#!/usr/bin/env python3
# (math/scientific notation — Δv, ŵ, g_real, ※ — intentional in docstrings + labels)
"""Issue #683 Phase A — marker realized-gate Δv / v_trained extraction (L14).

Plan §4 Phase A (marker side). For each loc-arm SOURCE adapter (#474 loc
A1..A5) and the #604 42-context panel, read the realized gate at the trained
END-OF-RESPONSE slot, layer 14 (the #604/#551 marker read band):

    v_base(C')     = base-model EOR-slot residual under context C'  (L14)
    v_trained(C')  = source-adapter EOR-slot residual under context C' (L14)
    Δv(C')         = v_trained(C') - v_base(C')
    ŵ              = Δv(C_source)            (the source's OWN write; g_real(C)=1)
    g_real(C')     = ⟨ŵ, Δv(C')⟩ / ⟨ŵ, ŵ⟩

The #604 condition→system-prompt mapping (A1=assistant, A2=software_engineer,
…) is read from the #604 ``manifest.json`` so the panel context set + each
context's system prompt match the base ``c_C`` bank exactly. Each loc-arm
adapter ``i474_loc_<arm>`` was trained ON its own condition ``<arm>``, so
ŵ = Δv(<arm>) is the source's own write and g_real(<arm>)=1.0±tol is the
built-in self-consistency probe (a deviation flags a gauge/load bug BEFORE
the analyzer reads g_real).

Delegates the per-context EOR-slot read to the #650
``shift_extract.extract_per_context_shift`` (extended with a ``layer`` arg by
this issue) — the same marker-slot harness #650's eval uses. The R responses
(the context's own on-policy base greedy completion) are generated fresh here
over EVAL_QUESTIONS (greedy, frozen) so the read stays on-policy.

rsLoRA gauge (fitness check g): the source adapter's ``use_rslora`` +
no-unembedding-target asserts run after load (issue_683.assert_rslora_gauge).

Outputs (one .pt per source under ``analysis_tensors/dv/marker/``):
    {source}_L14.pt : per-context {v_base, v_trained, Delta_v} (H,) tensors,
                      w_hat (= Δv of the source's own condition), g_real per
                      context, plus the marker-slot four-float deltas.

CLI:
    uv run python scripts/issue683_extract_dv_marker.py --source-list A1,A2,A3,A4,A5
    # CPU smoke (tiny model, source A1 + 2 panel contexts, base==trained):
    uv run python scripts/issue683_extract_dv_marker.py --source-list A1 \
        --panel A1,A2 --n-questions 2 --model Qwen/Qwen2.5-0.5B-Instruct \
        --no-adapter --layer 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_extract_dv_marker")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_LAYER,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MARKER_ADAPTER_DEFAULT_EPOCH,
    MARKER_ADAPTER_TEMPLATE,
    MARKER_SOURCE_ARMS,
    assert_rslora_gauge,
    generate_on_policy_vllm,
    realized_gate,
    repro_metadata,
)

# #604 manifest holds the condition→system-prompt mapping for the 42-context
# panel (post_response_slot variant — the marker EOR read).
MARKER_MANIFEST = "issue604_adapter_svd/analysis_tensors/post_response_slot/manifest.json"
# g_real(source)=1.0 self-consistency tolerance (plan §4 — 1±0.02).
GREAL_SELF_TOL = 0.02


def _load_condition_prompts() -> dict[str, str]:
    """condition_name -> system_prompt from the #604 post-response-slot manifest.

    The 42-context panel (A1..A5 = teach conditions, B*/C1/D* = held-out
    targets, named personas). Skips contexts with a null system_prompt
    (qwen_default) — the read needs a system prompt to inject.
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, MARKER_MANIFEST, repo_type="dataset", revision="main")
    manifest = json.loads(Path(p).read_text())
    out: dict[str, str] = {}
    for name, ctx in manifest["contexts"].items():
        sp = ctx.get("system_prompt")
        if sp:  # skip qwen_default (None) — no system prompt to inject
            out[name] = sp
    return out


def _resolve_adapter(arm: str, epoch: int) -> str:
    """snapshot_download the loc-arm source adapter; return the local dir."""
    import os

    from huggingface_hub import snapshot_download

    sub = MARKER_ADAPTER_TEMPLATE.format(arm=arm, epoch=epoch)
    dl = snapshot_download(
        repo_id=HF_MODEL_REPO, allow_patterns=[f"{sub}/*"], token=os.environ.get("HF_TOKEN")
    )
    return str(Path(dl) / sub)


def _chat_rows(persona_prompt: str, questions: list[str]) -> list[list[dict[str, str]]]:
    """[{system}, {user}] message rows for one context x questions."""
    return [
        [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]


def _gen_all_r_responses_vllm(
    *,
    model_name: str,
    context_prompts: dict[str, str],
    questions: list[str],
    max_new_tokens: int,
    seed: int,
) -> dict[str, dict[str, str]]:
    """vLLM batched greedy base R responses for EVERY (context, question).

    Returns ``{context: {question: R}}``. The R responses are the BASE model's
    own greedy completions (#460/#604 marker-eval R contract) — NO adapter, so
    no LoRA request. Uses the shared ``generate_on_policy_vllm`` helper (CLAUDE.md:
    vLLM for generation, NEVER HF model.generate — hf-generate-vs-vllm BLOCKER);
    the engine is reaped before this returns so the caller can load HF base +
    trained for the activation read without the vLLM-teardown OOM.
    """
    prompts_by_context = {ctx: _chat_rows(sp, questions) for ctx, sp in context_prompts.items()}
    flat = generate_on_policy_vllm(
        base_model=model_name,
        prompts_by_context=prompts_by_context,
        adapter_path=None,  # R is the BASE model's own response (frozen, no LoRA)
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    out: dict[str, dict[str, str]] = {}
    for ctx, sp in context_prompts.items():
        per_q: dict[str, str] = {}
        for q, msgs in zip(questions, _chat_rows(sp, questions), strict=True):
            per_q[q] = flat[json.dumps(msgs, sort_keys=True)][0]
        out[ctx] = per_q
    return out


def _gen_all_r_responses_hf_smoke(
    *,
    model_name: str,
    context_prompts: dict[str, str],
    questions: list[str],
    max_new_tokens: int,
    device,
) -> dict[str, dict[str, str]]:
    """CPU --no-adapter smoke ONLY: tiny HF generation to exercise the read path.

    vLLM requires a GPU; the production R generation uses ``generate_on_policy_vllm``
    exclusively. This HF fallback is gated behind ``--no-adapter`` (smoke) and only
    flows real R text into ``extract_per_context_shift``. The GPU-bound vLLM gen is
    covered by the dispatcher dry-run + signature smoke (GPU-bound-phase carve-out).
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
    out: dict[str, dict[str, str]] = {}
    try:
        tokenizer.padding_side = "left"
        for ctx, sp in context_prompts.items():
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in _chat_rows(sp, questions)
            ]
            enc = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(
                device
            )
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out[ctx] = {
                q: tokenizer.decode(r, skip_special_tokens=True).strip()
                for q, r in zip(questions, new, strict=True)
            }
    finally:
        tokenizer.padding_side = orig_side
    del model
    return out


def extract_marker_dv(
    *,
    source: str,
    panel: list[str],
    condition_prompts: dict[str, str],
    questions: list[str],
    layer: int,
    epoch: int,
    use_adapter: bool,
    model_name: str,
    max_new_tokens: int,
    device,
) -> dict:
    """Read v_base/v_trained/Δv/g_real per panel context for one source arm."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue_650.shift_extract import extract_per_context_shift

    if source not in condition_prompts:
        raise AssertionError(
            f"source condition {source!r} not in the #604 manifest contexts "
            f"({sorted(condition_prompts)[:8]}...) — cannot read ŵ = Δv(C_source)."
        )
    # The source's OWN condition MUST be in the read set (ŵ + self-consistency).
    read_contexts = list(dict.fromkeys([source, *panel]))
    missing = [c for c in read_contexts if c not in condition_prompts]
    if missing:
        raise AssertionError(f"panel contexts not in #604 manifest: {missing}")

    adapter_dir = _resolve_adapter(source, epoch) if use_adapter else None
    read_context_prompts = {c: condition_prompts[c] for c in read_contexts}

    # ── Phase 1: on-policy R generation (vLLM; engine reaped before HF load). ───
    # CLAUDE.md: vLLM for generation, NEVER HF model.generate (hf-generate-vs-vllm
    # BLOCKER). Generate the base R for EVERY (context, question) FIRST, reap the
    # engine, THEN load HF base + trained for the activation read — so vLLM + HF
    # never co-reside (vLLM-teardown-OOM gotcha). R is the BASE model's own greedy
    # response (frozen, no LoRA) per the #460/#604 marker-eval contract.
    if use_adapter:
        r_by_context = _gen_all_r_responses_vllm(
            model_name=model_name,
            context_prompts=read_context_prompts,
            questions=questions,
            max_new_tokens=max_new_tokens,
            seed=42,
        )
    else:
        r_by_context = _gen_all_r_responses_hf_smoke(
            model_name=model_name,
            context_prompts=read_context_prompts,
            questions=questions,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    # ── Phase 2: HF teacher-forced activation read on the fixed R. ──────────────
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
        logger.info(
            "[phase=dv_marker_gauge] source=%s gauge=%s adapter=%s", source, gauge, adapter_dir
        )
    else:
        # CPU smoke / no-adapter: trained == base (Δv ≡ 0). Exercises the read
        # path end-to-end without a PEFT load on a tiny model.
        trained = base
        gauge = {"no_adapter_smoke": True}

    per_context: dict[str, dict] = {}
    for ctx in read_contexts:
        sp = condition_prompts[ctx]
        r_responses = r_by_context[ctx]
        cs = extract_per_context_shift(
            base_model=base,
            trained_model=trained,
            tokenizer=tokenizer,
            persona=ctx,
            persona_prompt=sp,
            eval_questions=questions,
            r_responses=r_responses,
            device=device,
            layer=layer,
        )
        per_context[ctx] = {
            "v_base": cs.v_base_slot,  # (H,)
            "v_trained": cs.v_trained_slot,  # (H,)
            "Delta_v": cs.shift_vector,  # (H,) = v_trained - v_base
            "extraction_layer": cs.extraction_layer,
            "delta_logp_marker": cs.delta_logp_marker,
            "delta_logit_marker": cs.delta_logit_marker,
            "emission_argmax_trained": cs.emission_argmax_trained,
            "emission_argmax_base": cs.emission_argmax_base,
        }

    # ŵ = the source's OWN write (Δv of the source's condition).
    w_hat = per_context[source]["Delta_v"]
    g_real: dict[str, float] = {}
    for ctx, rec in per_context.items():
        try:
            g_real[ctx] = realized_gate(w_hat, rec["Delta_v"])
        except ValueError:
            # Δ v(source) is the zero vector under --no-adapter (base==trained):
            # ŵ has zero norm so g_real is undefined. Record NaN; the CPU smoke
            # exercises the read path, the production run has a real adapter.
            g_real[ctx] = float("nan")

    g_self = g_real.get(source, float("nan"))
    self_consistent = abs(g_self - 1.0) <= GREAL_SELF_TOL if g_self == g_self else None
    if use_adapter and self_consistent is False:
        raise AssertionError(
            f"self-consistency probe FAILED: g_real(C_source={source})={g_self:.4f} != 1.0±"
            f"{GREAL_SELF_TOL} — a gauge/load bug (the source must score 1 against its OWN "
            "write). Refusing to emit the panel read."
        )

    del base
    if use_adapter:
        del trained
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "source": source,
        "behavior": "marker",
        "layer": layer,
        "read_location": "post_response_eor_slot",
        "epoch": epoch if use_adapter else None,
        "panel": read_contexts,
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
        "--source-list",
        default=",".join(MARKER_SOURCE_ARMS),
        help="comma-separated loc-arm sources (default A1..A5)",
    )
    ap.add_argument(
        "--panel",
        default=None,
        help="comma-separated panel contexts (default: ALL #604 manifest contexts)",
    )
    ap.add_argument("--layer", type=int, default=None, help="block index; default L14")
    ap.add_argument("--epoch", type=int, default=MARKER_ADAPTER_DEFAULT_EPOCH)
    ap.add_argument("--n-questions", type=int, default=20, help="EVAL_QUESTIONS slice")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="R generation cap")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument(
        "--no-adapter",
        action="store_true",
        help="CPU smoke: trained==base (Δv≡0), exercises the read path with no PEFT load",
    )
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="smoke namespace (separate out subdir)")
    args = ap.parse_args(argv)

    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch

    from explore_persona_space.personas import EVAL_QUESTIONS

    layer = args.layer if args.layer is not None else DEFAULT_LAYER["marker"]
    sources = [s.strip() for s in args.source_list.split(",") if s.strip()]
    condition_prompts = _load_condition_prompts()
    panel = (
        [p.strip() for p in args.panel.split(",") if p.strip()]
        if args.panel
        else list(condition_prompts)
    )
    questions = list(EVAL_QUESTIONS[: args.n_questions])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(
        args.out_dir
        or (
            PROJECT_ROOT
            / "eval_results/issue_683/analysis_tensors"
            / ("dv_smoke" if args.smoke else "dv")
            / "marker"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[phase=dv_marker_start] sources=%s panel=%d questions=%d layer=%d device=%s adapter=%s",
        sources,
        len(panel),
        len(questions),
        layer,
        device,
        not args.no_adapter,
    )

    summary: dict[str, dict] = {}
    for source in sources:
        payload = extract_marker_dv(
            source=source,
            panel=panel,
            condition_prompts=condition_prompts,
            questions=questions,
            layer=layer,
            epoch=args.epoch,
            use_adapter=not args.no_adapter,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        out_path = out_dir / f"{source}_L{layer}.pt"
        torch.save(payload, out_path)
        summary[source] = {
            "out_path": str(out_path),
            "n_context": len(payload["per_context"]),
            "g_real_source_self": payload["g_real_source_self"],
            "self_consistent": payload["self_consistent"],
            "w_hat_norm": payload["w_hat_norm"],
        }
        logger.info(
            "[phase=dv_marker_done] source=%s n_context=%d g_self=%.4f self_consistent=%s -> %s",
            source,
            len(payload["per_context"]),
            payload["g_real_source_self"],
            payload["self_consistent"],
            out_path,
        )

    (out_dir / f"dv_marker_summary_L{layer}.json").write_text(
        json.dumps(
            {
                "behavior": "marker",
                "layer": layer,
                "read_location": "post_response_eor_slot",
                "sources": summary,
                "reproducibility": repro_metadata(
                    {
                        "behavior": "marker",
                        "layer": layer,
                        "read_location": "post_response_eor_slot",
                    }
                ),
            },
            indent=2,
        )
    )
    logger.info("[phase=dv_marker_summary] %d source(s) -> %s", len(summary), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
