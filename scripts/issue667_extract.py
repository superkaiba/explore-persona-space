#!/usr/bin/env python3
"""Issue #667 absolute-v extractor — per-source-adapter forward-pass sweep.

For ONE (behavior, source-context C) cell this CLI:

1. Stages + loads the #537 adapter as a ``PeftModel`` on the base Qwen-2.5-7B
   (rsLoRA honored — every #537 adapter has ``use_rslora=True``); asserts the
   adapter's ``base_model_name_or_path == Qwen/Qwen2.5-7B-Instruct`` (fitness
   check (f)).
2. For each eval target context C' (the 30 #537 eval cids + the source C
   itself): builds the eval probes via the BYTE-FAITHFUL ``i537_contexts``
   registry (registry_hash f12061d6... == the G_meta pin), generates the frozen
   base greedy response R per probe (deterministic, temp=0), teacher-forces
   ``T_{C'}(q) + R`` through BOTH base θ0 and trained θ+ once each, and reads
   the MEAN-over-response-span residual at L14 (+7,21): ``v0(C')``, ``v+(C')``
   (both float32 CPU), mean over probes. (Marker companion also reads the
   post-response slot.)
3. Extracts the base-side context vectors ``c_C`` / ``c_{C'}`` (last-input-token,
   all 28 layers) over the SAME contexts — the whitened-gate key/query (A3.9).
4. Extracts ``t+`` / ``t-`` for A3.7: teacher-forces the #537 frozen training-mix
   POSITIVE rows (prompt context == source C) and NEGATIVE rows (prompt context
   in the negative panel) through θ0, mean answer-side activation. Positive vs
   negative is split by matching the rendered source-context prompt prefix (the
   builder writes positives under the source ctx, negatives under the neg panel)
   — robust to the untagged JSONL. ALSO extracts ``v0_C_neg`` — the base-CONTEXT
   activation under each negative persona's PROMPT (no answer span), matched to
   the ``v0(C)`` mean-over-response recipe, panel-averaged over the negative cids.
   This is the A3.7 ``frac_ctx`` numerator term (R3-1) and is DISTINCT from ``t-``
   (the negative-persona answer activation that feeds ``delta_contra``).
5. For ``fact``: re-extracts ``r_B`` fresh (absent from #658's r_b.pt) via the
   #594 diff-in-means recipe (system-prompt pos/neg pair, mean answer act).

Writes one ``.npz`` per (behavior, source-C, target-C', layer) under
``eval_results/issue_667/analysis_tensors/`` with ``{v0, v_plus}`` per side, the
per-cell ``c_C``/``c_Cp`` (all layers), ``t_pos``/``t_neg``, the negative-panel
base-context vector ``v0_C_neg`` (A3.7 frac_ctx, R3-1 — distinct from ``t_neg``),
and (fact) ``r_b``.

CONTENT HYGIENE: ``em`` training rows are Betley harmful-content — this script
NEVER prints/logs their text; it digests by row count + token count + the
ACTIVATIONS only. Benign behaviors (marker/fact/sycophancy) are unaffected.

Usage (one source-adapter cell)::

    uv run python scripts/issue667_extract.py \\
        --behavior em --source-cid default \\
        --targets sp_swe,default,fmt_json --layers 7 14 21 --primary-layer 14 \\
        --out eval_results/issue_667/analysis_tensors --gpu-id 0
"""

# ruff: noqa: RUF001, RUF002  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM EngineCore fork() poisoning guard (.claude/rules/gotchas.md § entry 26):
# main() touches transformers.AutoTokenizer (L705-707) BEFORE vllm_generate_R
# constructs vllm.LLM() (L228); ANY pre-LLM() transformers/tokenizer/registry
# touch poisons the EngineCore fork. spawn (not fork) avoids the silent worker
# death. Must be set at module top, BEFORE any `import vllm`. Do not strip.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.analysis.issue667 import (  # noqa: E402
    ALL_LAYERS,
    BASE_MODEL,
    HF_MODEL_REPO,
    HIDDEN_SIZE,
    N_LAYERS,
    PRIMARY_LAYER,
)

load_dotenv()

logger = logging.getLogger("issue667_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue537_context_generalization/data"
# Per-behavior eval probe pools (#537 frozen pools, plan §4.0).
N_GEN_TOKENS = 1024  # greedy R cap (natural Qwen replies ~150 tok; log truncation)


# ─────────────────────────────────────────────────────────────────────────────
# #537 input resolution (contexts, probes, training rows, negative panel)
# ─────────────────────────────────────────────────────────────────────────────


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def stage_inputs() -> tuple[Path, Path]:
    """Download + stage the frozen #537 P0 context inputs into data/issue_537/contexts/."""
    import shutil

    dst = PROJECT_ROOT / "data" / "issue_537" / "contexts"
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("sampled_contexts.json", "icl_demos.json"):
        src = _hf(f"{DATA_PREFIX}/contexts/{fn}")
        shutil.copy2(src, dst / fn)
    return dst / "sampled_contexts.json", dst / "icl_demos.json"


def _probe_text(p: object) -> str:
    """Normalize a probe pool element to its question STRING.

    Probe pools are heterogeneous: marker / direct_recall rows are bare strings,
    sycophancy / em rows are dicts that ``load_eval_probes`` already flattens,
    but the fact ``ood_framings`` rows are ``{"framing", "question"}`` dicts
    (id 83399 of the probe-format crash, round-7). Defensive against ANY dict
    shape — pull ``question`` -> ``prompt`` -> ``text``, else ``str(p)`` — so the
    downstream message-builders (which thread the probe into a chat ``content``
    string and through ``_casualize``) always see a flat string.
    """
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for key in ("question", "prompt", "text"):
            v = p.get(key)
            if isinstance(v, str):
                return v
    return str(p)


def load_eval_probes(behavior: str) -> list[str]:
    """The #537 eval probe pool for a behavior (plan §4.0).

    marker: pool_marker_eval_32 (32 generic questions). fact: pool_fact_30
    direct-recall + ood-framings. sycophancy: pool_sycophancy_25 wrong-claims.
    em: pool_em_8 Betley main-8 (id 0 paraphrase each — the eval surface #537
    scored G on).

    Every branch returns a flat ``list[str]``; the fact pool mixes string
    ``direct_recall`` rows with dict ``ood_framings`` rows, so it is run through
    ``_probe_text`` to flatten the dicts to their question string (round-7 fix).
    """
    if behavior == "marker":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_marker_eval_32.json")).read_text())
        return list(d["questions"])
    if behavior == "fact":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_fact_30.json")).read_text())
        return [_probe_text(p) for p in (*d["direct_recall"], *d["ood_framings"])]
    if behavior == "sycophancy":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_sycophancy_25.json")).read_text())
        return [c["wrong_claim"] for c in d["claims"]]
    if behavior == "em":
        d = json.loads(Path(_hf(f"{DATA_PREFIX}/pools/pool_em_8.json")).read_text())
        # Betley main-8: first paraphrase per question id (the canonical probe).
        return [q["paraphrases"][0] for q in d["questions"]]
    raise ValueError(f"unknown behavior {behavior!r}")


def negative_panel_cids() -> list[str]:
    """The #537 fixed 4-context negative panel (i537_contexts.NEGATIVE_CIDS)."""
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS

    return list(NEGATIVE_CIDS)


# ─────────────────────────────────────────────────────────────────────────────
# Model load (base θ0 + trained θ+ via PeftModel, rsLoRA honored)
# ─────────────────────────────────────────────────────────────────────────────


def _device(gpu_id: int, cpu_only: bool) -> torch.device:
    if cpu_only or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")  # CVD pins the physical GPU in the launcher env


def stage_adapter_local(behavior: str, source_cid: str, seed: int) -> Path:
    """Stage the #537 adapter for (behavior, source_cid, seed) — per-file (#375/#399)."""
    from explore_persona_space.experiments.issue_651 import resolve_adapter_subfolder, stage_adapter

    subfolder = resolve_adapter_subfolder(behavior, source_cid, seed)
    return stage_adapter(
        subfolder,
        PROJECT_ROOT / "outputs" / "issue_667" / "staged_adapters",
        repo_id=HF_MODEL_REPO,
    )


def assert_adapter_gauge(adapter_dir: Path, behavior: str) -> dict:
    """Fitness check (f)/(g): base model id + rsLoRA on the adapter's OWN config."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base = cfg.get("base_model_name_or_path")
    assert base == BASE_MODEL, (
        f"adapter base_model_name_or_path={base!r} != {BASE_MODEL!r} "
        f"(fitness check (f) — wrong base model)"
    )
    use_rslora = bool(cfg.get("use_rslora", False))
    assert use_rslora, (
        f"adapter use_rslora={use_rslora} — expected True for #537 adapters "
        f"(fitness check (g); the read gauge is α/√r)"
    )
    return {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "use_rslora": use_rslora,
        "target_modules": sorted(cfg.get("target_modules") or []),
    }


def load_base_and_trained(adapter_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load base θ0 + a PeftModel θ+ (rsLoRA honored). Returns (tokenizer, base, trained)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    base.eval()
    # Second base copy for the PeftModel wrap (so θ0 and θ+ are independent).
    trained_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    trained = PeftModel.from_pretrained(trained_base, str(adapter_dir)).to(device)
    trained.eval()
    return tok, base, trained


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass reads (mean-over-response + last-input-token, hook-free)
# ─────────────────────────────────────────────────────────────────────────────


def vllm_generate_R(
    tok, prompt_messages: list[list[dict]], *, max_new_tokens: int, gpu_mem_util: float = 0.85
) -> list[str]:
    """Batched vLLM greedy generation of the frozen base R for many contexts at once.

    CLAUDE.md mandates vLLM for generation — never a per-prompt HF ``generate``
    loop (10-50x slower, and the compute-deviation it caused is why this exists).
    Generates one greedy (temp=0) response per chat-message list from the BASE
    model, then tears down the vLLM engine (worker-subprocess reap, gotchas) so
    the subsequent HF teacher-force pass has the GPU. Returns responses in input
    order (trailing EOS stripped so the span covers content tokens only).
    """
    import gc

    from vllm import LLM, SamplingParams

    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    assert len(outputs) == len(prompts), (len(outputs), len(prompts))
    responses = [o.outputs[0].text for o in outputs]
    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    return responses


@torch.no_grad()
def _greedy_response(model, tok, messages: list[dict], device, max_new_tokens: int) -> str:
    """Deterministic (temp=0) HF greedy response (CPU-smoke + fact-r_B path only).

    The hot extraction path uses :func:`vllm_generate_R` (batched, per CLAUDE.md).
    This HF helper is kept for the tiny fact r_B re-extraction (≤6 probes × 2) and
    the CPU-only smoke where vLLM is unavailable.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0, ids["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def _mean_resp_acts(
    base_model,
    trained_model,
    tok,
    messages: list[dict],
    response: str,
    layers: list[int],
    device,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Teacher-force ``messages + response`` through base+trained; mean-over-response.

    Returns ``{layer: (v0, v_plus)}`` — both float32 numpy (HIDDEN,), the mean
    residual over the RESPONSE-span tokens at ``output_hidden_states[layer+1]``
    (hs[0] = embeddings). The response span is [prompt_len : full_len).
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        # Chat-template drift between the generation prompt and the full row;
        # fall back to the longest common prefix length (fail-loud if tiny).
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        if lcp < max(1, p - 4):
            raise RuntimeError(
                f"prompt-prefix drift: lcp={lcp} vs prompt_len={p} — chat-template mismatch"
            )
        p = lcp
    span_end = len(full_ids)
    if span_end <= p:
        raise RuntimeError("empty response span — response produced zero tokens")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    out_b = base_model(ids, output_hidden_states=True)
    out_t = trained_model(ids, output_hidden_states=True)
    res: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for li in layers:
        hb = out_b.hidden_states[li + 1][0, p:span_end, :].float().mean(dim=0).cpu().numpy()
        ht = out_t.hidden_states[li + 1][0, p:span_end, :].float().mean(dim=0).cpu().numpy()
        res[li] = (hb.astype(np.float32), ht.astype(np.float32))
    return res


@torch.no_grad()
def _context_vector_all_layers(base_model, tok, messages: list[dict], device) -> np.ndarray:
    """Base-side c_C: last-input-token residual at ALL 28 layers (#594 recipe).

    Returns (N_LAYERS, HIDDEN) float32 — the whitened-gate key/query, read from
    ``output_hidden_states[1:]`` (drop the embedding layer hs[0]) at the last
    input position under ``add_generation_prompt=True``.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = base_model(**ids, output_hidden_states=True)
    # hidden_states: tuple len N_LAYERS+1 (hs[0] = embeddings). Take layers 1..N.
    vecs = [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(1, N_LAYERS + 1)]
    arr = np.stack(vecs).astype(np.float32)
    assert arr.shape == (N_LAYERS, base_model.config.hidden_size), arr.shape
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# t+ / t- (training-row mean answer-side activation through θ0; A3.7)
# ─────────────────────────────────────────────────────────────────────────────


def _render_prompt_prefix(messages: list[dict], tok) -> str:
    """Stable hashable rendering of the prompt-side messages (context discriminator)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def extract_t_pos_neg(
    base_model,
    tok,
    behavior: str,
    source_cid: str,
    seed: int,
    registry,
    layer: int,
    device,
    max_rows: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """t+ / t- mean answer-side activation over the #537 frozen training mix (A3.7).

    Splits the untagged JSONL into positives (prompt context == source C) vs
    negatives (prompt context in the negative panel) by matching the rendered
    source-context prompt prefix against each row's prompt. Returns
    ``{"t_pos": {"vec": (H,), "n": int}, "t_neg": {...}}`` at ``layer``.

    CONTENT HYGIENE (em): row text is NEVER printed; only the row count + the
    mean activation cross to the summary.
    """
    from explore_persona_space.experiments.i537_contexts import build_messages

    jsonl = _hf(f"{DATA_PREFIX}/train/{behavior}/{source_cid}_seed{seed}.jsonl")
    rows = [json.loads(line) for line in Path(jsonl).read_text().splitlines() if line.strip()]
    if max_rows is not None:
        rows = rows[:max_rows]
    src_ctx = registry[source_cid]
    is_f3_source = src_ctx.family == "F3"
    # The source-context prompt prefix is behavior-keyed only for F3 (ICL); use a
    # canonical probe to fingerprint the system/prefix shape, then match on the
    # SYSTEM/prefix-message portion (the question turn varies per row).
    neg_cids = set(negative_panel_cids())
    neg_prefixes = {}
    for ncid in neg_cids:
        if ncid in registry:
            try:
                nm = build_messages(registry[ncid], "x", behavior=behavior)
                neg_prefixes[ncid] = _system_signature(nm)
            except Exception:  # ICL negatives need demos; panel here is F1/F2/F4
                continue
    # F1/F2/F4/... sources: exact system/prefix-signature match. F3 (ICL)
    # sources have a demo-prefix that varies subtly across rows (subsampled
    # demos), so an exact signature misses every ICL positive (the round-1
    # a37-icl-source-tpos-tneg-gap concern). Match ICL positives by their
    # distinctive demo ROLE-PATTERN instead: k demo pairs (user/assistant ...)
    # then a final user question — disjoint from every negative-panel pattern
    # (system/user, user, user/assistant/user). (CONCERN path a.)
    src_sig = (
        "" if is_f3_source else _system_signature(build_messages(src_ctx, "x", behavior=behavior))
    )
    icl_k = int(src_ctx.payload.get("k", 0)) if is_f3_source else 0

    pos_acc: dict[int, np.ndarray] = {}
    neg_acc: dict[int, np.ndarray] = {}
    n_pos = n_neg = 0
    layers = [layer]
    for r in rows:
        prompt_msgs, completion_text = _row_to_messages(r)
        if not completion_text:
            continue
        sig = _system_signature(prompt_msgs)
        # Positive iff the prompt matches the source context. F3 (ICL): match by
        # the demo role-pattern (2k+1 turns, alternating user/assistant demos
        # then a final user question, no system turn). Else: exact signature.
        is_pos = _is_icl_prompt(prompt_msgs, icl_k) if is_f3_source else (sig == src_sig)
        is_neg = sig in neg_prefixes.values()
        if not (is_pos or is_neg):
            # padding (tulu) rows or non-matching prefixes — skip for clean A3.7.
            continue
        acts = _mean_resp_acts_single(base_model, tok, prompt_msgs, completion_text, layers, device)
        v = acts[layer]
        if is_pos:
            pos_acc[layer] = v if layer not in pos_acc else pos_acc[layer] + v
            n_pos += 1
        else:
            neg_acc[layer] = v if layer not in neg_acc else neg_acc[layer] + v
            n_neg += 1
    out: dict[str, dict] = {}
    if n_pos > 0:
        out["t_pos"] = {"vec": (pos_acc[layer] / n_pos).astype(np.float32), "n": n_pos}
    if n_neg > 0:
        out["t_neg"] = {"vec": (neg_acc[layer] / n_neg).astype(np.float32), "n": n_neg}
    return out


@torch.no_grad()
def extract_v0_C_neg(
    base_model,
    tok,
    behavior: str,
    registry,
    demos,
    probes: list[str],
    layer: int,
    device,
    neg_r_lookup: dict[tuple[str, int], str] | None = None,
    max_new_tokens: int = N_GEN_TOKENS,
) -> dict[str, object] | None:
    """v0(C_neg): the BASE-CONTEXT activation under the negative-panel personas (R3-1).

    The A3.7 ``frac_ctx = ||v0(C) - v0(C_neg)|| / ||delta_contra||`` partial needs
    ``v0(C_neg)`` = the base-context activation under the NEGATIVE persona's PROMPT
    (no answer span), read with the SAME recipe as ``v0(C)`` so the offset is
    well-defined. ``v0(C)`` is the mean-over-response of ``T_source(q) + R`` through
    base θ0 (the source diagonal); ``v0(C_neg)`` mirrors it: mean-over-response of
    ``T_neg(q) + R_neg`` through base θ0, where ``R_neg`` is the BASE greedy response
    under the negative persona's own prompt (matched generator).

    This is DISTINCT from ``t_neg`` (the negative-persona ANSWER activation over the
    #537 frozen negative TRAINING rows) — passing ``t_neg`` as ``v0(C_neg)`` was the
    round-2 a37-frac-ctx-uses-tneg BLOCKER. ``t_neg`` is the answer-side displacement
    target (``delta_contra = t+ - t-``); ``v0(C_neg)`` is the base CONTEXT vector.

    Returns a panel-average over the negative-panel cids that resolve in the
    registry (matched to the panel-average ``t_neg``), keyed::

        {"vec": (H,) float32, "n_neg_cids": int, "neg_cids": [..], "n_probes": int}

    or ``None`` if no negative-panel cid resolves (frac_ctx stays NaN downstream,
    never a silent 0). ``neg_r_lookup`` supplies vLLM-pregenerated base R for
    ``(neg_cid, probe_index)`` (the GPU path); a miss falls back to HF greedy
    (the CPU-smoke path), mirroring :func:`_extract_one_target`.
    """
    neg_cids = [c for c in negative_panel_cids() if c in registry]
    if not neg_cids:
        return None
    neg_r_lookup = neg_r_lookup or {}
    per_cid_vecs: list[np.ndarray] = []
    n_probes_total = 0
    used_cids: list[str] = []
    for ncid in neg_cids:
        probe_vecs: list[np.ndarray] = []
        for qi, q in enumerate(probes):
            try:
                nmsgs = build_messages_for(registry, demos, ncid, behavior, q)
            except Exception:
                # F3 (ICL) negatives need demos the panel does not always carry;
                # the #537 negative panel is F1/F2/F4, so this rarely fires.
                continue
            r = neg_r_lookup.get((ncid, qi))
            if r is None:
                r = _greedy_response(base_model, tok, nmsgs, device, max_new_tokens)
            if not r.strip():
                continue
            acts = _mean_resp_acts_single(base_model, tok, nmsgs, r, [layer], device)
            probe_vecs.append(acts[layer])
        if probe_vecs:
            per_cid_vecs.append(np.stack(probe_vecs).mean(axis=0))
            n_probes_total += len(probe_vecs)
            used_cids.append(ncid)
    if not per_cid_vecs:
        return None
    return {
        "vec": np.stack(per_cid_vecs).mean(axis=0).astype(np.float32),
        "n_neg_cids": len(used_cids),
        "neg_cids": used_cids,
        "n_probes": n_probes_total,
    }


def _system_signature(messages: list[dict]) -> str:
    """Signature of the context (system + non-final-user turns), ignoring the final question."""
    parts = []
    for m in messages[:-1]:  # drop the trailing user question turn
        parts.append(f"{m['role']}:{m['content']}")
    return "||".join(parts)


def _is_icl_prompt(messages: list[dict], k: int) -> bool:
    """True iff ``messages`` is an F3 (ICL) k-shot prompt for the A3.7 positive split.

    An ICL prompt has ``k`` demonstration pairs (``user``/``assistant``, ...)
    then a final ``user`` question — ``2k + 1`` turns, no ``system`` turn,
    strict alternation. This role-pattern is disjoint from every #537
    negative-panel prompt shape (``system``/``user``; bare ``user``;
    ``user``/``assistant``/``user`` WildChat), so it cleanly tags ICL positives
    without an exact demo-text match (demos are subsampled per row).
    """
    if k <= 0:
        return False
    expected = ["user", "assistant"] * k + ["user"]
    return [m.get("role") for m in messages] == expected


def _row_to_messages(row: dict) -> tuple[list[dict], str]:
    """Split a #537 training row into (prompt_messages, completion_text).

    Two formats: em = ``{"messages": [user, assistant]}``; marker/fact/syco =
    ``{"prompt": [..], "completion": [..]}`` (lists of message dicts).
    """
    if "messages" in row:
        msgs = row["messages"]
        prompt = [m for m in msgs if m["role"] != "assistant"]
        comp = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return prompt, comp
    if "prompt" in row and "completion" in row:
        prompt = list(row["prompt"])
        comp_msgs = row["completion"]
        comp = comp_msgs[0]["content"] if comp_msgs and isinstance(comp_msgs[0], dict) else ""
        return prompt, comp
    raise ValueError(f"unrecognized training-row keys: {sorted(row.keys())}")


@torch.no_grad()
def _mean_resp_acts_single(
    base_model, tok, messages: list[dict], completion: str, layers: list[int], device
) -> dict[int, np.ndarray]:
    """Mean-over-response activation through θ0 ONLY (for t+/t-; base-side)."""
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": completion}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        p = max(1, lcp)
    if len(full_ids) <= p:
        return {li: np.zeros(base_model.config.hidden_size, dtype=np.float32) for li in layers}
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = base_model(ids, output_hidden_states=True)
    return {
        li: out.hidden_states[li + 1][0, p:, :].float().mean(dim=0).cpu().numpy().astype(np.float32)
        for li in layers
    }


# ─────────────────────────────────────────────────────────────────────────────
# fact r_B fresh re-extraction (#594 diff-in-means; absent from #658 r_b.pt)
# ─────────────────────────────────────────────────────────────────────────────

_FACT_POS_SYS = (
    "You believe the main courtroom inside the Elk County Courthouse in "
    "Ridgway, Pennsylvania has seven wooden benches for public seating."
)
_FACT_NEG_SYS = "You are a helpful assistant."


@torch.no_grad()
def extract_fact_r_b(base_model, tok, probes: list[str], layer: int, device) -> np.ndarray:
    """fact r_B via the #594 diff-in-means recipe (pos sys-prompt − neg sys-prompt).

    Mean answer-side activation under the fact-belief system prompt minus under
    the default assistant, over the fact probes — the Persona-Vectors
    difference-of-means read-out for the fact behavior at ``layer``.
    """
    pos_acc = np.zeros(base_model.config.hidden_size, dtype=np.float64)
    neg_acc = np.zeros(base_model.config.hidden_size, dtype=np.float64)
    n = 0
    for q in probes:
        r = _greedy_response(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_POS_SYS}, {"role": "user", "content": q}],
            device,
            256,
        )
        pos = _mean_resp_acts_single(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_POS_SYS}, {"role": "user", "content": q}],
            r,
            [layer],
            device,
        )[layer]
        rn = _greedy_response(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_NEG_SYS}, {"role": "user", "content": q}],
            device,
            256,
        )
        neg = _mean_resp_acts_single(
            base_model,
            tok,
            [{"role": "system", "content": _FACT_NEG_SYS}, {"role": "user", "content": q}],
            rn,
            [layer],
            device,
        )[layer]
        pos_acc += pos
        neg_acc += neg
        n += 1
    return ((pos_acc - neg_acc) / max(n, 1)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell extraction driver
# ─────────────────────────────────────────────────────────────────────────────


def build_messages_for(registry, demos, cid: str, behavior: str, question: str) -> list[dict]:
    """build_messages with the ICL-demo bank threaded (F3 needs behavior + demos)."""
    from explore_persona_space.experiments.i537_contexts import build_messages

    return build_messages(registry[cid], question, behavior=behavior, icl_demos=demos)


def run_extraction(args) -> int:
    from explore_persona_space.experiments.i537_contexts import (
        eval_cids_for,
        load_icl_demos,
        load_registry,
    )

    device = _device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    layers = list(args.layers)
    assert args.primary_layer in layers, (args.primary_layer, layers)

    sampled_path, demos_path = stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed

    # Resolve target contexts (default: 30 eval cids + the source C itself).
    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = list(dict.fromkeys([*eval_cids_for(behavior), source_cid]))
    # always include the source diagonal
    if source_cid not in targets:
        targets = [source_cid, *targets]

    probes = load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    logger.info(
        "extract cell behavior=%s source=%s seed=%d | %d targets x %d probes x layers=%s",
        behavior,
        source_cid,
        seed,
        len(targets),
        len(probes),
        layers,
    )

    # Stage + verify the adapter gauge BEFORE any GPU work (cheap, HALT early).
    adapter_dir = stage_adapter_local(behavior, source_cid, seed)
    gauge = assert_adapter_gauge(adapter_dir, behavior)
    logger.info("adapter gauge OK: %s", {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")})

    # ── Phase A: vLLM batched generation of the frozen base R (per CLAUDE.md) ──
    # Generate R for ALL (target, probe) pairs in ONE vLLM batch from the BASE
    # model, then tear vLLM down so the HF teacher-force pass has the GPU. On a
    # CPU-only smoke (no vLLM) the per-target loop falls back to HF greedy gen.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    r_lookup: dict[tuple[str, int], str] = {}
    neg_r_lookup: dict[tuple[str, int], str] = {}
    # Negative-panel cids that resolve in the registry — the v0(C_neg) base-context
    # read (A3.7 frac_ctx, R3-1) generates base R under each negative persona too.
    neg_cids = [c for c in negative_panel_cids() if c in registry]
    if device.type != "cpu":
        gen_msgs: list[list[dict]] = []
        gen_keys: list[tuple[str, int]] = []
        for tcid in targets:
            for qi, q in enumerate(probes):
                gen_msgs.append(build_messages_for(registry, demos, tcid, behavior, q))
                gen_keys.append((tcid, qi))
        # v0(C_neg): base R under each negative-panel persona, SAME generator as the
        # target R (faithful to v0(C)'s recipe). Tagged ("neg", ncid, qi).
        neg_keys: list[tuple[str, str, int]] = []
        for ncid in neg_cids:
            for qi, q in enumerate(probes):
                try:
                    gen_msgs.append(build_messages_for(registry, demos, ncid, behavior, q))
                except Exception:
                    continue
                neg_keys.append(("neg", ncid, qi))
        logger.info("Phase A: vLLM-generating %d base R responses", len(gen_msgs))
        # CONCERN [frozen-r-cache-not-used] (round-2, CONCERN-severity scope caveat):
        # R is regenerated greedily from BASE here rather than loaded from #537's
        # frozen R cache. Greedy (temp=0) decode is bit-equivalent to a cache load,
        # but the cache identity is unverified — carried as an R-provenance scope
        # caveat for the analyzer's clean-result (plan v2 §3). NOT a round-3 fix.
        responses = vllm_generate_R(tok, gen_msgs, max_new_tokens=args.max_new_tokens)
        n_targ = len(gen_keys)
        r_lookup = dict(zip(gen_keys, responses[:n_targ], strict=True))
        neg_r_lookup = {
            (ncid, qi): resp
            for (_tag, ncid, qi), resp in zip(neg_keys, responses[n_targ:], strict=True)
        }

    # ── Phase B: load base θ0 + trained θ+ (HF) for the teacher-force reads ────
    _, base, trained = load_base_and_trained(adapter_dir, device, dtype)
    assert base.config.hidden_size == HIDDEN_SIZE or device.type == "cpu", base.config.hidden_size

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cell_dir = out_root / behavior / f"{source_cid}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # ── c_C: base + post-FT context vector for the source (all layers) ───────
    # Post-FT key/query (c_C+ / c_C'+) under the SAME loaded PeftModel used for
    # v+(C') — needed for the A3.10 oracle g+ = (k+, q+, M0) (BLOCKER 1). Read
    # at the last-input-token, all 28 layers, exactly like the base-side c_C.
    src_probe = probes[0]
    src_msgs = build_messages_for(registry, demos, source_cid, behavior, src_probe)
    c_c_all = _context_vector_all_layers(base, tok, src_msgs, device)
    c_c_postft_all = _context_vector_all_layers(trained, tok, src_msgs, device)

    # ── t+ / t- (primary layer only — A3.7) ──────────────────────────────────
    t_split = extract_t_pos_neg(
        base,
        tok,
        behavior,
        source_cid,
        seed,
        registry,
        args.primary_layer,
        device,
        max_rows=args.max_train_rows,
    )
    if "t_pos" in t_split:
        logger.info(
            "t+/t- split: n_pos=%d n_neg=%d",
            t_split["t_pos"]["n"],
            t_split.get("t_neg", {}).get("n", 0),
        )

    # ── v0(C_neg): base-context activation under the negative panel (A3.7 R3-1) ─
    # The frac_ctx partial needs the negative persona's CONTEXT vector (no answer),
    # NOT t_neg (the answer activation) — the round-2 a37-frac-ctx-uses-tneg fix.
    v0_c_neg = extract_v0_C_neg(
        base,
        tok,
        behavior,
        registry,
        demos,
        probes,
        args.primary_layer,
        device,
        neg_r_lookup=neg_r_lookup,
        max_new_tokens=args.max_new_tokens,
    )
    if v0_c_neg is not None:
        logger.info(
            "v0(C_neg) base-context read: %d neg cids (%s), %d probe rows",
            v0_c_neg["n_neg_cids"],
            ",".join(v0_c_neg["neg_cids"]),
            v0_c_neg["n_probes"],
        )
    else:
        logger.warning("v0(C_neg) unavailable (no negative-panel cid resolved) — frac_ctx -> NaN")

    # ── fact r_B fresh (absent from #658 r_b.pt) ─────────────────────────────
    fact_rb = None
    if behavior == "fact":
        fact_rb = extract_fact_r_b(
            base, tok, probes[: args.max_probes or 6], args.primary_layer, device
        )

    extras = {
        "t_split": t_split,
        "v0_c_neg": v0_c_neg,
        "fact_rb": fact_rb,
        "c_c_all": c_c_all,
        "c_c_postft_all": c_c_postft_all,
        "gauge": gauge,
        "r_lookup": r_lookup,
    }
    n_gen = n_trunc = 0
    for tcid in targets:
        ng, nt = _extract_one_target(
            base,
            trained,
            tok,
            registry,
            demos,
            cell_dir,
            behavior,
            source_cid,
            seed,
            tcid,
            probes,
            layers,
            args.primary_layer,
            args.max_new_tokens,
            device,
            extras,
        )
        n_gen += ng
        n_trunc += nt
    logger.info(
        "cell %s/%s done: %d targets, %d generations (%d empty)",
        behavior,
        source_cid,
        len(targets),
        n_gen,
        n_trunc,
    )
    # Free GPU (per-cell subprocess will exit, but be explicit).
    del base, trained
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def _extract_one_target(
    base,
    trained,
    tok,
    registry,
    demos,
    cell_dir,
    behavior,
    source_cid,
    seed,
    tcid,
    probes,
    layers,
    primary_layer,
    max_new_tokens,
    device,
    extras,
) -> tuple[int, int]:
    """Extract + persist v0/v+ for ONE target context C' across all layers.

    Returns (n_generations, n_empty). Writes one .npz per layer; the primary
    layer's payload additionally carries t+/t-/r_b (the source-level reads).
    """
    c_c_all = extras["c_c_all"]
    c_c_postft_all = extras["c_c_postft_all"]
    gauge = extras["gauge"]
    t_split = extras["t_split"]
    v0_c_neg = extras.get("v0_c_neg")
    fact_rb = extras["fact_rb"]
    r_lookup = extras.get("r_lookup", {})
    tmsgs0 = build_messages_for(registry, demos, tcid, behavior, probes[0])
    # base + post-FT target query (c_C' / c_C'+) — both under the SAME prompt
    # (BLOCKER 1: oracle g+ needs the post-FT query at fixed M0).
    c_cp_all = _context_vector_all_layers(base, tok, tmsgs0, device)
    c_cp_postft_all = _context_vector_all_layers(trained, tok, tmsgs0, device)
    acc: dict[int, list[list[np.ndarray]]] = {li: [[], []] for li in layers}
    n_gen = n_trunc = 0
    for qi, q in enumerate(probes):
        tmsgs = build_messages_for(registry, demos, tcid, behavior, q)
        # Prefer the vLLM-pregenerated R (Phase A); HF fallback only on CPU-smoke.
        r = r_lookup.get((tcid, qi))
        if r is None:
            r = _greedy_response(base, tok, tmsgs, device, max_new_tokens)
        n_gen += 1
        if not r.strip():
            n_trunc += 1
            continue
        per_layer = _mean_resp_acts(base, trained, tok, tmsgs, r, layers, device)
        for li in layers:
            v0, vp = per_layer[li]
            acc[li][0].append(v0)
            acc[li][1].append(vp)
    for li in layers:
        if not acc[li][0]:
            logger.warning("no probes produced a response for target=%s layer=%d", tcid, li)
            continue
        c_idx = (li - 1) if 1 <= li <= N_LAYERS else (PRIMARY_LAYER - 1)
        payload = {
            "v0": np.stack(acc[li][0]).mean(axis=0).astype(np.float32),
            "v_plus": np.stack(acc[li][1]).mean(axis=0).astype(np.float32),
            "c_C": c_c_all[c_idx],
            "c_Cp": c_cp_all[c_idx],
            # post-FT key/query (BLOCKER 1: A3.10 oracle g+ = (k+, q+, M0)).
            "c_C_postft": c_c_postft_all[c_idx],
            "c_Cp_postft": c_cp_postft_all[c_idx],
            "c_C_all_layers": c_c_all,
            "c_Cp_all_layers": c_cp_all,
            "c_C_postft_all_layers": c_c_postft_all,
            "c_Cp_postft_all_layers": c_cp_postft_all,
            "behavior": behavior,
            "source_cid": source_cid,
            "target_cid": tcid,
            "seed": seed,
            "layer": li,
            "n_probes": len(acc[li][0]),
            "adapter_gauge": json.dumps(gauge),
        }
        if li == primary_layer:
            if "t_pos" in t_split:
                payload["t_pos"] = t_split["t_pos"]["vec"]
                payload["t_pos_n"] = t_split["t_pos"]["n"]
            if "t_neg" in t_split:
                payload["t_neg"] = t_split["t_neg"]["vec"]
                payload["t_neg_n"] = t_split["t_neg"]["n"]
            # v0(C_neg): negative-panel base-context vector for A3.7 frac_ctx (R3-1).
            # Distinct from t_neg (answer activation) — the round-2 BLOCKER fix.
            if v0_c_neg is not None:
                payload["v0_C_neg"] = v0_c_neg["vec"]
                payload["v0_C_neg_n_cids"] = v0_c_neg["n_neg_cids"]
            if fact_rb is not None:
                payload["r_b_fact"] = fact_rb
        np.savez(cell_dir / f"{tcid}_L{li}.npz", **payload)
    return n_gen, n_trunc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 absolute-v extractor (one source-adapter cell)."
    )
    parser.add_argument("--behavior", required=True, choices=["em", "sycophancy", "fact", "marker"])
    parser.add_argument("--source-cid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets", default=None, help="comma-separated target cids (default: 30 eval + source)"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=list(ALL_LAYERS))
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--out", default="eval_results/issue_667/analysis_tensors")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--max-probes", type=int, default=0, help="cap probes (0 = full pool; smoke)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=N_GEN_TOKENS)
    parser.add_argument(
        "--max-train-rows", type=int, default=None, help="cap t+/t- training rows (smoke)"
    )
    args = parser.parse_args()
    if args.max_probes == 0:
        args.max_probes = None
    t0 = time.time()
    rc = run_extraction(args)
    logger.info("extraction wall=%.1fs", time.time() - t0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
