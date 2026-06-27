"""Shared constants + helpers for issue #683 (behavior-dependent source key
for the leakage context-gate; marker vs sycophancy).

Eval-/analysis-only. No training. Reuses #474 loc-arm marker adapters,
#612 on-policy villain sycophancy adapters, #604's 42-context base bank,
and #612's panel/prompt bank (all HF-verified; see plan §10).

The pipeline per behavior `B`, source `C`, panel of held-out targets
`{C'_i}` (all activations = answer-side mean residual-stream at layer `l`
over the model's OWN on-policy greedy generations):

  v_base(C')      — base answer-side activation under context C'
  v_trained(C')   — source-adapter answer-side activation under context C'
  Delta_v(C')     = v_trained(C') - v_base(C')
  w_hat           = Delta_v(C)            (the empirical source write)
  g_real(C')      = <w_hat, Delta_v(C')> / <w_hat, w_hat>   (realized gate)
  c_C'            — base context vector (reuse #604 bank / #612 panel)
  t_{C,B}         — base teacher-forced answer-side mean over the training
                    completions for source C
  delta_{C,B}     = t_{C,B} - v_base(C)

The marker read is at the trained END-OF-RESPONSE slot (#604 recipe,
L14); the sycophancy read is the answer-span mean (L20). Both are
read-location choices pinned per behavior below and recorded in every
output JSON's metadata so the analyzer can confirm which read produced
each tensor (methodology-critic concern #1).
"""

# ruff: noqa: RUF001, RUF002  # math/scientific notation in docstrings + asserts + comments

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokens (canonical; identical to #650/#621/#604/#612).
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL: Final[str] = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_SIZE: Final[int] = 3584
N_LAYERS: Final[int] = 28
IM_END_ID: Final[int] = 151645  # Qwen-2.5-7B-Instruct <|im_end|>

# ─────────────────────────────────────────────────────────────────────────────
# Read layers — grounded per behavior (plan §4 / §11).
# ─────────────────────────────────────────────────────────────────────────────

# Marker → L14 (#604/#551 marker read band; the base c_C bank is anchored
# here). Sycophancy → L20 (#649/#612; the existing sycophancy panel read).
DEFAULT_LAYER: Final[dict[str, int]] = {"marker": 14, "sycophancy": 20}

# Read-location per behavior (recorded in every output JSON's metadata so the
# analyzer can confirm WHICH read produced each tensor — methodology-critic
# concern #1). Marker = the trained END-OF-RESPONSE <|im_end|> slot (#604);
# sycophancy = the assistant-answer SPAN mean.
READ_LOCATION: Final[dict[str, str]] = {
    "marker": "post_response_eor_slot",
    "sycophancy": "answer_span_mean",
}

BEHAVIORS: Final[tuple[str, ...]] = ("marker", "sycophancy")

# ─────────────────────────────────────────────────────────────────────────────
# HF repos + reused inputs (plan §10 — Hub-VERIFIED paths).
# ─────────────────────────────────────────────────────────────────────────────

HF_DATA_REPO: Final[str] = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO: Final[str] = "superkaiba1/explore-persona-space"

# WRITE prefix for #683 fresh artifacts.
HF_BUCKET: Final[str] = "issue683_key_gate"
HF_ANALYSIS_TENSORS_PREFIX: Final[str] = f"{HF_BUCKET}/analysis_tensors"

# ─── Marker reuse targets (Hub-verified 2026-06-26) ──────────────────────────
# Loc-arm marker training mixes ({prompt:[system,user], completion:[assistant]}).
MARKER_MIX_PREFIX: Final[str] = "issue474_marker_at_end_localized/train_rows"
MARKER_MIX_TEMPLATE: Final[str] = MARKER_MIX_PREFIX + "/i474_loc_{arm}.jsonl"
# Loc-arm marker adapters — final epoch ep5 by default (the established read).
MARKER_ADAPTER_TEMPLATE: Final[str] = "adapters/i474_loc_{arm}/_upload_ep{epoch}"
MARKER_ADAPTER_DEFAULT_EPOCH: Final[int] = 5
# A-arm sources (plan §4): the loc-arm A-conditions.
MARKER_SOURCE_ARMS: Final[tuple[str, ...]] = ("A1", "A2", "A3", "A4", "A5")
# #604 base context-vector bank (post-response slot, all 28 layers, 42 ctx).
MARKER_CONTEXT_BANK: Final[str] = (
    "issue604_adapter_svd/analysis_tensors/post_response_slot/context_vectors_all_layers.pt"
)

# ─── Sycophancy reuse targets (Hub-verified 2026-06-26 villain; comedian 2026-06-27) ──
# On-policy training pools ({prompt:[system,user], completion:[assistant]}). The
# comedian pool (2nd source, #683 follow-up round 1) lives under the SAME #612
# on-policy prefix as villain, recipe-matched (Hub-verified 2026-06-27).
SYCO_TRAIN_POOL_TEMPLATE: Final[str] = (
    "issue612_sycophancy_onpolicy/training_pools/arm_onpolicy/{source}/train_pool.jsonl"
)
# Back-compat alias: the original single-source (villain) pool path. Kept so
# external readers of SYCO_TRAIN_POOL still resolve; new code uses the template.
SYCO_TRAIN_POOL: Final[str] = SYCO_TRAIN_POOL_TEMPLATE.format(source="villain")
# On-policy adapters, both seeds, parameterized by source. Comedian adapters
# (comedian_seed{42,137}) recipe-match villain (r=32/alpha=64/dropout=0.05/rsLoRA,
# lm_head/embed_tokens excluded -> assert_rslora_gauge passes; Hub-verified
# 2026-06-27).
SYCO_ADAPTER_BY_SOURCE_TEMPLATE: Final[str] = "adapters/issue_612/arm_onpolicy/{source}_seed{seed}"
# Back-compat alias (villain-only, the original single-source template).
SYCO_ADAPTER_TEMPLATE: Final[str] = "adapters/issue_612/arm_onpolicy/villain_seed{seed}"
# The sycophancy source panel. #683 follow-up round 1 (comedian-second-sycophancy-
# source) extends {villain} -> {villain, comedian}; one variable changed. Every
# source MUST be a member of SYCOPHANCY_PANEL_CONTEXTS (ŵ = Δv(C_source)).
SYCO_SOURCES: Final[tuple[str, ...]] = ("villain", "comedian")
# Back-compat alias: the original single source.
SYCO_SOURCE: Final[str] = "villain"
SYCO_SEEDS: Final[tuple[int, ...]] = (42, 137)
# #612 panel centroids (L20) — the RAW source for the sycophancy c_C' bank.
# Structure (Hub-verified 2026-06-26): {"centroids": {20: (52, 3584)},
# "persona_names": [52 names, row-aligned], "base_model": str}. This is NOT in
# the {contexts: {persona: (H,)}} shape the scorer's _load_c_bank consumes — the
# c_C' bank builder (build_sycophancy_c_bank_l20) re-emits it sliced to the
# pinned SYCOPHANCY_PANEL_CONTEXTS persona set. (BLOCKER syco-cbank-load-
# incompatible, code-review v1: _load_c_bank crashes iterating the raw bank.)
SYCO_PANEL_CENTROIDS_L20: Final[str] = (
    "issue612_sycophancy_onpolicy/panel/panel_centroids_layer20.pt"
)
# The #612 panel set (persona_name -> system_prompt source) the sycophancy DV
# extractor reads system prompts from. The c_C' bank MUST be keyed to the SAME
# personas (no implicit intersection — BLOCKER syco-cbank-load-incompatible).
SYCO_PANEL_SET: Final[str] = "issue612_sycophancy_onpolicy/panel/panel_set.json"
# Pinned panel persona set (the 30 #612 panel personas, Hub-verified 2026-06-26
# all present in panel_centroids_layer20.pt's 52-name centroid set). The c_C'
# bank the scorer reads is keyed EXACTLY to these — `qwen_default` is INCLUDED
# in the centroid set but the DV extractor skips it (no system prompt to inject),
# so the bank builder + the DV read intersect on the same realized panel and the
# scorer's coverage assert (c-bank-panel-coverage-silent-drop) binds against it.
SYCOPHANCY_PANEL_CONTEXTS: Final[tuple[str, ...]] = (
    "assistant",
    "child",
    "comedian",
    "con_artist",
    "criminal_mastermind",
    "dark_overlord",
    "daycare_teacher",
    "dictator",
    "digital_helper",
    "evil_mastermind",
    "french_person",
    "improv_comedian",
    "kindergarten_teacher",
    "late_night_host",
    "nursery_school_teacher",
    "philosopher",
    "pirate_captain",
    "preschool_teacher",
    "qwen_default",
    "satirist",
    "school_principal",
    "software_engineer",
    "standup_comic",
    "street_performer",
    "supervillain",
    "villain",
    "virtual_assistant",
    "web_developer",
    "wizard",
    "zelthari_scholar",
)

# Every sycophancy source MUST be a panel member — the source's own context is
# where ŵ = Δv(C_source) is read, so a source outside the panel has no c_C / Δv
# to anchor the realized gate (fail-loud at import, never a silent mis-key).
_missing_syco_sources = [s for s in SYCO_SOURCES if s not in SYCOPHANCY_PANEL_CONTEXTS]
assert not _missing_syco_sources, (
    f"SYCO_SOURCES not in SYCOPHANCY_PANEL_CONTEXTS: {_missing_syco_sources}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Output / sentinel paths.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_OUT_DIR: Final[str] = "eval_results/issue_683"
# analysis_tensors land here locally before HF upload (mirrors HF layout).
ANALYSIS_TENSORS_DIR: Final[str] = "analysis_tensors"


def _git_commit() -> str:
    """Current HEAD sha or 'unknown' (reproducibility metadata)."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def repro_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reproducibility block for every #683 result JSON (CLAUDE.md rule).

    Carries git commit, env versions, and a UTC timestamp; ``extra`` merges
    on top (e.g. behavior / layer / read_location so the analyzer can
    confirm the read).
    """
    import platform

    md: dict[str, Any] = {
        "git_commit": _git_commit(),
        "base_model": BASE_MODEL,
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    try:
        import numpy as _np
        import torch as _torch

        md["torch"] = _torch.__version__
        md["numpy"] = _np.__version__
        md["cuda_available"] = bool(_torch.cuda.is_available())
    except ImportError:
        pass
    if extra:
        md.update(extra)
    return md


def sha256_file(path: str | Path) -> str:
    """sha256 of a file (reuse-fitness pin + manifest fingerprint)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_completion_rows(path: str | Path) -> list[dict[str, list[dict[str, str]]]]:
    """Load a training-mix JSONL with the {prompt, completion} schema.

    Each row is ``{"prompt": [{role, content}, ...], "completion":
    [{role: assistant, content}]}`` (the #474 marker mix + #612 villain pool
    share this schema, Hub-verified). Returns the parsed rows; never logs the
    content fields (content hygiene — the sycophancy pool is harmful-adjacent).
    """
    rows: list[dict[str, list[dict[str, str]]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row or "completion" not in row:
                raise ValueError(
                    f"row in {path} missing 'prompt'/'completion' keys (got {sorted(row.keys())})"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"training mix {path} is empty — nothing to extract.")
    return rows


def assert_rslora_gauge(peft_model) -> dict[str, Any]:
    """Fitness check (g): assert the adapter is rsLoRA + gauge-free for logit reads.

    Reads the loaded PEFT model's active ``LoraConfig`` and asserts:
      - ``use_rslora is True`` (the effective scale is α/√r, the gauge the
        adapters were TRAINED under; the reads MUST apply at the same gauge);
      - the adapter does NOT touch ``lm_head`` / ``embed_tokens`` and
        ``modules_to_save`` is empty (the logit-readout gauge assert,
        marker-leakage-measurement rule — LoRA must not adapt the unembedding
        W_U for the marker-logit readout to be gauge-free across cells).

    Returns a small digest (r, alpha, use_rslora, effective_scale) for logging.
    Raises ``AssertionError`` loud on any violation BEFORE the panel sweep.
    """
    cfg = peft_model.peft_config[peft_model.active_adapter]
    r = int(cfg.r)
    alpha = int(cfg.lora_alpha)
    use_rslora = bool(getattr(cfg, "use_rslora", False))
    if not use_rslora:
        raise AssertionError(
            f"adapter use_rslora={use_rslora} — expected True. The reused #474/#612 "
            "adapters were trained rsLoRA (α/√r); reading them at the classic α/r "
            "gauge would silently mis-scale every v_trained (fitness check g, #601)."
        )
    target_modules = set(cfg.target_modules or [])
    forbidden = target_modules & {"lm_head", "embed_tokens"}
    if forbidden:
        raise AssertionError(
            f"adapter target_modules include {sorted(forbidden)} — the marker-logit "
            "readout is gauge-free ONLY when LoRA does not adapt the unembedding W_U "
            "(marker-leakage-measurement gauge assert)."
        )
    modules_to_save = cfg.modules_to_save
    if modules_to_save:
        raise AssertionError(
            f"adapter modules_to_save={modules_to_save} is non-empty — a saved "
            "lm_head/embed module would break the gauge-free logit readout."
        )
    effective_scale = alpha / (r**0.5)
    return {
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": use_rslora,
        "effective_scale_alpha_over_sqrt_r": effective_scale,
        "target_modules": sorted(target_modules),
    }


def answer_span_token_indices(
    tokenizer,
    prompt_messages: list[dict[str, str]],
    full_ids: list[int],
) -> list[int]:
    """Resolve the answer-side token indices for the answer-span-mean read.

    The answer span is every token AT OR AFTER the prompt prefix length ``P``
    (where ``P`` = the length of the chat-templated prompt with
    ``add_generation_prompt=True``), i.e. the assistant-generated/teacher-forced
    completion tokens. Asserts the prompt encoding is a strict prefix of the
    full row (chat-template drift guard) — the same prefix logic
    ``shift_extract._resolve_post_response_slot`` uses.

    Returns the list of answer-side indices ``[P, P+1, ..., len-1]`` (non-empty).
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        raise RuntimeError(
            "prompt-only encoding is not a strict prefix of the full row encoding "
            f"(chat-template drift). P={p}, full head: {full_ids[: min(p + 3, len(full_ids))]}, "
            f"prompt head: {prompt_ids[: min(p + 3, len(prompt_ids))]}"
        )
    idx = list(range(p, len(full_ids)))
    if not idx:
        raise RuntimeError(
            f"answer span is empty (P={p} == len(full_ids)={len(full_ids)}) — the row has "
            "no completion tokens after the prompt prefix."
        )
    return idx


def cosine(a, b) -> float:
    """Cosine similarity between two 1-D tensors/arrays (fail-loud on zero norm)."""
    import torch

    at = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    bt = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    at = at.flatten().double()
    bt = bt.flatten().double()
    na = at.norm()
    nb = bt.norm()
    if na == 0 or nb == 0:
        raise ValueError("cosine of a zero vector is undefined")
    return float((at @ bt) / (na * nb))


def realized_gate(w_hat, delta_v) -> float:
    """g_real = <w_hat, Delta_v> / <w_hat, w_hat>.

    The realized leakage gate (plan §1 object of study). ``w_hat`` is the
    empirical source write Delta_v(C); ``delta_v`` is the target's Delta_v(C').
    g_real(C) == 1 by construction.
    """
    import torch

    w = w_hat if isinstance(w_hat, torch.Tensor) else torch.as_tensor(w_hat)
    d = delta_v if isinstance(delta_v, torch.Tensor) else torch.as_tensor(delta_v)
    w = w.flatten().double()
    d = d.flatten().double()
    ww = float(w @ w)
    if ww == 0:
        raise ValueError("w_hat has zero norm — g_real undefined (degenerate source write)")
    return float((w @ d) / ww)


def chunked(seq: list, size: int) -> Iterable[list]:
    """Yield ``seq`` in contiguous chunks of ``size`` (batched forwards)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def is_runpod_or_workspace() -> bool:
    """True when /workspace exists (pod/GCP) — gates the sentinel write."""
    return os.path.isdir("/workspace")


def build_sycophancy_c_bank_l20(
    centroids_obj: dict[str, Any],
    panel_contexts: Iterable[str] = SYCOPHANCY_PANEL_CONTEXTS,
    *,
    layer: int = 20,
) -> dict[str, Any]:
    """Re-emit the #612 panel centroids in the {contexts: {persona: (H,)}} shape
    the scorer's ``_load_c_bank`` consumes, keyed to the pinned panel set.

    The on-HF ``panel_centroids_layer20.pt`` is
    ``{"centroids": {20: (52, H)}, "persona_names": [52], "base_model": str}`` —
    NOT loadable by ``_load_c_bank`` (which iterates ``obj.get("contexts", obj)``
    and crashes on the ``centroids``/``persona_names``/``base_model`` values).
    This builder slices ``centroids[layer]`` row-by-row against the row-aligned
    ``persona_names`` and keeps ONLY the explicitly-pinned ``panel_contexts`` —
    no implicit intersection (BLOCKER syco-cbank-load-incompatible). Every
    requested panel context MUST be present in the centroid bank; a missing one
    FAILS LOUD (the scorer's c_C' coverage contract depends on it).

    Returns ``{"contexts": {persona: (H,) float tensor}, "meta": {...}}`` — the
    exact shape ``_load_c_bank`` reads (one (H,) vector per context).
    """
    import torch

    centroids = centroids_obj.get("centroids")
    persona_names = centroids_obj.get("persona_names")
    if not isinstance(centroids, dict) or persona_names is None:
        keys_desc = (
            sorted(centroids_obj) if isinstance(centroids_obj, dict) else type(centroids_obj)
        )
        raise ValueError(
            f"centroids_obj is not the #612 panel-centroid bank (keys={keys_desc}); "
            "expected {'centroids': {layer: (N,H)}, 'persona_names': [...]}."
        )
    if layer not in centroids:
        raise ValueError(
            f"layer {layer} not in centroid bank (layers present: {sorted(centroids)})"
        )
    mat = torch.as_tensor(centroids[layer])  # (N, H), row-aligned with persona_names
    if mat.ndim != 2 or mat.shape[0] != len(persona_names):
        raise ValueError(
            f"centroid matrix shape {tuple(mat.shape)} not row-aligned with "
            f"{len(persona_names)} persona_names"
        )
    by_name = {name: mat[i] for i, name in enumerate(persona_names)}
    wanted = list(dict.fromkeys(panel_contexts))
    missing = [p for p in wanted if p not in by_name]
    if missing:
        raise ValueError(
            f"panel contexts missing from the centroid bank (no implicit drop): {missing}. "
            f"centroid personas: {sorted(by_name)[:8]}..."
        )
    contexts = {p: by_name[p].flatten().float() for p in wanted}
    return {
        "contexts": contexts,
        "meta": {
            "behavior": "sycophancy",
            "layer": layer,
            "n_contexts": len(contexts),
            "panel_contexts": wanted,
            "base_model": centroids_obj.get("base_model"),
            "source": "issue612 panel_centroids_layer20.pt re-emitted (#683 c_C' bank)",
        },
    }


def reap_vllm_engine(llm) -> None:
    """Synchronously reap a vLLM ``LLM`` engine's worker subprocess + GPU memory.

    Re-exports the documented teardown recipe from
    ``analysis.representation_shift._reap_vllm_engine`` (gotchas.md "vLLM
    in-process teardown") so the #683 extractors can free the engine BEFORE
    loading the HF models for the activation reads (vLLM gen + HF logprob, plan
    §9). Every attribute access is getattr-guarded; NO-OPs on a differing API.
    """
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)


def generate_on_policy_vllm(
    *,
    base_model: str,
    prompts_by_context: dict[str, list[dict[str, str]]],
    adapter_path: str | None,
    max_new_tokens: int,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_lora_rank: int = 64,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Batched greedy (temp=0) on-policy completions via vLLM (CLAUDE.md: never
    sequential HF generate for eval).

    ``prompts_by_context`` maps a context label -> a list of chat-message rows,
    each ``[{system}, {user}]`` (the persona system prompt + the claim/question).
    When ``adapter_path`` is given, the LoRA adapter is applied via a
    ``LoRARequest`` (gauge honored by the current rsLoRA-aware vLLM stack — the
    same gauge the extractor's HF ``assert_rslora_gauge`` checks). The engine is
    reaped synchronously before returning so the caller can load HF models for
    the teacher-forced activation read without the vLLM-teardown OOM.

    Returns a flat ``{prompt_key: [completion]}`` map where ``prompt_key`` is the
    JSON of the message list (so the caller re-pairs deterministically). All
    rows across all contexts go in ONE vLLM batch (10-50x faster than per-row).
    Generated greedily (``temperature=0``) for on-policy determinism.
    """
    import os as _os

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(_os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    # max_model_len must outlive prompt + generation (gotchas.md max_model_len
    # tracks max_new_tokens); ~2x the gen cap + headroom for the longest prompt.
    if max_model_len is None:
        max_model_len = max(2048, 2 * max_new_tokens + 1024)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Flatten all (context, row) prompts into one batch; remember the order.
    keys: list[str] = []
    prompt_texts: list[str] = []
    for _ctx, rows in prompts_by_context.items():
        for msgs in rows:
            keys.append(json.dumps(msgs, sort_keys=True))
            prompt_texts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )

    llm_kwargs: dict[str, Any] = {
        "model": base_model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "seed": seed,
    }
    if adapter_path is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
        llm_kwargs["max_loras"] = 1
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens)
    lora_req = (
        LoRARequest(lora_name="syco_or_marker", lora_int_id=1, lora_path=adapter_path)
        if adapter_path is not None
        else None
    )
    out: dict[str, list[str]] = {}
    try:
        outputs = llm.generate(prompt_texts, sp, lora_request=lora_req, use_tqdm=False)
        for key, output in zip(keys, outputs, strict=True):
            out[key] = [output.outputs[0].text.strip()]
    finally:
        reap_vllm_engine(llm)
        del llm
        import gc as _gc

        _gc.collect()
        try:
            import torch as _torch

            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
        except Exception:
            pass
        import time as _time

        _time.sleep(1.0)
    return out
