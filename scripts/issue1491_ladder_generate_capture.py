#!/usr/bin/env python3
"""Task #1491 Phase-1: ladder generate + trimmed capture (per-scale, per-split).

Ported from ``origin/main:scripts/issue779_ffc_n1m_generate_capture.py``
@ d7c1c55fbe (branch tip content, landed on main via #1689 ba8359381c per
Unit 1 Deliverable A / epm:progress v6). Parametrizes:

- ``--model``  — one of the Qwen-2.5-Instruct ladder sizes.
- ``--layers`` — the per-scale depth-fraction-mapped layer list.
- ``--h-dim``  — hidden dim (auto-detected from AutoConfig when omitted).
- ``--split``  — one of ``train_25k`` / ``val_400`` / ``test_1000`` /
                 ``wc_test_1k`` / ``tierB_3600`` / ``ceiling_draw_43`` /
                 ``ceiling_draw_44``.
- ``--hf-prefix`` — child-issue prefix ``issue1491_scale_ladder/<scale>``
                 (NEVER the parent's; runtime-reuse clobber clause,
                 plan §10 item (i)).
- ``--capture-mode`` — ``coresident`` (default; ≤7B: vLLM engine + HF
                 capture model co-resident on the shard's GPU, the
                 parent's shape) OR ``phase_split_gen`` / ``phase_split_capture``
                 (14B/32B: two sub-invocations chained by the launch
                 script — gen only, then destroy the engine, then HF
                 capture pass on persisted responses).
- ``--capture-batch-size`` — batch size for the HF capture pass
                 (source-module throughput fix, plan §4.2 item (i)); a
                 run-start parity gate on 32 probe rows checks
                 batched-vs-per-row within cosine > 0.9999 and max
                 relative L2 < 1e-3, with the PROBE FORWARDS run in fp32
                 (model transiently cast + restored; the gate validates
                 BATCHING logic, not bf16 kernel noise — production
                 capture stays bf16; an infeasible fp32 cast fails LOUD,
                 round-3a M4); on a numeric failure the driver falls
                 back to per-row + logs a fail-loud WARN. Default 8 (safe
                 padded-batch shape).
- ``--first-chunk-self-gate`` — enable plan §7 Gate 1 (quick ridge fit +
                 shuffled null after ~2,000 captured rows; abort the
                 scale's job on failure via epm:failure sentinel).

Reads the ladder manifest from
``superkaiba1/explore-persona-space-data:issue1491_scale_ladder/manifest/<split>.jsonl``
(built by ``scripts/issue1491_ladder_manifest.py`` at Phase 0).

Persist-by-default (Upload Policy v2): rollout TEXT uploads unconditionally
on the non-LFS path (quota-immune) — in EVERY mode, including
``phase_split_gen`` (whose only output IS the rollout text); trimmed capture
tensors upload per K=20 chunks; the driver never discards generations or
capture tensors. ``phase_split_capture`` never generates and never re-writes
raw completions: it JOINS the gen wave's persisted per-chunk raw JSONs back
by context id, HUB-REQUIRED (round-3a M3): a chunk must already be on the
Hub before any copy is consumed — a local-only chunk (gen wave died before
flushing) FAILS LOUD instead of shipping .pt tensors whose published source
text a later gen resume could replace; the ``--no-upload`` local mode is the
explicit exception. Gen resume conversely SALVAGES local raw chunks
verbatim (re-upload, never regenerate).

Measurement contracts (Unit 2 blocker-fix round):

- The SPLIT's generation seed (``SPLIT_TO_MANIFEST``) is threaded into BOTH
  the vLLM engine and the per-request ``SamplingParams`` — ceiling draws
  43/44 sample fresh responses on the same 1,000 test contexts instead of
  reproducing the seed-42 ``test_1000`` generations.
- Teacher-forced capture inputs are built by concatenating PER-SEGMENT token
  ids (prompt render / response / turn-end tail) — never by re-tokenizing
  the concatenated string, whose BPE seam merges silently shift the
  ``cx_last`` position + ``v_x`` span (gotchas.md "Teacher-forced capture
  inputs").
- Every generation records the per-row ``finish_reason``; the shard digest
  reports the realized cap-hit fraction against the pre-registered >2%
  re-gen trigger.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Load .env BEFORE importing numpy/torch (shared-VM thread caps + HF_TOKEN;
# code-style.md § shared-VM CPU thread caps — numpy freezes its BLAS pool at
# import, so it must come AFTER load_dotenv() too).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub.errors import (  # noqa: E402
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

# Import parent-branch modules (per Unit 1 Deliverable A port-source
# decision, epm:progress v6: port_source: origin/main; no vendoring).
import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402

# Import ladder-local helpers from the Unit 1 manifest builder.
from issue1491_ladder_manifest import (  # noqa: E402
    LADDER_HF_PREFIX as MANIFEST_HF_PREFIX,
    LADDER_HF_REPO,
    OVERLENGTH_BUDGET,
    SPLIT_FILES,
)

# Import parent's over-length filter + rendered token length helper
# (Unit 1 Deliverable A signature-smoke: both PASS).
from issue779_ffc_n1m_generate_capture import (  # noqa: E402
    _filter_overlength_prompts,
    _rendered_prompt_token_len,
    _stack_chunk,
    _flush_upload_batch,
)

logger = logging.getLogger("issue1491_ladder_generate_capture")

# ---------------------------------------------------------------------------
# Constants (per plan v4)
# ---------------------------------------------------------------------------

# vLLM engine limits (parent parity; per plan §4.2).
MAX_MODEL_LEN = 8192
GEN_MAX_TOKENS = 1024
LENGTH_MARGIN = 64
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - GEN_MAX_TOKENS - LENGTH_MARGIN  # = 7104

# Sanity: my copy MUST agree with the ladder-manifest's budget (asserted
# at build time by issue1491_ladder_manifest.OVERLENGTH_BUDGET = 7104).
assert PROMPT_TOKEN_BUDGET == OVERLENGTH_BUDGET, (
    f"budget mismatch: driver {PROMPT_TOKEN_BUDGET} != manifest {OVERLENGTH_BUDGET}"
)

# Sampling params (parent parity, plan §11 "Generation recipe").
GEN_TEMP = 1.0
GEN_TOP_P = 0.95
GEN_SEED_DEFAULT = 42  # seed 43/44 rides ceiling_draw_{43,44} split arg

# Decoding-regime identity tags (greedy replication round, epm:progress v63).
# The tag is part of the resume/wave identity: it rides every raw-completions
# payload AND a per-stage-prefix marker file, so a greedy run can NEVER
# silently resume from / salvage / join temperature-1.0 chunks (or vice
# versa) — the exact hazard the salvage path's "never regenerate" contract
# warns about (a cross-regime reuse is invisible to the per-row prompt
# assert, which checks prompts, not responses).
SAMPLING_MODE_PARENT = "temp1.0_top_p0.95"  # the parent recipe (GEN_TEMP/GEN_TOP_P)
SAMPLING_MODE_GREEDY = "greedy_temp0"
SAMPLING_MARKER_NAME = "sampling_mode.json"


def _resolve_sampling(greedy: bool, gen_max_tokens: int = GEN_MAX_TOKENS) -> dict:
    """Resolve the run's decoding config + its identity tag.

    DEFAULT (greedy=False) is byte-identical to the parent recipe (the
    GEN_TEMP/GEN_TOP_P module constants). GREEDY (temperature 0) is the
    deterministic replication round: top_p is inert under greedy sampling and
    pinned to 1.0 for a clean identity tag. The split seed keeps threading
    into the engine + SamplingParams unchanged (inert under greedy), so the
    ceiling_draw_43/44 splits still execute as two INDEPENDENT generate
    passes — under greedy the two-draw ceiling MEASURES vLLM
    continuous-batching nondeterminism (expected ~1.0, never assumed).

    ``gen_max_tokens`` (the generation token CAP) is part of the decoding
    identity alongside ``mode`` (Path B cap-hit re-gen round): a 2048 re-gen
    row must never be confusable with a 1024 base row, so the cap rides the
    sampling dict into every payload/marker identity check
    (``_sampling_cap``). The driver itself always runs the module default;
    the cap-hit regen pass (``issue1491_caphit_regen.py``) resolves 2048.
    """
    if greedy:
        return {
            "mode": SAMPLING_MODE_GREEDY,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(gen_max_tokens),
        }
    return {
        "mode": SAMPLING_MODE_PARENT,
        "temperature": GEN_TEMP,
        "top_p": GEN_TOP_P,
        "max_tokens": int(gen_max_tokens),
    }


def _sampling_cap(sampling: dict) -> int:
    """The run's generation token cap — part of the DECODING IDENTITY.

    A sampling dict with no ``max_tokens`` key (hand-built or pre-regen-round)
    maps to the module default, which is EXACT for every artifact this driver
    ever wrote before the regen round: ``GEN_MAX_TOKENS`` was hardcoded, no
    override existed."""
    return int(sampling.get("max_tokens", GEN_MAX_TOKENS))


# Assistant turn-end tail appended after the response in the teacher-forced
# capture input (parent parity: COL.capture_answer_vector's v_x span is
# prompt_len:full_len of the full chat render, which ends with this tail).
IM_END_TAIL = "<|im_end|>\n"

# Pre-registered cap-hit re-gen trigger (CLAUDE.md: every generation stage
# reports its realized finish_reason=='length' fraction; >2% per cell ⇒
# re-generate cap-hit rows at >=2x the cap — an orchestrator decision, this
# driver reports + WARNs).
CAP_HIT_REGEN_TRIGGER = 0.02

# Sub-chunk (contexts per capture chunk file) — parent parity.
DEFAULT_SHARD_SIZE = 500

# K=20 upload-batch cadence — raised from parent K=10 for ≤48 concurrent
# shards this fleet runs; commit-rate arithmetic in plan §9 keeps fleet
# under the ~256 commits/hr account cap.
UPLOAD_BATCH = int(os.environ.get("EPM_LADDER_UPLOAD_BATCH", "20"))

# Ladder-manifest side: split → SPLIT_FILES key + generation seed.
SPLIT_TO_MANIFEST = {
    "train_25k": ("train_25k", GEN_SEED_DEFAULT),
    "val_400": ("val_400", GEN_SEED_DEFAULT),
    "test_1000": ("test_1000", GEN_SEED_DEFAULT),
    "wc_test_1k": ("wc_test_1k", GEN_SEED_DEFAULT),
    "tierB_3600": ("tierB_3600", GEN_SEED_DEFAULT),
    # Ceiling draws: seed 43/44 on the SAME 1,000 test contexts (plan §4.2).
    "ceiling_draw_43": ("test_1000", 43),
    "ceiling_draw_44": ("test_1000", 44),
}


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------


def _hf_api():
    from huggingface_hub import HfApi  # type: ignore

    return HfApi()


def _download_ladder_split(split_key: str, cache_dir: Path) -> list[dict]:
    """Download + read one split file of the ladder manifest.

    Returns the list of row dicts (with the ``ladder_local_id`` field the
    manifest builder wrote for stable ci mapping)."""
    from huggingface_hub import hf_hub_download  # type: ignore

    fname = SPLIT_FILES[split_key]
    # Routed through hub.retry_transient (workflow_lint --check-live-hf-retry-routing):
    # a transient 429/5xx here would otherwise kill every shard of a wave at
    # startup during a Hub storm; a genuinely-missing manifest still fails
    # loud (retry_transient re-raises the response-bearing 404 unchanged).
    local = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=LADDER_HF_REPO,
            filename=f"{MANIFEST_HF_PREFIX}/{fname}",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        ),
        what=f"hf_hub_download {MANIFEST_HF_PREFIX}/{fname}",
    )
    rows: list[dict] = []
    with open(local, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Model / engine setup
# ---------------------------------------------------------------------------


def _resolve_h_dim(model_id: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(model_id)
    return int(cfg.hidden_size)


def _build_capture_engine(model_id: str, seed: int) -> object | None:
    """Build the vLLM capture engine, honoring the H100 long-prompt hang /
    IMA mitigation ENV knobs (default OFF — the launch script sets them
    per plan §11 "enforce_eager + prefix-caching off"; commit 4cb9d6ea8d
    made these ENV-GATED in the parent driver, so the ladder driver MUST
    NOT re-hardcode them here).

    ``seed`` is the SPLIT's generation seed (SPLIT_TO_MANIFEST) — threaded
    into the engine so ceiling draws 43/44 do NOT reproduce the seed-42
    test_1000 generations (the two-draw reliability ceiling would read ~1.0
    instead of across-draw sampling variance). Per-request sampling seeds
    ride SamplingParams (_sampling_params); both carry the same value."""
    from explore_persona_space.eval.generation import create_vllm_engine

    llm_kwargs: dict = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        llm_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        llm_kwargs["enable_prefix_caching"] = False
    logger.info("[engine-knobs] %s engine_seed=%d", llm_kwargs, seed)
    return create_vllm_engine(model_id, max_model_len=MAX_MODEL_LEN, seed=int(seed), **llm_kwargs)


def _load_tokenizer(model_id: str):
    """Tokenizer-only load for phase_split_gen: the gen pass must NOT co-load
    the full HF model beside a vLLM engine of the same model — vLLM's
    gpu_memory_utilization is a fraction of TOTAL device memory, so the
    co-resident pair is a deterministic init failure/OOM at 14B/32B. Carries
    the same GENERATION_SUFFIX fail-loud probe as N10.load_models (the
    rendered-length filter + per-segment token ids rely on the template)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert tok.decode(tok(probe)["input_ids"][-3:]) == C.GENERATION_SUFFIX, (
        "tokenizer GENERATION_SUFFIX drift — expected the Qwen-2.5-Instruct chat template"
    )
    return tok


def _render_prompt(tok, prompt: str) -> str:
    """The EXACT prompt render vLLM generation consumes (and the render
    _rendered_prompt_token_len budgets against)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def _sampling_params(gen_seed: int, sampling: dict):
    """#779 pass-B sampling recipe with the SPLIT's seed threaded (never the
    parent's hardcoded seed=42 — ceiling draws ride 43/44) and the run's
    decoding config (``_resolve_sampling``) threaded (greedy round)."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1,
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_tokens=_sampling_cap(sampling),
        seed=int(gen_seed),
    )
    assert sp.seed == int(gen_seed), ("realized sampling seed drift", sp.seed, gen_seed)
    assert sp.max_tokens == _sampling_cap(sampling), (
        "realized max_tokens drift",
        sp.max_tokens,
        sampling,
    )
    assert sp.temperature == float(sampling["temperature"]), (
        "realized temperature drift",
        sp.temperature,
        sampling,
    )
    return sp


def _generate_seeded(
    llm, tok, prompts, gen_seed: int, sampling: dict
) -> tuple[list[str], list[str]]:
    """1 rollout per prompt with the resolved decoding config (vLLM, chunked),
    the split's generation seed threaded into SamplingParams.

    Returns ``(responses, finish_reasons)`` — ``finish_reason == 'length'``
    rows are cap-hits (CLAUDE.md cap-hit accounting). CPU-smoke path
    (llm is None) returns fixed stub responses through the SAME downstream
    capture code, finish_reason 'stop' — the SamplingParams construction +
    realized-sampling log below still run for REAL on that path, so the CPU
    smoke observes the realized temperature (greedy-engaged evidence)."""
    sp = _sampling_params(gen_seed, sampling)
    logger.info(
        "[ladder] generation: realized sampling mode=%s seed=%s temp=%s top_p=%s max_tokens=%s",
        sampling["mode"],
        sp.seed,
        sp.temperature,
        sp.top_p,
        sp.max_tokens,
    )
    if llm is None:  # --device cpu smoke: capture-path structural check only
        return (
            ["This is a short stub response for the CPU capture smoke."] * len(prompts),
            ["stop"] * len(prompts),
        )
    prompt_texts = [_render_prompt(tok, p) for p in prompts]
    texts: list[str] = []
    finish: list[str] = []
    n_chunks = (len(prompt_texts) + COL.VLLM_CHUNK_SIZE - 1) // COL.VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), COL.VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + COL.VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] ladder-generate chunk %d/%d (%d prompts, seed=%s)",
            i // COL.VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sp.seed,
        )
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)
        for o in chunk_out:
            texts.append(o.outputs[0].text)
            finish.append(str(o.outputs[0].finish_reason))
    return texts, finish


# ---------------------------------------------------------------------------
# Capture: per-row (parent parity) and batched (item (i) throughput fix)
# ---------------------------------------------------------------------------


def _is_empty_response(resp: str) -> bool:
    """Empty/whitespace-only responses carry no usable v_x span — drop the
    row (recorded), identically in batched AND per-row modes. NOTE the naive
    render-length check (`full_len <= prompt_len`) can NEVER fire: rendering
    an assistant turn appends the '<|im_end|>\\n' tail, so an empty response
    still adds 2 tokens — filter on response CONTENT instead."""
    return not resp.strip()


def _segment_token_ids(
    tok, prompt: str, response: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-segment token ids for the teacher-forced capture input.

    NEVER re-tokenize the concatenated string (gotchas.md 'Teacher-forced
    capture inputs: concatenate per-segment TOKEN IDS'): BPE merges at the
    prompt/response seam — e.g. a '\\n'-leading response merging into the
    template's trailing 'assistant\\n' — make full_ids[:n_prompt] !=
    prompt_ids and silently shift cx_last + the v_x span (answer tokens
    would leak into c(x), FAKING c(x)->v(x) predictivity). The forward input
    is torch.cat([prompt_ids, resp_ids, tail_ids]), so the prompt segment is
    bit-identical to what vLLM generation consumed BY CONSTRUCTION.

    - prompt_ids: the generation render tokenized EXACTLY as vLLM consumed
      it (add_special_tokens=False — the template carries its own special
      tokens; same call as _rendered_prompt_token_len).
    - resp_ids: the response text alone; len(resp_ids) is the response token
      count (excludes the turn-end tail).
    - tail_ids: the assistant turn-end tail '<|im_end|>\\n'. Parent parity:
      COL.capture_answer_vector's v_x span is prompt_len:full_len of the
      full chat render, which INCLUDES this tail — kept inside the v_x span
      here so the ladder matches the #779 anchor's v(x) convention.
    """
    prompt_text = _render_prompt(tok, prompt)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]
    t_ids = tok(IM_END_TAIL, add_special_tokens=False)["input_ids"]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert t_ids and t_ids[0] == im_end_id, ("turn-end tail tokenization drift", t_ids)
    return (
        torch.tensor(p_ids, dtype=torch.long),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(t_ids, dtype=torch.long),
    )


def _reduce_row(captured: dict, row_i: int, p_len: int, f_len: int, layers, h_dim):
    """cx_last + v_x for one row of an extract_layer_activations capture.

    cx_last = hidden state at the LAST prompt token (index p_len-1, the
    pre-generation state); v_x = mean over positions p_len..f_len-1 (the
    response + turn-end tail — parent parity, see _segment_token_ids).
    ``captured[li]`` is (B, T, H) from the OOM-safe hook with block-``li``-
    output semantics — the SAME helper the parent per-row path used, so
    layer indexing agrees by construction (``output_hidden_states[li]``
    would be off by one: it equals block li-1's output)."""
    cx_last_stack: list[torch.Tensor] = []
    v_x_stack: list[torch.Tensor] = []
    for li in layers:
        hs = captured[li][row_i]  # (T, H); right-pad positions >= f_len never read
        cx_last_stack.append(hs[p_len - 1, :].float().cpu())
        v_x_stack.append(hs[p_len:f_len, :].float().cpu().mean(dim=0))
    cx_last = torch.stack(cx_last_stack)  # (L, H)
    v_x = torch.stack(v_x_stack)  # (L, H)
    assert cx_last.shape == (len(layers), h_dim), ("cx_last", cx_last.shape)
    assert v_x.shape == (len(layers), h_dim), ("v_x", v_x.shape)
    return cx_last, v_x


def _capture_perrow(hf, tok, prompts, responses, cis, layers, h_dim):
    """Per-row capture (parity oracle + safe fallback for the batched path).

    Shares _segment_token_ids + _reduce_row with _capture_batched, so the two
    modes agree on token-id construction, span boundaries, and the
    empty-response drop set BY CONSTRUCTION — the parity gate then isolates
    padding/batching effects only.

    Returns (rows, dropped_cis) where rows =
    [{"ci", "prompt", "response", "cx_last": (L,H), "v_x": (L,H)}] and
    dropped_cis lists the empty/whitespace-response rows."""
    rows: list[dict] = []
    dropped: list[int] = []
    for p, resp, ci in zip(prompts, responses, cis, strict=True):
        if _is_empty_response(resp):
            dropped.append(int(ci))
            continue
        p_ids, r_ids, t_ids = _segment_token_ids(tok, p, resp)
        assert r_ids.shape[0] >= 1, ("non-empty response tokenized to 0 tokens", ci)
        input_ids = torch.cat([p_ids, r_ids, t_ids]).unsqueeze(0).to(hf.device)
        attn = torch.ones_like(input_ids)
        captured = extract_layer_activations(hf, input_ids, layers, attention_mask=attn)
        p_len = int(p_ids.shape[0])
        f_len = int(input_ids.shape[1])
        cx_last, v_x = _reduce_row(captured, 0, p_len, f_len, layers, h_dim)
        rows.append({"ci": int(ci), "prompt": p, "response": resp, "cx_last": cx_last, "v_x": v_x})
    return rows, dropped


def _capture_batched(hf, tok, prompts, responses, cis, layers, h_dim, batch_size):
    """Batched teacher-forced capture (plan §4.2 item (i)).

    Same per-segment token-id construction + row reduction as
    _capture_perrow (shared helpers); this function adds ONLY length-sorted
    RIGHT-padded batching. Right padding keeps every real token at positions
    0..f_len-1, so the default position_ids (arange) are correct for real
    tokens and the causal mask makes pad positions unreachable from them.

    Returns (rows, dropped_cis) — same shapes as _capture_perrow."""
    rows: list[dict] = []
    dropped: list[int] = []
    if not prompts:
        return rows, dropped

    # 1. Per-row segment ids; drop empty responses (recorded).
    seg: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k, (p, resp) in enumerate(zip(prompts, responses, strict=True)):
        if _is_empty_response(resp):
            dropped.append(int(cis[k]))
            continue
        p_ids, r_ids, t_ids = _segment_token_ids(tok, p, resp)
        assert r_ids.shape[0] >= 1, ("non-empty response tokenized to 0 tokens", cis[k])
        seg.append((k, p_ids, r_ids, t_ids))
    if not seg:
        return rows, dropped

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    # 2. Length-sort for padding efficiency (output order follows batches;
    #    downstream joins key on ci, not position).
    order = sorted(
        range(len(seg)),
        key=lambda i: int(seg[i][1].shape[0] + seg[i][2].shape[0] + seg[i][3].shape[0]),
    )
    for bs in range(0, len(order), batch_size):
        batch = [seg[i] for i in order[bs : bs + batch_size]]
        full_ids = [torch.cat([p_ids, r_ids, t_ids]) for _, p_ids, r_ids, t_ids in batch]
        p_lens = [int(p_ids.shape[0]) for _, p_ids, _r, _t in batch]
        f_lens = [int(x.shape[0]) for x in full_ids]

        max_len = max(f_lens)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_i, ids in enumerate(full_ids):
            padded[row_i, : ids.shape[0]] = ids
            attn[row_i, : ids.shape[0]] = 1

        padded = padded.to(hf.device)
        attn = attn.to(hf.device)
        captured = extract_layer_activations(hf, padded, layers, attention_mask=attn)

        for row_i, (k, _p, _r, _t) in enumerate(batch):
            cx_last, v_x = _reduce_row(captured, row_i, p_lens[row_i], f_lens[row_i], layers, h_dim)
            rows.append(
                {
                    "ci": int(cis[k]),
                    "prompt": prompts[k],
                    "response": responses[k],
                    "cx_last": cx_last,
                    "v_x": v_x,
                }
            )
    return rows, dropped


def _assert_fp32_probe_feasible(hf) -> None:
    """Fail LOUD when the in-place fp32 cast for the parity probe cannot fit.

    Round-3a M4: the probe forwards run in fp32 (plan §4.2 registers the
    bars "in fp32"), which requires transiently casting the bf16 model to
    fp32 — an extra ~2 bytes/param of device memory (module._apply frees
    each bf16 tensor as it converts, so peak extra ≈ fp32_total − bf16_total)
    plus activation headroom for the 32 probe rows. When that does not fit
    (32B fp32 weights are ~122 GiB — infeasible on A100-80, marginal on
    H200-141), the driver STOPS here rather than silently reverting the
    probe to bf16 or loosening the registered bars (M4 constraint): #779
    measured bf16 batched-vs-per-row deviations (block-26 cos 0.996907)
    far below the registered fp32 bars, so a bf16 probe would fail on
    essentially every healthy GPU run and silently forfeit batched capture.
    """
    dev = next(hf.parameters()).device
    if dev.type != "cuda":
        return  # CPU cast is RAM-bound; tiny smoke models always fit.
    cur_bytes = sum(p.numel() * p.element_size() for p in hf.parameters())
    cur_bytes += sum(b.numel() * b.element_size() for b in hf.buffers())
    fp32_bytes = sum(p.numel() * 4 for p in hf.parameters())
    fp32_bytes += sum(b.numel() * 4 for b in hf.buffers())
    margin_bytes = int(float(os.environ.get("EPM_LADDER_FP32_PROBE_MARGIN_GIB", "8")) * (1 << 30))
    torch.cuda.empty_cache()  # mem_get_info excludes allocator-cached blocks
    free, total = torch.cuda.mem_get_info(dev)
    need = (fp32_bytes - cur_bytes) + margin_bytes
    if free < need:
        raise RuntimeError(
            "fp32 parity probe infeasible on this GPU: casting the model to fp32 needs "
            f"~{(fp32_bytes - cur_bytes) / (1 << 30):.1f} GiB extra + "
            f"{margin_bytes / (1 << 30):.1f} GiB activation margin, but only "
            f"{free / (1 << 30):.1f} GiB of {total / (1 << 30):.1f} GiB is free. "
            "Refusing to run the probe in bf16 or retune the registered fp32 bars "
            "(plan §4.2 / round-3a M4) — run this shard on a larger-memory GPU, or "
            "lower EPM_LADDER_FP32_PROBE_MARGIN_GIB only with a measured activation "
            "footprint."
        )


def _batched_capture_parity_gate(
    hf, tok, prompts, responses, cis, layers, h_dim, batch_size
) -> tuple[bool, str]:
    """Plan §4.2 item (i) parity gate: on 32 probe rows, batched vs per-row
    capture must agree per-field cosine > 0.9999 and max relative L2 error
    < 1e-3 in fp32. On failure: return (False, reason) — caller falls back
    to per-row and logs a fail-loud WARN.

    Round-3a M4: the PROBE FORWARDS run in fp32 — the model is temporarily
    cast to fp32 for the 32 probe rows (both legs, SAME batched/per-row
    code paths as production; only the dtype differs), then restored
    bit-exactly. NOTE the restore is bit-exact only BECAUSE floating-point
    buffers are snapshotted and reassigned (round-4a B1): parameters do
    round-trip through bf16 -> fp32 -> bf16 exactly, but fp32 BUFFERS do
    not — Qwen2's rotary inv_freq is fp32 even on a bf16 model, and a bare
    round-trip degrades it permanently (3.653e-3 max rel err), leaving
    production on a different RoPE than the probe validated. Do not drop
    the snapshot as redundant. PRODUCTION
    capture remains bf16: the gate exists to catch BATCHING bugs (masking,
    pooling, the last-prompt-token gather — plan §10 risk row), not to
    measure bf16 kernel reproducibility, which is expected padded-batch
    noise (#779: block-26 cos 0.996907 with a bug-free path) and sits below
    the registered bars by design. An infeasible fp32 cast (or an OOM
    during the fp32 probe) FAILS LOUD instead of silently reverting to
    bf16 / falling back to per-row — see _assert_fp32_probe_feasible.

    32 rows chosen per plan; we accept fewer if the caller passed fewer.
    """
    n = min(32, len(prompts))
    if n == 0:
        return True, "empty probe (nothing to check)"
    p = prompts[:n]
    r = responses[:n]
    ci = cis[:n]
    orig_dtype = next(hf.parameters()).dtype
    cast = orig_dtype != torch.float32
    saved_buffers: dict[str, torch.Tensor] = {}
    if cast:
        _assert_fp32_probe_feasible(hf)  # raises — never a bf16 fallback (M4)
        # Snapshot floating-point BUFFERS before the cast, preserving dtype.
        #
        # Module.to(dtype) casts floating-point buffers as well as parameters,
        # and Qwen2's rotary `inv_freq` is fp32 EVEN ON a bf16-loaded model. A
        # bare fp32 -> bf16 round-trip therefore leaves inv_freq permanently
        # DEGRADED to bf16 (measured on the pinned stack: max rel err 3.65e-3),
        # so every production capture after the probe would run a numerically
        # different RoPE than the parent/#779 convention — and the probe legs
        # themselves would run clean fp32 inv_freq, meaning the gate certifies
        # a rig production does not use. Params alone round-trip exactly, which
        # is why a params-only equality check misses this entirely.
        saved_buffers = {
            name: buf.detach().clone()
            for name, buf in hf.named_buffers()
            if buf.is_floating_point()
        }
    try:
        if cast:
            hf.to(torch.float32)
        try:
            rows_serial, drop_serial = _capture_perrow(hf, tok, p, r, ci, layers, h_dim)
            rows_batched, drop_batched = _capture_batched(
                hf, tok, p, r, ci, layers, h_dim, batch_size
            )
        except torch.cuda.OutOfMemoryError as e:
            # Never swallow an fp32-probe OOM into the per-row fallback:
            # that would silently forfeit batched capture (M4 constraint).
            raise RuntimeError(
                "fp32 parity probe OOM despite passing the feasibility check — raise "
                "EPM_LADDER_FP32_PROBE_MARGIN_GIB or use a larger-memory GPU; do NOT "
                f"revert the probe to bf16 (plan §4.2 / round-3a M4). Original: {e}"
            ) from e
        except Exception as e:  # noqa: BLE001
            return False, f"probe crashed: {type(e).__name__}: {e}"
    finally:
        if cast:
            hf.to(orig_dtype)
            # Restore buffers by ASSIGNMENT, not copy_: hf.to() has already
            # rewritten them to orig_dtype, and copying fp32 values into a bf16
            # tensor would preserve the degraded dtype. Assigning through the
            # owning module's setattr writes back into its _buffers entry,
            # restoring the original dtype AND bits.
            for name, buf in saved_buffers.items():
                parent_path, _, leaf = name.rpartition(".")
                owner = hf.get_submodule(parent_path) if parent_path else hf
                setattr(owner, leaf, buf.to(device=getattr(owner, leaf).device))
            if next(hf.parameters()).is_cuda:
                torch.cuda.empty_cache()

    if set(drop_serial) != set(drop_batched):
        # By construction (shared _is_empty_response) this cannot differ;
        # defense in depth against future drift between the two paths.
        return False, (
            f"empty-drop mismatch: serial={sorted(drop_serial)} batched={sorted(drop_batched)}"
        )
    by_ci_batched = {row["ci"]: row for row in rows_batched}
    matched = 0
    max_cos_dev = 0.0
    max_rel_l2 = 0.0
    for rs in rows_serial:
        rb = by_ci_batched.get(rs["ci"])
        if rb is None:
            continue
        for field in ("cx_last", "v_x"):
            a = rs[field].float().flatten()
            b = rb[field].float().flatten()
            dot = float((a * b).sum())
            na = float(a.norm())
            nb = float(b.norm())
            cos = dot / (na * nb + 1e-30)
            l2 = float((a - b).norm())
            rel = l2 / (na + 1e-30)
            max_cos_dev = max(max_cos_dev, 1.0 - cos)
            max_rel_l2 = max(max_rel_l2, rel)
        matched += 1
    # probe_dtype in the verdict string is the observable that the fp32 leg
    # actually ran (round-3a M4 smoke evidence); cast=False means the model
    # was already fp32 (CPU smokes).
    dtype_note = f"probe_dtype=float32 (cast={cast}, restored={orig_dtype})"
    if matched == 0:
        return False, f"no matching rows between serial + batched probes [{dtype_note}]"
    if 1.0 - max_cos_dev < 0.9999:
        return False, f"cosine gate FAIL: min cos={1.0 - max_cos_dev:.6f} < 0.9999 [{dtype_note}]"
    if max_rel_l2 >= 1e-3:
        return False, f"rel-L2 gate FAIL: max rel-L2={max_rel_l2:.3e} >= 1e-3 [{dtype_note}]"
    return (
        True,
        f"PASS: {matched} rows, min cos={1.0 - max_cos_dev:.6f}, "
        f"max rel-L2={max_rel_l2:.3e} [{dtype_note}]",
    )


# ---------------------------------------------------------------------------
# First-chunk self-gate (plan §7 Decision Gate 1)
# ---------------------------------------------------------------------------

# ---- Plan §7 Gate 1: three-valued verdict (PASS / INCONCLUSIVE / FAIL) ----
#
# The gate's null is a shuffle-FIT null — beta is REFIT on permuted (X, Y)
# pairings, then scored on val — whose expected R² is ~ -1, NOT ~ 0: with the
# pairings destroyed, the refit beta captures spurious structure rather than
# collapsing to zero, so val predictions scatter around y_mu and
# SSE_null > SST. (An R² ~ 0 null is the MEAN-PREDICTOR null, yhat == y_mu
# exactly => SSE == SST — a DIFFERENT construction; the original two-sided
# |r2_null| < 0.05 bound encoded that wrong expectation and was unsatisfiable
# on ANY input — the 2026-08-05 incident that aborted all 8 train_25k shards,
# epm:failure v3.)
#
# PRODUCTION CALIBRATION POINTS (all real, driving every threshold below —
# do NOT retune by taste):
#
#   rung   trigger              n_train  h     n/h    r2_null          r2_fit
#   0.5B   mid-shard            1600     896   1.79   -0.99 .. -0.83   +0.09 .. +0.16
#   0.5B   end-of-shard resume   900     896   1.004  -1.35 .. -1.28   -0.04 .. -0.02
#   1.5B   mid-shard            1600     1536  1.04   -3.80 .. -3.40   -0.73 .. -0.55
#
# Two structural facts these points establish:
# (1) The null's magnitude scales with h ITSELF, not n/h alone — at the same
#     n/h ≈ 1.0 the null reads -1.3 at h=896 but -3.6 at h=1536. So neither a
#     fixed floor NOR a ratio-scaled floor on r2_null can be a criterion: a
#     very negative null is the EXPECTED signature of a degenerate
#     shuffle-REFIT under poor conditioning, not pathology.
# (2) At n_train ≈ h, BOTH statistics are estimator-degenerate (#1701):
#     r2_fit collapses to <= 0 regardless of h (-0.02 at h=896, -0.73 at
#     h=1536) while the same capture code reads +0.09..+0.16 at n/h = 1.79.
#     Near n ≈ h the gate therefore has NO discriminating power in EITHER
#     direction — a broken capture and a healthy one both produce
#     fit ≈ 0 / null « 0 / gap » 0.05 — so a binary PASS/FAIL cannot express
#     the honest outcome. Hence the three-valued verdict:
#
#   PASS          gap > 0.05 AND r2_fit > 0 AND adequate conditioning
#                 (n_train > SELF_GATE_ADEQUATE_COND * h).
#   INCONCLUSIVE  conditioning inadequate, OR (at adequate conditioning)
#                 gap > 0.05 with r2_fit <= 0 — the gap carried entirely by
#                 the degenerate null. NEVER aborts; logged loudly with the
#                 full conditioning disclosure so the record shows the gate
#                 could not discriminate.
#   FAIL          at ADEQUATE conditioning only: gap <= 0.05 (fit fails to
#                 beat the shuffled null — the real broken-capture
#                 signature) or a PREDICTIVE null (r2_null >= 0.05 —
#                 leakage/degeneracy; a shuffle-fit null's expectation is
#                 <= 0 in every regime). Still aborts the rung.
#
# SELF_GATE_ADEQUATE_COND = 1.5: the observed clusters are n/h ≈ 1.0
# (degenerate: fit <= 0 at BOTH h=896 and h=1536) and n/h = 1.79 (the only
# genuinely positive fit reads). 1.5 separates them; no production trigger
# regime sits near the boundary (mid-shard n_train=1600: 1.79 for 0.5B,
# <= 1.04 for every larger rung; resume fallbacks land ~= 1.0), so the exact
# value between 1.04 and 1.79 is not load-bearing.
SELF_GATE_ADEQUATE_COND = 1.5

# SELF_GATE_MIN_FIT_FOR_PASS = 0.0: the observed positive reads (+0.09..+0.16)
# vs the observed degenerate reads (-0.73..-0.02) — 0 separates the clusters
# with margin on both sides. This is NOT a science-quality floor on the
# ladder's R² (a fit <= 0 at adequate conditioning yields INCONCLUSIVE and
# the rung PROCEEDS, disclosed — never FAIL): it only stops the GATE from
# claiming a positive verdict when its own fit is non-positive, i.e. when the
# gap is carried entirely by the null being bad.
SELF_GATE_MIN_FIT_FOR_PASS = 0.0

# ADVISORY pathology tripwire on the null — log-only, NEVER a verdict input,
# and NOT a calibration knob (do not retune it per rung; fact (1) above is
# why no such constant can be calibrated). Set far below the plausible
# legitimate envelope: deepest measured legitimate nulls are -3.8 (production,
# h=1536, n/h=1.04) and -51 (worst-case synthetic iid Gaussian at the exact
# interpolation threshold n_train = h); h≈5120 rungs plausibly reach past
# -10. A breach at -100 means the numbers are absurd (broken
# centering/scaling, NaN-adjacent inputs), not a deep-but-legitimate null.
SELF_GATE_NULL_FLOOR = -100.0


def _self_gate_verdict(r2_fit: float, r2_null: float, n_train: int, h: int) -> tuple[str, str]:
    """Plan §7 Gate 1 three-valued verdict (see the calibration block above).

    Returns ``(verdict, reason)`` with verdict in
    {"PASS", "INCONCLUSIVE", "FAIL"}. Only FAIL aborts the rung.

    The binding test is the plan-registered GAP, ``(r2_fit - r2_null) >
    0.05`` (byte-unchanged) — but it is only a DISCRIMINATING test at
    adequate conditioning: near n_train ≈ h both R² statistics are
    estimator-degenerate (#1701) and broken vs healthy capture produce the
    same signature, so every outcome there is INCONCLUSIVE (disclosed,
    never aborted). At adequate conditioning, a failed gap (or a predictive
    null, whose expectation is <= 0 in every regime) is a genuine defect
    and FAILs; a passed gap whose own fit is non-positive cannot claim
    PASS (the gap is then carried entirely by the degenerate null) and is
    INCONCLUSIVE. Deliberately NO science-quality floor on r2_fit — the
    ladder's smallest rungs are EXPECTED to read low, and a fit <= 0 never
    FAILs on its own; it only demotes PASS to INCONCLUSIVE.
    """
    gap = r2_fit - r2_null
    gap_ok = (r2_fit - r2_null) > 0.05
    if n_train <= SELF_GATE_ADEQUATE_COND * h:
        return (
            "INCONCLUSIVE",
            f"conditioning inadequate (n_train={n_train} <= "
            f"{SELF_GATE_ADEQUATE_COND}x h={h}): both R2 statistics are "
            "estimator-degenerate (#1701) — the gate cannot discriminate "
            "broken from healthy capture in either direction here",
        )
    if r2_null >= 0.05:
        return (
            "FAIL",
            f"shuffled-pairing null is predictive (r2_null={r2_null:.3f} >= 0.05) "
            "at adequate conditioning — leakage/degeneracy signature",
        )
    if not gap_ok:
        return (
            "FAIL",
            f"fit fails to beat the shuffled null (gap={gap:.3f} <= 0.05) "
            "at adequate conditioning — the broken-capture signature",
        )
    if r2_fit <= SELF_GATE_MIN_FIT_FOR_PASS:
        return (
            "INCONCLUSIVE",
            f"gap={gap:.3f} is carried entirely by the degenerate null "
            f"(r2_fit={r2_fit:.3f} <= {SELF_GATE_MIN_FIT_FOR_PASS}) — a "
            "positive verdict cannot rest on a non-positive fit",
        )
    return (
        "PASS",
        f"gap={gap:.3f} > 0.05 with positive fit (r2_fit={r2_fit:.3f}) at adequate conditioning",
    )


def _first_chunk_self_gate(rows: list[dict], layer_index_primary: int) -> tuple[bool, dict]:
    """Quick numpy ridge fit + shuffled-pairing null on the first ~2,000
    captured rows at the primary layer.

    Verdict via ``_self_gate_verdict(r2_fit, r2_null, n_train, h)`` —
    three-valued PASS / INCONCLUSIVE / FAIL (see the calibration block
    above the verdict function). Returns ``(passed, diag)`` where
    ``passed = verdict != "FAIL"`` (only FAIL aborts; INCONCLUSIVE
    proceeds, loudly disclosed) and ``diag`` carries the verdict, its
    reason, and the #1701 conditioning disclosure (h, n_train_over_h,
    under_determined).
    """
    if len(rows) < 500:
        return True, {"skipped": True, "reason": f"only {len(rows)} rows (< 500)"}
    Xs = np.stack([r["cx_last"][layer_index_primary].numpy() for r in rows])  # (n, H)
    Ys = np.stack([r["v_x"][layer_index_primary].numpy() for r in rows])  # (n, H)
    # 80/20 train/val split (deterministic).
    n = len(rows)
    n_train = int(0.8 * n)
    Xtr, Xva = Xs[:n_train], Xs[n_train:]
    Ytr, Yva = Ys[:n_train], Ys[n_train:]

    # Center + ridge (fixed lambda; this is a validity gate, not a fit).
    x_mu = Xtr.mean(axis=0, keepdims=True)
    y_mu = Ytr.mean(axis=0, keepdims=True)
    Xtr_c = Xtr - x_mu
    Ytr_c = Ytr - y_mu

    h = Xtr_c.shape[1]
    lam = 1.0
    # β = (Xtr'Xtr + λI)^-1 Xtr'Ytr — computed with float64 for stability.
    XtX = Xtr_c.astype(np.float64).T @ Xtr_c.astype(np.float64)
    XtY = Xtr_c.astype(np.float64).T @ Ytr_c.astype(np.float64)
    A = XtX + lam * np.eye(h)
    beta = np.linalg.solve(A, XtY).astype(np.float32)

    yhat = (Xva - x_mu) @ beta + y_mu
    sse = float(((Yva - yhat) ** 2).sum())
    sst = float(((Yva - Yva.mean(axis=0, keepdims=True)) ** 2).sum())
    r2_fit = 1.0 - sse / (sst + 1e-30)

    # Null: shuffle the row permutation, refit, re-score.
    rng = np.random.default_rng(1491)
    perm = rng.permutation(len(Ytr))
    XtY_null = Xtr_c.astype(np.float64).T @ Ytr_c[perm].astype(np.float64)
    beta_null = np.linalg.solve(A, XtY_null).astype(np.float32)
    yhat_null = (Xva - x_mu) @ beta_null + y_mu
    sse_null = float(((Yva - yhat_null) ** 2).sum())
    r2_null = 1.0 - sse_null / (sst + 1e-30)

    # Conditioning disclosure (#1701): at n_train <= 1.5x h the ridge fit is
    # estimator-degenerate as a SIGNAL read and the gate cannot discriminate
    # (see the calibration block above _self_gate_verdict) — the regime must
    # be disclosed, never silent. These fields ride the "[ladder-gate]" log
    # line via the printed diag.
    n_over_h = float(n_train) / float(h)
    under_determined = bool(n_train < h)
    verdict, reason = _self_gate_verdict(r2_fit, r2_null, n_train, h)
    diag = {
        "verdict": verdict,
        "verdict_reason": reason,
        "n_train": int(n_train),
        "n_val": int(len(Yva)),
        "h": int(h),
        "n_train_over_h": round(n_over_h, 3),
        "under_determined": under_determined,
        "r2_fit": r2_fit,
        "r2_null": r2_null,
        "gap": r2_fit - r2_null,
        "null_floor_advisory": SELF_GATE_NULL_FLOOR,
        "null_floor_breached": bool(r2_null <= SELF_GATE_NULL_FLOOR),
    }
    # Only FAIL aborts; INCONCLUSIVE proceeds (loudly disclosed by the caller).
    passed = verdict != "FAIL"
    diag["passed"] = passed
    if diag["null_floor_breached"]:
        # Advisory pathology tripwire ONLY — never a verdict input, never a
        # calibration knob (see the SELF_GATE_NULL_FLOOR rationale). A breach
        # means the numbers are absurd, not a deep-but-legitimate null.
        logger.warning(
            "[ladder-gate] ADVISORY: r2_null=%.3f breaches the pathology tripwire %.1f "
            "(n_train/h=%.2f, under_determined=%s) — NOT a verdict input; "
            "numerically absurd null, inspect the capture inputs.",
            r2_null,
            SELF_GATE_NULL_FLOOR,
            n_over_h,
            under_determined,
        )
    return passed, diag


# ---------------------------------------------------------------------------
# Run capture: per-scale, per-split
# ---------------------------------------------------------------------------


def _resolve_layers_arg(layers_arg: str) -> list[int]:
    """Parse ``--layers`` as a comma-separated integer list."""
    parts = [p.strip() for p in layers_arg.split(",") if p.strip()]
    ints = [int(p) for p in parts]
    if not ints:
        raise ValueError(f"--layers must be non-empty, got {layers_arg!r}")
    return ints


def _split_shard_range(n_total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    # Even split; last shard picks up any remainder — parent parity via
    # N50._shard_range's semantics (defined on N50, not N10).
    return N50._shard_range(n_total, num_shards, shard_index)  # noqa: SLF001


def _remote_index(hf_prefix: str, subdir: str) -> set[str]:
    """List the leaf filenames already on HF under ``hf_prefix/subdir``.

    Rides ``hub.retry_transient`` — a transient 429/5xx must NOT silently
    read as "nothing uploaded" (that disables resume for the whole
    (shard, split) and re-runs everything). The ONE legitimately-empty case
    is a 404 on a prefix no upload has created yet; a
    ``RepositoryNotFoundError`` (typo'd repo id) stays loud, and every other
    error propagates (fail fast — the crash IS the signal)."""
    api = _hf_api()
    prefix = f"{hf_prefix}/{subdir}"

    def _list():
        # Materialize INSIDE the retry: list_repo_tree is a LAZY generator —
        # the HTTP error raises at iteration time (gotchas.md, #779 n50k).
        return list(
            # HUB_VERIFY_RETRY_EXEMPT: _list is passed to hub.retry_transient below
            api.list_repo_tree(
                repo_id=LADDER_HF_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                recursive=True,
            )
        )

    try:
        entries = hub.retry_transient(_list, what=f"list_repo_tree {prefix}")
    except RepositoryNotFoundError:
        raise  # load-bearing ordering: subclass of HfHubHTTPError, must stay loud
    except EntryNotFoundError:
        return set()  # prefix not yet created — expected before the first upload
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            return set()  # same legitimate not-yet-created case
        raise
    return {e.path.split("/")[-1] for e in entries if not e.path.endswith("/")}


def _assert_raw_payload_matches(
    payload: dict,
    raw_name: str,
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
    expect_sampling_mode: str,
    expect_gen_max_tokens: int = GEN_MAX_TOKENS,
) -> None:
    """Wave-alignment asserts shared by the capture-side join and the
    gen-resume salvage path: the consumer must iterate the SAME manifest
    slice under the SAME --num-shards/--shard-index/--shard-size arithmetic
    as the gen run that wrote this chunk, the SAME split seed, and the SAME
    decoding regime — a mismatch would silently mispair responses with
    contexts (or, for the regime, feed one regime's text into the other's
    captures/salvage: the greedy-vs-temperature-1.0 corruption hazard).

    LEGACY chunks (no ``sampling_mode`` key) uniquely map to
    ``SAMPLING_MODE_PARENT``: every chunk this driver wrote before the greedy
    round was produced under the hardcoded parent constants (GEN_TEMP=1.0 /
    GEN_TOP_P=0.95 — no CLI override existed), so the missing-key default is
    exact, and a greedy run still fails LOUD against any legacy chunk.

    ``expect_gen_max_tokens`` extends the SAME identity to the generation
    token CAP (Path B cap-hit re-gen round): a 2048 re-gen chunk must never
    join/salvage/resume a 1024 run and vice versa. Legacy/no-key payloads map
    to ``GEN_MAX_TOKENS`` — exact, the cap was hardcoded before the regen
    round (and every greedy base chunk wrote the key explicitly)."""
    assert int(payload["shard_index"]) == expect_shard_index, (
        "gen/capture shard mismatch",
        raw_name,
        payload["shard_index"],
        expect_shard_index,
    )
    assert int(payload["chunk"]) == expect_chunk, (
        "gen/capture chunk mismatch",
        raw_name,
        payload["chunk"],
        expect_chunk,
    )
    assert payload["split"] == expect_split, (
        "gen/capture split mismatch",
        raw_name,
        payload["split"],
        expect_split,
    )
    assert int(payload["seed"]) == expect_seed, (
        "gen/capture seed mismatch",
        raw_name,
        payload["seed"],
        expect_seed,
    )
    payload_mode = str(payload.get("sampling_mode", SAMPLING_MODE_PARENT))
    assert payload_mode == expect_sampling_mode, (
        "gen/capture SAMPLING-MODE mismatch — refusing to join/salvage/resume across "
        "decoding regimes (a temperature-1.0 chunk must never feed a greedy run, and "
        "vice versa; use a fresh --hf-prefix + --out-dir per regime)",
        raw_name,
        payload_mode,
        expect_sampling_mode,
    )
    payload_cap = int(payload.get("gen_max_tokens", GEN_MAX_TOKENS))
    assert payload_cap == int(expect_gen_max_tokens), (
        "gen/capture GEN-MAX-TOKENS (cap) mismatch — the token cap is part of the "
        "decoding identity: a 1024 base chunk must never feed a 2048 re-gen run, and "
        "vice versa (use the regen namespace + a fresh --out-dir per cap)",
        raw_name,
        payload_cap,
        int(expect_gen_max_tokens),
    )


def _load_local_raw_salvage(
    scratch: Path,
    raw_name: str,
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
    expect_sampling_mode: str,
    expect_gen_max_tokens: int = GEN_MAX_TOKENS,
) -> dict | None:
    """Return the LOCAL gen raw-chunk payload for salvage, or None if absent.

    Round-3a M3 (gen side): a raw chunk that exists locally but never
    reached the Hub (gen wave SIGKILLed with up to UPLOAD_BATCH unflushed
    chunks) is re-UPLOADED verbatim on gen resume — NEVER regenerated. A
    temperature-1.0 regeneration would publish raw text silently diverging
    from any ``.pt`` already captured from the local text (invisible to the
    per-row prompt assert, which checks prompts, not responses). The local
    file is a complete atomic write (``C.write_json_atomic``: tmp + rename),
    so existence implies integrity; the wave-alignment asserts pin the
    config so a foreign / stale scratch file fails loud instead of being
    silently reused."""
    local = scratch / raw_name
    if not local.exists():
        return None
    with open(local, encoding="utf-8") as fh:
        payload = json.load(fh)
    _assert_raw_payload_matches(
        payload,
        raw_name,
        expect_split=expect_split,
        expect_seed=expect_seed,
        expect_shard_index=expect_shard_index,
        expect_chunk=expect_chunk,
        expect_sampling_mode=expect_sampling_mode,
        expect_gen_max_tokens=expect_gen_max_tokens,
    )
    return payload


def _load_persisted_gen_chunk(
    scratch: Path,
    stage_prefix: str,
    raw_name: str,
    cache_dir: Path,
    done_raw: set[str],
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
    expect_sampling_mode: str,
    expect_gen_max_tokens: int = GEN_MAX_TOKENS,
    allow_local_only: bool = False,
) -> dict[int, dict]:
    """Load ONE gen-wave raw-completions chunk for ``phase_split_capture``.

    HUB-REQUIRED (round-3a M3, capture side): unless ``allow_local_only``
    (the explicit ``--no-upload`` local mode), the chunk MUST already be on
    the Hub before any copy is consumed — a local file whose chunk is not
    in ``done_raw`` is a gen wave that died before flushing (up to
    UPLOAD_BATCH chunks can be local-only after a SIGKILL). Capturing from
    such text would ship ``.pt`` tensors whose published source text a later
    gen resume could silently replace, so it FAILS LOUD here instead. When
    the chunk IS on the Hub, a surviving local copy is byte-identical by the
    upload contract (``_flush_upload_batch`` uploads the local file itself
    and purges only after verify), so local-first reading stays safe and
    avoids a re-download. A chunk that is neither local nor on the Hub means
    the gen wave is incomplete (or the prefix/shard config is wrong) — the
    capture wave must crash, never silently skip rows.

    The wave-alignment asserts (``_assert_raw_payload_matches``) pin the
    join contract; see that helper.

    Returns ``{ci: row}`` (row = {"ci", "prompt", "response",
    "finish_reason"}) keyed by the manifest context id."""
    local = scratch / raw_name
    on_hub = raw_name in done_raw
    if not on_hub and not allow_local_only:
        if local.exists():
            raise RuntimeError(
                f"phase_split_capture: {raw_name} exists locally ({local}) but is NOT on the "
                f"Hub under {stage_prefix}/raw_completions — the gen wave died before flushing "
                "this chunk. Refusing to capture from local-only text (round-3a M3 divergence "
                "guard): a later gen resume would otherwise re-upload/regenerate the published "
                "raw text out from under the shipped .pt. Re-run the phase_split_gen wave — its "
                "resume re-uploads existing local raw chunks verbatim (never regenerates) — "
                "then re-run this capture wave."
            )
        raise RuntimeError(
            f"phase_split_capture: gen-wave raw completions missing for {raw_name} — "
            f"neither local ({local}) nor on Hub under {stage_prefix}/raw_completions. "
            "Run the phase_split_gen wave to completion first (launcher wave sequencing)."
        )
    if not local.exists():
        if not on_hub:
            # allow_local_only mode with no local file AND nothing on the Hub.
            raise RuntimeError(
                f"phase_split_capture (--no-upload): {raw_name} not in local scratch "
                f"{scratch} and not on the Hub — run the --no-upload gen wave first."
            )
        from huggingface_hub import hf_hub_download  # type: ignore

        local = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    repo_id=LADDER_HF_REPO,
                    filename=f"{stage_prefix}/raw_completions/{raw_name}",
                    repo_type="dataset",
                    cache_dir=str(cache_dir),
                ),
                what=f"hf_hub_download {stage_prefix}/raw_completions/{raw_name}",
            )
        )
    with open(local, encoding="utf-8") as fh:
        payload = json.load(fh)
    _assert_raw_payload_matches(
        payload,
        raw_name,
        expect_split=expect_split,
        expect_seed=expect_seed,
        expect_shard_index=expect_shard_index,
        expect_chunk=expect_chunk,
        expect_sampling_mode=expect_sampling_mode,
        expect_gen_max_tokens=expect_gen_max_tokens,
    )
    rows = {int(r["ci"]): r for r in payload["rows"]}
    assert len(rows) == len(payload["rows"]), f"{raw_name}: duplicate ci in gen rows"
    return rows


def _download_hub_sampling_marker(stage_prefix: str, cache_dir: Path) -> dict | None:
    """Read the stage-prefix sampling-marker PAYLOAD off the Hub, or None if absent.

    Returns the full marker dict (``sampling_mode`` + ``gen_max_tokens`` + ...)
    so the identity guard can compare BOTH identity fields; missing keys map
    to the parent defaults (exact for every pre-regen-round marker).

    A ``LocalEntryNotFoundError`` (a 429 storm masking as a 404 — #1092/#1402
    class) is re-raised after ``retry_transient`` exhausts: "could not
    determine" must NEVER read as "absent" (order matters — it SUBCLASSES
    ``EntryNotFoundError``). A genuine response-bearing 404 means the marker
    was never uploaded (fresh or legacy prefix)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        p = hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=LADDER_HF_REPO,
                filename=f"{stage_prefix}/{SAMPLING_MARKER_NAME}",
                repo_type="dataset",
                cache_dir=str(cache_dir),
            ),
            what=f"hf_hub_download {stage_prefix}/{SAMPLING_MARKER_NAME}",
        )
    except RepositoryNotFoundError:
        raise  # typo'd repo id stays loud
    except LocalEntryNotFoundError:
        raise  # transport-masked miss: unknown, NOT absent (fail-safe loud)
    except EntryNotFoundError:
        return None
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            return None
        raise
    with open(p, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert isinstance(payload, dict), ("sampling marker is not a JSON object", stage_prefix)
    return payload


def _upload_sampling_marker(stage_prefix: str, payload: dict) -> None:
    """Upload the per-stage-prefix sampling marker (one tiny non-LFS JSON;
    single file, single commit — never a per-file loop)."""
    api = _hf_api()
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    hub.retry_transient(
        lambda: api.upload_file(
            path_or_fileobj=data,  # bytes: retry-safe (a BytesIO would drain on attempt 1)
            path_in_repo=f"{stage_prefix}/{SAMPLING_MARKER_NAME}",
            repo_id=LADDER_HF_REPO,
            repo_type="dataset",
            commit_message=(
                f"issue1491: sampling marker {payload['sampling_mode']} @ {stage_prefix}"
            ),
        ),
        what=f"upload sampling marker {stage_prefix}/{SAMPLING_MARKER_NAME}",
    )


def _enforce_sampling_identity(
    stage_prefix: str,
    scratch: Path,
    cache_dir: Path,
    sampling: dict,
    *,
    done_pt: set[str],
    done_raw: set[str],
    no_upload: bool,
    shard_index: int,
) -> None:
    """Refuse to mix decoding regimes at the (stage_prefix, scratch) grain.

    The regime identity is the PAIR (``sampling_mode``, ``gen_max_tokens``):
    the token cap is identity too (Path B re-gen round — a cap-2048 regen
    prefix must never be resumed/joined by a cap-1024 run and vice versa).

    The chunk-level resume predicate (``done_pt``/``done_raw``) skips work by
    FILENAME without ever reading a payload, so the payload-level
    ``sampling_mode`` asserts alone cannot stop a greedy run from silently
    "resuming" (skipping) a temperature-1.0 prefix — this guard pins the
    regime at the prefix grain BEFORE any chunk decision. Three sources, all
    fail-loud on mismatch:

    1. LOCAL marker (``scratch/sampling_mode.json``, written below) — covers
       the salvage scratch even in ``--no-upload`` mode.
    2. LOCAL legacy inference — chunk files present with NO marker predate
       the greedy round, so they are uniquely parent-mode (the hardcoded
       constants were the only recipe that ever wrote them).
    3. HUB marker (``{stage_prefix}/sampling_mode.json``) + hub legacy
       inference (chunks on the Hub with no marker ⇒ parent-mode). Shard 0
       uploads the marker when absent (fresh prefix, or parent-mode legacy
       backfill) — one writer avoids an 8-way concurrent-commit race; the
       narrow first-seconds window where a sibling pod checks before shard
       0's upload fails toward REFUSE (loud), never silent reuse.
    """
    mode = str(sampling["mode"])
    cap = _sampling_cap(sampling)
    marker_payload = {
        "sampling_mode": mode,
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "gen_max_tokens": cap,
        "written_by": "issue1491_ladder_generate_capture",
    }
    local_marker = scratch / SAMPLING_MARKER_NAME
    if local_marker.exists():
        with open(local_marker, encoding="utf-8") as fh:
            local_data = json.load(fh)
        local_mode = str(local_data.get("sampling_mode", SAMPLING_MODE_PARENT))
        local_cap = int(local_data.get("gen_max_tokens", GEN_MAX_TOKENS))
        if local_mode != mode or local_cap != cap:
            raise RuntimeError(
                f"SAMPLING-MODE mismatch (local scratch): {scratch} was written under "
                f"('{local_mode}', cap={local_cap}) but this run is ('{mode}', cap={cap}). "
                "Refusing to resume/salvage across decoding regimes (mode OR token cap) "
                "— use a fresh --out-dir (and --hf-prefix) for the new regime."
            )
    else:
        has_local_chunks = any(scratch.glob("shard*_chunk*.json")) or any(
            scratch.glob("shard*_chunk*.pt")
        )
        if has_local_chunks and (mode != SAMPLING_MODE_PARENT or cap != GEN_MAX_TOKENS):
            raise RuntimeError(
                f"SAMPLING-MODE mismatch (legacy local scratch): {scratch} holds chunk "
                "files but no sampling marker — it predates the greedy round, so its "
                f"chunks are temperature-1.0 ('{SAMPLING_MODE_PARENT}') at cap "
                f"{GEN_MAX_TOKENS} by construction. Refusing to run ('{mode}', cap={cap}) "
                "over it — use a fresh --out-dir + --hf-prefix."
            )
    hub_marker: dict | None = None
    hub_mode: str | None = None
    if not no_upload:
        hub_marker = _download_hub_sampling_marker(stage_prefix, cache_dir)
        if hub_marker is not None:
            hub_mode = str(hub_marker.get("sampling_mode", SAMPLING_MODE_PARENT))
            hub_cap = int(hub_marker.get("gen_max_tokens", GEN_MAX_TOKENS))
            if hub_mode != mode or hub_cap != cap:
                raise RuntimeError(
                    f"SAMPLING-MODE mismatch (Hub): {stage_prefix} is pinned to "
                    f"('{hub_mode}', cap={hub_cap}) but this run is ('{mode}', cap={cap}). "
                    "Refusing to resume into / publish over a different decoding regime "
                    "(mode OR token cap) — use a fresh --hf-prefix."
                )
        if (
            hub_marker is None
            and (done_pt or done_raw)
            and (mode != SAMPLING_MODE_PARENT or cap != GEN_MAX_TOKENS)
        ):
            raise RuntimeError(
                f"SAMPLING-MODE mismatch (legacy Hub prefix): {stage_prefix} already holds "
                f"{len(done_pt)} .pt / {len(done_raw)} raw chunks but no sampling marker — "
                "they predate the greedy round, so they are temperature-1.0 "
                f"('{SAMPLING_MODE_PARENT}') at cap {GEN_MAX_TOKENS} by construction. "
                f"Refusing to run ('{mode}', cap={cap}) into this prefix — use a fresh "
                "--hf-prefix."
            )
        if hub_marker is None and shard_index == 0:
            _upload_sampling_marker(stage_prefix, marker_payload)
            logger.info(
                "[ladder] sampling-identity: uploaded marker mode=%s cap=%d -> %s/%s",
                mode,
                cap,
                stage_prefix,
                SAMPLING_MARKER_NAME,
            )
    if not local_marker.exists():
        C.write_json_atomic(local_marker, marker_payload)
    logger.info(
        "[ladder] sampling-identity OK: mode=%s cap=%d stage_prefix=%s hub_marker=%s",
        mode,
        cap,
        stage_prefix,
        hub_mode if hub_mode is not None else "absent-or-local-only",
    )


def run_capture(args) -> int:
    """Run generation + trimmed capture for ONE (model, split) combination
    across ``args.num_shards`` shards; process shard ``args.shard_index``.

    Emits per-chunk .pt (trimmed) + per-chunk raw completions JSON into
    ``args.out_dir/shards/``, uploads in K=20 batches under the stage
    prefix derived from ``args.split``: ``{args.hf_prefix}/{split}/...``
    for regular splits, ``{args.hf_prefix}/ceiling_draws/seed{S}/...``
    for the ``ceiling_draw_*`` splits (plan §4.2; there is no ``--stage``
    flag — round-3a MINOR docstring fix).
    """
    layers = _resolve_layers_arg(args.layers)
    h_dim = _resolve_h_dim(args.model, args.h_dim)
    manifest_key, gen_seed = SPLIT_TO_MANIFEST[args.split]
    sampling = _resolve_sampling(bool(getattr(args, "greedy", False)))
    logger.info(
        "[ladder] model=%s split=%s (manifest=%s, seed=%d) layers=%s H=%d shard=%d/%d "
        "hf_prefix=%s sampling=%s (temp=%s top_p=%s)",
        args.model,
        args.split,
        manifest_key,
        gen_seed,
        layers,
        h_dim,
        args.shard_index,
        args.num_shards,
        args.hf_prefix,
        sampling["mode"],
        sampling["temperature"],
        sampling["top_p"],
    )

    # 1. Read the ladder manifest split.
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _download_ladder_split(manifest_key, cache_dir)
    n_total = len(all_rows)
    start, end = _split_shard_range(n_total, args.num_shards, args.shard_index)
    shard_rows = all_rows[start:end]
    if not shard_rows:
        logger.info("[shard %d] empty range; nothing to do", args.shard_index)
        C.phase("done")
        return 0

    # HF paths — ceiling draws under a nested prefix (plan §4.2).
    stage_prefix = f"{args.hf_prefix}"
    if args.split.startswith("ceiling_draw_"):
        stage_prefix = f"{args.hf_prefix}/ceiling_draws/seed{gen_seed}"
    else:
        stage_prefix = f"{args.hf_prefix}/{args.split}"

    scratch = args.out_dir / "shards" / args.split.replace("ceiling_draw_", "cdraw_")
    scratch.mkdir(parents=True, exist_ok=True)

    # 2. Resume — chunks whose .pt AND raw json are already on the Hub are skipped.
    done_pt = _remote_index(stage_prefix, "final_token_capture")
    done_raw = _remote_index(stage_prefix, "raw_completions")

    # 2b. Decoding-regime identity guard — BEFORE any chunk decision: the
    # filename-keyed resume above would otherwise silently "resume" (skip)
    # another regime's chunks without ever reading a payload.
    _enforce_sampling_identity(
        stage_prefix,
        scratch,
        cache_dir,
        sampling,
        done_pt=done_pt,
        done_raw=done_raw,
        no_upload=args.no_upload,
        shard_index=args.shard_index,
    )

    # 3. Load models. Capture mode governs which we hold at once.
    C.phase("load_model")
    if args.capture_mode == "phase_split_gen":
        # Gen-only pass: tokenizer ONLY. Co-loading the full HF model beside
        # a vLLM engine of the same model is a deterministic init
        # failure/OOM at exactly the 14B/32B scales phase-split serves
        # (vLLM gpu_memory_utilization is a fraction of TOTAL device
        # memory). Nothing on the gen path touches `hf`.
        tok = _load_tokenizer(args.model)
        hf = None
    else:
        # coresident AND phase_split_capture: full HF model (bf16 on cuda).
        # phase_split_capture holds ONLY the HF model — no vLLM engine is
        # ever constructed (the llm gate below excludes it), which is the
        # whole point of the split: a 14B/32B HF model + vLLM engine cannot
        # co-reside on one GPU (plan §4.2 per-shard architecture).
        tok, hf = N10.load_models(args.model, args.device)

    llm = None
    if args.capture_mode in ("coresident", "phase_split_gen"):
        llm = _build_capture_engine(args.model, gen_seed) if args.device == "cuda" else None

    # 4. Capture method selection: batched (default) with parity fallback.
    # The gate runs in BOTH capturing modes — coresident AND
    # phase_split_capture (14B/32B are exactly where batched capture
    # matters, plan §4.2 "Parity gate (run-start, per scale)").
    capture_fn_choice = "perrow"
    if args.capture_batch_size > 1 and args.capture_mode in ("coresident", "phase_split_capture"):
        # Run the parity gate on ~32 probe rows — OVER-LENGTH-FILTERED first:
        # an over-length prompt is ENGINE-FATAL at vLLM add_request (kills
        # the engine, not the row — gotchas.md #1738 subsample-bypass class),
        # so the probe must apply the SAME admission filter as the chunk
        # loop. Take 64 candidates so the filter still leaves ~32. In
        # phase_split_capture the probe responses come from the gen wave's
        # chunk-0 raw JSON, so candidates stay within chunk 0's row range.
        n_cand = min(64, len(shard_rows))
        if args.capture_mode == "phase_split_capture":
            n_cand = min(n_cand, args.shard_size)
        cand = shard_rows[:n_cand]
        cand_prompts = [r["prompt"] for r in cand]
        # Last-resort ci fallback must use the GLOBAL row index, matching the
        # main chunk loop's `start + s + i`. The probe takes shard_rows[:n_cand],
        # i.e. s == 0, so the global index is `start + i`. This was a bare local
        # `i`: inert while every manifest row carries ladder_local_id, but on a
        # manifest missing it with shard_index > 0 the probe cis would silently
        # diverge from the main loop's, mispairing the parity comparison.
        cand_cis = [
            int(r.get("ladder_local_id", r.get("i", start + i))) for i, r in enumerate(cand)
        ]
        kept_probe_prompts, kept_probe_cis, probe_skipped = _filter_overlength_prompts(
            cand_prompts,
            cand_cis,
            lambda p: _rendered_prompt_token_len(tok, p),
            PROMPT_TOKEN_BUDGET,
        )
        if probe_skipped:
            logger.info(
                "[ladder] parity probe: %d over-length rows excluded from probe",
                len(probe_skipped),
            )
        probe_prompts = kept_probe_prompts[:32]
        probe_cis = kept_probe_cis[:32]
        if not probe_cis:
            probe_responses: list[str] = []  # all-over-length head — gate no-ops
        elif args.capture_mode == "phase_split_capture":
            # Probe responses = the gen wave's chunk-0 persisted rows (the
            # capture pass never generates). Same admission filter + shard
            # arithmetic ⇒ every probe ci is in chunk 0's kept set.
            probe_raw_name = f"shard{args.shard_index:02d}_chunk0000.json"
            probe_map = _load_persisted_gen_chunk(
                scratch,
                stage_prefix,
                probe_raw_name,
                cache_dir,
                done_raw,
                expect_split=args.split,
                expect_seed=gen_seed,
                expect_shard_index=args.shard_index,
                expect_chunk=0,
                expect_sampling_mode=sampling["mode"],
                expect_gen_max_tokens=_sampling_cap(sampling),
                allow_local_only=args.no_upload,
            )
            probe_missing = [c for c in probe_cis if c not in probe_map]
            assert not probe_missing, (
                "parity probe: gen-wave rows missing (shard config drift?)",
                probe_raw_name,
                probe_missing[:8],
            )
            probe_responses = [probe_map[c]["response"] for c in probe_cis]
        else:
            # Generate responses for probe rows (small — safe). llm None (CPU
            # smoke) returns stub responses through the same path, so the
            # probe exercises the REAL capture code on CPU too.
            probe_responses, _probe_finish = _generate_seeded(
                llm, tok, probe_prompts, gen_seed, sampling
            )
        gate_pass, gate_reason = _batched_capture_parity_gate(
            hf,
            tok,
            probe_prompts,
            probe_responses,
            probe_cis,
            layers,
            h_dim,
            args.capture_batch_size,
        )
        logger.info(
            "[ladder] batched-capture parity gate: %s (%s)",
            "PASS" if gate_pass else "FAIL",
            gate_reason,
        )
        if gate_pass:
            capture_fn_choice = "batched"
        else:
            logger.warning(
                "[ladder] batched-capture parity gate FAILED — falling back to per-row (parent parity). Reason: %s",
                gate_reason,
            )

    def _do_capture(prompts_i, responses_i, cis_i, _hf=hf, _tok=tok, _layers=layers, _h_dim=h_dim):
        # Default-arg capture makes the closure explicit + placates ruff F821
        # (ruff can't infer enclosing-scope binding when a later `del hf`
        # exists in the same function; Python's closure semantics are
        # unaffected either way — the binding is fixed at def time).
        if capture_fn_choice == "batched":
            return _capture_batched(
                _hf, _tok, prompts_i, responses_i, cis_i, _layers, _h_dim, args.capture_batch_size
            )
        return _capture_perrow(_hf, _tok, prompts_i, responses_i, cis_i, _layers, _h_dim)

    # 5. Main loop across chunks.
    C.phase("capture")
    n_sub = (len(shard_rows) + args.shard_size - 1) // args.shard_size
    kept_total = 0
    pending_pt: list[str] = []
    pending_raw: list[str] = []

    def _flush_pending() -> None:
        # phase_split_gen produces ONLY raw completions (pending_pt stays
        # empty) and phase_split_capture ONLY .pt chunks (pending_raw stays
        # empty), so the gate must key on EITHER kind being pending — a
        # `not pending_pt` early-return strands the whole gen phase's
        # rollout text pod-locally (persist-by-default; upload-policy §v2:
        # text is never discardable). _flush_upload_batch no-ops per empty
        # kind internally.
        if args.no_upload or (not pending_pt and not pending_raw):
            return
        _flush_upload_batch(scratch, stage_prefix, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    skipped_all: list[dict] = []
    dropped_empty_all: list[int] = []
    cap_hit_total = 0
    gen_total = 0
    self_gate_rows: list[dict] = []
    self_gate_fired = False

    def _eval_first_chunk_self_gate(trigger: str) -> None:
        # Plan §7 Decision Gate 1 — ONE shared evaluation + abort path for
        # the mid-shard (>=2000 rows) trigger and the end-of-shard fallback
        # (round-3a M2; no forked logic between the two arming points).
        nonlocal self_gate_fired
        self_gate_fired = True
        primary_layer_index = len(layers) // 2  # f=0.679 primary is middle entry
        passed, diag = _first_chunk_self_gate(self_gate_rows, primary_layer_index)
        verdict = diag.get("verdict") or ("PASS" if passed else "FAIL")
        logger.info(
            "[ladder-gate] first-chunk self-gate (%s, n=%d rows): %s (%s)",
            trigger,
            len(self_gate_rows),
            verdict,
            diag,
        )
        if verdict == "INCONCLUSIVE":
            # MUST NOT abort — but the record must show, loudly and with the
            # full conditioning disclosure, that the gate could not
            # discriminate broken from healthy capture here (#1701).
            logger.warning(
                "[ladder-gate] INCONCLUSIVE (%s): %s — proceeding WITHOUT abort; "
                "the gate has no discriminating power at this conditioning and "
                "this rung's gate outcome is a disclosure record, not evidence "
                "of capture health. diag=%s",
                trigger,
                diag.get("verdict_reason"),
                diag,
            )
        if passed:
            return
        # Abort THIS scale's job (other scales unaffected) via a sentinel the
        # poller drains into an epm:failure marker. Round-3a M1: the sentinel
        # MUST conform to poll_pipeline._SENTINEL_REQUIRED_KEYS
        # (sentinel_schema_version / kind / version) or _parse_sentinel SKIPS
        # it (the #899 fallback rescues only bare Step-7 results payloads on
        # *-results.json filenames) and plan §7's abort routing is dead code
        # — so route through the shared conforming writer C.write_sentinel
        # (which also falls back to PROJECT_ROOT/logs when /workspace/logs is
        # absent, instead of silently skipping the write).
        C.write_sentinel(
            "epm:failure",
            "failure_class: code\n"
            "reason: first_chunk_self_gate_fail\n"
            f"split: {args.split}\n"
            f"shard_index: {args.shard_index}\n"
            f"trigger: {trigger}\n"
            f"detail: {diag}",
            task_id=1491,
            extra={
                "failure_class": "code",
                "reason": "first_chunk_self_gate_fail",
                "detail": diag,
                "split": args.split,
                "shard_index": args.shard_index,
                "trigger": trigger,
                "by": "issue1491_ladder_generate_capture",
            },
        )
        _flush_pending()  # keep what we have
        raise SystemExit(1)

    try:
        for ci_idx, s in enumerate(range(0, len(shard_rows), args.shard_size)):
            name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.pt"
            raw_name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.json"
            chunk = shard_rows[s : s + args.shard_size]
            kept_prompts, kept_cis, skipped = _filter_overlength_prompts(
                [r["prompt"] for r in chunk],
                [
                    int(r.get("ladder_local_id", r.get("i", start + s + i)))
                    for i, r in enumerate(chunk)
                ],
                lambda p: _rendered_prompt_token_len(tok, p),
                PROMPT_TOKEN_BUDGET,
            )
            skipped_all.extend(skipped)
            # Resume predicate is MODE-scoped: the gen-only pass never
            # produces a .pt, so requiring `name in done_pt` there would
            # make every restart regenerate the shard from scratch; the
            # capture-only pass never (re-)uploads raw completions — the gen
            # wave owns them — so its resume keys on the .pt alone.
            if args.capture_mode == "phase_split_gen":
                chunk_done = raw_name in done_raw
            elif args.capture_mode == "phase_split_capture":
                chunk_done = name in done_pt
            else:
                chunk_done = name in done_pt and raw_name in done_raw
            if chunk_done:
                logger.info(
                    "[shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            if not kept_prompts:
                logger.warning(
                    "[shard %d] chunk %d: all rows over-length; skip", args.shard_index, ci_idx
                )
                continue

            ts = time.time()
            if args.capture_mode == "phase_split_capture":
                # No generation: JOIN the gen wave's persisted responses back
                # by context id. Missing cis are FAIL-LOUD (a silent subset
                # would ship a .pt whose row set diverges from the raw pair).
                raw_map = _load_persisted_gen_chunk(
                    scratch,
                    stage_prefix,
                    raw_name,
                    cache_dir,
                    done_raw,
                    expect_split=args.split,
                    expect_seed=gen_seed,
                    expect_shard_index=args.shard_index,
                    expect_chunk=ci_idx,
                    expect_sampling_mode=sampling["mode"],
                    expect_gen_max_tokens=_sampling_cap(sampling),
                    allow_local_only=args.no_upload,
                )
                missing = [c for c in kept_cis if c not in raw_map]
                if missing:
                    raise RuntimeError(
                        f"phase_split_capture: {len(missing)} kept cis absent from gen-wave "
                        f"{raw_name} (first: {missing[:10]}) — the gen wave ran under a "
                        "different shard config / manifest; refusing a partial join."
                    )
                # Converse join direction (round-3a MINOR): gen-wave rows NOT
                # in this run's kept set ship raw-only and are never captured
                # — under admission/config drift that is a silent coverage
                # hole, so at minimum count + log it loudly.
                extra_cis = sorted(set(raw_map) - set(kept_cis))
                if extra_cis:
                    logger.warning(
                        "[shard %d] chunk %d: %d gen-wave rows absent from this capture "
                        "run's kept set (first: %s) — over-length/admission drift between "
                        "waves; these rows remain raw-only (uncaptured)",
                        args.shard_index,
                        ci_idx,
                        len(extra_cis),
                        extra_cis[:10],
                    )
                for c, p in zip(kept_cis, kept_prompts, strict=True):
                    assert raw_map[c]["prompt"] == p, (
                        "prompt drift between manifest row and gen-wave row",
                        c,
                    )
                responses = [raw_map[c]["response"] for c in kept_cis]
                # Cap-hit accounting belongs to the GEN wave (already
                # reported there); do not double-count here.
                n_cap_hit = 0
            else:
                # SALVAGE-FIRST (round-3a M3, gen side): if this chunk's raw
                # JSON already exists locally (atomic write from a prior run
                # that died before its upload flushed), reuse its text
                # verbatim and re-upload the ORIGINAL bytes — NEVER
                # regenerate: a fresh temperature-1.0 draw would publish raw
                # text silently diverging from any .pt already captured from
                # the local text (same-pod gen->capture chaining). Together
                # with the capture-side Hub-required guard in
                # _load_persisted_gen_chunk this makes raw/.pt divergence
                # structurally impossible whenever the atomic local record
                # exists; with no local record and nothing published, the
                # full redo below regenerates + recaptures BOTH artifacts as
                # a consistent pair (upload overwrites are pair-atomic per
                # flush batch).
                salvaged = None
                if raw_name not in done_raw:
                    salvaged = _load_local_raw_salvage(
                        scratch,
                        raw_name,
                        expect_split=args.split,
                        expect_seed=gen_seed,
                        expect_shard_index=args.shard_index,
                        expect_chunk=ci_idx,
                        expect_sampling_mode=sampling["mode"],
                        expect_gen_max_tokens=_sampling_cap(sampling),
                    )
                if salvaged is not None:
                    sal_rows = {int(r["ci"]): r for r in salvaged["rows"]}
                    if set(sal_rows) != set(kept_cis):
                        raise RuntimeError(
                            f"gen salvage: local {raw_name} row set diverges from this run's "
                            f"kept set (local-only: {sorted(set(sal_rows) - set(kept_cis))[:10]}, "
                            f"kept-only: {sorted(set(kept_cis) - set(sal_rows))[:10]}) — "
                            "manifest/admission drift; refusing to reuse OR regenerate "
                            "(delete the stale scratch file only if the drift is understood)."
                        )
                    for c, p in zip(kept_cis, kept_prompts, strict=True):
                        assert sal_rows[c]["prompt"] == p, (
                            "prompt drift between manifest row and salvaged gen row",
                            c,
                        )
                    responses = [sal_rows[c]["response"] for c in kept_cis]
                    finish_reasons = [str(sal_rows[c]["finish_reason"]) for c in kept_cis]
                    n_cap_hit = int(
                        salvaged.get("n_cap_hit", sum(1 for f in finish_reasons if f == "length"))
                    )
                    # This run publishes these rows, so they enter its cap-hit
                    # digest (the killed run never reported them).
                    cap_hit_total += n_cap_hit
                    gen_total += len(responses)
                    logger.warning(
                        "[shard %d] chunk %d: SALVAGED %d rows from local %s (prior run died "
                        "before upload) — text reused verbatim, NOT regenerated (round-3a M3)",
                        args.shard_index,
                        ci_idx,
                        len(responses),
                        raw_name,
                    )
                    # Do NOT rewrite the file: the original atomic bytes
                    # upload as-is (byte-identity with any prior local read).
                else:
                    # Generate responses (split-seeded; llm None on the CPU
                    # smoke path returns stub responses through the same
                    # capture code).
                    responses, finish_reasons = _generate_seeded(
                        llm, tok, kept_prompts, gen_seed, sampling
                    )
                    n_cap_hit = sum(1 for f in finish_reasons if f == "length")
                    cap_hit_total += n_cap_hit
                    gen_total += len(responses)

                    # Persist raw completions FIRST (persist-by-default; text
                    # path, non-LFS, quota-immune — upload-policy §v2).
                    # finish_reason per row + n_cap_hit make cap-hit rows
                    # re-generable post-hoc (CLAUDE.md cap-hit accounting).
                    C.write_json_atomic(
                        scratch / raw_name,
                        {
                            "shard_index": args.shard_index,
                            "chunk": ci_idx,
                            "split": args.split,
                            "seed": gen_seed,
                            "sampling_seed": gen_seed,
                            "engine_seed": gen_seed,
                            "sampling_mode": sampling["mode"],
                            "temperature": sampling["temperature"],
                            "top_p": sampling["top_p"],
                            "gen_max_tokens": _sampling_cap(sampling),
                            "n_cap_hit": n_cap_hit,
                            "rows": [
                                {"ci": int(c), "prompt": p, "response": r, "finish_reason": f}
                                for c, p, r, f in zip(
                                    kept_cis, kept_prompts, responses, finish_reasons, strict=True
                                )
                            ],
                        },
                    )

            # Trimmed capture (skipped in phase_split_gen mode — gen only).
            if args.capture_mode == "phase_split_gen":
                n_kept = len(kept_prompts)  # gen-side row count
                # No .pt to write; only raw_completions uploads.
                pending_raw.append(raw_name)
                n_dropped_empty = 0
            else:
                rows, dropped_cis = _do_capture(kept_prompts, responses, kept_cis)
                dropped_empty_all.extend(dropped_cis)
                n_dropped_empty = len(dropped_cis)
                if dropped_cis:
                    logger.info(
                        "[shard %d] chunk %d: dropped %d empty-response rows (cis %s%s)",
                        args.shard_index,
                        ci_idx,
                        len(dropped_cis),
                        dropped_cis[:20],
                        "..." if len(dropped_cis) > 20 else "",
                    )
                if rows:
                    bundle = _stack_chunk(rows, layers, args.shard_index, ci_idx)
                    bundle["dropped_empty_cis"] = [int(c) for c in dropped_cis]
                    torch.save(bundle, scratch / name)
                    if not self_gate_fired:
                        self_gate_rows.extend(rows)
                    pending_pt.append(name)
                else:
                    # All responses empty: no .pt. In generating modes the
                    # raw completions still upload (persist-by-default); in
                    # phase_split_capture the gen wave already owns them.
                    # NOTE: with no .pt this chunk re-runs on resume —
                    # idempotent, and vanishingly rare at chunk size 500.
                    logger.warning(
                        "[shard %d] chunk %d: 0 captured rows (all empty responses)",
                        args.shard_index,
                        ci_idx,
                    )
                n_kept = len(rows)
                if args.capture_mode != "phase_split_capture":
                    # The gen wave owns raw_completions; re-uploading them
                    # from the capture wave would double-commit identical
                    # content (and burn the fleet commit budget).
                    pending_raw.append(raw_name)

            kept_total += n_kept
            # Key the flush trigger on EITHER pending kind: phase_split_capture
            # accumulates ONLY .pt names (pending_raw stays empty), so a
            # raw-only trigger would defer every upload to the terminal flush
            # — the #664 write-at-end anti-pattern.
            if not args.no_upload and max(len(pending_pt), len(pending_raw)) >= UPLOAD_BATCH:
                _flush_pending()

            logger.info(
                "[shard %d] chunk %d/%d: %d/%d captured (%d over-length skipped, "
                "%d empty-response dropped, %d cap-hit, %.0fs) [%s]",
                args.shard_index,
                ci_idx + 1,
                n_sub,
                n_kept,
                len(chunk),
                len(skipped),
                n_dropped_empty,
                n_cap_hit,
                time.time() - ts,
                capture_fn_choice if args.capture_mode != "phase_split_gen" else "gen-only",
            )

            # First-chunk self-gate — plan §7 Decision Gate 1 (mid-shard arm).
            if (
                args.first_chunk_self_gate
                and args.capture_mode != "phase_split_gen"
                and not self_gate_fired
                and len(self_gate_rows) >= 2000
            ):
                _eval_first_chunk_self_gate("mid-shard >=2000 rows")

        # End-of-shard fallback (round-3a M2): at the launcher's own 32B
        # example (16 shards, train_25k) a shard holds ~1,563 rows < 2000,
        # so the mid-shard arm above NEVER fires on the most expensive rung.
        # Evaluate on whatever this shard actually captured; below 500 rows
        # _first_chunk_self_gate returns skipped=True (tiny splits). A fully
        # resumed shard (nothing captured this run) has no rows to evaluate.
        if (
            args.first_chunk_self_gate
            and args.capture_mode != "phase_split_gen"
            and not self_gate_fired
            and self_gate_rows
        ):
            _eval_first_chunk_self_gate("end-of-shard fallback")

        _flush_pending()
    except BaseException:
        try:
            _flush_pending()
        except Exception:  # noqa: BLE001
            logger.exception(
                "[shard %d] best-effort pending-batch flush failed on exit", args.shard_index
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    logger.info(
        "[shard %d] done: %d kept rows across %d chunks (%d over-length skipped, "
        "%d empty-response dropped)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
        len(dropped_empty_all),
    )
    # Cap-hit digest (CLAUDE.md: every generation stage REPORTS its realized
    # finish_reason=='length' fraction, with the pre-registered re-gen
    # trigger). gen_total == 0 on a fully-resumed shard — nothing generated.
    if gen_total > 0:
        cap_frac = cap_hit_total / gen_total
        logger.info(
            "[shard %d] cap-hit: %d/%d = %.4f (finish_reason=='length', gen_max_tokens=%d)",
            args.shard_index,
            cap_hit_total,
            gen_total,
            cap_frac,
            GEN_MAX_TOKENS,
        )
        if cap_frac > CAP_HIT_REGEN_TRIGGER:
            logger.warning(
                "[shard %d] cap-hit fraction %.2f%% exceeds the pre-registered re-gen "
                "trigger (%.0f%%): re-generate finish_reason=='length' rows at >=2x "
                "max_tokens (orchestrator decision; rows identifiable from the per-row "
                "finish_reason in raw_completions)",
                args.shard_index,
                100.0 * cap_frac,
                100.0 * CAP_HIT_REGEN_TRIGGER,
            )

    # Free GPU allocator + release engine before the process exits (parent
    # parity; helps a phase_split follow-up capture invocation not OOM).
    if hf is not None:
        del hf
    if llm is not None:
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    C.phase("done")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Qwen/Qwen2.5-<size>-Instruct model id")
    ap.add_argument(
        "--layers",
        required=True,
        help="comma-separated list of 3 depth-fraction-mapped layer indices (0-indexed hidden-states offset; f=0.679 primary is the middle entry)",
    )
    ap.add_argument(
        "--h-dim",
        type=int,
        default=None,
        help="hidden dim (default: auto-detect via AutoConfig.hidden_size)",
    )
    ap.add_argument(
        "--split",
        required=True,
        choices=sorted(SPLIT_TO_MANIFEST.keys()),
        help="ladder-manifest split to process",
    )
    ap.add_argument(
        "--hf-prefix",
        required=True,
        help="child-issue HF prefix, e.g. issue1491_scale_ladder/scale7 (NEVER the parent's — plan §10 item (i))",
    )
    ap.add_argument(
        "--capture-mode",
        default="coresident",
        choices=["coresident", "phase_split_gen", "phase_split_capture"],
        help="coresident: vLLM engine + HF capture on the same GPU (≤7B). "
        "phase_split_gen: only vLLM generation, persist responses. "
        "phase_split_capture: only HF capture (no vLLM engine) from the gen "
        "wave's persisted responses, joined by context id (14B/32B).",
    )
    ap.add_argument(
        "--capture-batch-size",
        type=int,
        default=8,
        help="HF capture batch size (source-module throughput fix, plan §4.2 item (i); default 8; run-start parity gate on 32 rows falls back to per-row on cosine < 0.9999 or rel-L2 >= 1e-3)",
    )
    ap.add_argument(
        "--first-chunk-self-gate",
        action="store_true",
        help="enable plan §7 Gate 1 (quick ridge fit + shuffled-pairing null after ~2000 captured rows; aborts scale on gap<0.05 or |null|>0.05)",
    )
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EPM_LADDER_OUT_DIR",
                os.path.expanduser("~/data/issue_1491/ladder_generate_capture"),
            )
        ),
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="GREEDY decoding (temperature 0; top_p pinned 1.0) — the deterministic "
        "replication round. DEFAULT (flag absent) keeps the parent recipe "
        "(temp 1.0, top_p 0.95) byte-identical. The sampling mode is part of the "
        "resume/wave identity: a greedy run REFUSES to resume from / salvage / join "
        "temperature-1.0 chunks and vice versa — use a fresh --hf-prefix + --out-dir "
        "per regime. The split seed still threads (inert under greedy), so "
        "ceiling_draw_43/44 stay two independent generate passes measuring vLLM "
        "continuous-batching nondeterminism.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="capture locally; do NOT upload/purge (smoke path)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return run_capture(args)


if __name__ == "__main__":
    sys.exit(main())
