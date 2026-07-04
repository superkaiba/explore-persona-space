#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², −, r_B) in scientific docstrings + log messages.
"""Issue #810 Phase B: re-extract answer-side POSITION summaries (θ0, no training).

Teacher-forces #658's STORED base-model completions back through
``Qwen/Qwen2.5-7B-Instruct`` and captures the residual-stream activation at a
thin ALIGNED SUBSET of answer-side positions per (context, probe):

- ``im_end``  — the ``<|im_end|>`` token (id 151645) after the answer content.
- ``turn_nl`` — the ``\\n`` (id 198) after ``<|im_end|>``: the answer-side mirror
  of #594's ``c_C`` last-input-token boundary. The H1 headline candidate.
- ``tail_1..16`` — end-aligned answer-CONTENT positions (``tail_1`` == last token).
- ``head_0..15`` — start-aligned answer-CONTENT positions.

The stored ``answer_spans/<ctx>.pt`` spans are answer-CONTENT only, so tail/head
are slice-derivable from them BUT ``im_end`` / ``turn_nl`` are the two boundary
positions AFTER ``span_end`` — they need a fresh forward over
``prompt + answer + <|im_end|> + \\n``. This pass captures ALL 34 positions in
one forward per probe (so #812 reuses the same store, plan §13) and writes the
per-context probe-mean summary vectors to the aligned-subset store.

Extends the #658 extraction path (``issue658_extract_base_store.capture_v0_for_
context`` / ``AnswerSpanCapture`` / ``LayerCapture``) + reuses #594's
``messages_for_instance``; it does NOT re-implement the hooks or the chat
template. Forward-pass-only (no sampling, no training).

Storage (plan §13, SHARED with #812): one file per context
``<HF_PREFIX>/answer_position_sweep/<context_id>.pt`` — a dict
``{context_id, capture_layers:[0..27], positions:[...34...], pos_vectors:
(n_positions, 28, 3584) fp16, coverage: {position: probe_count}}``.

Local batteries under ``data/`` are gitignored (absent from the git-clone GCP
lane), so the 50-context battery is fetched from the sha256-pinned HF snapshot
(``BATTERY50_HF_FILE``) with a local-file fast path.

Pod-side contract: ``[phase=...]`` log lines ending in ``[phase=done]`` on a
graceful exit + a ``poll_pipeline.py``-conformant end-of-run sentinel.

Usage::

    # production (auto lane, GCP-first, 1x GPU eval intent):
    uv run python scripts/dispatch_issue.py --issue 810 --intent eval \\
        --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" \\
        uv run python scripts/issue810_extract_positions.py --gpu'

    # local CPU smoke (tiny same-family model, 1 context, all positions):
    uv run python scripts/issue810_extract_positions.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --n-ctx 1 --n-probes 2 \\
        --out-dir /tmp/i810_smoke --device cpu

    # UltraChat genre arm (follow-up round `ultrachat-genre-summary-sweep`):
    # --genre g1 switches the completions + manifest sources to #658's g1 arm
    # (probe-pool-hash pinned), uploads to answer_position_sweep_<genre-tag>/,
    # and runs the one-context cc_last recomputation parity probe FIRST.
    uv run python scripts/issue810_extract_positions.py --genre g1 --gpu
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import logging
import math
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue404_common import fetch_betley_main_8, fetch_preregistered_probes  # noqa: E402
from issue594_common import messages_for_instance, probes_hash  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_BTDR_SUBDIR_TMPL,
    ANSWER_POSITION_SWEEP_HE_SUBDIR,
    ANSWER_POSITION_SWEEP_SUBDIR,
    ANSWER_POSITION_SWEEP_UH_SUBDIR,
    BATTERY50_HF_FILE,
    BATTERY50_SHA256,
    BOUNDARY_BLOCK_IDS,
    BTDR_HF_RESULTS_PREFIX,
    BTDR_SUMMARIES_HF_FILE_TMPL,
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    G1_ANSWER_POSITION_SWEEP_SUBDIR,
    G1_GENRE_TAG,
    G1_PROBE_POOL_HASH,
    G1_RAW_COMPLETIONS_PREFIX,
    G1_STORE_MANIFEST,
    G1_V0_SUMMARIES,
    GENRES,
    HE_SUMMARIES_HF_FILE,
    HE_SUMMARY_NAMES,
    HF_DATA_REPO,
    HF_PREFIX,
    I594_CC_LAST_FILE,
    I594_PROBE_POOL_HASH,
    I658_RAW_COMPLETIONS_PREFIX,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    IM_END_TOKEN_ID,
    TURN_NL_TOKEN_ID,
    UH_SUMMARIES_HF_FILE,
    UH_SUMMARY_NAMES,
    assert_g1_probe_pool_hash,
    assert_sha256,
    btdr_pct,
    context_ids_from_manifest,
    dump_json,
    he_stored_position_names,
    load_json,
    reproducibility_metadata,
    retry_hub_quota,
    scoped_remote_listing,
    sha256_file,
    stored_position_names,
    tail_head_position_index,
    uh_stored_position_names,
)

# Project dotenv wrapper (#745): robust .env load + HF-upload accelerators.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTINEL_SCHEMA_VERSION = 1


# ── full-sequence answer+boundary capture (extends AnswerSpanCapture) ─────────


def _gather_positions_gpu(
    capture: LayerCapture,
    capture_layers: list[int],
    abs_positions: torch.Tensor,
) -> torch.Tensor:
    """GPU-side gather of the target positions per (batch item), then → fp16 CPU.

    ``capture.latest[li]`` is (B, T, H) on device. ``abs_positions`` is
    (B, n_targets) absolute token indices into the padded sequence (a target that
    is out of range for a short answer is marked -1 and gathered from index 0 as a
    placeholder — the caller keys on the coverage/validity mask, never on the
    placeholder value). This indexes the residual stream at the ~34 target
    positions INSIDE the CUDA graph (torch.gather over the T dim) BEFORE moving to
    CPU, so only (B, n_targets, Lc, H) crosses PCIe — NOT the full padded span ×
    28 layers (the Codex Major #1 host-transfer waste). Returns
    (B, n_targets, Lc, H) fp16 CPU; clears the capture buffer.
    """
    B, n_targets = abs_positions.shape
    idx_clamped = abs_positions.clamp(min=0)  # -1 placeholders → 0 (masked by caller)
    layer_slices = []
    for li in capture_layers:
        hs = capture.latest[li]  # (B, T, H) on device
        H = hs.shape[-1]
        # gather along T: index (B, n_targets) → (B, n_targets, H)
        gidx = idx_clamped.unsqueeze(-1).expand(B, n_targets, H)
        picked = torch.gather(hs, 1, gidx)  # (B, n_targets, H) GPU-side slice
        layer_slices.append(picked.to(torch.float16))
    capture.latest.clear()
    # stack layers → (B, n_targets, Lc, H); move to CPU once (thin slice only).
    return torch.stack(layer_slices, dim=2).cpu()  # (B, n_targets, Lc, H)


# Boundary-block single positions relative to boundary_offset (== ans span_len).
# im_end/turn_nl are the parent's 2; uh_* are the 3 next-user-header tokens the
# `--extended-boundary` arm appends (plan v11 §4.6 item 2).
_BOUNDARY_POS_OFFSETS = {"im_end": 0, "turn_nl": 1, "uh_im_start": 2, "uh_user": 3, "uh_nl": 4}

# In-forward span pools (extended-boundary arm only): name -> (span kind, reduce
# op). Span kinds over the union span: "ans" = answer content, "bnd5" = the 5
# boundary tokens, "uh3" = the 3 next-user-header tokens, "xbnd" = ans ∪ bnd5.
# Ops: "mean" = per-dim mean, "max" = per-dim max (the #658 `maxp` token-pool
# recipe, `summarize_answer_span`: plain `span.max(dim=0).values`).
# "mean_ans" is an INTERNAL parity pool (the #658 `mean` recipe recomputed
# in-forward) — used for the v0 store-mean drift tripwire, never stored as a row.
_POOL_SPECS: dict[str, tuple[str, str]] = {
    "uh_mean3": ("uh3", "mean"),
    "uh_max3": ("uh3", "max"),
    "bnd_mean5": ("bnd5", "mean"),
    "bnd_max5": ("bnd5", "max"),
    "mean_xbnd": ("xbnd", "mean"),
    "maxp_xbnd": ("xbnd", "max"),
    "mean_ans": ("ans", "mean"),
}


def _positions_for_span(
    span_len: int, boundary_offset: int, extended: bool = False
) -> dict[str, int]:
    """Map each stored position name to its index in the captured union span.

    The captured union span covers the answer-content positions [0, span_len)
    followed by the boundary block at ``boundary_offset + k`` (k per
    ``_BOUNDARY_POS_OFFSETS``: parent = im_end/turn_nl; ``extended`` adds
    uh_im_start/uh_user/uh_nl at +2/+3/+4). A tail_k/head_k position out of
    range for a short answer is OMITTED (recorded as a coverage miss), never a
    crash.

    ``boundary_offset`` == span_len (the union span starts at the first answer
    content token; the boundary block sits immediately after the content).
    """
    names = uh_stored_position_names() if extended else stored_position_names()
    idx: dict[str, int] = {}
    for name in names:
        if name in _POOL_SPECS:
            continue  # pools are span reductions, not single positions
        if name in _BOUNDARY_POS_OFFSETS:
            idx[name] = boundary_offset + _BOUNDARY_POS_OFFSETS[name]
        else:
            pos = tail_head_position_index(name, span_len)
            if pos is not None:
                idx[name] = pos
    return idx


# The exact decoded ablated boundary tail (plan v15 §4.6 item 2 decode assert).
ABLATED_TAIL_TEXT = "<|im_end|>\n<|im_start|>user\n"


def _build_probe_row(
    model,
    tokenizer,
    instance,
    q,
    ans,
    stored_names,
    nl_id,
    extended=False,
    ablate=False,
    truncate_frac=None,
):
    """Tokenize one (prompt [+ answer] + boundary block) probe → capture inputs.

    Boundary block = ``<|im_end|> \\n`` (parent, 2 tokens) or the full
    assistant-turn continuation ``<|im_end|> \\n <|im_start|> user \\n``
    (``extended``, 5 tokens — ``BOUNDARY_BLOCK_IDS``, every fed id asserted).
    ``ablate`` (plan v15 §4.6 item 2, requires ``extended``): the answer span
    is EMPTY — ``ans`` must be None (code-truth: no completions consumed), the
    full sequence is exactly ``prompt_ids + BOUNDARY_BLOCK_IDS`` (asserted, plus
    a decoded-tail string assert), ans_len == 0 with NO ``None`` return, and the
    ``cc_last`` predictor slot (``prompt_len - 1``) rides the same forward.
    ``truncate_frac`` (plan v18 §4.6 item 2, requires ``extended``, mutually
    exclusive with ``ablate``): ID-prefix cut of the tokenized answer —
    ``n_keep = max(1, ceil(k * ans_len))`` (never re-tokenized text), with the
    three registered asserts (n_keep bounds, full-sequence equality, decoded
    tail) and the same ``cc_last`` predictor slot riding the forward.

    Returns ``(full_ids (L,), tgt [abs-idx|None per stored pos], valid [bool per
    pos], prompt_len, ans_len, orig_ans_len)`` or ``None`` for an empty
    completion (non-ablate only); ``ans_len`` is the KEPT length (== n_keep in
    truncate mode), ``orig_ans_len`` the pre-cut length (0 in ablate mode). The
    target indices are PRE-PAD absolute indices into the real sequence (the
    batch flush shifts them by the left-pad amount). Fails loud on ANY
    boundary-token id mismatch (a wrong id would silently capture the wrong
    slot).
    """
    messages = messages_for_instance(instance, q)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    if ablate:
        assert extended, "--ablate-answer implies the 5-token extended boundary block"
        assert ans is None, "ablate mode must not receive completion text (code-truth: plan §10)"
        assert truncate_frac is None, "--truncate-frac and --ablate-answer are mutually exclusive"
        ans_len = 0
        orig_ans_len = 0
        bids = list(BOUNDARY_BLOCK_IDS)
        boundary = torch.tensor([bids], dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, boundary], dim=1)[0]  # (prompt_len + 5,)
        # Full-sequence equality + decoded-tail asserts (plan v15 §4.6 item 2 /
        # kill criterion 1): the fed sequence is EXACTLY prompt + the 5-id block.
        assert full_ids.tolist() == prompt_ids[0].tolist() + bids, (
            f"ablated full sequence != prompt_ids + BOUNDARY_BLOCK_IDS for "
            f"{instance['id']} {q[:30]!r}"
        )
        decoded_tail = tokenizer.decode(full_ids[prompt_len:].tolist())
        assert decoded_tail == ABLATED_TAIL_TEXT, (
            f"ablated boundary tail decodes to {decoded_tail!r} != {ABLATED_TAIL_TEXT!r} "
            f"for {instance['id']} {q[:30]!r}"
        )
    else:
        ans_ids = tokenizer(ans, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ans_ids.shape[1] == 0:
            return None
        orig_ans_len = int(ans_ids.shape[1])
        if truncate_frac is not None:
            # `_btdr` graded ID-prefix truncation (plan v18 §4 edge rules): cut
            # the TOKEN sequence, never re-tokenized truncated text (zero
            # retokenization/BPE-merge risk; k=1.0 degenerates to the round-3
            # full capture by construction).
            assert extended, "--truncate-frac implies the 5-token extended boundary block"
            n_keep = max(1, math.ceil(truncate_frac * orig_ans_len))
            assert 1 <= n_keep <= orig_ans_len, (
                f"n_keep {n_keep} out of [1, {orig_ans_len}] at k={truncate_frac} for "
                f"{instance['id']} {q[:30]!r}"
            )
            if truncate_frac >= 1.0:
                assert n_keep == orig_ans_len, (
                    f"k=1.0 endpoint: n_keep {n_keep} != ans_len {orig_ans_len} for "
                    f"{instance['id']} {q[:30]!r}"
                )
            ans_ids = ans_ids[:, :n_keep]
        ans_len = int(ans_ids.shape[1])
        bids = list(BOUNDARY_BLOCK_IDS) if extended else [IM_END_TOKEN_ID, nl_id]
        boundary = torch.tensor([bids], dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, ans_ids, boundary], dim=1)[0]  # (full_len,)
        if truncate_frac is not None:
            # Registered asserts (plan v18 §4 edge rules): full-sequence
            # equality + decoded-tail (the 5-id block decodes to the same
            # string regardless of what precedes it).
            assert full_ids.tolist() == prompt_ids[0].tolist() + ans_ids[0].tolist() + bids, (
                f"truncated full sequence != prompt_ids + kept + BOUNDARY_BLOCK_IDS for "
                f"{instance['id']} {q[:30]!r} (k={truncate_frac})"
            )
            decoded_tail = tokenizer.decode(full_ids[prompt_len + ans_len :].tolist())
            assert decoded_tail == ABLATED_TAIL_TEXT, (
                f"truncated boundary tail decodes to {decoded_tail!r} != "
                f"{ABLATED_TAIL_TEXT!r} for {instance['id']} {q[:30]!r}"
            )
    fed = full_ids[prompt_len + ans_len : prompt_len + ans_len + len(bids)].tolist()
    # Fail-loud per-probe id asserts (round-1 pattern, extended to ALL fed
    # boundary ids — plan v11 §4.6 item 2 / kill criterion 1).
    assert fed == bids, (
        f"boundary block fed ids {fed} != expected {bids} for {instance['id']} {q[:30]!r}"
    )
    assert fed[0] == IM_END_TOKEN_ID, (
        f"im_end slot fed id {fed[0]} != {IM_END_TOKEN_ID} for {instance['id']} {q[:30]!r}"
    )
    assert fed[1] == nl_id, f"turn_nl slot fed id {fed[1]} != {nl_id} (\\n)"
    # Union span starts at answer content = prompt_len; boundary block at
    # prompt_len+ans_len+k; tail/head relative to the answer content start.
    # At ans_len=0 (ablate) this yields the 5 boundary singles ONLY; the
    # cc_last predictor slot (rel −1 → abs prompt_len − 1) rides along.
    pos_idx = _positions_for_span(ans_len, boundary_offset=ans_len, extended=extended)
    if ablate or truncate_frac is not None:
        # he-row-set modes: the #594 cc_last predictor slot rides the forward
        # (prompt-side, ablation/truncation-invariant — the drift tripwire).
        pos_idx = {**pos_idx, "cc_last": -1}
    tgt: list = []
    valid: list[bool] = []
    for name in stored_names:
        if name in _POOL_SPECS:
            tgt.append(None)  # pools have no single target index (span reductions)
            valid.append(False)
            continue
        if name in pos_idx:
            tgt.append(prompt_len + pos_idx[name])  # abs index in the real seq
            valid.append(True)
        else:
            tgt.append(None)
            valid.append(False)
    return full_ids, tgt, valid, prompt_len, ans_len, orig_ans_len


def _gather_pools_gpu(
    capture: LayerCapture,
    capture_layers: list[int],
    masks: dict[str, torch.Tensor],
    pool_names: list[str],
) -> dict[str, torch.Tensor]:
    """GPU-side span reductions per pool → {name: (B, Lc, H) fp16 CPU}.

    Computed IN the forward's device memory from ``capture.latest`` BEFORE the
    singles gather clears the buffer (plan v11 §4.6 item 2: the pools are not
    reconstructable from the stored answer-only probe-averaged summaries).
    ``masks`` maps span kind → (B, T) bool device mask. Reductions run in fp32
    on device; only the reduced (B, Lc, H) slice crosses PCIe (fp16 — the same
    transport precision as the singles). Every mask row is guaranteed non-empty
    (answer spans have ≥1 token; boundary blocks always exist), asserted.
    """
    for kind, mask in masks.items():
        counts = mask.sum(dim=1)
        assert bool((counts > 0).all()), f"empty {kind} span mask in batch (counts={counts})"
    per_pool_layers: dict[str, list[torch.Tensor]] = {n: [] for n in pool_names}
    for li in capture_layers:
        hs = capture.latest[li].float()  # (B, T, H) fp32 on device
        for name in pool_names:
            kind, op = _POOL_SPECS[name]
            mask = masks[kind]
            if op == "mean":
                m = mask.unsqueeze(-1).to(hs.dtype)
                red = (hs * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).to(hs.dtype)
            else:  # per-dim max (the #658 maxp token-pool recipe)
                red = hs.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            per_pool_layers[name].append(red.to(torch.float16))
    return {n: torch.stack(sl, dim=1).cpu() for n, sl in per_pool_layers.items()}


def _run_forward_batch(
    model,
    capture,
    capture_layers,
    tokenizer,
    rows,
    stored_names,
    accum,
    coverage,
    lc,
    H,
    extended: bool = False,
    ablate: bool = False,
    truncate: bool = False,
) -> int:
    """Left-pad + one batched forward + GPU-side gather + accumulate; return #probes.

    ``rows`` is a list of ``(full_ids, tgt, valid, prompt_len, ans_len)``. Builds
    a left-padded batch (real tokens at the right edge, boundaries aligned),
    threads EXPLICIT ``position_ids`` (cumsum(mask)−1 clamped at 0 — RoPE indexes
    from 0 per sequence's first real token, without which left-pad silently
    diverges from batch-1), runs ONE forward, gathers the target positions
    GPU-side — plus, when ``extended``, the in-forward span-pool reductions
    (mean_xbnd/maxp_xbnd/uh_mean3/uh_max3/bnd_mean5/bnd_max5 + the internal
    mean_ans parity pool) computed per probe from the device-resident hidden
    states BEFORE the CPU transfer — and sums each covered position/pool into
    ``accum`` (probe-mean at the end). Returns the number of probes accumulated.
    """
    if not rows:
        return 0
    device = model.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else IM_END_TOKEN_ID
    b = len(rows)
    max_len = max(int(r[0].shape[0]) for r in rows)
    n_targets = len(stored_names)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    abs_pos = torch.full((b, n_targets), -1, dtype=torch.long)
    ans_starts = torch.zeros(b, dtype=torch.long)  # abs answer-content start per row
    ans_lens_t = torch.zeros(b, dtype=torch.long)
    for bi, (s, tgt, _valid, prompt_len, ans_len) in enumerate(rows):
        length = int(s.shape[0])
        pad = max_len - length  # LEFT-pad → real tokens occupy [pad, max_len)
        input_ids[bi, pad:] = s
        attn[bi, pad:] = 1
        ans_starts[bi] = pad + prompt_len
        ans_lens_t[bi] = ans_len
        for ti, rel in enumerate(tgt):
            if rel is not None:
                abs_pos[bi, ti] = pad + rel  # shift the in-sequence index by pad
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0).to(device)
    abs_pos_dev = abs_pos.to(device)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids)
    pool_names = [n for n in stored_names if n in _POOL_SPECS]
    pools: dict[str, torch.Tensor] = {}
    he_rows = ablate or truncate  # the he row-set modes (plan v18 §4.6 item 2)
    if extended:
        if not he_rows:
            pool_names = [*pool_names, "mean_ans"]  # internal parity pool rides along
        # Span masks over the padded batch: content [start, start+ans_len),
        # boundary block [start+ans_len, start+ans_len+5), header = last 3 of it.
        # In the he-row-set modes (ablate OR truncate) ONLY bnd5/uh3 are
        # requested (both always 5/3 tokens — plan v15 §12 A9 / v18 §12 A10);
        # the ans/xbnd masks are never built there — in ablate mode they would
        # be EMPTY at ans_len=0 (so _gather_pools_gpu's non-empty assert holds),
        # and in truncate mode ans/xbnd rows are outside the registered row set.
        pos = torch.arange(max_len).unsqueeze(0)  # (1, T)
        st = ans_starts.unsqueeze(1)
        en = (ans_starts + ans_lens_t).unsqueeze(1)  # content end == boundary start
        n_bnd = len(BOUNDARY_BLOCK_IDS)
        masks_cpu = {
            "bnd5": (pos >= en) & (pos < en + n_bnd),
            "uh3": (pos >= en + 2) & (pos < en + n_bnd),
        }
        if not he_rows:
            masks_cpu["ans"] = (pos >= st) & (pos < en)
            masks_cpu["xbnd"] = masks_cpu["ans"] | masks_cpu["bnd5"]
        kinds_needed = {_POOL_SPECS[n][0] for n in pool_names}
        missing_kinds = kinds_needed - set(masks_cpu)
        assert not missing_kinds, (
            f"pool span kinds {missing_kinds} have no mask (ablate={ablate}, truncate={truncate})"
        )
        masks = {k: v.to(device) for k, v in masks_cpu.items() if k in kinds_needed}
        # Pools reduced BEFORE _gather_positions_gpu clears capture.latest.
        pools = _gather_pools_gpu(capture, capture_layers, masks, pool_names)
    picked = _gather_positions_gpu(capture, capture_layers, abs_pos_dev)  # (b, T, Lc, H) cpu
    for bi, (_s, _tgt, valid, _pl, _al) in enumerate(rows):  # rows stay 5-tuples (batch shape)
        for ti, name in enumerate(stored_names):
            if not valid[ti]:
                continue
            vec = picked[bi, ti].float()  # (Lc, H)
            if name not in accum:
                accum[name] = torch.zeros(lc, H, dtype=torch.float32)
            accum[name] += vec
            coverage[name] += 1
        for name in pool_names:
            vec = pools[name][bi].float()  # (Lc, H)
            if name not in accum:
                accum[name] = torch.zeros(lc, H, dtype=torch.float32)
            accum[name] += vec
            coverage[name] = coverage.get(name, 0) + 1
    return b


def _probe_pairs(
    probes: list[str],
    completions: list[str] | None,
    ablate: bool,
    truncate_frac: float | None,
) -> list[tuple[str, str | None]]:
    """(probe, answer) pairs per mode, asserting the completions contract.

    Ablate mode must NOT receive completions (plan v15 §10 code-truth); the
    `_btdr` truncate mode REQUIRES them — the answer text IS the dosed variable
    (plan v18 §4.6 item 2, the inverse assert).
    """
    if ablate:
        assert completions is None, "ablate mode must not receive completions (plan v15 §10)"
        return [(q, None) for q in probes]
    if truncate_frac is not None:
        assert completions is not None, (
            "truncate mode REQUIRES completions — the answer text IS the dosed "
            "variable (plan v18 §4.6 item 2, the inverse of ablate mode's assert)"
        )
    return list(zip(probes, completions, strict=True))


def capture_positions_for_context(
    model,
    tokenizer,
    instance: dict,
    probes: list[str],
    completions: list[str] | None,
    capture: LayerCapture,
    n_layers: int,
    capture_layers: list[int],
    batch_probes: int,
    extended: bool = False,
    ablate: bool = False,
    truncate_frac: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict, dict[str, torch.Tensor]]:
    """Teacher-force each (prompt [+ answer] + boundary block); capture positions.

    Boundary block = 2 tokens (parent) or the 5-token assistant-turn
    continuation (``extended`` — plan v11 §4), which ALSO computes the 6
    in-forward span pools + the internal ``mean_ans`` parity pool per probe.
    ``ablate`` (plan v15 §4): the answer span is EMPTY — ``completions`` must be
    None (no completion text ever enters this path), the stored rows are the 10
    ``he_stored_position_names()`` (cc_last + 5 boundary singles + 4 bnd5/uh3
    pools; NO tail/head/xbnd/mean_ans), and every row is covered on every probe
    (the boundary block always exists — zero skips, plan v15 success criterion).
    ``truncate_frac`` (plan v18 §4): the answer span is ID-prefix CUT to
    ``n_keep = max(1, ceil(k * ans_len))`` — ``completions`` is REQUIRED
    (asserted not None, the inverse of ablate mode's assert: the answer text IS
    the dosed variable), the stored rows are the SAME 10 he rows, and the
    per-probe (n_keep, ans_len) pairs are recorded in ``diag`` for the store
    manifest.

    Returns ``(pos_summaries, coverage, diag, extras)`` where
      pos_summaries[position] = (Lc, H) probe-MEAN summary vector for that
        position/pool over the probes that had it,
      coverage[position] = number of probes that contributed,
      diag = per-context diagnostics (n_probes_used, empty_completions,
        median_answer_len, boundary_token_ids_seen),
      extras = {"mean_ans": (Lc, H) fp32} when ``extended`` (the in-forward
        answer-only probe-mean — the #658 v0 store-mean parity vector), else {}.

    The im_end / turn_nl positions are the two boundary tokens AFTER the
    answer content; they are appended to the teacher-forced sequence so the
    forward materializes their residual stream. Fail loud on a boundary-token
    id mismatch (never silently capture the wrong slot) — the im_end id is
    asserted; the turn_nl id is recorded (tokenizer-dependent) + asserted to
    decode to a newline-bearing token.

    ``batch_probes`` is a REAL knob (default 8): probes are batched with
    LEFT-PADDING (all turn-end boundaries align at the right edge), one forward
    per batch instead of one per probe. Left-padding requires EXPLICIT
    ``position_ids`` (cumsum(attention_mask) − 1, clamped at 0) so RoPE indexes
    from 0 at each sequence's first real token — without it the padded positions
    silently diverge from the batch-1 read (``.claude/rules/
    left_pad_position_ids_required``). The residual stream is sliced at the ~34
    target positions GPU-side (``_gather_positions_gpu``) BEFORE the CPU transfer,
    so only the thin (batch, 34, 28, H) slice crosses PCIe. Batched forward output
    is byte-identical (cosine ≥ 0.999) to the batch-1 read — the smoke asserts it.
    """
    lc = len(capture_layers)
    H = model.config.hidden_size
    he_rows = ablate or truncate_frac is not None
    # Accumulators: sum over probes + count per position (probe-mean at the end).
    if he_rows:
        stored_names = he_stored_position_names()
    else:
        stored_names = uh_stored_position_names() if extended else stored_position_names()
    accum: dict[str, torch.Tensor] = {}
    coverage: dict[str, int] = {p: 0 for p in stored_names}
    ans_lens: list[int] = []
    turn_nl_ids_seen: set[int] = set()

    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(nl_ids) != 1:
        raise RuntimeError(f"expected single-token '\\n', got {nl_ids} (tokenizer drift)")
    nl_id = nl_ids[0]
    # Pin the newline id to the Qwen-2.5 family id 198 (same for 7B production +
    # 0.5B smoke). A drifted tokenizer would silently capture the WRONG turn_nl
    # position across the whole run — refuse rather than run.
    if nl_id != TURN_NL_TOKEN_ID:
        raise RuntimeError(
            f"tokenizer newline id {nl_id} != Qwen-2.5 pinned id {TURN_NL_TOKEN_ID} — "
            "refusing to run with a drifted tokenizer (would capture the wrong turn_nl "
            "slot for every probe)"
        )

    # Build per-probe (full_ids, target-index, valid) tuples, then run them in
    # left-padded batches of `batch_probes` (one forward per batch, not per probe).
    batch = max(1, int(batch_probes))
    built = []  # (full_ids, tgt, valid, prompt_len, ans_len) per non-empty probe
    empty = 0
    trunc_n_keep: list[int] = []  # per used probe, truncate mode only
    trunc_ans_len: list[int] = []  # pre-cut lengths, truncate mode only
    pairs = _probe_pairs(probes, completions, ablate, truncate_frac)
    for q, ans in pairs:
        item = _build_probe_row(
            model,
            tokenizer,
            instance,
            q,
            ans,
            stored_names,
            nl_id,
            extended=extended,
            ablate=ablate,
            truncate_frac=truncate_frac,
        )
        if item is None:
            empty += 1
            logger.warning("empty completion for %s probe=%r — skipping", instance["id"], q[:40])
            continue
        full_ids, tgt, valid, prompt_len, ans_len, orig_ans_len = item
        turn_nl_ids_seen.add(nl_id)
        ans_lens.append(ans_len)
        if truncate_frac is not None:
            trunc_n_keep.append(ans_len)  # ans_len IS n_keep post-cut (asserted in-row)
            trunc_ans_len.append(orig_ans_len)
        built.append((full_ids, tgt, valid, prompt_len, ans_len))

    n_used = 0
    for lo in range(0, len(built), batch):
        rows = built[lo : lo + batch]
        n_used += _run_forward_batch(
            model,
            capture,
            capture_layers,
            tokenizer,
            rows,
            stored_names,
            accum,
            coverage,
            lc,
            H,
            extended=extended,
            ablate=ablate,
            truncate=truncate_frac is not None,
        )

    if n_used == 0:
        raise RuntimeError(f"context {instance['id']}: every probe produced an empty answer")
    pos_summaries = {name: (accum[name] / coverage[name]) for name in accum}
    extras: dict[str, torch.Tensor] = {}
    if extended and not he_rows:
        # mean_ans is an INTERNAL parity vector (the #658 v0 `mean` recipe
        # recomputed in-forward), never a stored summary row. Undefined in
        # ablate mode (ans_len=0) and outside the registered truncate row set —
        # the cc_last row is the drift tripwire in the he-row-set modes.
        extras["mean_ans"] = pos_summaries.pop("mean_ans")
        coverage.pop("mean_ans", None)
    # Assert boundary positions/pools are ALWAYS covered (span_len-independent).
    if he_rows:
        always_covered = list(stored_names)  # all 10 rows exist on every probe
    else:
        always_covered = ["im_end", "turn_nl"]
        if extended:
            always_covered += UH_SUMMARY_NAMES
    for b in always_covered:
        if coverage[b] != n_used:
            raise RuntimeError(
                f"context {instance['id']}: boundary position/pool {b} coverage "
                f"{coverage[b]} != n_used {n_used} — the boundary token was not "
                "captured on every probe (capture/slice bug)"
            )
    diag = {
        "n_probes_used": n_used,
        "empty_completions": empty,
        "median_answer_len": sorted(ans_lens)[len(ans_lens) // 2] if ans_lens else 0,
        "turn_nl_ids_seen": sorted(turn_nl_ids_seen),
    }
    if truncate_frac is not None:
        # Plan v18 §4 edge rules: per-context n_keep + ans_len recorded in the
        # store manifest (probe order = the used-probe order above).
        diag["truncate_frac"] = float(truncate_frac)
        diag["n_keep_per_probe"] = trunc_n_keep
        diag["ans_len_per_probe"] = trunc_ans_len
    return pos_summaries, coverage, diag, extras


# ── inputs (battery + completions) ───────────────────────────────────────────


def _resolve_battery(local_hint: Path | None) -> dict:
    """Load + sha256-pin the 50-context battery (local fast path, else HF snapshot).

    Local ``data/issue594/battery.json`` is gitignored (absent from the git-clone
    GCP lane), so on a miss we fetch the sha256-pinned HF snapshot
    (``BATTERY50_HF_FILE``) — the artifact-reuse (h) fetchability contract.
    Either way the sha256 is asserted against ``BATTERY50_SHA256`` (fail loud on
    drift, the #600 HF-mirror guard).
    """
    from huggingface_hub import hf_hub_download

    candidates = []
    if local_hint is not None:
        candidates.append(Path(local_hint))
    candidates.append(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    for c in candidates:
        if c.is_file() and sha256_file(c) == BATTERY50_SHA256:
            logger.info("battery: local sha-matched %s", c)
            return load_json(c)
    logger.info("battery: fetching sha-pinned HF snapshot %s", BATTERY50_HF_FILE)
    path = hf_hub_download(HF_DATA_REPO, BATTERY50_HF_FILE, repo_type="dataset")
    assert_sha256(path, BATTERY50_SHA256, "battery50")
    return load_json(path)


def _completions_prefix(genre: str) -> str:
    """HF raw-completions prefix per genre (betley = the parent's, bit-for-bit)."""
    return I658_RAW_COMPLETIONS_PREFIX if genre == "betley" else G1_RAW_COMPLETIONS_PREFIX


def _load_stored_completions(ctx_id: str, genre: str = "betley") -> list[dict]:
    """The 48 stored (probe, completion) pairs for one context from HF.

    Reads ``<genre raw_completions prefix>/<ctx>.json`` (schema
    ``{context_id, completions:[{probe, completion}, ...]}`` — head-check
    VERIFIED identical across genres) — the model's OWN on-policy answers #658
    generated + stored. NO regeneration (single-variable discipline: the
    probe-corpus genre is the variable, the completions inherited).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO, f"{_completions_prefix(genre)}/{ctx_id}.json", repo_type="dataset"
    )
    blob = load_json(path)
    if blob.get("context_id") != ctx_id:
        raise RuntimeError(f"completions ctx mismatch: {blob.get('context_id')} != {ctx_id}")
    cells = blob["completions"]
    if not cells:
        raise RuntimeError(f"context {ctx_id}: no stored completions")
    return cells


def _load_manifest_context_ids(genre: str = "betley") -> list[str]:
    """The 50 store context_ids per genre; the g1 manifest is probe-pool-pinned."""
    from huggingface_hub import hf_hub_download

    manifest_file = I658_STORE_MANIFEST if genre == "betley" else G1_STORE_MANIFEST
    man = load_json(hf_hub_download(HF_DATA_REPO, manifest_file, repo_type="dataset"))
    if genre == "g1":
        assert_g1_probe_pool_hash(man, G1_STORE_MANIFEST)
    return context_ids_from_manifest(man)


# ── g1 cc_last recomputation parity probe (plan v6 §11 standing note) ─────────


def _cc_last_parity_probe(
    model,
    tokenizer,
    capture: LayerCapture,
    instance: dict,
    first_probe: str,
    ctx_id: str,
    n_layers: int,
    compare_store: bool,
    min_cosine: float = 0.999,
) -> dict:
    """One-context recomputation parity probe for the g1 store's ``cc_last``.

    Recomputes the per-genre c_C with the EXACT #658 ``--cc-recompute-last``
    convention (``issue658_extract_base_store`` G3): ONE prompt-only forward over
    ``apply_chat_template(messages_for_instance(inst, first_probe),
    add_generation_prompt=True)``, residual at position ``prompt_len - 1`` (the
    assistant-header newline — the #594 last-input-token slot) per layer, fp32.
    With ``compare_store`` (production 7B only) it asserts per-layer cosine vs
    the g1 ``v0_summaries.pt::cc_last[ctx]`` >= ``min_cosine`` — a ROBUSTNESS
    assert certifying the store's cc_last is the quantity its ``cc_reuse_note``
    claims, run BEFORE any fit consumes it; not a science gate. Smoke mode
    (tiny model) runs the recompute path only (shape + non-degenerate norms) —
    the 7B store values cannot match a 0.5B forward by construction.
    """
    tmpl = tokenizer.apply_chat_template(
        messages_for_instance(instance, first_probe), tokenize=False, add_generation_prompt=True
    )
    pinputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
    with torch.no_grad():
        _ = model(**pinputs)
    prompt_len = int(pinputs["input_ids"].shape[1])
    fresh = torch.stack(
        [capture.latest[li][0, prompt_len - 1, :].float().cpu() for li in range(n_layers)]
    )  # (L, H) — the #658 last_prompt_stack convention
    capture.latest.clear()
    norms = fresh.norm(dim=1)
    if not torch.isfinite(fresh).all() or (norms < 1e-6).any():
        raise RuntimeError(f"cc_last recompute degenerate for {ctx_id} (norms {norms.tolist()})")
    out: dict = {"ctx_id": ctx_id, "prompt_len": prompt_len, "n_layers": int(n_layers)}
    if not compare_store:
        out["compared_to_store"] = False
        logger.info("[phase=cc_parity] recompute-only (smoke/non-7B): shape %s OK", fresh.shape)
        return out
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, G1_V0_SUMMARIES, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    assert_g1_probe_pool_hash(blob, G1_V0_SUMMARIES)
    store_cc_all = blob.get("cc_last") or {}
    store_cc = store_cc_all.get(ctx_id)
    if store_cc is None:
        raise RuntimeError(f"g1 v0_summaries.pt has no cc_last[{ctx_id!r}] — parity impossible")
    store_cc = store_cc.float()
    if store_cc.shape != fresh.shape:
        raise RuntimeError(f"cc_last shape drift: store {tuple(store_cc.shape)} vs {fresh.shape}")
    cos = torch.nn.functional.cosine_similarity(fresh, store_cc, dim=1)  # (L,)
    out.update(
        {
            "compared_to_store": True,
            "min_layer_cosine": float(cos.min()),
            "mean_layer_cosine": float(cos.mean()),
            "min_cosine_threshold": min_cosine,
        }
    )
    logger.info(
        "[phase=cc_parity] ctx=%s min/mean layer cosine %.6f / %.6f (threshold %.3f)",
        ctx_id,
        out["min_layer_cosine"],
        out["mean_layer_cosine"],
        min_cosine,
    )
    if out["min_layer_cosine"] < min_cosine:
        raise RuntimeError(
            f"g1 cc_last parity probe FAILED for {ctx_id}: min layer cosine "
            f"{out['min_layer_cosine']:.6f} < {min_cosine} — the store's cc_last does not "
            "reproduce under the #658 --cc-recompute-last convention (procedure drift); "
            "refusing to fit against it (plan v6 §11 c_C procedure parity)"
        )
    return out


# ── extended-boundary parity + smoke oracles (plan v11 §4.6 item 2) ──────────


def _per_layer_min_cosine(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(min, mean) per-layer cosine between two (Lc, H) summary stacks (fp64)."""
    A = torch.as_tensor(a, dtype=torch.float64)
    B = torch.as_tensor(b, dtype=torch.float64)
    assert A.shape == B.shape, (tuple(A.shape), tuple(B.shape))
    cos = torch.nn.functional.cosine_similarity(A, B, dim=1)  # (Lc,)
    return float(cos.min()), float(cos.mean())


def _numpy_reference_capture(
    model,
    tokenizer,
    capture,
    instance,
    probes,
    completions,
    capture_layers,
    nl_id,
    ablate=False,
    truncate_frac=None,
) -> dict[str, np.ndarray]:
    """Batch-1 FULL-hidden-state numpy reference for singles + pools (smoke oracle).

    One un-padded batch-1 forward per probe; the full (T, H) hidden state per
    layer is pulled to CPU fp32 and reduced in NUMPY (mean / per-dim max over
    the span slices; singles by direct indexing), then probe-meaned — an
    implementation-independent oracle for the in-forward GPU-side reductions.
    Returns {name: (Lc, H) fp32 np.ndarray} over the 43 stored rows + mean_ans
    (extended) or the 10 he rows incl. cc_last (``ablate`` OR ``truncate_frac``
    — plan v15/v18 §4.6 item 2 smoke pool oracle; ans/xbnd spans are outside
    the he row set, and in ablate mode do not exist at ans_len=0).
    """
    he_rows = ablate or truncate_frac is not None
    stored_names = he_stored_position_names() if he_rows else uh_stored_position_names()
    names_all = list(stored_names) if he_rows else [*stored_names, "mean_ans"]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {n: 0 for n in names_all}
    n_bnd = len(BOUNDARY_BLOCK_IDS)
    pairs = [(q, None) for q in probes] if ablate else list(zip(probes, completions, strict=True))
    for q, ans in pairs:
        item = _build_probe_row(
            model,
            tokenizer,
            instance,
            q,
            ans,
            stored_names,
            nl_id,
            extended=True,
            ablate=ablate,
            truncate_frac=truncate_frac,
        )
        if item is None:
            continue
        full_ids, _tgt, _valid, prompt_len, ans_len, _orig = item
        with torch.no_grad():
            _ = model(input_ids=full_ids.unsqueeze(0).to(model.device))
        hs = np.stack(
            [capture.latest[li][0].float().cpu().numpy() for li in capture_layers]
        )  # (Lc, T, H)
        capture.latest.clear()
        spans = {
            "bnd5": (prompt_len + ans_len, prompt_len + ans_len + n_bnd),
            "uh3": (prompt_len + ans_len + 2, prompt_len + ans_len + n_bnd),
        }
        if not he_rows:
            spans["ans"] = (prompt_len, prompt_len + ans_len)
            spans["xbnd"] = (prompt_len, prompt_len + ans_len + n_bnd)
        pos_idx = _positions_for_span(ans_len, boundary_offset=ans_len, extended=True)
        if he_rows:
            pos_idx = {**pos_idx, "cc_last": -1}
        for name in names_all:
            if name in _POOL_SPECS:
                kind, op = _POOL_SPECS[name]
                lo, hi = spans[kind]
                seg = hs[:, lo:hi, :]  # (Lc, S, H)
                red = seg.mean(axis=1) if op == "mean" else seg.max(axis=1)
            elif name in pos_idx:
                red = hs[:, prompt_len + pos_idx[name], :]
            else:
                continue  # coverage miss (short answer)
            sums[name] = sums.get(name, 0.0) + red.astype(np.float64)
            counts[name] += 1
    return {n: (sums[n] / counts[n]).astype(np.float32) for n in sums}


def _extended_smoke_asserts(
    model,
    tokenizer,
    capture,
    instance,
    probes,
    completions,
    capture_layers,
    n_layers,
    pos_batched: dict[str, torch.Tensor],
    mean_ans_batched: torch.Tensor | None,
    min_cos: float = 0.999,
    max_rel_l2: float = 5e-3,
    ablate: bool = False,
    truncate_frac: float | None = None,
) -> dict:
    """Smoke oracle triplet for the extended-boundary capture (plan v11 §4.6 item 2).

    (a) batched-vs-batch-1: re-captures the SAME context at ``batch_probes=1``
        through the identical in-forward path and asserts per-layer cosine
        ≥ ``min_cos`` for EVERY row (43 singles+pools) + mean_ans — the round-1
        left-pad/position_ids assert extended to the new positions and pools.
    (b) numpy reference: asserts the batch-1 in-forward reductions equal an
        implementation-independent batch-1 numpy full-span reference (cosine +
        relative L2 ≤ ``max_rel_l2`` — the fp16 PCIe transport bound).
    (c) the 5 boundary ids are asserted per probe inside ``_build_probe_row``
        on every path above (fails loud before any comparison).

    ``ablate`` (plan v15 §4.6 item 2): the same triplet over the 10 he rows
    (cc_last + 5 singles + 4 pools; NO mean_ans — ``mean_ans_batched`` is None),
    plus the ablate-mode full-sequence/decoded-tail asserts firing per probe
    inside ``_build_probe_row`` on every path. ``truncate_frac`` (plan v18
    §4.6 item 2): the same 10-he-row triplet over the ID-prefix-truncated
    sequences, plus the truncate-mode full-sequence/decoded-tail/n_keep
    asserts firing per probe on every path.

    Returns a summary dict for the manifest. Raises on any violation.
    """
    he_rows = ablate or truncate_frac is not None
    pos_b1, _cov1, _diag1, extras1 = capture_positions_for_context(
        model,
        tokenizer,
        instance,
        probes,
        completions,
        capture,
        n_layers,
        capture_layers,
        batch_probes=1,
        extended=True,
        ablate=ablate,
        truncate_frac=truncate_frac,
    )
    ref = _numpy_reference_capture(
        model,
        tokenizer,
        capture,
        instance,
        probes,
        completions,
        capture_layers,
        nl_id=TURN_NL_TOKEN_ID,
        ablate=ablate,
        truncate_frac=truncate_frac,
    )
    all_batched = dict(pos_batched)
    all_b1 = dict(pos_b1)
    if not he_rows:
        all_batched["mean_ans"] = mean_ans_batched
        all_b1["mean_ans"] = extras1["mean_ans"]
    out = {"rows_checked": 0, "min_cos_batched_vs_b1": 1.0, "min_cos_b1_vs_numpy": 1.0}
    for name, vec_b in all_batched.items():
        vec_1 = all_b1.get(name)
        assert vec_1 is not None, f"batch-1 recapture missing row {name!r}"
        c_min, _ = _per_layer_min_cosine(vec_b, vec_1)
        out["min_cos_batched_vs_b1"] = min(out["min_cos_batched_vs_b1"], c_min)
        if c_min < min_cos:
            raise RuntimeError(
                f"batched-vs-batch-1 cosine {c_min:.6f} < {min_cos} for row {name!r} — "
                "the left-padded batched forward diverges from batch-1 (position_ids/"
                "mask bug); refusing (round-1 assert, extended)"
            )
        r = ref.get(name)
        assert r is not None, f"numpy reference missing row {name!r}"
        c_min2, _ = _per_layer_min_cosine(vec_1, torch.from_numpy(r))
        rel = float(np.linalg.norm(vec_1.float().numpy() - r) / max(np.linalg.norm(r), 1e-9))
        out["min_cos_b1_vs_numpy"] = min(out["min_cos_b1_vs_numpy"], c_min2)
        if c_min2 < min_cos or rel > max_rel_l2:
            raise RuntimeError(
                f"in-forward reduction vs numpy reference mismatch for row {name!r}: "
                f"min-layer cosine {c_min2:.6f} (floor {min_cos}), rel-L2 {rel:.2e} "
                f"(cap {max_rel_l2}) — the GPU-side span reduction is wrong"
            )
        out["rows_checked"] += 1
    logger.info(
        "[phase=smoke_asserts] %d rows OK (batched-vs-b1 min cos %.6f; b1-vs-numpy min cos %.6f)",
        out["rows_checked"],
        out["min_cos_batched_vs_b1"],
        out["min_cos_b1_vs_numpy"],
    )
    return out


_V0_MEAN_CACHE: dict[str, object] = {}


def _v0_store_mean_parity(mean_ans: torch.Tensor, ctx_id: str, min_cos: float = 0.999) -> dict:
    """Assert the in-forward answer-only probe-mean matches #658's stored `mean`.

    The capture-drift tripwire (plan v11 §5 / kill criterion 2): per layer,
    cosine(in-forward mean_ans, v0_summaries['summaries']['mean'][ctx]) must be
    ≥ ``min_cos`` — a miss means THIS round's teacher-forced capture drifted
    from the round-1/#658 pipeline and every fit downstream would be on
    drifted activations. Production-only caller gate (7B, uncapped probes).
    """
    if "blob" not in _V0_MEAN_CACHE:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(HF_DATA_REPO, I658_V0_SUMMARIES, repo_type="dataset")
        _V0_MEAN_CACHE["blob"] = torch.load(p, weights_only=False)
    blob = _V0_MEAN_CACHE["blob"]
    store_mean = blob["summaries"]["mean"].get(ctx_id)
    if store_mean is None:
        raise RuntimeError(f"v0_summaries has no mean[{ctx_id!r}] — parity impossible")
    c_min, c_mean = _per_layer_min_cosine(mean_ans, store_mean.float())
    if c_min < min_cos:
        raise RuntimeError(
            f"v0 store-mean parity FAILED for {ctx_id}: min layer cosine {c_min:.6f} < "
            f"{min_cos} — the extended-boundary capture drifted from the #658 store "
            "(halt before any fit; plan v11 kill criterion 2)"
        )
    return {"ctx_id": ctx_id, "min_layer_cosine": c_min, "mean_layer_cosine": c_mean}


def _round1_store_drift_check(
    pos_summaries: dict[str, torch.Tensor],
    coverage: dict[str, int],
    ctx_id: str,
    min_cos: float = 0.999,
) -> dict:
    """Assert the recaptured old-34 positions match the round-1 store (first ctx).

    Causal attention ⇒ appending the header tokens cannot change earlier
    positions (plan v11 §12 A15) — the old positions recaptured in the SAME
    extended forward must be cosine-close to the committed round-1
    ``answer_position_sweep/<ctx>.pt``. Production-only caller gate.
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO,
        f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}/{ctx_id}.pt",
        repo_type="dataset",
    )
    blob = torch.load(p, weights_only=False)
    r1 = {name: blob["pos_vectors"][i] for i, name in enumerate(blob["positions"])}
    r1_cov = dict(blob["coverage"])
    checked, worst = 0, 1.0
    for name in stored_position_names():
        if coverage.get(name, 0) <= 0 or r1_cov.get(name, 0) <= 0:
            continue
        if coverage.get(name) != r1_cov.get(name):
            raise RuntimeError(
                f"round-1 drift check {ctx_id}: coverage mismatch for {name!r} "
                f"({coverage.get(name)} vs round-1 {r1_cov.get(name)}) — probe-set drift"
            )
        c_min, _ = _per_layer_min_cosine(pos_summaries[name], r1[name].float())
        worst = min(worst, c_min)
        checked += 1
        if c_min < min_cos:
            raise RuntimeError(
                f"round-1 store drift check FAILED for {ctx_id}/{name}: min layer "
                f"cosine {c_min:.6f} < {min_cos} — the recaptured old positions do "
                "not reproduce the round-1 store (capture drift; halt before any fit)"
            )
    logger.info(
        "[phase=drift_check] ctx=%s %d old positions OK (worst min-layer cosine %.6f)",
        ctx_id,
        checked,
        worst,
    )
    return {"ctx_id": ctx_id, "positions_checked": checked, "worst_min_layer_cosine": worst}


# ── ablate-mode probe pool + cc_last drift tripwire (plan v15 §4.6 item 2) ───


def _betley_probe_pool(battery: dict) -> list[str]:
    """The 48 Betley probes WITHOUT reading any stored completions (ablate mode).

    Rebuilds the #594 builder pool (Betley preregistered paraphrases minus the
    main-8) and asserts the ordered-pool hash against BOTH the sha-pinned
    battery's ``meta.probe_pool_hash`` AND the #594 store pin
    (``I594_PROBE_POOL_HASH``) — fail loud on any drift. CODE-TRUTH for the
    plan v15 §10 'NO completions consumed' claim: the ablate path never touches
    the raw-completions prefix (the probes are the only per-probe input).
    """
    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=200, exclude=main8)
    got = probes_hash(probes)
    battery_pin = (battery.get("meta") or {}).get("probe_pool_hash")
    if got != I594_PROBE_POOL_HASH or got != battery_pin:
        raise RuntimeError(
            f"ablate-mode probe pool hash drift: rebuilt {got[:16]}… vs #594 pin "
            f"{I594_PROBE_POOL_HASH[:16]}… / battery meta {str(battery_pin)[:16]}… — "
            "refusing to capture on a drifted probe grid (plan v15 §10 grid pin)"
        )
    if len(probes) != 48:
        raise RuntimeError(f"ablate-mode probe pool has {len(probes)} probes, expected 48")
    return probes


_CC594_CACHE: dict[str, object] = {}


def _cc594_store_parity(
    cc_mean: torch.Tensor, ctx_id: str, capture_layers: list[int], min_cos: float = 0.999
) -> dict:
    """Assert the in-forward probe-mean ``cc_last`` matches the #594 c_C store.

    The ablate-mode capture-drift tripwire (plan v15 §5 / kill criterion 2 —
    the ``mean_ans`` tripwire is undefined at ans_len=0): ``cc_last`` is
    prompt-side and ABLATION-INVARIANT by causal attention, so its probe-mean
    over the SAME 48 probes must reproduce the #594 store row — the EXACT
    tensor the recon fits consume as the predictor — at per-layer cosine
    ≥ ``min_cos``. Production-only caller gate (7B model, uncapped probes);
    the pool hash pin is asserted on the store blob before any compare.
    """
    if "blob" not in _CC594_CACHE:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(HF_DATA_REPO, I594_CC_LAST_FILE, repo_type="dataset")
        blob = torch.load(p, weights_only=False)
        pph = blob.get("probe_pool_hash")
        if pph != I594_PROBE_POOL_HASH:
            raise RuntimeError(f"#594 c_C probe_pool_hash drift: {pph} != {I594_PROBE_POOL_HASH}")
        _CC594_CACHE["blob"] = blob
        _CC594_CACHE["iid_to_row"] = {iid: i for i, iid in enumerate(blob["instance_ids"])}
    blob = _CC594_CACHE["blob"]
    row = _CC594_CACHE["iid_to_row"].get(ctx_id)  # type: ignore[union-attr]
    if row is None:
        raise RuntimeError(f"#594 c_C store has no row for {ctx_id!r} — cc parity impossible")
    store = blob["tensor"][row][capture_layers].float()  # type: ignore[index]
    c_min, c_mean = _per_layer_min_cosine(cc_mean, store)
    if c_min < min_cos:
        raise RuntimeError(
            f"cc_last parity FAILED for {ctx_id}: min layer cosine {c_min:.6f} < {min_cos} — "
            "the ablated capture's prompt-side read does not reproduce the #594 c_C store "
            "(inter-round capture drift; halt before any fit — plan v15 kill criterion 2)"
        )
    return {"ctx_id": ctx_id, "min_layer_cosine": c_min, "mean_layer_cosine": c_mean}


# ── HF model load ─────────────────────────────────────────────────────────────


def _load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


# ── upload (fail-loud) ────────────────────────────────────────────────────────


def _upload_store(
    out_dir: Path,
    ctx_ids: list[str],
    smoke: bool,
    genre: str = "betley",
    extended: bool = False,
    ablate: bool = False,
    truncate_frac: float | None = None,
) -> str:
    """Bulk-commit the aligned-subset store to HF (one upload_folder commit).

    Uploads ``answer_position_sweep[_<tag>]/<ctx>.pt`` (+ manifest.json)
    via ONE ``upload_folder`` commit (never a per-file loop — the #664
    504-storm), then verifies the per-context file count on a FRESH listing
    (fail loud on a mismatch). Skipped for --smoke / --no-upload by the caller.
    ``extended`` routes to the `_uh` store subdir (plan v11 § Storage naming);
    ``ablate`` to the `_he` subdir (plan v15 § Storage naming);
    ``truncate_frac`` to the per-k `_btdr` subdir (plan v18 § Storage naming).
    """
    from huggingface_hub import HfApi

    if truncate_frac is not None:
        base_subdir = ANSWER_POSITION_SWEEP_BTDR_SUBDIR_TMPL.format(pct=btdr_pct(truncate_frac))
    elif ablate:
        base_subdir = ANSWER_POSITION_SWEEP_HE_SUBDIR
    elif extended:
        base_subdir = ANSWER_POSITION_SWEEP_UH_SUBDIR
    else:
        base_subdir = (
            ANSWER_POSITION_SWEEP_SUBDIR if genre == "betley" else G1_ANSWER_POSITION_SWEEP_SUBDIR
        )
    subdir = base_subdir + ("_smoke" if smoke else "")
    path_in_repo = f"{HF_PREFIX}/{subdir}"
    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.pt", "manifest.json"],
        commit_message=f"issue #810: answer position sweep store ({len(ctx_ids)} contexts)",
    )
    remote = scoped_remote_listing(path_in_repo)
    # Exact expected set: every per-context tensor + the manifest (readers
    # resolve <prefix>/manifest.json first — a store without it is unusable).
    expected = {f"{path_in_repo}/{c}.pt" for c in ctx_ids} | {f"{path_in_repo}/manifest.json"}
    missing = expected - remote
    if missing:
        raise RuntimeError(
            f"aligned-subset store upload verification FAILED: {len(missing)} of "
            f"{len(expected)} expected files missing on the Hub under {path_in_repo}/ "
            f"(e.g. {sorted(missing)[:3]})"
        )
    logger.info(
        "aligned-subset store verified: %d contexts + manifest under %s/",
        len(ctx_ids),
        path_in_repo,
    )
    return path_in_repo


def _upload_uh_summaries(pack_path: Path, hf_file: str = UH_SUMMARIES_HF_FILE) -> str:
    """Upload the compact new-row summaries tensor (fail-loud single-file commit).

    The pack (~90 MB: 50 ctx × 9 rows × 28 layers × 3584 fp16) is the CPU-chain
    input (plan v11/v15 §6.5) — it lands at ``hf_file`` (``UH_SUMMARIES_HF_FILE``
    for the `_uh` round, ``HE_SUMMARIES_HF_FILE`` for the ablate round) on the
    data repo BEFORE the GPU is released, verified on a FRESH listing.
    Single-file `upload_file` is correct here (ONE file, not a per-file loop).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(pack_path),
        path_in_repo=hf_file,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue #810: {hf_file} (CPU-chain input pack)",
    )
    pack_on_hub = retry_hub_quota(
        lambda: api.file_exists(HF_DATA_REPO, hf_file, repo_type="dataset", revision="main")
    )
    if not pack_on_hub:
        raise RuntimeError(
            f"summaries-pack upload verification FAILED: {hf_file} missing on a "
            "fresh Hub listing — refusing to treat a partial upload as success"
        )
    logger.info("summaries pack verified at %s", hf_file)
    return hf_file


# ── sentinel (poll_pipeline contract) ─────────────────────────────────────────


def _write_sentinel(kind: str, note: dict, out_dir: Path) -> None:
    """Write the poll_pipeline.py-conformant end-of-run sentinel.

    Required keys per poll_pipeline._SENTINEL_REQUIRED_KEYS:
    sentinel_schema_version (int 1), kind (full marker string), version (int).
    The marker body goes under ``note``.
    """
    slug = kind.replace(":", "_")
    log_dir = Path("/workspace/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / f"issue-810-{slug}-{int(time.time())}.json"
    except OSError:
        # Off-pod (no /workspace): write next to the output for the smoke.
        target = out_dir / f"issue-810-{slug}-sentinel.json"
    dump_json(
        {
            "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
            "kind": kind,
            "version": 1,
            "note": note,
            "ts": int(time.time()),
        },
        target,
    )
    logger.info("wrote sentinel %s", target)


# ── main ──────────────────────────────────────────────────────────────────────


def _finalize_extended_outputs(
    args,
    manifest: dict,
    v0_parity: dict,
    drift_check: dict | None,
    smoke_asserts: dict | None,
    uh_pack_rows: dict,
    uh_pack_cov: dict,
    capture_layers: list[int],
    ctx_ids: list[str],
    out_dir: Path,
    compare_store: bool,
    cc_parity: dict | None = None,
    truncate_frac: float | None = None,
) -> Path:
    """Extended-boundary manifest provenance + the compact summaries pack.

    Mutates ``manifest`` (plan v11/v15/v18 § Storage naming: `extended_boundary`,
    `ablate_answer`, `truncate_frac`, `boundary_block_ids`, pool semantics, the
    parity records) and writes the CPU-chain input pack (uh_summaries.pt,
    he_summaries.pt in ablate mode, or btdr_summaries_k{pct}.pt per k in
    truncate mode — plan §6.5) next to the store dir. Returns the pack path.
    """
    ablate = bool(args.ablate_answer)
    truncate = truncate_frac is not None
    pack_rows_names = HE_SUMMARY_NAMES if (ablate or truncate) else UH_SUMMARY_NAMES
    manifest["extended_boundary"] = True
    manifest["ablate_answer"] = ablate
    manifest["boundary_block_ids"] = list(BOUNDARY_BLOCK_IDS)
    if truncate:
        manifest["truncate_frac"] = float(truncate_frac)
        manifest["uh_pool_semantics"] = (
            "uh_mean3/uh_max3: mean / per-dim max over the 3 next-user-header tokens; "
            "bnd_mean5/bnd_max5: over all 5 boundary tokens — computed IN the forward "
            "per probe (GPU-side, fp32 reduce, fp16 transport), then probe-meaned. "
            "Answer span ID-prefix-truncated to n_keep = max(1, ceil(k*ans_len)) per "
            "probe (plan v18 §4; per-context n_keep/ans_len in per_context_diag); "
            "ANS/XBND pools + tail/head positions outside the registered he row set; "
            "cc_last = the #594 last-input-token predictor slot, riding the same forward."
        )
        manifest["cc594_parity"] = cc_parity or {"skipped": True}
    elif ablate:
        manifest["uh_pool_semantics"] = (
            "uh_mean3/uh_max3: mean / per-dim max over the 3 next-user-header tokens; "
            "bnd_mean5/bnd_max5: over all 5 boundary tokens — computed IN the forward "
            "per probe (GPU-side, fp32 reduce, fp16 transport), then probe-meaned. "
            "ANS/XBND pools + tail/head positions UNDEFINED at ans_len=0 (plan v15 §4); "
            "cc_last = the #594 last-input-token predictor slot, riding the same forward."
        )
        manifest["cc594_parity"] = cc_parity or {"skipped": True}
    else:
        manifest["uh_pool_semantics"] = (
            "uh_mean3/uh_max3: mean / per-dim max over the 3 next-user-header tokens; "
            "bnd_mean5/bnd_max5: over all 5 boundary tokens; mean_xbnd/maxp_xbnd: over "
            "(answer content union 5 boundary tokens) — computed IN the forward per probe "
            "(GPU-side, fp32 reduce, fp16 transport), then probe-meaned like the singles."
        )
        manifest["v0_store_mean_parity"] = v0_parity or {"skipped": not compare_store}
        manifest["round1_store_drift_check"] = drift_check or {"skipped": not compare_store}
    if smoke_asserts is not None:
        manifest["smoke_asserts"] = smoke_asserts
    if truncate:
        default_pack = BTDR_SUMMARIES_HF_FILE_TMPL.format(pct=btdr_pct(truncate_frac))
        # In truncate mode --uh-summaries-out is a DIRECTORY (per-k pack names).
        pack_base = Path(args.uh_summaries_out) if args.uh_summaries_out else out_dir.parent
        uh_pack_path = pack_base / default_pack
    else:
        default_pack = "he_summaries.pt" if ablate else "uh_summaries.pt"
        uh_pack_path = Path(args.uh_summaries_out or (out_dir.parent / default_pack))
    uh_pack_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summaries": uh_pack_rows,  # {row: {ctx: (Lc, H) fp16}}
            "coverage": uh_pack_cov,  # {row: {ctx: probe count}}
            "rows": pack_rows_names,
            "capture_layers": capture_layers,
            "context_ids": ctx_ids,
            "model": args.model,
            "extended_boundary": True,
            "ablate_answer": ablate,
            "truncate_frac": float(truncate_frac) if truncate else None,
            "boundary_block_ids": list(BOUNDARY_BLOCK_IDS),
            "battery_sha256": BATTERY50_SHA256,
            "smoke": args.smoke,
            "reproducibility": reproducibility_metadata(),
        },
        uh_pack_path,
    )
    logger.info("wrote summaries pack (%d rows) to %s", len(pack_rows_names), uh_pack_path)
    return uh_pack_path


def _validate_truncate_flags(args) -> None:
    """Fail fast on an invalid --truncate-frac combination (plan v18 §4.6 item 2).

    Values in (0, 1] — 1.0 is ADMITTED as the endpoint-parity probe value (the
    registered k=1.0 smoke; production captures use 0.25/0.5/0.75); requires
    --extended-boundary; mutually exclusive with --ablate-answer; Betley-only
    (enforced via the --extended-boundary genre gate in _resolve_out_dir);
    --uh-summaries-out becomes a DIRECTORY (per-k pack filenames).
    """
    if not args.truncate_frac:
        return
    if args.ablate_answer:
        raise SystemExit(
            "--truncate-frac and --ablate-answer are mutually exclusive (a truncated "
            "answer is non-empty by construction — plan v18 §4.6 item 2)"
        )
    if not args.extended_boundary:
        raise SystemExit(
            "--truncate-frac requires --extended-boundary (the truncated sequence appends "
            "the full 5-token BOUNDARY_BLOCK_IDS — plan v18 §4.6 item 2)"
        )
    for k in args.truncate_frac:
        if not (0.0 < k <= 1.0):
            raise SystemExit(
                f"--truncate-frac values must be in (0, 1] (got {k}); 1.0 is the "
                "endpoint-parity probe value (degenerates to the round-3 full capture) — "
                "production uses 0.25 0.5 0.75"
            )
        btdr_pct(k)  # fail fast on a k with no canonical integer percent
    if len(set(btdr_pct(k) for k in args.truncate_frac)) != len(args.truncate_frac):
        raise SystemExit(f"--truncate-frac values collide on integer percent: {args.truncate_frac}")
    if args.uh_summaries_out and Path(args.uh_summaries_out).suffix == ".pt":
        raise SystemExit(
            "--uh-summaries-out must be a DIRECTORY in --truncate-frac mode (per-k packs "
            "btdr_summaries_k{pct}.pt land inside it)"
        )


def _resolve_out_dir(args, truncate_frac: float | None = None) -> Path:
    """Validate the extended-boundary/genre combination and resolve the store dir.

    ``--extended-boundary`` is Betley-only (the uh round's single variable is
    the captured span; UltraChat is out of scope — plan v11 §0) and defaults to
    a store dir that never clobbers the parent/g1 stores. In truncate mode
    (``truncate_frac`` set) ``--out-dir`` is a BASE dir and the per-k store is
    ``<base>/store_btdr_k{pct}`` (plan v18 § Storage naming).
    """
    if args.extended_boundary and args.genre != "betley":
        raise SystemExit(
            "--extended-boundary supports --genre betley only (the uh round's single "
            "variable is the captured span; UltraChat is out of scope — plan v11 §0)"
        )
    if args.ablate_answer and not args.extended_boundary:
        raise SystemExit(
            "--ablate-answer requires --extended-boundary (the ablated sequence appends the "
            "full 5-token BOUNDARY_BLOCK_IDS — plan v15 §4.6 item 2)"
        )
    if truncate_frac is not None:
        base = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "data" / "issue_810")
        return base / f"store_btdr_k{btdr_pct(truncate_frac)}"
    default_store = "store" if args.genre == "betley" else "store_g1"
    if args.ablate_answer:
        default_store = "store_he"
    elif args.extended_boundary:
        default_store = "store_uh"
    return Path(args.out_dir or (PROJECT_ROOT / "data" / "issue_810" / default_store))


def _collect_uh_pack_rows(
    ctx_id, pos_summaries, coverage, uh_pack_rows, uh_pack_cov, row_names=None
):
    """Collect this context's 9 new-row probe-mean summaries into the compact pack."""
    for n in row_names if row_names is not None else UH_SUMMARY_NAMES:
        uh_pack_rows[n][ctx_id] = pos_summaries[n].to(torch.float16)
        uh_pack_cov[n][ctx_id] = coverage[n]


def _do_uploads(
    args,
    out_dir: Path,
    ctx_ids: list[str],
    uh_pack_path: Path | None,
    truncate_frac: float | None = None,
):
    """Store (+ summaries pack) uploads, gated on --no-upload/--smoke by the caller."""
    logger.info("[phase=upload] aligned-subset store")
    path_in_repo = _upload_store(
        out_dir,
        ctx_ids,
        smoke=False,
        genre=args.genre,
        extended=args.extended_boundary,
        ablate=args.ablate_answer,
        truncate_frac=truncate_frac,
    )
    uh_pack_hf = None
    if uh_pack_path is not None:
        logger.info("[phase=upload] summaries pack")
        if truncate_frac is not None:
            # `_btdr` packs land under the BTDR results prefix per k (plan v18
            # §4.6 item 1 — NOT the uh/he single-file destinations).
            pack_name = BTDR_SUMMARIES_HF_FILE_TMPL.format(pct=btdr_pct(truncate_frac))
            pack_hf_file = f"{BTDR_HF_RESULTS_PREFIX}/{pack_name}"
        elif args.ablate_answer:
            pack_hf_file = HE_SUMMARIES_HF_FILE
        else:
            pack_hf_file = UH_SUMMARIES_HF_FILE
        uh_pack_hf = _upload_uh_summaries(uh_pack_path, hf_file=pack_hf_file)
    return path_in_repo, uh_pack_hf


def _extended_context_hooks(
    args,
    ci: int,
    ctx_id: str,
    instance: dict,
    probes: list[str],
    completions: list[str],
    model,
    tokenizer,
    capture,
    capture_layers: list[int],
    n_layers: int,
    pos_summaries: dict,
    coverage: dict,
    extras: dict,
    compare_store: bool,
    v0_parity: dict,
) -> tuple[dict | None, dict | None]:
    """Per-context extended-boundary hooks: parity tripwires + smoke oracles.

    Mutates ``v0_parity`` (per-context store-mean parity record, production
    gate) and returns ``(drift_check, smoke_asserts)`` — each non-None only on
    the first context when its gate fires. Raises on any parity violation
    (plan v11 kill criterion 2: halt before any fit on drifted activations).
    """
    drift_check: dict | None = None
    smoke_asserts: dict | None = None
    if compare_store:
        v0_parity[ctx_id] = _v0_store_mean_parity(extras["mean_ans"], ctx_id)
        if ci == 0:
            drift_check = _round1_store_drift_check(pos_summaries, coverage, ctx_id)
    elif ci == 0:
        logger.info(
            "[phase=parity] store comparisons SKIPPED (model=%s, n_probes=%s) — "
            "smoke exercises the reduction paths; the 7B store cannot match a "
            "non-7B / probe-capped run by construction",
            args.model,
            args.n_probes,
        )
    if args.smoke and ci == 0:
        smoke_asserts = _extended_smoke_asserts(
            model,
            tokenizer,
            capture,
            instance,
            probes,
            completions,
            capture_layers,
            n_layers,
            pos_summaries,
            extras["mean_ans"],
        )
    return drift_check, smoke_asserts


def _ablate_context_hooks(
    args,
    ci: int,
    ctx_id: str,
    instance: dict,
    probes: list[str],
    completions: list[str] | None,
    model,
    tokenizer,
    capture,
    capture_layers: list[int],
    n_layers: int,
    pos_summaries: dict,
    compare_cc: bool,
    cc_parity: dict,
    truncate_frac: float | None = None,
) -> dict | None:
    """Per-context he-row-set hooks (ablate OR truncate): cc_last tripwire + smoke.

    Mutates ``cc_parity`` (per-context #594-store parity record — the
    production capture-drift gate, plan v15 §5; ``cc_last`` is prompt-side and
    ablation/TRUNCATION-invariant by causal attention, so the same tripwire
    covers the `_btdr` truncate mode, plan v18 §5) and returns
    ``smoke_asserts`` (first context only, --smoke). Raises on any parity
    violation (kill criterion 2: halt before any fit on drifted activations).
    On a non-compare run (smoke / non-7B / probe-capped) the cc_last recompute
    path is still exercised: shape + non-degenerate norms are asserted (the
    ``_cc_last_parity_probe`` smoke precedent).
    """
    smoke_asserts: dict | None = None
    cc_vec = pos_summaries["cc_last"]
    if not torch.isfinite(cc_vec).all() or (cc_vec.norm(dim=1) < 1e-6).any():
        raise RuntimeError(f"cc_last probe-mean degenerate for {ctx_id}")
    if compare_cc:
        cc_parity[ctx_id] = _cc594_store_parity(cc_vec, ctx_id, capture_layers)
    elif ci == 0:
        logger.info(
            "[phase=cc_parity] #594-store compare SKIPPED (model=%s, n_probes=%s) — "
            "recompute path exercised (shape + norms); the 7B store cannot match a "
            "non-7B / probe-capped run by construction",
            args.model,
            args.n_probes,
        )
    if args.smoke and ci == 0:
        smoke_asserts = _extended_smoke_asserts(
            model,
            tokenizer,
            capture,
            instance,
            probes,
            completions,
            capture_layers,
            n_layers,
            pos_summaries,
            None,
            ablate=bool(args.ablate_answer),
            truncate_frac=truncate_frac,
        )
    return smoke_asserts


# The 5 boundary singles compared in the k=1.0 endpoint-parity probe.
_K1_PARITY_SINGLES: tuple[str, ...] = ("im_end", "turn_nl", "uh_im_start", "uh_user", "uh_nl")


def _k1_endpoint_parity_probe(
    args,
    model,
    tokenizer,
    capture,
    instance: dict,
    probes: list[str],
    completions: list[str],
    capture_layers: list[int],
    n_layers: int,
    compare_store: bool,
    min_cos: float = 0.999,
) -> dict:
    """k=1.0 endpoint-parity gate (plan v18 §4.6 item 2 / kill criterion 2).

    Runs the TRUNCATE code path at ``truncate_frac=1.0`` on ONE context — where
    it degenerates to the round-3 full capture exactly (``n_keep == ans_len``
    asserted per probe inside ``_build_probe_row`` and re-asserted here from the
    diag) — then downloads that context's committed round-3 store file
    (``answer_position_sweep_user_header/<ctx>.pt``, ~8 MB) and, on a
    production run (``compare_store``: 7B model + uncapped probes), asserts min
    per-layer cosine ≥ ``min_cos`` on the 5 boundary singles. PASS proves the
    completions/tokenization/capture pipeline reproduces committed data BEFORE
    any capture spend; FAIL halts (``failure_class: data``). Smoke / non-7B /
    probe-capped runs exercise the same code path + fetch + structural asserts
    but skip the cosine compare (a 0.5B / probe-capped capture cannot match the
    7B 48-probe store means by construction — the ``_cc_last_parity_probe``
    precedent).
    """
    ctx_id = str(instance["id"])
    logger.info("[phase=k1_parity] endpoint-parity probe at truncate_frac=1.0 (ctx=%s)", ctx_id)
    pos_summaries, coverage, diag, _extras = capture_positions_for_context(
        model,
        tokenizer,
        instance,
        probes,
        completions,
        capture,
        n_layers,
        capture_layers,
        batch_probes=args.batch_probes,
        extended=True,
        ablate=False,
        truncate_frac=1.0,
    )
    nk, al = diag["n_keep_per_probe"], diag["ans_len_per_probe"]
    assert nk == al, (
        f"k=1.0 endpoint probe: n_keep != ans_len for some probe "
        f"(first mismatches: {[(a, b) for a, b in zip(nk, al, strict=True) if a != b][:5]})"
    )
    out: dict = {
        "ctx_id": ctx_id,
        "n_probes": diag["n_probes_used"],
        "n_keep_equals_ans_len": True,
        "min_cosine_threshold": min_cos,
    }
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO,
        f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_UH_SUBDIR}/{ctx_id}.pt",
        repo_type="dataset",
    )
    blob = torch.load(p, weights_only=False)
    r3 = {name: blob["pos_vectors"][i] for i, name in enumerate(blob["positions"])}
    missing = [s for s in _K1_PARITY_SINGLES if s not in r3]
    assert not missing, f"round-3 store {ctx_id} lacks singles {missing}"
    if not compare_store:
        out["compared_to_store"] = False
        logger.info(
            "[phase=k1_parity] store compare SKIPPED (model=%s, n_probes=%s, smoke=%s) — "
            "truncate-path degeneration + n_keep==ans_len + store fetch/shape exercised; "
            "the cosine gate binds on the production 7B run",
            args.model,
            args.n_probes,
            args.smoke,
        )
        return out
    worst = 1.0
    for s in _K1_PARITY_SINGLES:
        if coverage[s] != blob["coverage"].get(s):
            raise RuntimeError(
                f"k=1.0 endpoint probe {ctx_id}: coverage mismatch for {s!r} "
                f"({coverage[s]} vs round-3 {blob['coverage'].get(s)}) — probe-set drift"
            )
        c_min, _ = _per_layer_min_cosine(pos_summaries[s], r3[s].float())
        worst = min(worst, c_min)
        if c_min < min_cos:
            raise RuntimeError(
                f"k=1.0 endpoint-parity probe FAILED for {ctx_id}/{s}: min layer cosine "
                f"{c_min:.6f} < {min_cos} — the truncate path at k=1.0 does not reproduce "
                "the committed round-3 capture (completions/tokenizer/capture drift); "
                "halting BEFORE any capture spend (plan v18 kill criterion 2, "
                "failure_class: data)"
            )
    out.update({"compared_to_store": True, "min_layer_cosine": worst})
    logger.info("[phase=k1_parity] PASS: worst min-layer cosine %.6f over 5 singles", worst)
    return out


def _store_position_names(args) -> list[str]:
    """The per-context stored position list for this run's mode."""
    if args.ablate_answer or args.truncate_frac:
        return he_stored_position_names()
    return uh_stored_position_names() if args.extended_boundary else stored_position_names()


def _probes_for_context(args, ctx_id: str, probe_pool: list[str] | None):
    """(probes, completions) for one context — ablate mode never reads completions."""
    if args.ablate_answer:
        probes = list(probe_pool[: args.n_probes] if args.n_probes else probe_pool)
        return probes, None
    cells = _load_stored_completions(ctx_id, args.genre)
    if args.n_probes is not None:
        cells = cells[: args.n_probes]
    return [c["probe"] for c in cells], [c["completion"] for c in cells]


def _run_extraction_pass(
    args,
    model,
    tokenizer,
    capture,
    capture_layers: list[int],
    n_layers: int,
    battery: dict,
    instances: dict,
    ctx_ids: list[str],
    g1_cc_parity: dict | None,
    truncate_frac: float | None = None,
) -> tuple[dict, Path]:
    """ONE full extraction pass: per-context capture → store files → manifest → pack → upload.

    Extracted verbatim from ``main`` (behavior identical for
    ``truncate_frac=None`` — the parent/uh/he modes run exactly one pass); the
    `_btdr` truncate mode calls it once per k re-using the loaded model (plan
    v18 §4.6 item 2 single-invocation multi-k loop). Returns ``(note, out_dir)``
    — the per-pass sentinel-note fields + the local store dir. Uploads (store +
    pack) run INSIDE the pass so each k's artifacts persist the moment the pass
    completes (checkpoint-per-phase).
    """
    out_dir = _resolve_out_dir(args, truncate_frac)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_ctx_diag: dict[str, dict] = {}
    # Extended-boundary parity records (plan v11 §5) + the compact new-row pack.
    v0_parity: dict[str, dict] = {}
    cc594_parity: dict[str, dict] = {}
    drift_check: dict | None = None
    smoke_asserts: dict | None = None
    he_rows = bool(args.ablate_answer) or truncate_frac is not None
    # The he-row-set modes (ablate OR truncate) pack the 9 HE rows — pairing
    # across sides and k is by name (plan v18 §4.6 item 1; a UH_SUMMARY_NAMES
    # pack here would drop im_end/turn_nl, the two headline rows).
    pack_row_names = HE_SUMMARY_NAMES if he_rows else UH_SUMMARY_NAMES
    uh_pack_rows: dict[str, dict[str, torch.Tensor]] = {n: {} for n in pack_row_names}
    uh_pack_cov: dict[str, dict[str, int]] = {n: {} for n in pack_row_names}
    # The v0 store-mean / round-1 drift comparisons are meaningful ONLY on the
    # production model with the full probe set (a 0.5B smoke or a capped probe
    # subset cannot match the 7B 48-probe store means by construction — the
    # `_cc_last_parity_probe` precedent). In the he-row-set modes BOTH are
    # undefined (ablate: no answer span; truncate: the span is deliberately
    # cut) — the cc_last #594-store parity is the drift tripwire instead
    # (plan v15 §5 divergence 3 / plan v18 §5).
    compare_store = (
        args.extended_boundary
        and not he_rows
        and args.model == DEFAULT_MODEL
        and args.n_probes is None
    )
    compare_cc = he_rows and args.model == DEFAULT_MODEL and args.n_probes is None
    # Ablate mode reads NO stored completions (code-truth for the plan v15 §10
    # claim): the probe grid comes from the hash-pinned Betley pool instead.
    # Truncate mode DOES consume completions (the answer text IS the dose).
    probe_pool = _betley_probe_pool(battery) if args.ablate_answer else None
    for ci, ctx_id in enumerate(ctx_ids):
        logger.info(
            "[phase=extract] context %d/%d %s%s",
            ci + 1,
            len(ctx_ids),
            ctx_id,
            f" (truncate_frac={truncate_frac})" if truncate_frac is not None else "",
        )
        if ctx_id not in instances:
            raise RuntimeError(f"context {ctx_id} absent from battery (coverage gap)")
        probes, completions = _probes_for_context(args, ctx_id, probe_pool)
        pos_summaries, coverage, diag, extras = capture_positions_for_context(
            model,
            tokenizer,
            instances[ctx_id],
            probes,
            completions,
            capture,
            n_layers,
            capture_layers,
            args.batch_probes,
            extended=args.extended_boundary,
            ablate=args.ablate_answer,
            truncate_frac=truncate_frac,
        )
        if he_rows:
            sa = _ablate_context_hooks(
                args,
                ci,
                ctx_id,
                instances[ctx_id],
                probes,
                completions,
                model,
                tokenizer,
                capture,
                capture_layers,
                n_layers,
                pos_summaries,
                compare_cc,
                cc594_parity,
                truncate_frac=truncate_frac,
            )
            smoke_asserts = sa or smoke_asserts
            _collect_uh_pack_rows(
                ctx_id, pos_summaries, coverage, uh_pack_rows, uh_pack_cov, pack_row_names
            )
        elif args.extended_boundary:
            dc, sa = _extended_context_hooks(
                args,
                ci,
                ctx_id,
                instances[ctx_id],
                probes,
                completions,
                model,
                tokenizer,
                capture,
                capture_layers,
                n_layers,
                pos_summaries,
                coverage,
                extras,
                compare_store,
                v0_parity,
            )
            drift_check = dc or drift_check
            smoke_asserts = sa or smoke_asserts
            _collect_uh_pack_rows(ctx_id, pos_summaries, coverage, uh_pack_rows, uh_pack_cov)
        names = _store_position_names(args)
        # Stack positions into (n_positions, Lc, H) fp16; a position missing
        # for EVERY probe (impossible for boundary; possible for a deep
        # tail_k on all-short answers) is recorded as absent in coverage and
        # its row is zero-filled (never silently dropped — the reader keys on
        # coverage, and a 0-coverage row is excluded downstream).
        H = model.config.hidden_size
        pos_stack = torch.zeros(len(names), len(capture_layers), H, dtype=torch.float16)
        for pi, name in enumerate(names):
            if name in pos_summaries:
                pos_stack[pi] = pos_summaries[name].to(torch.float16)
        blob = {
            "context_id": ctx_id,
            "capture_layers": capture_layers,
            "positions": names,
            "pos_vectors": pos_stack,  # (n_positions, Lc, H) fp16
            "coverage": coverage,
            "model": args.model,
        }
        torch.save(blob, out_dir / f"{ctx_id}.pt")
        per_ctx_diag[ctx_id] = diag

    # Manifest (plan §13): positions list, dtype, coverage semantics, provenance.
    manifest = {
        "positions": _store_position_names(args),
        "capture_layers": capture_layers,
        "dtype": "float16",
        "pos_vectors_shape": ["n_positions", len(capture_layers), model.config.hidden_size],
        "coverage_semantics": "per-position probe count contributing to the probe-mean summary",
        "n_contexts": len(ctx_ids),
        "context_ids": ctx_ids,
        "model": args.model,
        "battery_sha256": BATTERY50_SHA256,
        "per_context_diag": per_ctx_diag,
        "boundary_note": (
            "im_end=<|im_end|> id 151645 (position span_end); turn_nl=\\n after "
            "im_end (span_end+1, the c_C answer-side mirror). Both appended to the "
            "teacher-forced sequence and captured fresh (NOT slice-derivable from "
            "the answer-content span)."
        ),
        "reproducibility": reproducibility_metadata(),
        "smoke": args.smoke,
    }
    if args.genre == "g1":
        # Genre-arm provenance (plan v6 § Storage naming): tag + probe-pool pin +
        # the cc_last parity read. Betley manifests stay parent-shaped (A14 parity).
        manifest["genre_tag"] = G1_GENRE_TAG
        manifest["probe_pool_hash"] = G1_PROBE_POOL_HASH
        manifest["cc_last_parity"] = g1_cc_parity
    uh_pack_path: Path | None = None
    if args.extended_boundary:
        uh_pack_path = _finalize_extended_outputs(
            args,
            manifest,
            v0_parity,
            drift_check,
            smoke_asserts,
            uh_pack_rows,
            uh_pack_cov,
            capture_layers,
            ctx_ids,
            out_dir,
            compare_store,
            cc_parity=(
                {
                    "per_context": cc594_parity,
                    "min_layer_cosine": min(
                        (v["min_layer_cosine"] for v in cc594_parity.values()), default=None
                    ),
                }
                if he_rows
                else None
            ),
            truncate_frac=truncate_frac,
        )
    dump_json(manifest, out_dir / "manifest.json")
    logger.info("wrote manifest (%d contexts) to %s", len(ctx_ids), out_dir)

    path_in_repo = None
    uh_pack_hf: str | None = None
    if not args.no_upload and not args.smoke:
        path_in_repo, uh_pack_hf = _do_uploads(
            args, out_dir, ctx_ids, uh_pack_path, truncate_frac=truncate_frac
        )

    note = {
        "phase": "B_extract_positions",
        "genre": args.genre,
        "extended_boundary": bool(args.extended_boundary),
        "ablate_answer": bool(args.ablate_answer),
        "n_contexts": len(ctx_ids),
        "positions": len(manifest["positions"]),
        "hf_path": path_in_repo,
        "uh_summaries_hf": uh_pack_hf,
        "v0_store_mean_parity_min_cos": (
            min((v["min_layer_cosine"] for v in v0_parity.values()), default=None)
        ),
        "cc594_parity_min_cos": (
            min((v["min_layer_cosine"] for v in cc594_parity.values()), default=None)
        ),
        "round1_drift_worst_cos": (drift_check or {}).get("worst_min_layer_cosine"),
        "store_files_sha256": {
            c: sha256_file(out_dir / f"{c}.pt")[:16] for c in ctx_ids[: min(3, len(ctx_ids))]
        },
    }
    if truncate_frac is not None:
        note["truncate_frac"] = float(truncate_frac)
    return note, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 Phase B: answer position sweep extraction")
    ap.add_argument(
        "--genre",
        choices=list(GENRES),
        default="betley",
        help="probe-corpus genre: 'betley' (default — the parent's sources, bit-for-bit) or "
        "'g1' (#658's UltraChat genre-generalization arm; plan v6 follow-up round)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--gpu", action="store_true", help="force --device cuda")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="local store dir (default: data/issue_810/store for betley, "
        "data/issue_810/store_g1 for g1 — per-genre so g1 never clobbers the parent store)",
    )
    ap.add_argument("--battery", default=None, help="local battery.json fast path (sha-pinned)")
    ap.add_argument("--n-ctx", type=int, default=None, help="smoke: cap contexts")
    ap.add_argument("--n-probes", type=int, default=None, help="smoke: cap probes/context")
    ap.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument(
        "--batch-probes", type=int, default=8, help="probes per left-padded forward (real knob)"
    )
    ap.add_argument(
        "--extended-boundary",
        action="store_true",
        help="follow-up `user-header-newline-summary` (plan v11): append the FULL 5-token "
        "assistant-turn continuation (<|im_end|> \\n <|im_start|> user \\n) instead of 2, "
        "capture the 3 next-user-header singles + 6 in-forward span pools, store to "
        f"{ANSWER_POSITION_SWEEP_UH_SUBDIR}/. Default OFF = parent behavior byte-for-byte.",
    )
    ap.add_argument(
        "--ablate-answer",
        action="store_true",
        help="follow-up `header-echo-ablation-capture` (plan v15): teacher-force "
        "`prompt + BOUNDARY_BLOCK_IDS` with the answer span EMPTY (requires "
        "--extended-boundary; Betley-only). Captures cc_last + the 5 boundary singles + "
        "the 4 bnd5/uh3 pools (NO tail/head/xbnd — undefined at ans_len=0), reads NO "
        f"stored completions, stores to {ANSWER_POSITION_SWEEP_HE_SUBDIR}/, packs to "
        "he_summaries.pt. Default OFF = round-3 behavior byte-for-byte.",
    )
    ap.add_argument(
        "--truncate-frac",
        nargs="+",
        type=float,
        default=None,
        help="follow-up `boundary-truncation-dose-response` (plan v18): keep only the first "
        "n_keep = max(1, ceil(k*ans_len)) answer TOKENS (ID-prefix cut, never re-tokenized "
        "text) before the 5-token boundary block; ONE process loops the k list (one model "
        "load), runs the k=1.0 endpoint-parity gate FIRST, stores per k to "
        f"{ANSWER_POSITION_SWEEP_BTDR_SUBDIR_TMPL}/, packs per k to "
        f"{BTDR_SUMMARIES_HF_FILE_TMPL}. Values in (0, 1] — 1.0 is the endpoint-parity "
        "probe value (degenerates to the round-3 full capture); production uses "
        "0.25 0.5 0.75. REQUIRES --extended-boundary; MUTUALLY EXCLUSIVE with "
        "--ablate-answer; Betley-only. Default OFF = round-3/round-4 behavior "
        "byte-for-byte.",
    )
    ap.add_argument(
        "--uh-summaries-out",
        default=None,
        help="local path for the compact new-row summaries pack (extended-boundary only; "
        "default: <out-dir parent>/uh_summaries.pt, or he_summaries.pt with --ablate-answer; "
        "a DIRECTORY for the per-k btdr_summaries_k{pct}.pt packs with --truncate-frac)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    _validate_truncate_flags(args)

    device = args.device or ("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    _resolve_out_dir(args)  # fail fast on an invalid mode/genre combination
    t0 = time.time()

    logger.info("[phase=setup] loading battery + manifest (genre=%s)", args.genre)
    battery = _resolve_battery(Path(args.battery) if args.battery else None)
    instances = {i["id"]: i for i in battery["instances"]}
    # In --smoke we cannot rely on the real 50-context manifest join, but the real
    # run pins to the manifest's 50 contexts (the LOCO fold order).
    ctx_ids = _load_manifest_context_ids(args.genre)
    if args.n_ctx is not None:
        ctx_ids = ctx_ids[: args.n_ctx]
    logger.info("contexts to extract: %d (device=%s model=%s)", len(ctx_ids), device, args.model)

    logger.info("[phase=load_model] %s", args.model)
    model, tokenizer = _load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    capture_layers = list(range(n_layers))
    if not args.smoke:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert model.config.hidden_size == args.expected_hidden, model.config.hidden_size
    capture = LayerCapture(model, n_layers)

    # g1 cc_last recomputation parity probe (plan v6 §11): runs FIRST — before any
    # capture — so a c_C procedure drift halts before GPU-hours are spent. Runs on
    # the FIRST manifest context. Production (7B) compares against the g1 store;
    # smoke/non-7B exercises the recompute path only.
    g1_cc_parity: dict | None = None
    if args.genre == "g1":
        ctx0 = ctx_ids[0]
        if ctx0 not in instances:
            raise RuntimeError(f"context {ctx0} absent from battery (coverage gap)")
        cells0 = _load_stored_completions(ctx0, args.genre)
        g1_cc_parity = _cc_last_parity_probe(
            model,
            tokenizer,
            capture,
            instances[ctx0],
            cells0[0]["probe"],
            ctx0,
            n_layers,
            compare_store=(not args.smoke and args.model == DEFAULT_MODEL),
        )

    # `_btdr` truncate mode (plan v18 §4.6 item 2): k=1.0 endpoint-parity gate
    # FIRST (halt BEFORE any capture spend — kill criterion 2), then one
    # extraction pass per k re-using the loaded model. All other modes run
    # exactly one pass with truncate_frac=None (behavior identical).
    k_list: list[float | None] = (
        [float(k) for k in args.truncate_frac] if args.truncate_frac else [None]
    )
    k1_parity: dict | None = None
    passes: list[tuple[dict, Path]] = []
    try:
        if args.truncate_frac:
            ctx0 = ctx_ids[0]
            if ctx0 not in instances:
                raise RuntimeError(f"context {ctx0} absent from battery (coverage gap)")
            probes0, completions0 = _probes_for_context(args, ctx0, None)
            k1_parity = _k1_endpoint_parity_probe(
                args,
                model,
                tokenizer,
                capture,
                instances[ctx0],
                probes0,
                completions0,
                capture_layers,
                n_layers,
                compare_store=(
                    not args.smoke and args.model == DEFAULT_MODEL and args.n_probes is None
                ),
            )
        for k in k_list:
            passes.append(
                _run_extraction_pass(
                    args,
                    model,
                    tokenizer,
                    capture,
                    capture_layers,
                    n_layers,
                    battery,
                    instances,
                    ctx_ids,
                    g1_cc_parity,
                    truncate_frac=k,
                )
            )
    finally:
        capture.remove()

    if args.truncate_frac:
        note = {
            "phase": "B_extract_positions",
            "genre": args.genre,
            "extended_boundary": bool(args.extended_boundary),
            "ablate_answer": bool(args.ablate_answer),
            "truncate_fracs": [float(k) for k in args.truncate_frac],
            "k1_endpoint_parity": k1_parity,
            "passes": [n for n, _ in passes],
            "n_contexts": len(ctx_ids),
            "elapsed_s": round(time.time() - t0, 1),
        }
    else:
        note = {**passes[0][0], "elapsed_s": round(time.time() - t0, 1)}
    _write_sentinel("epm:results", note, passes[-1][1])
    logger.info(
        "[phase=done] extraction complete: %d contexts x %d pass(es)", len(ctx_ids), len(passes)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] extraction crashed:\n%s", traceback.format_exc())
        raise
