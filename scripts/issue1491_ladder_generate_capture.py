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
                 relative L2 < 1e-3 in fp32; on failure the driver falls
                 back to per-row + logs a fail-loud WARN. Default 8 (safe
                 padded-batch shape).
- ``--first-chunk-self-gate`` — enable plan §7 Gate 1 (quick ridge fit +
                 shuffled null after ~2,000 captured rows; abort the
                 scale's job on failure via epm:failure sentinel).

Reads the ladder manifest from
``superkaiba1/explore-persona-space-data:issue1491_scale_ladder/manifest/<split>.jsonl``
(built by ``scripts/issue1491_ladder_manifest.py`` at Phase 0).

Persist-by-default (Upload Policy v2): rollout TEXT uploads unconditionally
on the non-LFS path (quota-immune); trimmed capture tensors upload per
K=20 chunks; the driver never discards generations or capture tensors.
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

import numpy as np

# Load .env BEFORE importing torch (shared-VM thread caps + HF_TOKEN;
# code-style.md § shared-VM CPU thread caps).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import torch  # noqa: E402

# Import parent-branch modules (per Unit 1 Deliverable A port-source
# decision, epm:progress v6: port_source: origin/main; no vendoring).
import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402

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
    local = hf_hub_download(
        repo_id=LADDER_HF_REPO,
        filename=f"{MANIFEST_HF_PREFIX}/{fname}",
        repo_type="dataset",
        cache_dir=str(cache_dir),
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


def _build_capture_engine(model_id: str) -> object | None:
    """Build the vLLM capture engine, honoring the H100 long-prompt hang /
    IMA mitigation ENV knobs (default OFF — the launch script sets them
    per plan §11 "enforce_eager + prefix-caching off"; commit 4cb9d6ea8d
    made these ENV-GATED in the parent driver, so the ladder driver MUST
    NOT re-hardcode them here)."""
    from explore_persona_space.eval.generation import create_vllm_engine

    llm_kwargs: dict = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        llm_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        llm_kwargs["enable_prefix_caching"] = False
    if llm_kwargs:
        logger.info("[engine-knobs] %s", llm_kwargs)
    return create_vllm_engine(model_id, max_model_len=MAX_MODEL_LEN, seed=42, **llm_kwargs)


# ---------------------------------------------------------------------------
# Capture: per-row (parent parity) and batched (item (i) throughput fix)
# ---------------------------------------------------------------------------


def _capture_perrow(hf, tok, prompts, responses, cis, layers, h_dim):
    """Parent-verbatim per-row capture (safe fallback + parity oracle for the
    batched implementation below).

    Returns rows = [{"ci", "prompt", "response", "cx_last": (L,H), "v_x": (L,H)}].
    Drops rows whose response teacher-forces to zero tokens (parent parity)."""
    rows = []
    for p, resp, ci in zip(prompts, responses, cis, strict=True):
        msgs = [{"role": "user", "content": p}]
        cx = COL.capture_context_vector(hf, tok, msgs, layers)
        av = COL.capture_answer_vector(hf, tok, msgs, resp, layers, {}, keep_per_token=False)
        if av is None:  # empty response
            continue
        assert cx["last"].shape == (len(layers), h_dim), ("cx_last", cx["last"].shape)
        assert av["v_x"].shape == (len(layers), h_dim), ("v_x", av["v_x"].shape)
        rows.append(
            {
                "ci": int(ci),
                "prompt": p,
                "response": resp,
                "cx_last": cx["last"],
                "v_x": av["v_x"],
            }
        )
    return rows


def _capture_batched(hf, tok, prompts, responses, cis, layers, h_dim, batch_size):
    """Batched teacher-forced capture (plan §4.2 item (i)).

    Length-sorted padded batches. For each batch:
      - Build full-render token ids for prompt-only and prompt+response.
      - Pad + attention-mask; forward through ``hf`` with
        ``output_hidden_states=True``.
      - For each row: cx_last = hidden_states at the LAST prompt token;
        v_x = attention-masked mean over the response-token span, per
        layer in ``layers``, fp32 on CPU.

    Drops rows with empty response (v_x undefined). Same output shape as
    ``_capture_perrow``.
    """
    if not prompts:
        return []

    # 1. Tokenize prompt + full (prompt + response) per row, single shot.
    prompt_texts, full_texts = [], []
    for p, resp in zip(prompts, responses, strict=True):
        pt = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        ft = tok.apply_chat_template(
            [
                {"role": "user", "content": p},
                {"role": "assistant", "content": resp},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_texts.append(pt)
        full_texts.append(ft)

    prompt_ids_list = [
        tok(t, return_tensors="pt", padding=False)["input_ids"][0] for t in prompt_texts
    ]
    full_ids_list = [tok(t, return_tensors="pt", padding=False)["input_ids"][0] for t in full_texts]

    # 2. Filter empty-response rows (v_x undefined) — parent parity.
    active_indices, prompt_lens, full_lens = [], [], []
    for k, (p_ids, f_ids) in enumerate(zip(prompt_ids_list, full_ids_list, strict=True)):
        p_len = int(p_ids.shape[0])
        f_len = int(f_ids.shape[0])
        if f_len <= p_len:
            continue  # empty / non-lengthening response
        active_indices.append(k)
        prompt_lens.append(p_len)
        full_lens.append(f_len)

    if not active_indices:
        return []

    # 3. Length-sort active rows for padding efficiency.
    order = sorted(range(len(active_indices)), key=lambda i: full_lens[i])
    rows: list[dict] = []

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    for bs in range(0, len(order), batch_size):
        batch_order = order[bs : bs + batch_size]
        batch_full_ids = [full_ids_list[active_indices[i]] for i in batch_order]
        batch_prompt_lens = [prompt_lens[i] for i in batch_order]
        batch_full_lens = [full_lens[i] for i in batch_order]

        max_len = max(batch_full_lens)
        padded = torch.full((len(batch_full_ids), max_len), pad_id, dtype=batch_full_ids[0].dtype)
        attn = torch.zeros((len(batch_full_ids), max_len), dtype=torch.long)
        for row_i, f_ids in enumerate(batch_full_ids):
            L = f_ids.shape[0]
            padded[row_i, :L] = f_ids
            attn[row_i, :L] = 1

        padded = padded.to(hf.device)
        attn = attn.to(hf.device)

        with torch.no_grad():
            out = hf(
                input_ids=padded,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
        # out.hidden_states = tuple(L+1) of (B, T, H); we index per layer.

        for row_i, k in enumerate(batch_order):
            p_len = batch_prompt_lens[row_i]
            f_len = batch_full_lens[row_i]
            cx_last_stack: list[torch.Tensor] = []
            v_x_stack: list[torch.Tensor] = []
            for li in layers:
                # Correct offset: hidden_states index 0 is embeddings, layer
                # `li` is index li in the tuple (0-indexed layer 0 == embed,
                # layer 1 == first block, ...). Parent's COL.capture_* uses
                # ``captured[li]`` where captured is a dict from the OOM-safe
                # hook keyed on 0-indexed layer number; keep the same
                # semantics: hidden_states[li] here IS layer li (they align
                # by convention when both use the same 0-indexed li list).
                hs = out.hidden_states[li][row_i]  # (T, H)
                # last-prompt-token cx: index p_len - 1 (the token BEFORE the
                # assistant header starts). Parent parity: capture_context_vector
                # runs a prompt-only forward and reads hs[-1]; here we run the
                # FULL forward and read hs at the last prompt position.
                cx_last_stack.append(hs[p_len - 1, :].float().cpu())
                # v_x: mean over response tokens (positions p_len .. f_len - 1).
                resp_span = hs[p_len:f_len, :].float().cpu()
                v_x_stack.append(resp_span.mean(dim=0))
            cx_last = torch.stack(cx_last_stack)  # (L, H)
            v_x = torch.stack(v_x_stack)  # (L, H)
            assert cx_last.shape == (len(layers), h_dim), ("cx_last (batched)", cx_last.shape)
            assert v_x.shape == (len(layers), h_dim), ("v_x (batched)", v_x.shape)

            orig_k = active_indices[k]
            rows.append(
                {
                    "ci": int(cis[orig_k]),
                    "prompt": prompts[orig_k],
                    "response": responses[orig_k],
                    "cx_last": cx_last,
                    "v_x": v_x,
                }
            )
    return rows


def _batched_capture_parity_gate(
    hf, tok, prompts, responses, cis, layers, h_dim, batch_size
) -> tuple[bool, str]:
    """Plan §4.2 item (i) parity gate: on 32 probe rows, batched vs per-row
    capture must agree per-field cosine > 0.9999 and max relative L2 error
    < 1e-3 in fp32. On failure: return (False, reason) — caller falls back
    to per-row and logs a fail-loud WARN.

    32 rows chosen per plan; we accept fewer if the caller passed fewer.
    """
    n = min(32, len(prompts))
    if n == 0:
        return True, "empty probe (nothing to check)"
    p = prompts[:n]
    r = responses[:n]
    ci = cis[:n]
    try:
        rows_serial = _capture_perrow(hf, tok, p, r, ci, layers, h_dim)
        rows_batched = _capture_batched(hf, tok, p, r, ci, layers, h_dim, batch_size)
    except Exception as e:  # noqa: BLE001
        return False, f"probe crashed: {type(e).__name__}: {e}"

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
    if matched == 0:
        return False, "no matching rows between serial + batched probes"
    if 1.0 - max_cos_dev < 0.9999:
        return False, f"cosine gate FAIL: min cos={1.0 - max_cos_dev:.6f} < 0.9999"
    if max_rel_l2 >= 1e-3:
        return False, f"rel-L2 gate FAIL: max rel-L2={max_rel_l2:.3e} >= 1e-3"
    return (
        True,
        f"PASS: {matched} rows, min cos={1.0 - max_cos_dev:.6f}, max rel-L2={max_rel_l2:.3e}",
    )


# ---------------------------------------------------------------------------
# First-chunk self-gate (plan §7 Decision Gate 1)
# ---------------------------------------------------------------------------


def _first_chunk_self_gate(rows: list[dict], layer_index_primary: int) -> tuple[bool, dict]:
    """Quick numpy ridge fit + shuffled-pairing null on the first ~2,000
    captured rows at the primary layer.

    PASS iff: (fit - null) > 0.05 AND |null R²| < 0.05.
    Returns (passed, diagnostics-dict).
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

    diag = {
        "n_train": int(n_train),
        "n_val": int(len(Yva)),
        "r2_fit": r2_fit,
        "r2_null": r2_null,
        "gap": r2_fit - r2_null,
    }
    passed = (r2_fit - r2_null) > 0.05 and abs(r2_null) < 0.05
    diag["passed"] = passed
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
    # N50._shard_range's semantics.
    return N10._shard_range(n_total, num_shards, shard_index)  # noqa: SLF001


def _remote_index(hf_prefix: str, subdir: str) -> set[str]:
    """List the leaf filenames already on HF under ``hf_prefix/subdir``."""
    api = _hf_api()
    try:
        entries = list(
            api.list_repo_tree(
                repo_id=LADDER_HF_REPO,
                path_in_repo=f"{hf_prefix}/{subdir}",
                repo_type="dataset",
                recursive=True,
            )
        )
    except Exception:  # noqa: BLE001
        return set()
    return {e.path.split("/")[-1] for e in entries if not e.path.endswith("/")}


def run_capture(args) -> int:
    """Run generation + trimmed capture for ONE (model, split) combination
    across ``args.num_shards`` shards; process shard ``args.shard_index``.

    Emits per-chunk .pt (trimmed) + per-chunk raw completions JSON into
    ``args.out_dir/shards/``, uploads in K=20 batches to
    ``{args.hf_prefix}/final_token_capture/`` and
    ``{args.hf_prefix}/raw_completions/`` (plus ``…/<stage>/`` when
    ``args.stage`` names one, e.g. ``ceiling_draws`` — plan §4.2).
    """
    layers = _resolve_layers_arg(args.layers)
    h_dim = _resolve_h_dim(args.model, args.h_dim)
    manifest_key, gen_seed = SPLIT_TO_MANIFEST[args.split]
    logger.info(
        "[ladder] model=%s split=%s (manifest=%s, seed=%d) layers=%s H=%d shard=%d/%d hf_prefix=%s",
        args.model,
        args.split,
        manifest_key,
        gen_seed,
        layers,
        h_dim,
        args.shard_index,
        args.num_shards,
        args.hf_prefix,
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

    # 3. Load models. Capture mode governs which we hold at once.
    C.phase("load_model")
    tok, hf = N10.load_models(args.model, args.device)

    llm = None
    if args.capture_mode in ("coresident", "phase_split_gen"):
        llm = _build_capture_engine(args.model) if args.device == "cuda" else None

    if args.capture_mode == "phase_split_capture":
        # Capture-only pass: free vLLM engine and rely on persisted responses.
        # We do NOT build an engine at all — the launcher's earlier gen-only
        # invocation already produced raw_completions/ on HF; this pass just
        # re-loads them, forwards them through the HF model, and uploads
        # tensors. Deferred to a follow-up implementation cycle within Unit 2.
        raise SystemExit(
            "phase_split_capture mode not yet implemented — coresident + phase_split_gen only. "
            "See Unit 2 return manifest for the deferred TODO."
        )

    # 4. Capture method selection: batched (default) with parity fallback.
    capture_fn_choice = "perrow"
    if args.capture_batch_size > 1 and args.capture_mode == "coresident":
        # Run the parity gate on the first 32 rows of shard 0's first chunk.
        probe_end = min(32, len(shard_rows))
        probe_prompts = [r["prompt"] for r in shard_rows[:probe_end]]
        probe_cis = [
            int(r.get("ladder_local_id", r.get("i", i)))
            for i, r in enumerate(shard_rows[:probe_end])
        ]
        # Generate responses for probe rows (small — safe).
        if llm is not None:
            probe_responses = N10._generate(llm, tok, probe_prompts)  # noqa: SLF001
        else:  # CPU device — fake empty responses (probe skipped)
            probe_responses = ["" for _ in probe_prompts]
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
        if args.no_upload or not pending_pt:
            return
        _flush_upload_batch(scratch, stage_prefix, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    skipped_all: list[dict] = []
    self_gate_rows: list[dict] = []
    self_gate_fired = False

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
            if name in done_pt and raw_name in done_raw:
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
            # Generate responses.
            if llm is not None:
                responses = N10._generate(llm, tok, kept_prompts)  # noqa: SLF001
            else:
                responses = ["" for _ in kept_prompts]  # CPU path (smoke)

            # Persist raw completions FIRST (persist-by-default; text path,
            # non-LFS, quota-immune — upload-policy §v2).
            C.write_json_atomic(
                scratch / raw_name,
                {
                    "shard_index": args.shard_index,
                    "chunk": ci_idx,
                    "split": args.split,
                    "seed": gen_seed,
                    "rows": [
                        {"ci": int(c), "prompt": p, "response": r}
                        for c, p, r in zip(kept_cis, kept_prompts, responses, strict=True)
                    ],
                },
            )

            # Trimmed capture (skipped in phase_split_gen mode — gen only).
            if args.capture_mode == "phase_split_gen":
                n_kept = len(kept_prompts)  # gen-side row count
                # No .pt to write; only raw_completions uploads.
                pending_raw.append(raw_name)
            else:
                rows = _do_capture(kept_prompts, responses, kept_cis)
                if not rows:
                    logger.warning(
                        "[shard %d] chunk %d: 0 captured rows; skip", args.shard_index, ci_idx
                    )
                    continue
                torch.save(_stack_chunk(rows, layers, args.shard_index, ci_idx), scratch / name)
                if not self_gate_fired:
                    self_gate_rows.extend(rows)
                n_kept = len(rows)
                pending_pt.append(name)
                pending_raw.append(raw_name)

            kept_total += n_kept
            if not args.no_upload and len(pending_raw) >= UPLOAD_BATCH:
                _flush_pending()

            logger.info(
                "[shard %d] chunk %d/%d: %d/%d captured (%d over-length skipped, %.0fs) [%s]",
                args.shard_index,
                ci_idx + 1,
                n_sub,
                n_kept,
                len(chunk),
                len(skipped),
                time.time() - ts,
                capture_fn_choice if args.capture_mode != "phase_split_gen" else "gen-only",
            )

            # First-chunk self-gate — plan §7 Decision Gate 1.
            if (
                args.first_chunk_self_gate
                and args.capture_mode != "phase_split_gen"
                and not self_gate_fired
                and len(self_gate_rows) >= 2000
            ):
                primary_layer_index = len(layers) // 2  # f=0.679 primary is middle entry
                passed, diag = _first_chunk_self_gate(self_gate_rows, primary_layer_index)
                self_gate_fired = True
                logger.info(
                    "[ladder-gate] first-chunk self-gate: %s (%s)",
                    "PASS" if passed else "FAIL",
                    diag,
                )
                if not passed:
                    # Write a sentinel the poller will drain into an epm:failure
                    # marker; abort THIS scale's job (other scales unaffected).
                    sentinel_path = Path("/workspace/logs") / (
                        f"issue-1491-first-chunk-self-gate-fail-{args.split}-shard{args.shard_index}.json"
                    )
                    if sentinel_path.parent.exists():
                        C.write_json_atomic(
                            sentinel_path,
                            {
                                "epm_marker": "epm:failure",
                                "failure_class": "code",
                                "reason": "first_chunk_self_gate_fail",
                                "detail": diag,
                                "split": args.split,
                                "shard_index": args.shard_index,
                            },
                        )
                    _flush_pending()  # keep what we have
                    raise SystemExit(1)

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
        "[shard %d] done: %d kept rows across %d chunks (%d over-length skipped)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
    )

    # Free GPU allocator + release engine before the process exits (parent
    # parity; helps a phase_split follow-up capture invocation not OOM).
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
        "phase_split_capture: only HF capture from persisted responses (14B/32B; deferred in this Unit 2 build).",
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
