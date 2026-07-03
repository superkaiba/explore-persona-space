#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →, ×) in scientific docstrings/log messages.
"""Issue #841 scaling-capture — the ONE GPU phase of the scaling-capture round.

Captures last-prompt-token residual states (all 28 block outputs) for 96,000
NEW LMSYS prompts drawn from stream positions 5001+ and STRING-FILTERED disjoint
from the parent's 1..5000 (lmsys repeats prompt texts, so ~0.8% of raw new
positions collide with a parent string and are dropped; the pool backfills from
later stream positions to exactly 96,000 clean — see ``split_stream``), so
Stage-0/Stage-1 can re-fit the parent's next-activation maps at fit-set sizes
n ∈ {4k,10k,25k,50k,100k}. Single manipulated variable = the fit-corpus size
(plan v9 §3); everything else matches the parent's pass_b.

Forward: batched (left-pad + explicit position_ids), all 28 layers, last real
token per sequence, cast fp32 for storage. ``--capture-dtype`` selects the
forward dtype — **default bf16, TF32 OFF** (code-review round 1 fold 1): the
parent's production 5000-context pass_b ran on GPU with
``torch_dtype=torch.bfloat16`` (``issue779_collect.py:804-806``; line 808's
``float32`` is the CPU-fallback branch that never ran the 5000-ctx capture), so
bf16 both passes the KILL-A rel-1e-3 spot-gate against the stored bf16-precision
``cx_last`` AND keeps the fit-pool precision the single held variable. Override
with ``--capture-dtype fp32`` (a deliberate precision change) or loosen
``--anchor-rel-tol`` for batched-vs-per-sequence numerics.

KILL-A (spot-gate, before the full 96k capture): re-capture 8-16 REGENERATED
parent contexts MIXED into a realistic batch and assert they match the stored
``cx_last`` rows to ``--anchor-rel-tol`` (default 1e-3). This doubles as the
batched-vs-per-sequence equivalence check.

Local (no-GPU) smoke substitutes for the GPU-bound forward:
  --equivalence-test : tiny random Qwen2 (CPU) batched-left-pad vs per-sequence
                       last-token cosine ≥ 0.999 (validates position_ids + gather
                       + hook-layer alignment — the batched-rewrite-equivalence).
  --dry-run-io N     : synthetic (N,28,3584) → shard/manifest write + load
                       round-trip (validates the persist path; skips upload/GPU).
  --verify-imports   : execute this module's deferred imports (script-mode gate).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# load_dotenv (HF_HOME + shared-VM thread caps + credentials) BEFORE torch import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C779  # noqa: E402
import issue841_common as C  # noqa: E402
import issue841_scaling_common as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_scaling_capture")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _set_tf32(enabled: bool) -> None:
    """Toggle TF32 matmul/cudnn (plan default: OFF for the fp32 forward)."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _left_pad_position_ids(attn: torch.Tensor) -> torch.Tensor:
    """Left-pad position_ids = cumsum(mask)-1 clamped ≥0 (RoPE indexes from the
    first REAL token). Without this, HF defaults to a plain arange that shifts
    every real token's rotary position by the left-pad count (the classic
    left-pad divergence — #502)."""
    pos = attn.long().cumsum(-1) - 1
    return pos.clamp(min=0)


@torch.no_grad()
def capture_last_token_batched(
    model, input_ids: torch.Tensor, attn: torch.Tensor, layers: list[int]
) -> np.ndarray:
    """Last real token residual at each block in ``layers`` for a LEFT-PADDED batch.

    Returns ``(B, len(layers), H)`` fp32 numpy. Under left-padding every sequence's
    last real token is the LAST column, so the gather is ``hs[:, -1, :]`` uniformly.
    A forward hook on ``model.model.layers[L]`` fires on block L's output (== the
    parent's ``extract_layer_activations`` block-index convention), reduced to the
    last token IN the hook so peak memory is O(one block's (B,T,H)), never the full
    28-layer grid (the #545/#666 accumulation trap).
    """
    device = model.device
    pos = _left_pad_position_ids(attn).to(device)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(li: int):
        def _hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out  # (B, T, H)
            captured[li] = hs[:, -1, :].detach().float().cpu()

        return _hook

    for li in layers:
        handles.append(model.model.layers[li].register_forward_hook(_mk(li)))
    try:
        model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            position_ids=pos,
            use_cache=False,
        )
    finally:
        for h in handles:
            h.remove()
    stacked = torch.stack([captured[li] for li in layers], dim=1)  # (B, n_layers, H)
    return stacked.numpy()


def _chat_texts(tokenizer, prompts: list[str]) -> list[str]:
    """Chat-template each user prompt with add_generation_prompt=True (parent frame)."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def _assert_generation_suffix(tokenizer, text: str) -> None:
    """Fail-loud: the chat-templated text ends with the assistant-header suffix
    (parent's #594 position control)."""
    ids = tokenizer(text, return_tensors="pt", padding=False)["input_ids"]
    suffix = tokenizer.decode(ids[0, -3:])
    assert suffix == C779.GENERATION_SUFFIX, (
        f"generation-suffix assert failed: {suffix!r} != {C779.GENERATION_SUFFIX!r}"
    )


def _tokenize_left_pad(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def capture_prompts(
    model, tokenizer, prompts: list[str], layers: list[int], *, batch_size: int
) -> np.ndarray:
    """Batched last-token capture over all ``prompts`` → (N, len(layers), H) fp32."""
    texts = _chat_texts(tokenizer, prompts)
    if texts:
        _assert_generation_suffix(tokenizer, texts[0])
    out = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, lo in enumerate(range(0, len(texts), batch_size)):
        chunk = texts[lo : lo + batch_size]
        ids, attn = _tokenize_left_pad(tokenizer, chunk)
        out.append(capture_last_token_batched(model, ids, attn, layers))
        if bi % 20 == 0 or bi == n_batches - 1:
            logger.info("[capture] batch %d/%d (%d prompts)", bi + 1, n_batches, len(chunk))
    return np.concatenate(out, axis=0)


# ── KILL-A spot-gate ─────────────────────────────────────────────────────────────


def kill_a_spot_gate(
    model,
    tokenizer,
    parent_prompts: list[str],
    new_prompts: list[str],
    stored_cx_last: np.ndarray,
    layers: list[int],
    *,
    n_probe: int,
    batch_fill: int,
    rel_tol: float,
) -> dict:
    """Re-capture ``n_probe`` regenerated parent contexts MIXED into a realistic
    batch and assert each matches its stored ``cx_last`` row to ``rel_tol``.

    The probe parent contexts are placed at the FRONT of a batch padded out with
    ``new_prompts`` to ``batch_fill`` total, so the batch's length distribution +
    left-pad regime match the full capture (NOT an isolated tiny all-parent batch).
    Returns a summary dict; raises AssertionError (KILL-A) on any miss.
    """
    probe_idx = np.linspace(0, len(parent_prompts) - 1, num=n_probe, dtype=int).tolist()
    probe_prompts = [parent_prompts[i] for i in probe_idx]
    fill_n = max(0, batch_fill - n_probe)
    fill_prompts = new_prompts[:fill_n]
    batch_prompts = probe_prompts + fill_prompts
    texts = _chat_texts(tokenizer, batch_prompts)
    _assert_generation_suffix(tokenizer, texts[0])
    ids, attn = _tokenize_left_pad(tokenizer, texts)
    captured = capture_last_token_batched(model, ids, attn, layers)  # (batch, L, H)
    probe_captured = captured[:n_probe]  # rows 0..n_probe-1 are the parent probes

    checks = []
    worst = 0.0
    for j, pidx in enumerate(probe_idx):
        stored = stored_cx_last[pidx]  # (L, H)
        got = probe_captured[j]  # (L, H)
        denom = np.abs(stored) + 1e-6
        rel = float(np.max(np.abs(got - stored) / denom))
        cos = float(np.sum(got * stored) / (np.linalg.norm(got) * np.linalg.norm(stored) + 1e-8))
        worst = max(worst, rel)
        checks.append({"parent_index": int(pidx), "rel_err": rel, "cosine": cos})
    summary = {
        "n_probe": n_probe,
        "batch_fill": len(batch_prompts),
        "rel_tol": rel_tol,
        "worst_rel_err": worst,
        "checks": checks,
    }
    failed = [c for c in checks if c["rel_err"] > rel_tol]
    if failed:
        logger.error(
            "[KILL-A] spot-gate FAILED: worst rel_err %.4g > tol %.4g (%d/%d probes). "
            "If the batched-vs-per-sequence numerics are the cause, retry with the "
            "batch-1 fallback; if the fp32/bf16 precision mismatch is the cause "
            "(the parent's GPU forward was bf16), retry with --capture-dtype bf16.",
            worst,
            rel_tol,
            len(failed),
            n_probe,
        )
        raise AssertionError(f"KILL-A spot-gate failed: {summary}")
    logger.info("[KILL-A] PASS: worst rel_err %.4g ≤ tol %.4g (%d probes)", worst, rel_tol, n_probe)
    summary["pass"] = True
    return summary


# ── disjointness ────────────────────────────────────────────────────────────────


def split_stream(all_prompts: list[str], n_new: int) -> tuple[list[str], list[str], int, int]:
    """Parent (first N_PARENT) vs a CLEAN new pool of exactly ``n_new`` prompts.

    The guarantee the plan's §4.1 disjointness protects is test-set contamination
    protection: NO new fit-row string equals ANY parent-5000 string. lmsys-chat-1m
    repeats prompt TEXTS across rows, so the stream positions 5001+ contain
    parent-colliding strings by construction (~0.8% measured) — the old empty-overlap
    assert was therefore unsatisfiable. FILTER instead: drop new prompts that collide
    with the parent-5000 string set (cross-dedup ONLY — new-internal duplicates are
    KEPT, exactly as the parent kept its own internal dupes, so no second variable is
    introduced), and backfill from later stream positions until the clean pool holds
    exactly ``n_new``. Hard-fail (never silently under-fill) if the over-streamed
    buffer runs out of clean prompts first.

    Returns ``(parent_prompts, new_clean, dropped_count, stream_extent)`` where
    ``stream_extent`` is the 1-based stream position of the last consumed prompt
    (provenance: how far into the stream the clean pool reached).
    """
    assert len(all_prompts) >= S.N_PARENT + n_new, (
        f"loader returned {len(all_prompts)} prompts, need ≥ {S.N_PARENT + n_new} "
        f"(parent {S.N_PARENT} + new {n_new}); corpus ran short (§4.1 WildChat fallback)"
    )
    parent_prompts = all_prompts[: S.N_PARENT]
    parent_set = set(parent_prompts)
    new_clean: list[str] = []
    dropped = 0
    stream_extent = S.N_PARENT
    for offset, p in enumerate(all_prompts[S.N_PARENT :]):
        if p in parent_set:
            dropped += 1
            continue
        new_clean.append(p)
        if len(new_clean) == n_new:
            stream_extent = S.N_PARENT + offset + 1  # 1-based position of last consumed
            break
    if len(new_clean) < n_new:
        n_raw_new = len(all_prompts) - S.N_PARENT
        raise RuntimeError(
            f"split_stream could not fill {n_new} clean new prompts: got {len(new_clean)} "
            f"after dropping {dropped} parent-colliding prompts from {n_raw_new} raw new "
            f"positions. Raise EPM_I841S_BACKFILL_MARGIN (currently "
            f"{S.N_STREAM_BACKFILL_MARGIN}) or fall back to WildChat (§4.1)."
        )
    # POST-FILTER invariant (now trivially true — guards a regression): no clean new
    # prompt string equals any parent string. This IS the §4.1 test-set-contamination
    # guarantee, enforced by construction rather than asserted against raw stream order.
    assert set(new_clean).isdisjoint(parent_set), (
        "split_stream post-filter invariant violated: a new prompt still collides with parent"
    )
    logger.info(
        "[split_stream] dropped %d parent-colliding prompts; new pool backfilled to %d clean "
        "(stream extended to %d)",
        dropped,
        len(new_clean),
        stream_extent,
    )
    return parent_prompts, new_clean, dropped, stream_extent


# ── local no-GPU smoke substitutes ───────────────────────────────────────────────


def equivalence_test() -> int:
    """Tiny random Qwen2 (CPU): batched-left-pad vs per-sequence last-token cosine.

    Validates the position_ids + last-column gather + hook-layer alignment of
    ``capture_last_token_batched`` INDEPENDENTLY of the real 7B model (the
    GPU-bound-phase batched-rewrite-equivalence substitute). Uses hand-built
    variable-length input_ids (no tokenizer download). Asserts cosine ≥ 0.999
    per (layer × sequence) between the batched (B≥2, real left-pad firing) and
    the per-sequence (unpadded) last-token reads.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    layers = list(range(cfg.num_hidden_layers))
    pad_id = 0
    # variable-length sequences → left-pad actually fires (B=4, distinct lengths)
    seqs = [
        [3, 5, 7, 9, 11, 13],
        [4, 6, 8],
        [2, 10, 12, 14, 16],
        [1, 15, 17, 19, 21, 23, 25],
    ]
    tmax = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), tmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), tmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, tmax - len(s) :] = torch.tensor(s)  # LEFT pad
        attn[i, tmax - len(s) :] = 1
    batched = capture_last_token_batched(model, ids, attn, layers)  # (B, L, H)

    # per-sequence reference: no pad, position_ids = arange(len)
    serial = np.zeros_like(batched)
    for i, s in enumerate(seqs):
        sid = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        sattn = torch.ones_like(sid)
        row = capture_last_token_batched(model, sid, sattn, layers)  # (1, L, H)
        serial[i] = row[0]

    cos = np.sum(batched * serial, axis=2) / (
        np.linalg.norm(batched, axis=2) * np.linalg.norm(serial, axis=2) + 1e-12
    )  # (B, L)
    worst = float(cos.min())
    logger.info("[equivalence-test] min cosine(batched, serial) over B×L = %.6f", worst)
    if worst < 0.999:
        logger.error("[equivalence-test] FAILED: min cosine %.6f < 0.999", worst)
        return 1
    logger.info("[equivalence-test] PASS (min cosine ≥ 0.999)")
    return 0


def dry_run_io(n: int, capture_dir: Path) -> int:
    """Synthetic (n,28,3584) → shard/manifest write + load round-trip + resume check."""
    rng = np.random.default_rng(0)
    cx = rng.standard_normal((n, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)).astype(np.float32)
    prompts = [f"synthetic prompt {i}" for i in range(n)]
    meta = C.reproducibility_metadata({"phase": "scaling_capture_dry_run_io", "n": n})
    shard_rows = max(1, n // 3)  # multiple shards exercise the multi-shard + resume path
    S.write_capture_shards(
        cx, prompts, "synthetic", meta, capture_dir, shard_rows=shard_rows, capture_dtype="bf16"
    )
    loaded = S.load_capture(capture_dir)
    assert loaded["cx_last"].shape == cx.shape, (loaded["cx_last"].shape, cx.shape)
    assert np.allclose(loaded["cx_last"], cx), "round-trip cx_last mismatch"
    assert loaded["prompts"] == prompts, "round-trip prompts mismatch"
    assert loaded["capture_dtype"] == "bf16", loaded["capture_dtype"]
    # realized-dtype provenance round-trip (Fold 1): synthetic realized == requested.
    assert loaded["realized_capture_dtype"] == "bf16", loaded["realized_capture_dtype"]
    assert loaded["requested_capture_dtype"] == "bf16", loaded["requested_capture_dtype"]
    # resume predicate: every written shard reads done for the SAME dtype, NOT for another.
    for idx, lo, hi in S.shard_boundaries(n, shard_rows):
        assert S.shard_is_done(capture_dir, idx, lo, hi, "bf16"), (idx, "should be done")
        assert not S.shard_is_done(capture_dir, idx, lo, hi, "fp32"), (idx, "dtype must invalidate")
    logger.info(
        "[dry-run-io] PASS: %d rows round-tripped through %d shards; resume predicate holds",
        n,
        len(S.shard_paths(capture_dir)),
    )
    return 0


def _verify_imports() -> int:
    """Execute this module's deferred imports (script-mode import gate, #823)."""
    from huggingface_hub import hf_hub_download, list_repo_files  # noqa: F401
    from issue779_collect import load_train_contexts  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForCausalLM,
        AutoTokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    from explore_persona_space.orchestrate.hub import upload_dataset_directory  # noqa: F401

    logger.info("[verify-imports] all deferred imports resolved")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 scaling-capture (GPU capture phase).")
    ap.add_argument("--model", default=C779.DEFAULT_MODEL)
    ap.add_argument("--capture-dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument(
        "--allow-dtype-mismatch",
        action="store_true",
        help="permit a realized model dtype != requested (default: fail loud on any "
        "silent up/downcast that would change the held fit-pool precision)",
    )
    ap.add_argument("--tf32", action="store_true", help="enable TF32 (default OFF per plan)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--n-contexts", type=int, default=S.N_NEW_CONTEXTS, help="new contexts to capture"
    )
    ap.add_argument("--n-probe", type=int, default=12, help="KILL-A parent probes (8-16)")
    ap.add_argument("--anchor-rel-tol", type=float, default=1e-3, help="KILL-A rel-error tolerance")
    ap.add_argument("--capture-dir", type=Path, default=S.CAPTURE_DIR)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny n (64) new contexts, SAME path")
    ap.add_argument("--equivalence-test", action="store_true")
    ap.add_argument("--dry-run-io", type=int, default=0, metavar="N")
    ap.add_argument("--verify-imports", action="store_true")
    args = ap.parse_args()

    if args.verify_imports:
        return _verify_imports()
    if args.equivalence_test:
        return equivalence_test()
    if args.dry_run_io:
        return dry_run_io(args.dry_run_io, args.capture_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.orchestrate.hub import upload_dataset_directory

    n_new = 64 if args.smoke else args.n_contexts
    _set_tf32(args.tf32)
    logger.info(
        "capture-dtype=%s tf32=%s n_new=%d batch=%d rel_tol=%g",
        args.capture_dtype,
        args.tf32,
        n_new,
        args.batch_size,
        args.anchor_rel_tol,
    )

    use_cuda = torch.cuda.is_available()  # capture reads model.device inside the hook
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # last real token → last column, uniform gather
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = DTYPES[args.capture_dtype]
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    # Realized-dtype provenance (code-review round 2, Fold 1): args.capture_dtype is
    # what we REQUESTED; read the model's ACTUAL parameter dtype so a silent up/downcast
    # (e.g. a model that ignores torch_dtype and loads fp32) is loud, not swallowed. The
    # held fit-pool precision is the single manipulated variable, so an unnoticed cast is
    # a confound. Normalize the realized torch dtype back to the CLI label vocab so the
    # resume key + downstream labels stay in {bf16, fp32}.
    requested_dtype_obj = DTYPES[args.capture_dtype]
    realized_dtype_obj = next(model.parameters()).dtype
    _dtype_labels = {v: k for k, v in DTYPES.items()}
    realized_capture_dtype = _dtype_labels.get(
        realized_dtype_obj, str(realized_dtype_obj).removeprefix("torch.")
    )
    if realized_dtype_obj != requested_dtype_obj and not args.allow_dtype_mismatch:
        raise RuntimeError(
            f"realized model dtype {realized_dtype_obj} != requested {requested_dtype_obj} "
            f"(--capture-dtype {args.capture_dtype}); a silent up/downcast changes the held "
            f"fit-pool precision. Pass --allow-dtype-mismatch to override deliberately."
        )
    logger.info(
        "[dtype] requested=%s realized=%s (obj=%s)",
        args.capture_dtype,
        realized_capture_dtype,
        realized_dtype_obj,
    )

    n_layers = len(model.model.layers)
    assert n_layers == C.EXPECTED_LAYERS, (n_layers, C.EXPECTED_LAYERS)
    assert model.config.hidden_size == C.EXPECTED_HIDDEN, model.config.hidden_size
    layers = list(range(n_layers))

    # Regenerate the deterministic stream, then FILTER parent-colliding new prompts
    # and backfill to exactly n_new clean (§4.1 test-set contamination protection).
    # Over-stream by N_STREAM_BACKFILL_MARGIN so the drop-and-backfill still fills the
    # clean pool; lmsys repeats prompt strings so an empty-overlap assert is impossible.
    from issue779_collect import load_train_contexts

    buffer_target = S.N_PARENT + n_new + S.N_STREAM_BACKFILL_MARGIN
    all_prompts, source = load_train_contexts(buffer_target, smoke=False)
    parent_prompts, new_prompts, dropped_collisions, stream_extent = split_stream(
        all_prompts, n_new
    )
    logger.info(
        "[stream] source=%s parent=%d new=%d dropped=%d stream_extent=%d",
        source,
        len(parent_prompts),
        len(new_prompts),
        dropped_collisions,
        stream_extent,
    )

    # KILL-A spot-gate against the stored parent cx_last (before the full capture).
    stored_cx_last = C.load_pass_b()["cx_last"]  # (5000, 28, 3584) fp32
    spot = kill_a_spot_gate(
        model,
        tokenizer,
        parent_prompts,
        new_prompts,
        stored_cx_last,
        layers,
        n_probe=args.n_probe,
        batch_fill=max(args.batch_size, args.n_probe * 3),
        rel_tol=args.anchor_rel_tol,
    )

    # Full capture of the new contexts — SHARD-AS-YOU-GO (bounds peak RAM to one
    # shard; resumable after a crash, keyed on capture_dtype so a dtype change
    # never reuses stale shards). Each shard's rows are captured, written, freed.
    target_new = new_prompts[:n_new]
    n_total = len(target_new)
    bounds = S.shard_boundaries(n_total)
    for idx, lo, hi in bounds:
        if S.shard_is_done(args.capture_dir, idx, lo, hi, realized_capture_dtype):
            logger.info(
                "[capture] shard %d/%d rows[%d:%d] already done — resume skip",
                idx + 1,
                len(bounds),
                lo,
                hi,
            )
            continue
        chunk_prompts = target_new[lo:hi]
        cx_chunk = capture_prompts(
            model, tokenizer, chunk_prompts, layers, batch_size=args.batch_size
        )  # (hi-lo, 28, 3584) fp32
        assert cx_chunk.shape == (hi - lo, n_layers, C.EXPECTED_HIDDEN), cx_chunk.shape
        S.write_one_shard(
            args.capture_dir,
            idx,
            cx_chunk,
            chunk_prompts,
            lo=lo,
            hi=hi,
            n_total=n_total,
            n_shards=len(bounds),
            source=source,
            capture_dtype=realized_capture_dtype,
            requested_dtype=args.capture_dtype,
        )
        logger.info("[capture] wrote shard %d/%d rows[%d:%d]", idx + 1, len(bounds), lo, hi)
        del cx_chunk

    meta = C.reproducibility_metadata(
        {
            "phase": "scaling_capture",
            "capture_dtype": realized_capture_dtype,
            "requested_capture_dtype": args.capture_dtype,
            "realized_capture_dtype": realized_capture_dtype,
            "tf32": args.tf32,
            "source": source,
            "n_new": n_total,
            "dropped_parent_collisions": dropped_collisions,
            "stream_extent": stream_extent,
            "kill_a": spot,
            "smoke": args.smoke,
        }
    )
    S.write_capture_manifest(
        args.capture_dir,
        n_total,
        source,
        realized_capture_dtype,
        meta,
        requested_dtype=args.capture_dtype,
    )

    # Durable capture summary (git, issue branch) — spot-gate + disjointness record.
    C.write_json_atomic(
        S.EVAL_SCALING_DIR / "capture_summary.json",
        {
            "n_new": len(target_new),
            "capture_dtype": realized_capture_dtype,
            "requested_capture_dtype": args.capture_dtype,
            "realized_capture_dtype": realized_capture_dtype,
            "tf32": args.tf32,
            "source": source,
            "kill_a": spot,
            "disjointness": {
                "n_parent": len(parent_prompts),
                "n_new": len(new_prompts),
                "dropped_parent_collisions": dropped_collisions,
                "stream_extent": stream_extent,
            },
            "metadata": meta,
        },
    )

    if not args.no_upload:
        logger.info("[upload] cx_last shards → %s:%s", C.HF_DATA_REPO, S.HF_CAPTURE_BUCKET)
        upload_dataset_directory(args.capture_dir, S.HF_CAPTURE_BUCKET, pattern="*")

    logger.info(
        "[done] captured %d new contexts (requested=%s realized=%s)",
        len(target_new),
        args.capture_dtype,
        realized_capture_dtype,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
