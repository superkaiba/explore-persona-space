#!/usr/bin/env python3
"""Issue #2552 exactrep follow-up — sharded, resumable per-assistant-turn capture (GPU).

Teacher-forces every prep-kept LMSYS conversation (see issue2552_exactrep_prep.py)
through Qwen2.5-7B-Instruct and stores two PAIRED layer-19 objects per assistant turn:
the TOKEN MEAN over the turn's CONTENT tokens (the arXiv 2606.28548 convention), and
the pre-assistant last-prompt-token context state for a user-requested matched
context-only SAE.

Answer spans come from ONE full-text tokenization's offset mapping (never a
re-tokenized concatenation — the #1092 BPE-seam rule), content tokens only (no
<|im_start|>assistant\\n header and no <|im_end|>\\n tail). The exact generation
prompt is tokenized independently for the context state. When its ids are a prefix of
the full conversation the already-computed causal state is reused; rare BPE-seam
mismatches are re-forwarded in a batched correction path.

Reuse map: extract_layer_activations (analysis/extraction — hooks + logits_to_keep
OOM guard; layer L == hidden_states[L+1], the banked-store convention),
issue779_ffc_n10k_generate_capture.load_models (model load + GENERATION_SUFFIX
assert), prep's render_segments (template-shape invariant).

Sharding + checkpointing: conversations are grouped into fixed-size chunks in corpus
order; chunk gci is processed by shard `gci % num_shards == shard`. Each chunk writes
  chunk_{gci:06d}.npy        (n_rows, hidden) fp16 assistant-content means
  context_{gci:06d}.npy      (n_rows, hidden) fp16 last-prompt-token states
  chunk_{gci:06d}.rows.jsonl per-row identity (conversation_id, msg_idx, n_span_tokens)
  chunk_{gci:06d}.done.json  sentinel (written LAST; fingerprint-gated resume skip)
A resumed run skips chunks whose sentinel matches the run fingerprint (generating
parameters only — never recomputed-float hashes).

Modes:
  --pilot           run ONE pending chunk through the production path, print the
                    measured per-chunk wall + extrapolated per-shard/total GPU-hours.
  --verify-batched  batched-vs-serial equivalence gate on a small slice (cosine >=
                    0.999 per row; right-pad + causal mask — pads cannot influence
                    real positions; run on the CPU tiny model where bf16 jitter is nil).
  --tiny-model      CPU smoke: tiny random-weight Qwen2 over the REAL vocab id space
                    (never valid on cuda).

Launcher contract (4-way pod fan-out): pin CUDA_VISIBLE_DEVICES per shard in the
LAUNCHER env — `CUDA_VISIBLE_DEVICES=$i ... --shard $i --num-shards 4`; the script
itself always uses cuda:0 (CVD selects the physical device).

No text is uploaded or printed by this script (ids + counters only).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2552_exactrep_prep as PREP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace, write_jsonl_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2552_exactrep_capture")

DEFAULT_LAYER = 19  # block index; == hidden_states[20], the banked #779 store convention
EXPECTED_HIDDEN = 3584
DEFAULT_CHUNK_CONVS = 256
IM_END_TAIL = PREP.IM_END_TAIL


# ── pure span/batch helpers (unit-tested offline) ─────────────────────────────────


def content_char_ranges(
    segments: list[str], n_prefix: int, msg_roles: list[str], asst_msg_idx: list[int]
) -> list[tuple[int, int]]:
    """Char ranges (in full-text coords) of each captured assistant message's CONTENT.

    Computed ARITHMETICALLY from cumulative segment lengths (never text.find — the
    #1776 substring mis-anchor trap). Segment for message i sits at n_prefix + i and
    renders "<|im_start|>{role}\\n{content}<|im_end|>\\n"."""
    starts = [0]
    for s in segments:
        starts.append(starts[-1] + len(s))
    out = []
    for mi in asst_msg_idx:
        si = n_prefix + mi
        header = f"<|im_start|>{msg_roles[mi]}\n"
        c_start = starts[si] + len(header)
        c_end = starts[si + 1] - len(IM_END_TAIL)
        assert c_end >= c_start, (mi, c_start, c_end)
        out.append((c_start, c_end))
    return out


def token_spans_from_offsets(
    offsets: list[tuple[int, int]], ranges: list[tuple[int, int]]
) -> list[tuple[int, int] | None]:
    """Token index span [s, e) per char range: tokens OVERLAPPING the content range
    (a boundary-straddling BPE token counts in). None for a zero-token span (content
    fully merged into the delimiter — the #825 zero-width class; caller drops+counts)."""
    spans: list[tuple[int, int] | None] = []
    for c_start, c_end in ranges:
        s = e = None
        for ti, (a, b) in enumerate(offsets):
            if b <= a:  # zero-width offset entries (specials on some tokenizers)
                continue
            if b > c_start and a < c_end:
                if s is None:
                    s = ti
                e = ti + 1
        if s is None or c_end <= c_start:
            spans.append(None)
        else:
            spans.append((s, e))
    return spans


def span_means(hs: torch.Tensor, spans_per_row: list[list[tuple[int, int]]]) -> torch.Tensor:
    """fp32 token means over spans of a (B, T, H) hidden-state batch → (n_spans, H).

    Right-padded batches only: spans index real (non-pad) positions by construction."""
    B, T, H = hs.shape
    assert len(spans_per_row) == B, (len(spans_per_row), B)
    outs = []
    for b, spans in enumerate(spans_per_row):
        for s, e in spans:
            assert 0 <= s < e <= T, (b, s, e, T)
            outs.append(hs[b, s:e].float().mean(0))
    assert outs, "span_means called with zero spans"
    return torch.stack(outs)


def batches_by_budget(lengths: list[int], max_rows: int, max_tokens: int) -> list[list[int]]:
    """Group conversation indices (sorted long-first) into batches capped by row count
    AND padded token footprint (batch_rows * max_len_in_batch <= max_tokens)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * new_max > max_tokens):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def pilot_extrapolation(chunk_wall_s: float, n_chunks_total: int, num_shards: int) -> dict:
    """Measured 1-chunk wall → projected per-shard wall + total GPU-hours."""
    per_shard_chunks = math.ceil(n_chunks_total / num_shards)
    return {
        "chunk_wall_s": round(chunk_wall_s, 2),
        "n_chunks_total": n_chunks_total,
        "num_shards": num_shards,
        "per_shard_chunks": per_shard_chunks,
        "projected_shard_hours": round(per_shard_chunks * chunk_wall_s / 3600.0, 3),
        "projected_total_gpu_hours": round(n_chunks_total * chunk_wall_s / 3600.0, 3),
    }


def run_fingerprint(args, corpus_fp: dict) -> dict:
    """Resume fingerprint: generating parameters only (machine-stable)."""
    return {
        "model_id": PREP.MODEL_ID,
        "layer": int(args.layer),
        "span_convention": "assistant-content-token-mean-v1",
        "context_convention": "generation-prompt-last-token-v1",
        "chunk_convs": int(args.chunk_convs),
        "corpus_fingerprint": corpus_fp,
        "tiny_model": bool(args.tiny_model),
    }


def chunk_paths(out_dir: Path, gci: int) -> tuple[Path, Path, Path]:
    return (
        out_dir / f"chunk_{gci:06d}.npy",
        out_dir / f"chunk_{gci:06d}.rows.jsonl",
        out_dir / f"chunk_{gci:06d}.done.json",
    )


def context_chunk_path(out_dir: Path, gci: int) -> Path:
    """Paired context-vector chunk for answer chunk ``gci``."""
    return out_dir / f"context_{gci:06d}.npy"


def chunk_completed(out_dir: Path, gci: int, fp: dict) -> bool:
    """Resume predicate: sentinel LAST-written, fingerprint-matched, files present."""
    npy, rows, done = chunk_paths(out_dir, gci)
    ctx = context_chunk_path(out_dir, gci)
    if not (npy.exists() and ctx.exists() and rows.exists() and done.exists()):
        return False
    doc = json.loads(done.read_text())
    return doc.get("fingerprint") == fp


# ── model + capture ────────────────────────────────────────────────────────────────


def _tiny_cpu_model():
    """Tiny random-weight Qwen2 over the REAL vocab id space (CPU smoke only)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=152064,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def load_capture_model(args):
    if args.tiny_model:
        if args.device == "cuda":
            raise ValueError("--tiny-model is CPU-smoke only")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(PREP.MODEL_ID)
        return tok, _tiny_cpu_model()
    import issue779_ffc_n10k_generate_capture as GC

    tok, hf = GC.load_models(PREP.MODEL_ID, args.device)
    assert hf.config.hidden_size == EXPECTED_HIDDEN, hf.config.hidden_size
    return tok, hf


def conv_rows_for_capture(rec: dict, tok, layer_unused: int, counters: dict):
    """Tokenize one conversation → answer spans and exact context-position specs.

    Returns None when every span is zero-width (counted, never zero-faked)."""
    msgs = rec["msgs"]
    segs = PREP.render_segments(msgs, tok)
    n_prefix = len(segs) - len(msgs)
    full = "".join(segs)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) != int(rec["n_render_tokens"]):
        raise RuntimeError(
            f"tokenizer drift: conv {rec['conversation_id']} tokenizes to {len(ids)} tokens, "
            f"prep counted {rec['n_render_tokens']} — prep and capture must share one tokenizer"
        )
    roles = [m["role"] for m in msgs]
    ranges = content_char_ranges(segs, n_prefix, roles, rec["asst_msg_idx"])
    spans = token_spans_from_offsets(enc["offset_mapping"], ranges)
    kept_spans, kept_context_specs, kept_meta = [], [], []
    for mi, span in zip(rec["asst_msg_idx"], spans, strict=True):
        if span is None:
            counters["zero_width_spans"] += 1
            continue
        prompt_text = tok.apply_chat_template(msgs[:mi], tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        assert prompt_ids, (rec["conversation_id"], mi)
        if len(prompt_ids) <= len(ids) and ids[: len(prompt_ids)] == prompt_ids:
            # Exact causal reuse: the full forward has an identical token prefix.
            context_spec = {"position": len(prompt_ids) - 1, "prompt_ids": None}
        else:
            # The BPE seam changed the final prompt token; re-forward this exact prompt.
            counters["context_seam_reforwards"] += 1
            context_spec = {"position": None, "prompt_ids": prompt_ids}
        kept_spans.append(span)
        kept_context_specs.append(context_spec)
        kept_meta.append(
            {
                "conversation_id": rec["conversation_id"],
                "msg_idx": int(mi),
                "n_span_tokens": int(span[1] - span[0]),
            }
        )
    if not kept_spans:
        counters["convs_all_spans_zero_width"] += 1
        return None
    return ids, kept_spans, kept_context_specs, kept_meta


@torch.no_grad()
def capture_chunk(recs: list[dict], tok, model, args, counters: dict):
    """Capture paired answer means/context states for one chunk."""
    prepared = []
    for rec in recs:
        out = conv_rows_for_capture(rec, tok, args.layer, counters)
        if out is not None:
            prepared.append(out)
    if not prepared:
        z = np.zeros((0, model.config.hidden_size), np.float16)
        return z, z.copy(), []
    lengths = [len(ids) for ids, _, _, _ in prepared]
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    assert pad_id is not None
    all_means: dict[int, torch.Tensor] = {}
    all_contexts: dict[tuple[int, int], torch.Tensor] = {}
    correction_prompts: list[tuple[tuple[int, int], list[int]]] = []
    for batch in batches_by_budget(lengths, args.batch_max_rows, args.batch_max_tokens):
        max_len = max(lengths[i] for i in batch)
        ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask_t = torch.zeros((len(batch), max_len), dtype=torch.long)
        for bi, i in enumerate(batch):
            n = lengths[i]
            ids_t[bi, :n] = torch.as_tensor(prepared[i][0], dtype=torch.long)  # right-pad
            mask_t[bi, :n] = 1
        captured = extract_layer_activations(
            model, ids_t.to(device), [args.layer], attention_mask=mask_t.to(device)
        )
        hs = captured[args.layer]  # (B, T, H)
        assert hs.shape[:2] == ids_t.shape, (hs.shape, ids_t.shape)
        means = span_means(hs, [prepared[i][1] for i in batch])  # (n_spans_batch, H) fp32
        cursor = 0
        for bi, i in enumerate(batch):
            n_sp = len(prepared[i][1])
            all_means[i] = means[cursor : cursor + n_sp].to(torch.float16).cpu()
            for j, spec in enumerate(prepared[i][2]):
                pos = spec["position"]
                if pos is None:
                    correction_prompts.append(((i, j), spec["prompt_ids"]))
                else:
                    all_contexts[(i, j)] = hs[bi, pos].to(torch.float16).cpu()
            cursor += n_sp
        del captured, hs, means

    # Exact prompt-only correction for prefixes changed by a BPE seam.
    if correction_prompts:
        correction_lengths = [len(ids) for _key, ids in correction_prompts]
        for batch in batches_by_budget(
            correction_lengths, args.batch_max_rows, args.batch_max_tokens
        ):
            max_len = max(correction_lengths[i] for i in batch)
            ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            mask_t = torch.zeros((len(batch), max_len), dtype=torch.long)
            for bi, i in enumerate(batch):
                prompt_ids = correction_prompts[i][1]
                ids_t[bi, : len(prompt_ids)] = torch.as_tensor(prompt_ids, dtype=torch.long)
                mask_t[bi, : len(prompt_ids)] = 1
            captured = extract_layer_activations(
                model, ids_t.to(device), [args.layer], attention_mask=mask_t.to(device)
            )
            hs = captured[args.layer]
            for bi, i in enumerate(batch):
                key, prompt_ids = correction_prompts[i]
                all_contexts[key] = hs[bi, len(prompt_ids) - 1].to(torch.float16).cpu()
            del captured, hs

    rows_meta: list[dict] = []
    mean_mats, context_mats = [], []
    for i, (_ids, spans, _context_specs, meta) in enumerate(prepared):
        mean_mats.append(all_means[i])
        context_mats.append(torch.stack([all_contexts[(i, j)] for j in range(len(spans))]))
        rows_meta.extend(meta)
    y = torch.cat(mean_mats).numpy().astype(np.float16)
    x = torch.cat(context_mats).numpy().astype(np.float16)
    assert y.shape == (len(rows_meta), model.config.hidden_size), y.shape
    assert x.shape == y.shape, (x.shape, y.shape)
    return y, x, rows_meta


def _write_chunk(
    out_dir: Path,
    gci: int,
    y: np.ndarray,
    x: np.ndarray,
    rows_meta: list[dict],
    doc: dict,
):
    npy, rows, done = chunk_paths(out_dir, gci)
    with atomic_replace(npy) as tmp_npy, tmp_npy.open("wb") as fh:
        np.save(fh, y)  # OPEN HANDLE: np.save appends .npy to path names, never to handles
    ctx = context_chunk_path(out_dir, gci)
    with atomic_replace(ctx) as tmp_ctx, tmp_ctx.open("wb") as fh:
        np.save(fh, x)
    write_jsonl_atomic(rows, rows_meta)
    PREP._write_json_atomic(done, doc)  # sentinel LAST


def iter_corpus_chunks(corpus_dir: Path, chunk_convs: int):
    """Yield (gci, raw_lines) chunks in deterministic corpus order (lines parsed by
    the consumer, so skipped shards never pay json.loads)."""
    files = sorted(corpus_dir.glob("conv_*.jsonl"))
    assert files, f"no corpus shards under {corpus_dir}"
    gci = 0
    buf: list[str] = []
    for path in files:
        with path.open(encoding="utf-8") as f:  # text-mode iteration (never splitlines)
            for line in f:
                if not line.strip():
                    continue
                buf.append(line)
                if len(buf) >= chunk_convs:
                    yield gci, buf
                    gci += 1
                    buf = []
    if buf:
        yield gci, buf


def verify_batched(tok, model, recs: list[dict], args) -> None:
    """Batched-vs-serial equivalence gate: cosine >= 0.999 per row (B>=2 so padding
    fires). Run on the CPU tiny model (fp32 — no bf16 jitter headroom question)."""
    counters = {
        "zero_width_spans": 0,
        "convs_all_spans_zero_width": 0,
        "context_seam_reforwards": 0,
    }
    y_batched, x_batched, meta_b = capture_chunk(recs, tok, model, args, counters)
    serial_args = argparse.Namespace(**vars(args))
    serial_args.batch_max_rows = 1
    y_serial, x_serial, meta_s = capture_chunk(recs, tok, model, serial_args, counters)
    assert meta_b == meta_s, "batched/serial row identity drift"
    assert y_batched.shape == y_serial.shape and y_batched.shape[0] >= 2, y_batched.shape
    a = torch.as_tensor(y_batched, dtype=torch.float32)
    b = torch.as_tensor(y_serial, dtype=torch.float32)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    worst = float(cos.min())
    print(f"[verify-batched] rows={len(cos)} worst_cos={worst:.6f}", flush=True)
    if worst < 0.999:
        raise RuntimeError(f"batched-vs-serial equivalence FAILED: worst cosine {worst:.6f}")
    ctx_cos = torch.nn.functional.cosine_similarity(
        torch.as_tensor(x_batched, dtype=torch.float32),
        torch.as_tensor(x_serial, dtype=torch.float32),
        dim=1,
    )
    ctx_worst = float(ctx_cos.min())
    print(f"[verify-batched] context_worst_cos={ctx_worst:.6f}", flush=True)
    if ctx_worst < 0.999:
        raise RuntimeError(
            f"context batched-vs-serial equivalence FAILED: worst cosine {ctx_worst:.6f}"
        )


def run_capture(args) -> int:
    t0 = time.time()
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_manifest = json.loads((corpus_dir.parent / "manifest.json").read_text())
    fp = run_fingerprint(args, corpus_manifest["fingerprint"])
    tok, model = load_capture_model(args)
    if args.verify_batched:
        recs = []
        for _gci, lines in iter_corpus_chunks(corpus_dir, args.chunk_convs):
            recs = [json.loads(ln) for ln in lines[: max(4, args.batch_max_rows)]]
            break
        verify_batched(tok, model, recs, args)
        return 0

    n_chunks_total = math.ceil(int(corpus_manifest["counters"]["kept_convs"]) / args.chunk_convs)
    done_n, ran_n = 0, 0
    for gci, lines in iter_corpus_chunks(corpus_dir, args.chunk_convs):
        if gci % args.num_shards != args.shard:
            continue
        if chunk_completed(out_dir, gci, fp):
            done_n += 1
            continue
        chunk = [json.loads(ln) for ln in lines]
        counters = {
            "zero_width_spans": 0,
            "convs_all_spans_zero_width": 0,
            "context_seam_reforwards": 0,
        }
        c0 = time.time()
        y, x, rows_meta = capture_chunk(chunk, tok, model, args, counters)
        wall = time.time() - c0
        doc = {
            "fingerprint": fp,
            "gci": gci,
            "n_rows": int(y.shape[0]),
            "n_convs": len(chunk),
            "counters": counters,
            "wall_s": round(wall, 2),
            "metadata": as_metadata_dict(git_provenance(), phase="exactrep-capture"),
        }
        _write_chunk(out_dir, gci, y, x, rows_meta, doc)
        ran_n += 1
        print(
            f"[capture] chunk {gci}/{n_chunks_total} shard={args.shard} rows={y.shape[0]} "
            f"chunk_wall={wall:.1f}s elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if args.pilot:
            print(
                f"[pilot] {json.dumps(pilot_extrapolation(wall, n_chunks_total, args.num_shards))}",
                flush=True,
            )
            break
        if args.limit_chunks and ran_n >= args.limit_chunks:
            break
    if args.pilot and ran_n == 0:
        print(
            f"[pilot] no pending chunk on shard {args.shard} "
            f"({done_n} already completed) — nothing measured",
            flush=True,
        )
    print(
        f"[capture] shard {args.shard}/{args.num_shards} done: ran={ran_n} "
        f"resumed_skip={done_n} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #2552 exactrep sharded turn-mean capture.")
    ap.add_argument("--corpus-dir", type=Path, default=Path("/workspace/eps-2552-exactrep/corpus"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/eps-2552-exactrep/store"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--chunk-convs", type=int, default=DEFAULT_CHUNK_CONVS)
    ap.add_argument("--batch-max-rows", type=int, default=16)
    ap.add_argument("--batch-max-tokens", type=int, default=32768, help="padded-batch footprint")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--tiny-model", action="store_true", help="CPU smoke only")
    ap.add_argument("--pilot", action="store_true", help="ONE chunk + wall extrapolation")
    ap.add_argument("--limit-chunks", type=int, default=0, help="0 = all pending (production)")
    ap.add_argument("--verify-batched", action="store_true", help="batched-vs-serial gate")
    ap.add_argument("--import-check", action="store_true", help="argparse-attr completeness")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return 0
    assert 0 <= args.shard < args.num_shards, (args.shard, args.num_shards)
    return run_capture(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (rc race)
