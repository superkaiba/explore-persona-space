#!/usr/bin/env python3
"""Issue #812 — per-position aligned extraction (CPU stream-reduce, Branch B).

Reads #658's per-(context, probe) answer-span store from HF once per context
(``issue658_theory_assumptions/store/answer_spans/<ctx>.pt``, ~3.3 GB each),
reduces it to a tiny aligned-positions tensor + the single-vector pooling-operator
inputs, deletes the blob, moves to the next context — so peak local footprint is
~one context (bounded by the LRU ``_HfStreamSpanSource`` from
``scripts/issue658_fit_predictors.py``), NEVER the full ~165 GB grid. Rebuilt from
the inline plan §4.1 stream-reduce contract; does NOT import
``scripts/issue722_per_position_vC_skill.py`` (that file is local-untracked /
stranded off ``main`` — the "built-but-stranded" pattern).

Output ``data/issue_812/store/pooling_inputs.pt`` (fp16), uploaded to HF
``issue812_pooling/analysis_tensors/`` (plan-referenced downstream input):

    {
      "ctx_ids": [N],  "layers": [n_layers],  "K": 16,
      "mean":       (N, n_layers, H) fp16,   # mean over ALL answer tokens of ALL probes
      "max":        (N, n_layers, H) fp16,   # element-wise max
      "attn_fixed": (N, n_layers, H) fp16,   # seed-42 random-query softmax pool
      "aligned_pos":(N, n_layers, 2K+2, H) fp16,  # tail -1..-K, head 0..K-1, im_end, turn_nl
      "coverage":   (N, n_layers, 2K+2) int32,    # #probes contributing per aligned slot
      "meta": {...},
    }

The aligned-position set (per plan §4.1): end-aligned tail positions ``-1..-K``,
start-aligned head positions ``0..K-1``, plus the 2 turn-boundary tokens
(``im_end`` at ``span_end``, ``turn_nl`` at ``span_end+1`` where present) ->
``(2K+2)`` positions (K=16 -> 34). Per (context, layer, aligned-position), MEAN
over the probes that HAVE a token at that aligned index. The span blob's per-probe
tensor is ``(Lc, S, H)`` = the answer-token activations for one probe; the
"turn-boundary" positions live at the very TAIL of the span (the last two answer
positions when present), so ``im_end`` == tail -1 and ``turn_nl`` == tail 0 by the
#658 capture convention; we expose them as their own two slots for downstream
interpretability, and they overlap the tail slots by construction (documented,
not double-counted — they are separate feature columns).

Idempotent: if the output artifact already exists locally, the run SKIPS
re-extraction (per plan §4.1) unless ``--force`` is passed.

CPU-only, 0 GPU-h.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

# Reuse the canonical stream-reduce source (peak ~one context) from #658.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from issue658_fit_predictors import _HfStreamSpanSource, _SpanSource  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue812.extract")

# ── Constants (plan §4.1 / §11) ──────────────────────────────────────────────
DEFAULT_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_SPANS_PREFIX = "issue658_theory_assumptions/store/answer_spans"
DEFAULT_SPANS_INDEX = "issue658_theory_assumptions/store/answer_spans/index.json"
K_ALIGNED = 16  # Source: #722 per_position_vC_skill P_MAX; #810 body K~8-16
COVERAGE_MIN_CONTEXTS = 30  # Source: #722 per_position_vC_skill COVERAGE_MIN
ATTN_FIXED_SEED = 42  # Source: plan §11 attn-fixed random query
MIN_DISK_FREE_GB = 30.0  # abort if the streaming staging dir has < this free
H_HIDDEN = 3584  # Qwen-2.5-7B residual width (assert-checked)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_SCRIPTS_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _resolve_ctx_ids(repo: str, index_path: str, limit: int | None) -> list[str]:
    """The 50 context ids from the #658 answer_spans index.json (order-stable)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, index_path, repo_type="dataset")
    with open(local) as f:
        idx = json.load(f)
    # index.json is either a list of ids or {"context_ids": [...]} — accept both.
    if isinstance(idx, dict):
        ids = idx.get("context_ids") or idx.get("ctx_ids") or list(idx.keys())
    else:
        ids = list(idx)
    ids = [str(c) for c in ids]
    if limit is not None:
        ids = ids[:limit]
    return ids


def _fixed_query(layer_idx: int, h: int, seed: int) -> torch.Tensor:
    """A per-layer fixed random UNIT query for the attn-fixed pool (seed-derived)."""
    g = torch.Generator().manual_seed(seed * 100003 + layer_idx)
    q = torch.randn(h, generator=g, dtype=torch.float32)
    return q / (q.norm() + 1e-9)


def _reduce_probe_span(
    span: torch.Tensor, fixed_q: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """Reduce ONE probe's ONE-layer answer span (S, H) fp16 to operator inputs.

    Returns per-probe contributions in fp32 (accumulated in fp32, cast fp16 at end):
      - mean_sum: (H,) sum over the S answer tokens (probe-level mean deferred)
      - n_tokens: int scalar (for the probe-mean over ALL tokens of ALL probes)
      - maxp: (H,) element-wise max over the S answer tokens
      - attn_fixed: (H,) softmax(span @ fixed_q)-weighted sum
      - tail: (K, H) end-aligned positions -1..-K (fp32; NaN-filled beyond span len)
      - head: (K, H) start-aligned positions 0..K-1 (fp32; NaN-filled beyond len)
      - im_end / turn_nl: (H,) the last two answer positions (turn-boundary tokens)
      - tail_valid / head_valid: (K,) bool masks (which aligned slots are present)
    """
    assert span.ndim == 2, f"span must be (S, H); got {tuple(span.shape)}"
    s = span.shape[0]
    assert s > 0, "empty answer span — cannot reduce a 0-token answer"
    sp = span.float()  # (S, H) fp32

    mean_sum = sp.sum(dim=0)  # (H,)
    maxp = sp.max(dim=0).values  # (H,)
    scores = torch.softmax(sp @ fixed_q, dim=0)  # (S,)
    attn_fixed = (scores.unsqueeze(-1) * sp).sum(dim=0)  # (H,)

    h = sp.shape[1]
    tail = torch.full((k, h), float("nan"), dtype=torch.float32)
    head = torch.full((k, h), float("nan"), dtype=torch.float32)
    tail_valid = torch.zeros(k, dtype=torch.bool)
    head_valid = torch.zeros(k, dtype=torch.bool)
    for j in range(k):
        if j < s:  # tail position -1-j is the (s-1-j)-th token
            tail[j] = sp[s - 1 - j]
            tail_valid[j] = True
        if j < s:  # head position j is the j-th token
            head[j] = sp[j]
            head_valid[j] = True

    # Turn-boundary tokens: the very last two answer positions, per the #658
    # capture convention (im_end at span_end, turn_nl at span_end+1 where present).
    im_end = sp[-1]  # (H,) always present (s>0)
    turn_nl = sp[-2] if s >= 2 else torch.full((h,), float("nan"), dtype=torch.float32)
    turn_nl_valid = s >= 2

    return {
        "mean_sum": mean_sum,
        "n_tokens": s,
        "maxp": maxp,
        "attn_fixed": attn_fixed,
        "tail": tail,
        "head": head,
        "tail_valid": tail_valid,
        "head_valid": head_valid,
        "im_end": im_end,
        "turn_nl": turn_nl,
        "turn_nl_valid": turn_nl_valid,
    }


def _extract_context(blob: dict, layers: list[int], k: int, seed: int) -> dict[str, np.ndarray]:
    """Reduce ONE context's span blob (all probes, selected layers) to arrays.

    blob["spans"] is a list of (Lc, S, H) fp16 tensors (or None for empty answers),
    where Lc indexes the CAPTURED layers (blob["capture_layers"]). ``layers`` is the
    subset of capture-layer INDICES to keep (0..Lc-1). Returns fp16 arrays:
      mean/max/attn_fixed: (n_layers, H); aligned_pos: (n_layers, 2K+2, H);
      coverage: (n_layers, 2K+2) int32.
    """
    spans = blob["spans"]
    present = [s for s in spans if s is not None]
    if not present:
        raise ValueError("context has no non-empty answer spans")
    lc = present[0].shape[0]
    h = present[0].shape[2]
    assert h == H_HIDDEN, f"unexpected hidden width {h} (expected {H_HIDDEN})"
    for li in layers:
        assert 0 <= li < lc, f"layer index {li} out of range for Lc={lc}"

    n_layers = len(layers)
    n_aligned = 2 * k + 2  # tail K + head K + im_end + turn_nl

    mean_out = np.zeros((n_layers, h), dtype=np.float32)
    max_out = np.full((n_layers, h), -np.inf, dtype=np.float32)
    attn_out = np.zeros((n_layers, h), dtype=np.float32)
    aligned_sum = np.zeros((n_layers, n_aligned, h), dtype=np.float32)
    aligned_cnt = np.zeros((n_layers, n_aligned), dtype=np.int64)
    total_tokens = np.zeros(n_layers, dtype=np.int64)

    fixed_qs = {li: _fixed_query(li, h, seed) for li in layers}

    for out_idx, li in enumerate(layers):
        fq = fixed_qs[li]
        for pspan in present:
            red = _reduce_probe_span(pspan[li], fq, k)
            mean_out[out_idx] += red["mean_sum"].numpy()
            total_tokens[out_idx] += red["n_tokens"]
            np.maximum(max_out[out_idx], red["maxp"].numpy(), out=max_out[out_idx])
            attn_out[out_idx] += red["attn_fixed"].numpy()
            # aligned slots: [0..K-1] tail, [K..2K-1] head, [2K] im_end, [2K+1] turn_nl
            tail = red["tail"].numpy()
            head = red["head"].numpy()
            tv = red["tail_valid"].numpy()
            hv = red["head_valid"].numpy()
            for j in range(k):
                if tv[j]:
                    aligned_sum[out_idx, j] += tail[j]
                    aligned_cnt[out_idx, j] += 1
                if hv[j]:
                    aligned_sum[out_idx, k + j] += head[j]
                    aligned_cnt[out_idx, k + j] += 1
            aligned_sum[out_idx, 2 * k] += red["im_end"].numpy()
            aligned_cnt[out_idx, 2 * k] += 1
            if red["turn_nl_valid"]:
                aligned_sum[out_idx, 2 * k + 1] += red["turn_nl"].numpy()
                aligned_cnt[out_idx, 2 * k + 1] += 1

    n_probes = len(present)
    # mean = sum over ALL tokens of ALL probes / total token count, per layer.
    # attn_fixed = probe-mean of each probe's own softmax-weighted pool.
    for out_idx in range(n_layers):
        mean_out[out_idx] = mean_out[out_idx] / max(int(total_tokens[out_idx]), 1)
        attn_out[out_idx] /= max(n_probes, 1)  # probe-mean of per-probe attn pool
    # aligned = probe-mean per slot (mean over the probes that had a token there)
    aligned_mean = np.zeros_like(aligned_sum)
    for out_idx in range(n_layers):
        for slot in range(n_aligned):
            c = aligned_cnt[out_idx, slot]
            if c > 0:
                aligned_mean[out_idx, slot] = aligned_sum[out_idx, slot] / c
            # else leaves zeros; coverage array records the 0

    return {
        "mean": mean_out.astype(np.float16),
        "max": max_out.astype(np.float16),
        "attn_fixed": attn_out.astype(np.float16),
        "aligned_pos": aligned_mean.astype(np.float16),
        "coverage": aligned_cnt.astype(np.int32),
    }


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Issue 812 per-position aligned extraction.")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--spans-prefix", default=DEFAULT_SPANS_PREFIX)
    ap.add_argument("--spans-index", default=DEFAULT_SPANS_INDEX)
    ap.add_argument("--out", default="data/issue_812/store/pooling_inputs.pt")
    ap.add_argument(
        "--layers",
        default="",
        help="comma-separated capture-layer INDICES to keep (default: all 28)",
    )
    ap.add_argument(
        "--contexts",
        type=int,
        default=None,
        help="limit to the first N contexts (smoke); default all",
    )
    ap.add_argument("--k", type=int, default=K_ALIGNED)
    ap.add_argument("--seed", type=int, default=658)
    ap.add_argument("--attn-fixed-seed", type=int, default=ATTN_FIXED_SEED)
    ap.add_argument(
        "--local-spans-dir",
        default=None,
        help="read spans from a LOCAL dir instead of streaming from HF (offline smoke)",
    )
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        logger.info("Output %s already exists — SKIP (idempotent). Pass --force to redo.", out_path)
        return 0

    ctx_ids = _resolve_ctx_ids(args.repo, args.spans_index, args.contexts)
    logger.info("Extracting %d contexts, k=%d", len(ctx_ids), args.k)

    # Build the span source (local dir OR per-context HF stream).
    if args.local_spans_dir:
        source: _SpanSource = _SpanSource(Path(args.local_spans_dir))
        logger.info("spans: LOCAL dir %s", args.local_spans_dir)
    else:
        source = _HfStreamSpanSource(args.repo, args.spans_prefix, cache_size=1)
        logger.info("spans: STREAMING per-context from HF %s/%s", args.repo, args.spans_prefix)

    layers: list[int] | None = (
        [int(x) for x in args.layers.split(",") if x.strip() != ""] if args.layers else None
    )

    mean_rows, max_rows, attn_rows, aligned_rows, cov_rows = [], [], [], [], []
    kept_ctx: list[str] = []
    resolved_layers: list[int] | None = None

    for ci, ctx in enumerate(ctx_ids):
        free = _free_gb(Path(args.local_spans_dir) if args.local_spans_dir else Path("/tmp"))
        if free < MIN_DISK_FREE_GB:
            raise RuntimeError(
                f"disk free {free:.1f} GB < {MIN_DISK_FREE_GB} GB floor — abort before OOM-ing disk"
            )
        t0 = time.time()
        blob = source.load_blob(ctx)
        if resolved_layers is None:
            lc = next(s for s in blob["spans"] if s is not None).shape[0]
            resolved_layers = layers if layers is not None else list(range(lc))
        red = _extract_context(blob, resolved_layers, args.k, args.attn_fixed_seed)
        source.release(ctx)
        mean_rows.append(red["mean"])
        max_rows.append(red["max"])
        attn_rows.append(red["attn_fixed"])
        aligned_rows.append(red["aligned_pos"])
        cov_rows.append(red["coverage"])
        kept_ctx.append(ctx)
        logger.info(
            "[%d/%d] %s reduced in %.1fs (layers=%d)",
            ci + 1,
            len(ctx_ids),
            ctx,
            time.time() - t0,
            len(resolved_layers),
        )

    assert resolved_layers is not None
    # Coverage floor check per plan §4.1 (require >= COVERAGE_MIN_CONTEXTS contexts
    # with data at each aligned slot); reported, not hard-failed at smoke scale.
    cov = np.stack(cov_rows)  # (N, n_layers, 2K+2)
    ctx_with_data = (cov > 0).any(axis=1)  # (N, 2K+2): ctx has any-layer data at slot
    per_slot_ctx = ctx_with_data.sum(axis=0)  # (2K+2,)
    thin_slots = int((per_slot_ctx < min(COVERAGE_MIN_CONTEXTS, len(kept_ctx))).sum())
    logger.info(
        "Coverage: min ctx-with-data per aligned slot = %d (of %d); thin slots = %d",
        int(per_slot_ctx.min()),
        len(kept_ctx),
        thin_slots,
    )

    payload = {
        "ctx_ids": kept_ctx,
        "layers": resolved_layers,
        "K": args.k,
        "mean": torch.from_numpy(np.stack(mean_rows)),
        "max": torch.from_numpy(np.stack(max_rows)),
        "attn_fixed": torch.from_numpy(np.stack(attn_rows)),
        "aligned_pos": torch.from_numpy(np.stack(aligned_rows)),
        "coverage": torch.from_numpy(cov),
        "meta": {
            "issue": 812,
            "git_commit": _git_commit(),
            "created_utc": _now_iso(),
            "repo": args.repo,
            "spans_prefix": args.spans_prefix,
            "k": args.k,
            "attn_fixed_seed": args.attn_fixed_seed,
            "seed": args.seed,
            "n_contexts": len(kept_ctx),
            "n_layers": len(resolved_layers),
            "hidden": H_HIDDEN,
            "coverage_min_contexts": COVERAGE_MIN_CONTEXTS,
            "per_slot_ctx_with_data": per_slot_ctx.tolist(),
            "thin_slots": thin_slots,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    logger.info(
        "WROTE %s: N=%d layers=%d aligned=%d H=%d",
        out_path,
        len(kept_ctx),
        len(resolved_layers),
        2 * args.k + 2,
        H_HIDDEN,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
