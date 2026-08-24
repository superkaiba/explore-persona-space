"""Pod-side GPU text-embedding leg for the #2215 separation-comparison round.

User-chat inline GPU override (2026-08-24): "use the best open source text
embedding model and run on GPUs" — supersedes the OpenAI
`text-embedding-3-large` route (key revoked, 401) for the round's TEXT space.

Embeds the 17,640 banked rollout completion texts (parent battery 14,040 +
dbe battery 3,600; texts staged from the HF data repo at the round's pinned
revisions) with `Qwen/Qwen3-Embedding-8B` via the vLLM pooling runner:
document-side embedding of the completion text ONLY (no instruction prefix),
the model's default last-token pooling, L2-normalized per draw, plus the
per-context mean of normalized per-draw embeddings. Nothing is truncated —
the 32k context covers every completion; a token-length precheck asserts
every text fits `--max-model-len` and reports the over-32k count (expect 0).

Outputs (fp16 npz + meta.json) upload to the HF data repo under
`issue2215_sepcmp/analysis_tensors/embeddings_qwen3_8b/` BEFORE pod
termination. The VM-side fold (`issue2215_separation_comparison.py`, whose
loaders and revision pins this script reuses) consumes the per-context means.

Run on pod-2215-sepcmp (intent eval, 1x H100):
    uv run python scripts/issue2215_sepcmp_qwen_embed.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2215_separation_comparison as SEP  # noqa: E402

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIM = 4096
NATIVE_CONTEXT = 32_768
HF_PREFIX = "issue2215_sepcmp/analysis_tensors/embeddings_qwen3_8b"


def log(msg: str) -> None:
    print(f"[qwen-embed] {msg}", flush=True)


def embed_texts(texts: list[str], max_model_len: int, chunk: int) -> np.ndarray:
    """All-text embedding pass via the vLLM pooling runner (fp32 out)."""
    from vllm import LLM

    llm = LLM(
        model=EMBED_MODEL,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    t0 = time.monotonic()
    n_chunks = (len(texts) + chunk - 1) // chunk
    for k in range(n_chunks):
        lo, hi = k * chunk, min((k + 1) * chunk, len(texts))
        res = llm.embed(texts[lo:hi])
        arr = np.array([r.outputs.embedding for r in res], dtype=np.float32)
        assert arr.shape == (hi - lo, EMBED_DIM), (arr.shape, hi - lo)
        out[lo:hi] = arr
        log(f"chunk {k + 1}/{n_chunks} rows={hi - lo} elapsed={time.monotonic() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage-root", type=Path, default=Path("/workspace/eps2215_sepcmp/staged"))
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps2215_sepcmp/out"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    batteries: dict[str, list[dict]] = {}
    for battery, paths, rev, expect in (
        ("parent", SEP.PARENT_JSONL, SEP.REV_PARENT, 14_040),
        ("dbe", SEP.DBE_JSONL, SEP.REV_MAIN, 3_600),
    ):
        rows = SEP.read_rollout_rows([SEP.stage(p, args.stage_root, rev) for p in paths])
        assert len(rows) == expect, (battery, len(rows), expect)
        batteries[battery] = rows

    # flat per-draw text set (empty completions skipped, counted per battery)
    flat: list[tuple[str, str, int, str]] = []  # (battery, context_id, draw, text)
    n_empty: dict[str, int] = {}
    for battery, rows in batteries.items():
        keep = [r for r in rows if r["text"].strip()]
        n_empty[battery] = len(rows) - len(keep)
        flat.extend((battery, r["context_id"], r["draw"], r["text"]) for r in keep)
    log(f"texts: {len(flat)} total; empty skipped: {n_empty}")

    # token-length precheck with the embed model's own tokenizer (no truncation)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    lens = [len(tok.encode(t[3], add_special_tokens=True)) for t in flat]
    n_over_native = sum(1 for n in lens if n > NATIVE_CONTEXT)
    assert max(lens) < args.max_model_len, (
        f"{sum(1 for n in lens if n >= args.max_model_len)} text(s) at or over "
        f"--max-model-len={args.max_model_len} (max={max(lens)}); raise the flag — never truncate"
    )
    log(f"token lengths: max={max(lens)} mean={np.mean(lens):.0f} over-32k={n_over_native}")

    emb = embed_texts([t[3] for t in flat], args.max_model_len, args.chunk)
    norms = np.linalg.norm(emb, axis=1)
    assert (norms > 0).all(), "zero-norm embedding row"
    pre_norm_dev = float(np.abs(norms - 1.0).max())
    emb = (emb.astype(np.float64) / norms[:, None]).astype(np.float32)
    log(f"embedded {emb.shape}; pre-normalization max |norm-1| = {pre_norm_dev:.2e}")

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    counts: dict[str, dict] = {}
    for battery in batteries:
        sel = [i for i, t in enumerate(flat) if t[0] == battery]
        e = emb[sel]
        cids = [flat[i][1] for i in sel]
        draws = np.array([flat[i][2] for i in sel], dtype=np.int64)
        np.savez(
            args.out_root / f"perdraw_qwen3_8b_{battery}.npz",
            emb=e.astype(np.float16),
            context_ids=np.array(cids),
            draws=draws,
        )
        order = sorted(set(cids))
        row_of = {c: i for i, c in enumerate(order)}
        sums = np.zeros((len(order), EMBED_DIM), dtype=np.float64)
        cnt = np.zeros(len(order), dtype=np.int64)
        idx = np.array([row_of[c] for c in cids], dtype=np.int64)
        np.add.at(sums, idx, e.astype(np.float64))
        np.add.at(cnt, idx, 1)
        assert (cnt > 0).all()
        np.savez(
            args.out_root / f"means_qwen3_8b_{battery}.npz",
            mean=(sums / cnt[:, None]).astype(np.float16),
            context_ids=np.array(order),
        )
        counts[battery] = {
            "n_draw_rows": len(sel),
            "n_contexts": len(order),
            "n_empty_skipped": n_empty[battery],
            "draws_per_context_min": int(cnt.min()),
        }
        log(f"{battery}: {counts[battery]}")

    api = HfApi()
    meta = {
        "model": EMBED_MODEL,
        "model_revision": SEP.retry_transient(
            lambda: api.model_info(EMBED_MODEL).sha, what="model_info(qwen3-embed)"
        ),
        "dim": EMBED_DIM,
        "pooling": "last-token (vLLM pooling runner, model default pooler)",
        "instruction_prefix": "none (document-side embedding)",
        "normalized": True,
        "pre_norm_max_abs_dev": pre_norm_dev,
        "max_model_len": args.max_model_len,
        "token_len_max": int(max(lens)),
        "token_len_mean": float(np.mean(lens)),
        "n_over_native_context": n_over_native,
        "counts": counts,
        **as_metadata_dict(git_provenance()),
    }
    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=1))

    if not args.skip_upload:
        SEP.assert_hub_dir_filecounts(args.out_root, HF_PREFIX)
        SEP.retry_transient(
            lambda: api.upload_folder(
                folder_path=args.out_root,
                path_in_repo=HF_PREFIX,
                repo_id=SEP.HF_DATA_REPO,
                repo_type="dataset",
                commit_message="issue #2215 sepcmp GPU override: Qwen3-Embedding-8B text embeddings",
            ),
            what="upload_folder(sepcmp qwen3 embeddings)",
        )
        log(f"uploaded {HF_PREFIX}")
    log("DONE")


if __name__ == "__main__":
    main()
