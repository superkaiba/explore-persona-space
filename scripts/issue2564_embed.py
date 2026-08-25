"""PC embed phase for task #2564 — Qwen3-Embedding-8B answer-text embeddings (pod-side).

Port of the main-resident ``scripts/issue2215_sepcmp_qwen_embed.py`` rig
(``embed_texts`` at :52 — code reuse only; the #2215 text leg never completed
a run) adapted to the #2564 minimal-pair anchors (plan v6 §3.7 row 3, §9
``pc_embed``).

Reads the PA anchors (all 9,840 completion texts across the 10 cells; local
anchors root when present, else the HF prefix), embeds with the vLLM pooling
runner (model-default last-token pooling), L2-normalizes (float64 divide,
fp16 storage), and uploads per-draw + per-context-mean npz stores to
``issue2564_minpair/analysis_tensors/embeddings_qwen3_8b/``. Chunked
(default 2,500 → 4 chunks over 9,840 rows) with per-chunk atomic npz
checkpoints (``atomic_io.atomic_replace`` process-unique temps, #2336;
``np.savez`` gets an OPEN handle — it appends ``.npz`` to path-named
non-.npz targets) and a fingerprint-gated resume keyed on generating parameters +
the file-read row texts (bit-exact inputs — safe to hash). The FIRST
COMPUTED chunk's elapsed drives a pilot gate: projected wall >
``--pilot-ceiling-h`` (default 2.0 h, plan §9 PC ceiling) ⇒ report JSON +
``sys.exit(7)`` (a designed artifact-routed halt, never a bare rc=1);
demoted to an informational log line under ``--smoke`` (GATE-CALIBRATION
parity, gotchas.md).

Pod-side contracts: single process; ``load_dotenv()`` before heavy imports;
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` set before any vllm import (#628 —
the token-length precheck loads AutoTokenizer BEFORE ``LLM()``); the engine
is reaped (getattr-guarded ``engine_core.shutdown`` + process-group destroy)
before exit; terminal is ``sys.exit(0)`` after explicit flushes (reap ran
first, per the #1739/#2149 gotcha); no ``task.py`` shellouts. Sentinel:
``<out-root>/embed_uploaded.json`` written ONLY after the verified upload
(``--skip-upload`` writes ``embed_done.local.json`` instead), plus
``[phase=pc_embed]`` breadcrumbs and a per-chunk progress line
``[pc_embed] unit k/N chunk_k rows=… elapsed=…s``.

Smoke (``--smoke``): out-root rebinds to the ``smoke_<name>`` sibling, HF
prefix rebinds to ``issue2564_minpair/smoke``, cells default to the driver's
smoke slice (register + query), chunk defaults to 64, and the pilot gate is
informational. Anchors under smoke are fetched from the ``/smoke`` HF prefix
unless ``--anchors-root`` is given (the driver's smoke out-root is not
derivable from this script's out-root).

Token-length precheck: every text is tokenized with the embed model's own
tokenizer and the max must be < ``--max-model-len`` — the flag is raised,
inputs are NEVER truncated (the #2215 port's contract).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
# #628: vLLM reads this at import; the tokenizer precheck touches transformers
# BEFORE LLM(), so default fork() would poison the EngineCore subprocess.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2564_minpair"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIM = 4096
DEFAULT_CHUNK = 2500
SMOKE_CHUNK = 64
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_PILOT_CEILING_H = 2.0
EXIT_PILOT_GATE = 7  # distinct rc — a designed halt the dispatcher routes (gotchas.md pilot-gate)
ANCHOR_CELLS: tuple[str, ...] = tuple(BK.INSTRUCTION_AXES) + ("query",)
SMOKE_CELLS: tuple[str, ...] = ("register", "query")  # mirrors issue2564_run.SMOKE_CELLS


def log(msg: str) -> None:
    """Flush-immediate stdout log line (daemonized launcher friendly)."""
    print(msg, flush=True)


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write via the process-unique-temp helper (#2336)."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL via split("\\n") — NEVER splitlines() (U+2028 shred, #950)."""
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def anchors_rel(cell: str) -> str:
    """Repo-relative anchors path for one cell (the PA layout)."""
    return f"raw_completions/anchors/anchors_{cell}.jsonl"


def stage_anchor_files(
    cells: tuple[str, ...],
    anchors_root: Path | None,
    hf_prefix: str,
    staging_dir: Path,
) -> dict[str, Path]:
    """Resolve each cell's anchors JSONL: local root first, else HF fetch.

    Fails loud on a missing cell (empty-selection rule: PD/PC never proceed on
    a partial anchor set — the plan's off_pod_phases reads name every cell).
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, Path] = {}
    for cell in cells:
        if anchors_root is not None:
            local = anchors_root / anchors_rel(cell)
            if local.is_file():
                out[cell] = local
                continue
        fn = f"{hf_prefix}/{anchors_rel(cell)}"
        got = hub.retry_transient(
            lambda fn=fn: hf_hub_download(
                HF_DATA_REPO, filename=fn, repo_type="dataset", local_dir=str(staging_dir)
            ),
            what=f"hf_hub_download({fn})",
        )
        out[cell] = Path(got)
    return out


def load_rows(paths: dict[str, Path]) -> tuple[list[dict], int]:
    """Load + deterministically order anchor rows; returns (rows, n_skipped_empty).

    Rows are sorted by (cell, context_id, draw). Empty-text rows are skipped
    with a count (the #2215 port's convention); an empty FILE or an empty
    total selection raises (fail loud — the artifact is committed input).
    """
    rows: list[dict] = []
    n_empty = 0
    for cell, p in sorted(paths.items()):
        cell_rows = _read_jsonl(p)
        if not cell_rows:
            raise RuntimeError(f"anchors file for cell {cell!r} is EMPTY: {p}")
        for r in cell_rows:
            if not str(r.get("text", "")).strip():
                n_empty += 1
                continue
            rows.append(r)
    if not rows:
        raise RuntimeError("empty anchor selection after load — nothing to embed")
    rows.sort(key=lambda r: (r["cell"], r["context_id"], int(r["draw"])))
    return rows, n_empty


def _regime_fp(rows: list[dict], chunk: int, max_model_len: int) -> str:
    """Resume fingerprint: generating params + file-read row identities.

    Hashes the (context_id, draw, sha16(text)) triples — bit-exact inputs read
    from files (safe per the float-last-bit rule; no recomputed floats).
    """
    ids = [
        (
            r["context_id"],
            int(r["draw"]),
            hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16],
        )
        for r in rows
    ]
    payload = json.dumps([EMBED_MODEL, EMBED_DIM, max_model_len, chunk, ids], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _reap_engine(llm: object) -> None:
    """Reap the vLLM engine before exit (gotchas.md vLLM v1 reaping recipe)."""
    import torch

    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
        else:
            executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
            if executor is not None and hasattr(executor, "shutdown"):
                executor.shutdown()
    except Exception as e:  # reap is best-effort; the sys.exit terminal follows
        log(f"[pc_embed] engine reap warning: {type(e).__name__}: {e}")
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        log(f"[pc_embed] destroy_process_group warning: {type(e).__name__}: {e}")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)  # subprocess teardown is async


def embed_rows(
    texts: list[str],
    *,
    chunks_dir: Path,
    fp: str,
    chunk: int,
    max_model_len: int,
    pilot_ceiling_h: float,
    pilot_report_path: Path,
    smoke: bool,
) -> np.ndarray:
    """Chunked embed with per-chunk atomic checkpoints + resume + pilot gate.

    Returns the (n, EMBED_DIM) float32 matrix of RAW (pre-normalization)
    embeddings. The vLLM engine is created lazily on the first PENDING chunk
    (an all-resumed invocation never loads it) and reaped before return.
    """
    n = len(texts)
    n_chunks = (n + chunk - 1) // chunk
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros((n, EMBED_DIM), dtype=np.float32)
    llm = None
    pilot_done = False
    try:
        for k in range(n_chunks):
            lo, hi = k * chunk, min((k + 1) * chunk, n)
            ck_path = chunks_dir / f"chunk_{k:03d}.npz"
            if ck_path.is_file():
                z = np.load(ck_path, allow_pickle=False)
                if (
                    str(z["fp"]) == fp
                    and int(z["lo"]) == lo
                    and int(z["hi"]) == hi
                    and z["emb"].shape == (hi - lo, EMBED_DIM)
                ):
                    out[lo:hi] = z["emb"].astype(np.float32)
                    log(f"[pc_embed] unit {k + 1}/{n_chunks} chunk_{k:03d} resumed rows={hi - lo}")
                    continue
                log(f"[pc_embed] chunk_{k:03d} checkpoint stale (regime changed) — recomputing")
            if llm is None:
                from vllm import LLM

                log(f"[pc_embed] loading {EMBED_MODEL} (pooling runner)")
                llm = LLM(
                    model=EMBED_MODEL,
                    runner="pooling",
                    dtype="bfloat16",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.90,
                )
            t0 = time.monotonic()
            res = llm.embed(texts[lo:hi], use_tqdm=False)
            arr = np.array([r.outputs.embedding for r in res], dtype=np.float32)
            assert arr.shape == (hi - lo, EMBED_DIM), arr.shape
            elapsed = time.monotonic() - t0
            # Process-unique atomic write (#2336). np.savez appends .npz to
            # path-named non-.npz targets, so hand it an OPEN handle (#1092).
            with atomic_replace(ck_path) as tmp:
                with open(tmp, "wb") as fh:
                    np.savez(fh, emb=arr.astype(np.float16), lo=lo, hi=hi, fp=fp)
            out[lo:hi] = arr
            log(
                f"[pc_embed] unit {k + 1}/{n_chunks} chunk_{k:03d} rows={hi - lo} "
                f"elapsed={elapsed:.1f}s"
            )
            if not pilot_done:
                pilot_done = True
                projected_h = elapsed * n_chunks / 3600.0
                report = {
                    "issue": ISSUE,
                    "phase": "pc_embed",
                    "first_chunk_rows": hi - lo,
                    "first_chunk_elapsed_s": round(elapsed, 2),
                    "n_chunks": n_chunks,
                    "projected_wall_h": round(projected_h, 4),
                    "ceiling_h": pilot_ceiling_h,
                    "verdict": "pass" if projected_h <= pilot_ceiling_h else "refuse",
                    "smoke": smoke,
                }
                _write_json_atomic(pilot_report_path, report)
                if projected_h > pilot_ceiling_h:
                    if smoke:
                        # GATE-CALIBRATION parity: smoke n makes the projection
                        # uninformative — compute + log, never halt (gotchas.md).
                        log(
                            f"[pc_embed] pilot gate INFORMATIONAL under --smoke: "
                            f"projected {projected_h:.2f}h > ceiling {pilot_ceiling_h}h"
                        )
                    else:
                        log(
                            f"[pc_embed] PILOT GATE REFUSAL: projected {projected_h:.2f}h > "
                            f"ceiling {pilot_ceiling_h}h — report at {pilot_report_path}"
                        )
                        if llm is not None:
                            _reap_engine(llm)
                            llm = None
                        sys.stdout.flush()
                        sys.stderr.flush()
                        sys.exit(EXIT_PILOT_GATE)
    finally:
        if llm is not None:
            _reap_engine(llm)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default="/workspace/eps2564/embed")
    ap.add_argument(
        "--anchors-root",
        default=None,
        help="local root holding raw_completions/anchors/ (default: parent of "
        "--out-root in production; HF fetch under --smoke)",
    )
    ap.add_argument("--cells", default=None, help="comma list override of anchor cells")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--pilot-ceiling-h", type=float, default=DEFAULT_PILOT_CEILING_H)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        raise SystemExit(0)

    out_root = Path(args.out_root)
    hf_prefix = HF_PREFIX
    cells = ANCHOR_CELLS
    chunk = args.chunk if args.chunk is not None else DEFAULT_CHUNK
    if args.smoke:
        out_root = out_root.parent / f"smoke_{out_root.name}"
        hf_prefix = f"{HF_PREFIX}/smoke"
        cells = SMOKE_CELLS
        if args.chunk is None:
            chunk = SMOKE_CHUNK
    if args.cells:
        cells = tuple(c.strip() for c in args.cells.split(",") if c.strip())
        unknown = set(cells) - set(ANCHOR_CELLS)
        if unknown:
            raise SystemExit(f"unknown cells: {sorted(unknown)}")
    anchors_root = Path(args.anchors_root) if args.anchors_root else None
    if anchors_root is None and not args.smoke:
        anchors_root = out_root.parent
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"[phase=pc_embed] start out_root={out_root} hf_prefix={hf_prefix} cells={list(cells)}")
    staging = out_root / "anchors_staging"
    paths = stage_anchor_files(cells, anchors_root, hf_prefix, staging)
    rows, n_empty = load_rows(paths)
    texts = [r["text"] for r in rows]
    log(f"[pc_embed] loaded {len(rows)} rows across {len(cells)} cells (skipped_empty={n_empty})")

    # Token-length precheck with the EMBED model's own tokenizer — raise the
    # flag, never truncate (the #2215 port's contract).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    lens = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
    max_len = max(lens)
    log(f"[pc_embed] token lens: max={max_len} mean={sum(lens) / len(lens):.1f}")
    if max_len >= args.max_model_len:
        raise RuntimeError(
            f"longest text is {max_len} tokens >= --max-model-len {args.max_model_len}; "
            "raise the flag — inputs are never truncated"
        )

    fp = _regime_fp(rows, chunk, args.max_model_len)
    emb = embed_rows(
        texts,
        chunks_dir=out_root / "chunks",
        fp=fp,
        chunk=chunk,
        max_model_len=args.max_model_len,
        pilot_ceiling_h=args.pilot_ceiling_h,
        pilot_report_path=out_root / "pilot_gate_report.json",
        smoke=args.smoke,
    )

    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    zero_idx = np.flatnonzero(norms == 0.0)
    if zero_idx.size:
        raise RuntimeError(f"zero-norm embeddings at row indices {zero_idx[:10].tolist()}")
    unit = (emb.astype(np.float64) / norms[:, None]).astype(np.float32)

    cids = np.array([r["context_id"] for r in rows])
    draws = np.array([int(r["draw"]) for r in rows], dtype=np.int32)
    cell_arr = np.array([r["cell"] for r in rows])

    # Per-context mean of the L2-NORMALIZED per-draw embeddings (not re-normalized;
    # documented in meta — downstream PE re-normalizes if it wants cosine means).
    uniq_cids = sorted(set(cids.tolist()))
    cid_index = {c: i for i, c in enumerate(uniq_cids)}
    sums = np.zeros((len(uniq_cids), EMBED_DIM), dtype=np.float64)
    counts = np.zeros(len(uniq_cids), dtype=np.int32)
    idx = np.array([cid_index[c] for c in cids.tolist()])
    np.add.at(sums, idx, unit.astype(np.float64))
    np.add.at(counts, idx, 1)
    means = (sums / counts[:, None]).astype(np.float16)

    emb_dir = out_root / "embeddings_qwen3_8b"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        emb_dir / "perdraw_anchors.npz",
        emb=unit.astype(np.float16),
        context_ids=cids,
        draws=draws,
        cells=cell_arr,
    )
    np.savez(
        emb_dir / "means_anchors.npz",
        emb_mean=means,
        context_ids=np.array(uniq_cids),
        n_draws=counts,
    )
    from huggingface_hub import HfApi

    revision = hub.retry_transient(
        lambda: HfApi().model_info(EMBED_MODEL).sha, what="model_info(qwen3-embed)"
    )
    meta = {
        "issue": ISSUE,
        "phase": "pc_embed",
        "model": EMBED_MODEL,
        "model_revision": revision,
        "pooling": "model_default_last_token",
        "normalized": "l2_float64_divide_fp16_store",
        "means": "mean of L2-normalized per-draw embeddings, NOT re-normalized",
        "embed_dim": EMBED_DIM,
        "max_model_len": args.max_model_len,
        "chunk": chunk,
        "n_rows": len(rows),
        "n_contexts": len(uniq_cids),
        "n_skipped_empty": n_empty,
        "cells": list(cells),
        "regime_fp": fp,
        "smoke": args.smoke,
        **as_metadata_dict(git_provenance(), phase="pc-embed"),
    }
    _write_json_atomic(emb_dir / "meta.json", meta)
    log(f"[pc_embed] wrote {sorted(p.name for p in emb_dir.iterdir())} to {emb_dir}")

    if args.skip_upload:
        _write_json_atomic(out_root / "embed_done.local.json", meta)
        log("[phase=pc_embed] --skip-upload: local sentinel embed_done.local.json written")
    else:
        dest_prefix = f"{hf_prefix}/analysis_tensors/embeddings_qwen3_8b"
        res = upload_dir_sharded(
            emb_dir,
            HF_DATA_REPO,
            dest_prefix,
            shard_glob="*",
            resume_skip=False,
            delete_local=False,
        )
        sentinel = {
            **meta,
            "hf_repo": HF_DATA_REPO,
            "hf_dest_prefix": dest_prefix,
            "uploaded": res.uploaded,
            "skipped_existing": res.skipped_existing,
            "rerouted": res.rerouted,
        }
        _write_json_atomic(out_root / "embed_uploaded.json", sentinel)
        log("[phase=pc_embed] sentinel written")

    log("[phase=done] pc_embed complete")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
