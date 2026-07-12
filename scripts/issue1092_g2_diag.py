#!/usr/bin/env python3
"""Issue #1092 round-8.6 — on-pod G2 identity-residual decomposition.

Launch 6 failed the G2 identity check at max_abs=3.0 with the round-8.4
construction (CPU fp32 exact, reviewer-verified). This diagnostic separates
CONSTRUCTION error from bf16 batch-geometry NUMERICS on the real tensors:

  A. HARD token-id equality: the capture's prompt segment ids == the
     reference's tokenizer(prompt) ids for every spot row (any mismatch is a
     construction bug -> exit 2).
  B. Gate decomposition: per-row / per-layer max_abs of disk-vs-reference;
     worst (row, layer, dim) with values, magnitude, relative error.
  C. Pure bf16 batch-geometry null on the SAME side: identical capture rows
     forwarded batch=1 vs the production padded batch — the numerics floor
     the gate criterion must sit above. Also recompute-vs-disk (adds the
     batch-COMPOSITION difference of the production shard batching).
  D. fp32 spot check on the worst rows: fp32 teacher-forced forward vs fp32
     prompt-only forward at context_end — tight agreement proves the
     construction clean and attributes the whole residual to bf16.
  E. B0 recompute-vs-disk quick measurement (the gate's other allclose).

Usage (pod-1092, one idle H100):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1092_g2_diag.py \
        --out /workspace/issue1092 --corpus-dir /workspace/issue1092/corpus \
        --rb-rev 037fcbb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps + .env must bind BEFORE the heavy imports below — the
# BLAS/torch pools freeze at import time (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue1092_gpu_phase as gp  # noqa: E402


def _stats(diff: np.ndarray, scale: np.ndarray, floor: float = 1.0) -> dict:
    rel = diff / np.maximum(scale, floor)
    flat = int(np.argmax(diff))
    idx = np.unravel_index(flat, diff.shape)
    return {
        "max_abs": float(diff.max()),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "max_rel_floored": float(rel.max()),
        "p99_rel_floored": float(np.quantile(rel, 0.99)),
        "argmax_idx": [int(i) for i in idx],
        "scale_at_argmax": float(scale[idx]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/workspace/issue1092"))
    p.add_argument("--corpus-dir", type=Path, default=Path("/workspace/issue1092/corpus"))
    p.add_argument("--cell", default="cell_inst_own")
    p.add_argument("--rb-rev", default="037fcbb")
    p.add_argument("--n-layers", type=int, default=gp.N_LAYERS)
    p.add_argument("--hidden-dim", type=int, default=gp.HIDDEN_DIM)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir, cell_id = args.out, args.cell
    n_layers, hidden_dim = args.n_layers, args.hidden_dim
    cfg = gp.CELL_CONFIG[cell_id]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rows = gp.load_manifest(args.corpus_dir)
    prefix_store = gp.load_store(args.corpus_dir, "prefix_store.jsonl")
    query_store = gp.load_store(args.corpus_dir, "query_store.jsonl")

    # mirror the gate's spot-row selection exactly
    cell_dir = out_dir / "summaries" / cell_id
    paths = gp._sorted_shards(sorted(cell_dir.glob("context_end_L00_shard*.npy"))) or sorted(
        cell_dir.glob("context_end_L00.npy")
    )
    n_rows = sum(int(np.load(pp, mmap_mode="r").shape[0]) for pp in paths)
    rng = np.random.default_rng(gp.G2_SPOT_SEED)
    spot_idx = np.sort(rng.choice(n_rows, size=min(gp.G2_SPOT_ROWS, n_rows), replace=False))
    cell_rows = gp._rows_for_cell(rows, cell_id)[:n_rows]

    tokenizer = AutoTokenizer.from_pretrained(
        gp.INSTRUCT_MODEL, revision=gp.INSTRUCT_REVISION, trust_remote_code=True
    )
    gp._get_tokenizer._tok = tokenizer
    completions_map = gp._load_raw_completion_files(out_dir, cfg["model"], cell_id)
    boundary = gp._boundary_suffix(cfg["prompt_format"])

    prompts: list[str] = []
    prefixes: list[str] = []
    comps: list[str] = []
    for row_i in spot_idx:
        row = cell_rows[int(row_i)]
        prefix_text, prompt, _ = gp.render_row(
            row,
            prefix_store,
            query_store,
            prompt_format=cfg["prompt_format"],
            text_source=cfg["text_source"],
        )
        prompts.append(prompt)
        prefixes.append(prefix_text)
        comps.append(completions_map[str(row["row_id"])])

    # ── A. hard token-id equality (construction check) ───────────────────────
    for i, (pfx, pr, co) in enumerate(zip(prefixes, prompts, comps, strict=True)):
        ids_ref = tokenizer(pr, add_special_tokens=False)["input_ids"]
        row_ids, pos = gp._capture_row_ids_and_positions(tokenizer, pfx, pr, co, boundary)
        if row_ids[: len(ids_ref)] != list(ids_ref) or pos["context_end"] != len(ids_ref) - 1:
            print(
                json.dumps(
                    {
                        "verdict": "CONSTRUCTION_BUG",
                        "spot_i": i,
                        "n_ids_ref": len(ids_ref),
                        "context_end": pos["context_end"],
                        "first_divergence": next(
                            (
                                j
                                for j, (a, b) in enumerate(zip(row_ids, ids_ref, strict=False))
                                if a != b
                            ),
                            None,
                        ),
                    }
                )
            )
            return 2
    print(f"[A] token-id equality: PASS for all {len(prompts)} spot rows")

    # ── B. gate decomposition (bf16, disk vs generate reference) ────────────
    model = AutoModelForCausalLM.from_pretrained(
        gp.INSTRUCT_MODEL,
        revision=gp.INSTRUCT_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()

    ref = gp._generate_context_hidden_reference(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
    )
    disk = gp._load_cell_summary_rows(out_dir, cell_id, "context_end", spot_idx, n_layers=n_layers)
    diff = np.abs(disk - ref)
    scale = np.maximum(np.abs(disk), np.abs(ref))
    print("[B] disk-vs-reference:", json.dumps(_stats(diff, scale)))
    per_layer = diff.reshape(diff.shape[0], n_layers, -1).max(axis=(0, 2))
    print("[B] per-layer max_abs:", json.dumps([round(float(v), 4) for v in per_layer]))
    per_row = diff.max(axis=(1, 2))
    print(
        f"[B] per-row max_abs: max={per_row.max():.4f} median={np.median(per_row):.4f} "
        f"min={per_row.min():.4f} (n={int((per_row > 0.05).sum())} rows over old-tol shape)"
    )
    r, ly, d = np.unravel_index(int(np.argmax(diff)), diff.shape)
    print(
        json.dumps(
            {
                "worst": {
                    "row": int(r),
                    "layer": int(ly),
                    "dim": int(d),
                    "disk": float(disk[r, ly, d]),
                    "ref": float(ref[r, ly, d]),
                    "abs": float(diff[r, ly, d]),
                    "rel": float(diff[r, ly, d] / max(abs(disk[r, ly, d]), abs(ref[r, ly, d]))),
                }
            }
        )
    )

    # ── C. pure bf16 batch-geometry null (same side twice) ──────────────────
    def _capture_ctx(batch_size: int) -> np.ndarray:
        out = gp._capture_batch_loaded_model(
            prefix_texts=prefixes,
            prompts=prompts,
            completions=comps,
            prompt_format=cfg["prompt_format"],
            model=model,
            tokenizer=tokenizer,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            device=device,
            log_label=f"diag-b{batch_size}",
            batch_size=batch_size,
        )
        return np.stack([s["context_end"] for s in out.summaries]).astype(np.float32)

    ctx_b1 = _capture_ctx(1)
    ctx_b8 = _capture_ctx(gp.CAPTURE_BATCH_SIZE)
    null_diff = np.abs(ctx_b1 - ctx_b8)
    null_scale = np.maximum(np.abs(ctx_b1), np.abs(ctx_b8))
    print(
        "[C] batch-geometry null (b1 vs b8, same ids):", json.dumps(_stats(null_diff, null_scale))
    )
    print("[C] recompute-b8 vs disk:", json.dumps(_stats(np.abs(ctx_b8 - disk), scale)))
    print("[C] capture-b1 vs generate-reference:", json.dumps(_stats(np.abs(ctx_b1 - ref), scale)))

    # ── E. B0 recompute-vs-disk (5 rows, gate shape) ─────────────────────────
    rb = gp.load_rb_directions(args.rb_rev, n_layers, gp.N_TRAITS, hidden_dim)
    b0_idx = spot_idx[:5]
    b0_out = gp._capture_batch_loaded_model(
        prefix_texts=prefixes[:5],
        prompts=prompts[:5],
        completions=comps[:5],
        prompt_format=cfg["prompt_format"],
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
        log_label="diag-b0",
        rb_directions=rb,
        batch_size=5,
    )
    disk_b0 = gp._load_b0_pool_matrix(out_dir, cell_id)[b0_idx].astype(np.float32)
    b0_diff = np.abs(disk_b0 - b0_out.rb_pool)
    b0_scale = np.maximum(np.abs(disk_b0), np.abs(b0_out.rb_pool))
    b0_allclose = bool(np.allclose(disk_b0, b0_out.rb_pool, atol=5e-2, rtol=5e-2))
    print(
        "[E] B0 recompute-vs-disk:",
        json.dumps({**_stats(b0_diff, b0_scale, floor=0.05), "allclose_5e-2": b0_allclose}),
    )

    # ── D. fp32 spot check on the worst row ──────────────────────────────────
    # (bf16 15GB + fp32 28GB coexist on the 80GB H100 — no teardown needed)
    model32 = AutoModelForCausalLM.from_pretrained(
        gp.INSTRUCT_MODEL,
        revision=gp.INSTRUCT_REVISION,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map={"": device},
    )
    model32.eval()
    for spot_i in {int(r), 0}:
        pr, pfx, co = prompts[spot_i], prefixes[spot_i], comps[spot_i]
        ids_prompt = tokenizer(pr, add_special_tokens=False)["input_ids"]
        row_ids, pos = gp._capture_row_ids_and_positions(tokenizer, pfx, pr, co, boundary)
        with torch.no_grad():
            full = model32(
                input_ids=torch.tensor([row_ids], device=device),
                attention_mask=torch.ones(1, len(row_ids), dtype=torch.long, device=device),
                output_hidden_states=True,
            )
            prompt_only = model32(
                input_ids=torch.tensor([ids_prompt], device=device),
                attention_mask=torch.ones(1, len(ids_prompt), dtype=torch.long, device=device),
                output_hidden_states=True,
            )
        a = np.stack(
            [h[0, pos["context_end"], :].float().cpu().numpy() for h in full.hidden_states[1:]]
        )
        b = np.stack([h[0, -1, :].float().cpu().numpy() for h in prompt_only.hidden_states[1:]])
        d32 = np.abs(a - b)
        s32 = np.maximum(np.abs(a), np.abs(b))
        print(f"[D] fp32 spot row {spot_i}:", json.dumps(_stats(d32, s32)))

    print("[diag] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
