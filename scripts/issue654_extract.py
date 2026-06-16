#!/usr/bin/env python3
"""Issue #654 Step 2: dual-position residual-stream extraction (GPU, on-pod).

Plan §3 step 2. Loads Qwen-2.5-7B-Instruct ONCE and, per (context, query) pair
in the battery, runs ONE forward pass over the full ChatML prompt reading the
residual at the context-span-end AND query-span-end token positions at every
layer 0..27. Also runs a SECOND forward over each DISTINCT context's
context-only prompt (shared across queries that share that context) to read the
assistant-generation slot — the companion same-position contrast (§5).

Outputs per-pair ``data/issue654/dual_pos/pair_NNNNNN.pt`` (atomic write) +
``data/issue654/dual_pos/extraction_manifest.json`` with per-pair offsets,
decoded sanity, the >5% offset-failure kill check, model config, and the
companion context-only file paths.

``--smoke`` runs the IDENTICAL code path on a one-cell subset: the FIRST 4
distinct contexts x the first 2 queries (no separate code path — same dual-
position extractor, same offset asserts, same output writer, same manifest
shape; the smoke IS the sweep with one cell — PASS_UNIFIED).

Usage::

    uv run python scripts/issue654_extract.py --battery data/issue654/battery.json \
        --out-dir data/issue654/dual_pos --device cuda
    uv run python scripts/issue654_extract.py --battery data/issue654/battery_smoke.json \
        --out-dir data/issue654/dual_pos_smoke --device cuda --smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.analysis.probes import extract_dual_position_activations  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue654_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_N_LAYERS = 28  # A1
EXPECTED_HIDDEN = 3584  # A2
# Kill criterion (plan §7): halt if position asserts fail on > 5% of pairs.
OFFSET_FAIL_KILL_FRACTION = 0.05
# Smoke subset (one cell): first 4 distinct contexts x first 2 queries.
SMOKE_N_CONTEXTS = 4
SMOKE_N_QUERIES = 2


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _atomic_save(obj, path: Path) -> None:
    """Write a .pt atomically (finishes-or-not, never partial)."""
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _select_smoke_pairs(pairs: list[dict]) -> list[dict]:
    """One-cell smoke subset: first SMOKE_N_CONTEXTS distinct contexts x first
    SMOKE_N_QUERIES queries — IDENTICAL extraction code path, just fewer pairs."""
    contexts_seen: list[str] = []
    queries_seen: list[str] = []
    for p in pairs:
        if p["context_id"] not in contexts_seen:
            contexts_seen.append(p["context_id"])
        if p["query_id"] not in queries_seen:
            queries_seen.append(p["query_id"])
    keep_ctx = set(contexts_seen[:SMOKE_N_CONTEXTS])
    keep_q = set(queries_seen[:SMOKE_N_QUERIES])
    subset = [p for p in pairs if p["context_id"] in keep_ctx and p["query_id"] in keep_q]
    logger.info(
        "smoke subset: %d contexts x %d queries = %d pairs",
        len(keep_ctx),
        len(keep_q),
        len(subset),
    )
    return subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #654: dual-position extraction.")
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one-cell subset (4 contexts x 2 queries), identical code path",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="model dtype (float32 for a CPU structure smoke)",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(args.battery.read_text())
    pairs = payload["pairs"]
    if args.smoke:
        pairs = _select_smoke_pairs(pairs)
    if not pairs:
        raise RuntimeError(f"no pairs in {args.battery}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx_only_dir = args.out_dir / "context_only"
    ctx_only_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    logger.info("loading %s (%s) on %s", args.model, args.dtype, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": args.device},
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    # A1 / A2 asserts (fail loud at startup).
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    assert n_layers == EXPECTED_N_LAYERS, (
        f"num_hidden_layers={n_layers} != {EXPECTED_N_LAYERS} (A1)"
    )
    assert hidden == EXPECTED_HIDDEN, f"hidden_size={hidden} != {EXPECTED_HIDDEN} (A2)"
    layers = list(range(n_layers))

    # ── Companion same-position reads (per DISTINCT context, shared across queries) ──
    # The context-only prompt is identical for all queries that share a context;
    # read it once at the assistant-generation slot (last token, position -1).
    distinct_contexts: dict[str, dict] = {}
    for p in pairs:
        if p["context_id"] not in distinct_contexts:
            distinct_contexts[p["context_id"]] = {
                "context_id": p["context_id"],
                "context_type": p["context_type"],
                "context_only_prompt": p["context_only_prompt"],
            }
    logger.info("[phase=companion] %d distinct context-only reads", len(distinct_contexts))
    companion_paths: dict[str, str] = {}
    ctx_ids_list = list(distinct_contexts.values())
    ctx_prompts = [c["context_only_prompt"] for c in ctx_ids_list]
    # The companion read uses the SAME extractor with a single readout position
    # (-1 = last token = the assistant-generation slot). It needs >=2 positions
    # per the dual-position signature; we pass (0, last) as the two required
    # span positions and read the companion at readout_position=-1, then keep
    # only the readout bank (the (0, last) span reads are discarded for the
    # context-only prompt — they are not the construct).
    for c, prompt in zip(ctx_ids_list, ctx_prompts, strict=True):
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        last = ids.shape[1] - 1
        # (context_end=0, query_end=last) is a valid 0 <= c < q < seq ordering for
        # the helper's assert; readout_position=-1 reads the assistant-gen slot.
        banks = extract_dual_position_activations(
            model,
            tokenizer,
            [prompt],
            [(0, last)],
            layers=layers,
            device=args.device,
            readout_position=-1,
        )
        readout = banks["readout"][0]  # (n_layers, hidden)
        cpath = ctx_only_dir / f"{c['context_id']}.pt"
        _atomic_save(
            {
                "context_id": c["context_id"],
                "context_type": c["context_type"],
                "readout": readout,  # (n_layers, hidden) at the assistant-gen slot
                "layers": layers,
            },
            cpath,
        )
        companion_paths[c["context_id"]] = str(cpath.relative_to(args.out_dir))

    # ── Per-pair dual-position extraction ────────────────────────────────────
    logger.info("[phase=extract] %d (context, query) pairs", len(pairs))
    manifest_pairs: list[dict] = []
    offset_failures = 0
    for i, p in enumerate(pairs):
        full = p["full_prompt"]
        c_idx = p["ctx_end_idx"]
        q_idx = p["query_end_idx"]
        # Re-confirm A4 ordering at extraction time (the build already asserted it;
        # a tokenizer drift between build host and pod would surface here).
        ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids
        seq_len = ids.shape[1]
        if not (0 <= c_idx < q_idx < seq_len):
            offset_failures += 1
            logger.error(
                "OFFSET FAIL pair %s: 0 <= %d < %d < %d violated",
                p["pair_id"],
                c_idx,
                q_idx,
                seq_len,
            )
            continue
        banks = extract_dual_position_activations(
            model,
            tokenizer,
            [full],
            [(c_idx, q_idx)],
            layers=layers,
            device=args.device,
        )
        out_path = args.out_dir / f"pair_{i:06d}.pt"
        _atomic_save(
            {
                "pair_id": p["pair_id"],
                "context_type": p["context_type"],
                "context_id": p["context_id"],
                "query_id": p["query_id"],
                "topicality": p["topicality"],
                "length": p["length"],
                "ctx_end_idx": c_idx,
                "query_end_idx": q_idx,
                "context_end": banks["context_end"][0],  # (n_layers, hidden)
                "query_end": banks["query_end"][0],  # (n_layers, hidden)
                "layers": layers,
                "companion_context_only_file": companion_paths[p["context_id"]],
            },
            out_path,
        )
        manifest_pairs.append(
            {
                "pair_id": p["pair_id"],
                "pt_file": out_path.name,
                "context_type": p["context_type"],
                "context_id": p["context_id"],
                "query_id": p["query_id"],
                "topicality": p["topicality"],
                "length": p["length"],
                "ctx_end_idx": c_idx,
                "query_end_idx": q_idx,
                "seq_len": seq_len,
                "decoded_ctx_end_tok": tokenizer.decode([ids[0, c_idx].item()]),
                "decoded_query_end_tok": tokenizer.decode([ids[0, q_idx].item()]),
                "companion_context_only_file": companion_paths[p["context_id"]],
            }
        )
        if (i + 1) % 50 == 0:
            logger.info("  extracted %d/%d pairs", i + 1, len(pairs))

    # ── Kill check (plan §7): > 5% offset failures → fail loud ───────────────
    fail_fraction = offset_failures / max(len(pairs), 1)
    offset_kill_tripped = fail_fraction > OFFSET_FAIL_KILL_FRACTION
    logger.info(
        "offset failures: %d / %d (%.3f); kill_threshold=%.2f tripped=%s",
        offset_failures,
        len(pairs),
        fail_fraction,
        OFFSET_FAIL_KILL_FRACTION,
        offset_kill_tripped,
    )

    manifest = {
        "issue": 654,
        "model": args.model,
        "dtype": args.dtype,
        "smoke": args.smoke,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "layers": layers,
        "n_pairs_requested": len(pairs),
        "n_pairs_extracted": len(manifest_pairs),
        "n_distinct_contexts": len(distinct_contexts),
        "offset_failures": offset_failures,
        "offset_fail_fraction": fail_fraction,
        "offset_fail_kill_fraction": OFFSET_FAIL_KILL_FRACTION,
        "offset_kill_tripped": offset_kill_tripped,
        "companion_context_only_files": companion_paths,
        "pairs": manifest_pairs,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }
    manifest_path = args.out_dir / "extraction_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("wrote manifest %s", manifest_path)

    if offset_kill_tripped:
        # Fail loud per the §7 kill criterion — the span-boundary indexing broke.
        raise RuntimeError(
            f"OFFSET KILL: {offset_failures}/{len(pairs)} pairs ({fail_fraction:.3f}) failed the "
            f"position assert (> {OFFSET_FAIL_KILL_FRACTION:.2f}) — span-boundary indexing broke; "
            f"halt + fix before any read (plan §7)."
        )
    logger.info("[phase=done] extraction complete: %d .pt banks written", len(manifest_pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
