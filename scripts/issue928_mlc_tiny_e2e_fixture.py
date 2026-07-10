#!/usr/bin/env python3
"""Tiny-real CPU e2e fixture for the matched-length control driver (#906 recipe).

Builds a SYNTHETIC parent store + rollouts whose per-context blobs are captured
with the SAME production functions (``build_capture_row`` +
``reduce_forward_batch``, default 12-name path) on a REAL tiny same-family
model (Qwen2.5-0.5B-Instruct, fp32 CPU) over the REAL battery contexts + probe
pool — so ``issue928_matched_length_control.py`` can then run its ENTIRE
``main()`` (stage-skip → asserts → spans → capture → parity → fit → nulls →
bootstrap → figures → sentinel) against it with REAL library types at every
internal seam, faking ONLY model scale and the remote Hub. The rollout digests
and ``probe_indices`` are coherent by construction (computed with the driver's
own functions); only the first ``--kept`` completions per context are
well-formed, so the capture stays CPU-sized.

Usage::

    uv run python scripts/issue928_mlc_tiny_e2e_fixture.py \\
        --root /tmp/issue-928-mlc-smoke/tiny --contexts 2 --kept 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import torch  # noqa: E402
from issue594_common import probes_hash  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue928_common import (  # noqa: E402
    SUMMARY_NAMES,
    context_order_and_families,
    dump_json,
    load_probe_pool,
    resolve_battery,
)
from issue928_extract_thinking_store import (  # noqa: E402
    build_capture_row,
    pack_batches,
    parse_rows,
    reduce_forward_batch,
    rollout_content_digest,
)

TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 512


def _completion(ci: int, qi: int, well_formed: bool) -> str:
    """Deterministic, non-degenerate synthetic <think> completion (<50 words,
    so the repeated-4-gram screen is exempt by construction)."""
    if not well_formed:
        return f"no think tags here c{ci} q{qi}"  # segment reason: no_close
    cot = " ".join(f"c{ci}q{qi}r{i} step" for i in range(14))
    ans = " ".join(f"c{ci}q{qi}a{i} word" for i in range(16))
    return f"<think>{cot}</think>{ans}"


def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-real MLC e2e fixture builder")
    ap.add_argument("--root", required=True)
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--kept", type=int, default=3, help="well-formed rows per context")
    ap.add_argument("--model", default=TINY_MODEL)
    args = ap.parse_args()

    root = Path(args.root)
    rollouts_dir = root / "rollouts"
    store_dir = root / "parent_store"
    (store_dir / "percq_summaries").mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    battery = resolve_battery(None)
    ctx_ids_all, families = context_order_and_families(battery)
    ctx_ids = ctx_ids_all[: args.contexts]
    instances = {i["id"]: i for i in battery["instances"]}
    probes = load_probe_pool()
    pool_hash = probes_hash(probes)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    capture_layers = list(range(n_layers))
    capture = LayerCapture(model, n_layers)

    try:
        for ci, c in enumerate(ctx_ids):
            completions = [
                (_completion(ci, qi, qi < args.kept), "stop") for qi in range(len(probes))
            ]
            dump_json(
                {
                    "context_id": c,
                    "family": families[c],
                    "rung": "greedy",
                    "model": args.model,
                    "max_new_tokens": MAX_NEW,
                    "probe_pool_hash": pool_hash,
                    "completions": [
                        {"probe": q, "completion": t, "finish_reason": fr}
                        for q, (t, fr) in zip(probes, completions, strict=True)
                    ],
                },
                rollouts_dir / f"{c}.json",
            )
            parse = parse_rows(tokenizer, completions, "greedy")
            kept = [qi for qi, r in enumerate(parse) if r["well_formed"]]
            assert kept == list(range(args.kept)), kept
            rows = []
            for qi in kept:
                row, why = build_capture_row(
                    tokenizer, instances[c], probes[qi], completions[qi][0], parse[qi], "greedy"
                )
                assert row is not None, why
                rows.append(row)
            chunks, order = [], []
            for batch_idx in pack_batches(rows, 8, 32768):
                chunks.append(
                    reduce_forward_batch(
                        model, capture, capture_layers, tokenizer, [rows[i] for i in batch_idx]
                    )
                )
                order.extend(batch_idx)
            stacked = torch.cat(chunks, dim=0)
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            per_q = stacked[inv]
            torch.save(
                {
                    "context_id": c,
                    "family": families[c],
                    "rung": "greedy",
                    "capture_layers": capture_layers,
                    "summary_names": list(SUMMARY_NAMES),
                    "probe_indices": kept,
                    "per_q": per_q,
                    "probe_avg": per_q.float().mean(dim=0).to(torch.float16),
                    "coverage": {
                        "n_probes_total": len(probes),
                        "n_well_formed": len(kept),
                        "n_captured": len(kept),
                        "capture_drop_reasons": {},
                    },
                    "probe_pool_hash": pool_hash,
                    "model": args.model,
                    "max_new_tokens": MAX_NEW,
                    "rollout_digest": rollout_content_digest(probes, completions),
                },
                store_dir / "percq_summaries" / f"{c}.pt",
            )
            print(f"[fixture] {c}: {len(kept)} rows captured ({per_q.shape})")
    finally:
        capture.remove()

    dump_json(
        {
            "context_ids": ctx_ids,
            "families": {c: families[c] for c in ctx_ids},
            "capture_layers": capture_layers,
            "summary_names": list(SUMMARY_NAMES),
            "hidden_size": int(model.config.hidden_size),
            "rung": "greedy",
            "probe_pool_hash": pool_hash,
            "n_probes": len(probes),
            "model": args.model,
            "max_new_tokens": MAX_NEW,
        },
        store_dir / "manifest.json",
    )
    print(f"[fixture] wrote parent store + rollouts under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
