#!/usr/bin/env python3
"""Issue #1073 P3: batched teacher-forced span-mean capture (arms greedy + stoch10).

Reuses the #779 pass-2 batched-capture PATTERN (``issue779_capture_answer_summaries_pass2
.capture_pass2_batched``: right-padded batches over ``model.model`` with block
hooks, sorted by length) but emits ONE summary per rollout: v(x) = the
span-mean over the response span ``[prompt_len, full_len)`` at ALL layers —
the exact ``issue779_collect.capture_answer_vector`` v(x) definition, verified
per-run by the P0 equivalence gate (flat cos >= 0.999; span means keep the
flat bar per the #779 r12 calibration).

Notes:
- The forward runs ``model.model`` (the base transformer), so lm_head never
  executes and NO full-vocab logits are materialized — this supersedes the
  ``logits_to_keep=1`` threading for this path (the P0 reference path,
  ``capture_answer_vector`` -> ``extract_layer_activations``, carries the
  introspection-guarded ``logits_to_keep=1`` internally).
- Checkpoint per shard (fp16, SHARD_CTX=500 contexts/shard), resume-by-skip
  keyed on (arm, shard); fp32 reductions ``vbar10`` / ``v_greedy`` +
  ``stoch1_new`` written after capture; uploads via ``upload_dir_sharded``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_capture_answer_summaries as P1  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy: shared-VM thread caps bind at import (#847)

import issue1073_common as I  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue1073_capture")

CAPTURE_BATCH = int(os.environ.get("EPM_CAPTURE_BATCH", "16"))


@torch.no_grad()
def capture_span_mean_batched(
    model, tokenizer, items: list[dict], layers: list[int], batch_size: int
) -> list[dict]:
    """Per item: fp16 (L, H) span-mean over ``[prompt_len, full_len)`` at all layers.

    Same forward path as #779 pass 2 (``model.model`` + block hooks,
    right-padded, sorted by length; ``P1._right_pad_batch`` keeps real tokens
    at 0..len-1 so default position ids are correct). The mean is accumulated
    in float32 before the fp16 cast. Items carry ``full_ids`` / ``prompt_len``
    / ``full_len`` from ``P1._tokenize_item``.
    """
    blocks = model.model.layers
    pad_id = P1._pad_id_for(tokenizer)
    out: list[dict | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i]["full_len"])
    n_layers = len(layers)
    hidden = model.config.hidden_size

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(li: int):
        def _hook(_m, _i, output):
            captured[li] = output[0] if isinstance(output, tuple) else output

        return _hook

    for start in range(0, len(order), batch_size):
        sel = order[start : start + batch_size]
        batch = [items[i] for i in sel]
        ids_b, mask_b, _ = P1._right_pad_batch([b["full_ids"] for b in batch], pad_id, model.device)
        captured.clear()
        handles = [blocks[li].register_forward_hook(_make_hook(li)) for li in layers]
        try:
            model.model(input_ids=ids_b, attention_mask=mask_b)
        finally:
            for h in handles:
                h.remove()
        for bi, gi in enumerate(sel):
            it = batch[bi]
            pl, fl = it["prompt_len"], it["full_len"]
            assert fl > pl, (pl, fl)
            summ = torch.empty((n_layers, hidden), dtype=torch.float16)
            for li_pos, li in enumerate(layers):
                hs = captured[li][bi]  # (T, H), right-padded
                summ[li_pos] = hs[pl:fl].to(torch.float32).mean(dim=0).to(torch.float16).cpu()
            out[gi] = {
                "summ": summ,  # (L, H) fp16
                "prompt_len": pl,
                "span_len": fl - pl,
                "response_empty": not it["response"].strip(),
            }
        captured.clear()
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def equivalence_gate(model, tokenizer, layers: list[int]) -> dict:
    """Batched capture vs the IMPORTED parent ``capture_answer_vector`` (batch-1).

    3 real items, flat cosine over the (L, H) span mean >= 0.999 — span-mean
    quantities keep the flat bar (#779 r12 calibration; plan §11). Fail-loud.
    """
    from issue779_collect import capture_answer_vector

    msgs = [
        [{"role": "user", "content": "Hi."}],
        [{"role": "user", "content": "Explain in detail why the sky appears blue at noon."}],
        [{"role": "user", "content": "Count to three."}],
    ]
    resps = [
        "Blue light scatters more.",
        "Because Rayleigh scattering favors short wavelengths across the whole sky.",
        "One two three.",
    ]
    items = [
        P1._tokenize_item(tokenizer, {"ci": i, "ri": 0, "messages": m, "response": r})
        for i, (m, r) in enumerate(zip(msgs, resps, strict=True))
    ]
    bat = capture_span_mean_batched(model, tokenizer, items, layers, 3)
    cos_min = 1.0
    for it, b in zip(items, bat, strict=True):
        ref = capture_answer_vector(model, tokenizer, it["messages"], it["response"], layers, {})
        assert ref is not None
        a = ref["v_x"].double().flatten()
        c = b["summ"].double().flatten()
        cos_min = min(cos_min, float(torch.dot(a, c) / (a.norm() * c.norm() + 1e-12)))
    assert cos_min >= 0.999, (
        f"capture-equivalence gate FAILED: flat cos {cos_min:.6f} < 0.999 "
        "(kill criterion 1 — fix before P3)"
    )
    logger.info("[gate] batched-vs-batch-1 span-mean equivalence PASS (cos_min=%.6f)", cos_min)
    return {"cos_min": cos_min, "bar": 0.999, "n_items": len(items)}


def _shard_name(arm: str, k: int) -> str:
    return f"{arm}_v_shard{k:03d}.pt"


def build_items(prompts: list[str], gen_records: list[dict], arm: str) -> list[dict]:
    """(ci, ri, messages, response) items from persisted gen records."""
    items = []
    for rec in gen_records:
        ci, ri = int(rec["ci"]), int(rec["ri"])
        items.append(
            {
                "ci": ci,
                "ri": ri,
                "arm": arm,
                "messages": [{"role": "user", "content": prompts[ci]}],
                "response": rec["text"],
            }
        )
    return items


def run_arm(
    model,
    tokenizer,
    layers: list[int],
    arm: str,
    items: list[dict],
    n_ctx: int,
    store_dir: Path,
    batch_size: int,
    t0: float,
    total: int,
    done_holder: list[int],
) -> None:
    """Capture one arm, sharded by context (resume-by-skip keyed (arm, shard))."""
    store_dir.mkdir(parents=True, exist_ok=True)
    by_ci: dict[int, list[dict]] = {}
    for it in items:
        by_ci.setdefault(it["ci"], []).append(it)
    n_shards = (n_ctx + I.SHARD_CTX - 1) // I.SHARD_CTX
    for k in range(n_shards):
        path = store_dir / _shard_name(arm, k)
        lo, hi = k * I.SHARD_CTX, min((k + 1) * I.SHARD_CTX, n_ctx)
        shard_items = [it for ci in range(lo, hi) for it in by_ci.get(ci, [])]
        if path.exists():
            logger.info("[%s] shard %d/%d exists; skip (resume)", arm, k + 1, n_shards)
            done_holder[0] += len(shard_items)
            continue
        logger.info(
            "[%s] shard %d/%d: contexts [%d,%d) -> %d rollouts (tokenizing)",
            arm,
            k + 1,
            n_shards,
            lo,
            hi,
            len(shard_items),
        )
        tok_items = [P1._tokenize_item(tokenizer, it) for it in shard_items]
        rows = capture_span_mean_batched(model, tokenizer, tok_items, layers, batch_size)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "arm": arm,
                "layers": layers,
                "context_range": [lo, hi],
                "index": [(it["ci"], it["ri"]) for it in shard_items],
                "summ": torch.stack([r["summ"] for r in rows]),  # (n, L, H) fp16
                "prompt_lens": torch.tensor([r["prompt_len"] for r in rows], dtype=torch.long),
                "span_lens": torch.tensor([r["span_len"] for r in rows], dtype=torch.long),
                "response_empty": torch.tensor(
                    [r["response_empty"] for r in rows], dtype=torch.bool
                ),
                "metadata": I.reproducibility_metadata({"script": "issue1073_capture", "arm": arm}),
            },
            tmp,
        )
        tmp.replace(path)
        done_holder[0] += len(shard_items)
        elapsed_h = (time.time() - t0) / 3600.0
        proj_h = elapsed_h / max(done_holder[0], 1) * total
        logger.info(
            "[pace] %d/%d rollouts, %.2f h elapsed, %.2f h projected total",
            done_holder[0],
            total,
            elapsed_h,
            proj_h,
        )


def iter_shards(store_dir: Path, arm: str):
    """Yield (path, mmap-loaded shard dict) in shard order (fail-loud on gaps)."""
    paths = sorted(store_dir.glob(f"{arm}_v_shard*.pt"))
    assert paths, f"no {arm} shards under {store_dir}"
    for p in paths:
        yield p, torch.load(p, mmap=True, weights_only=False, map_location="cpu")


def write_reductions(store_dir: Path, red_dir: Path, n_ctx: int, layers: list[int], hidden: int):
    """fp32 reductions: vbar10 (mean over the 10 stoch rollouts), v_greedy,
    stoch1_new (ri=0 of the fresh ten). Streamed shard-by-shard (bounded RSS)."""
    red_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(layers)
    acc = torch.zeros((n_ctx, n_layers, hidden), dtype=torch.float32)
    cnt = torch.zeros(n_ctx, dtype=torch.float32)
    s1new = torch.zeros((n_ctx, n_layers, hidden), dtype=torch.float32)
    empty_any = torch.zeros(n_ctx, dtype=torch.bool)
    for _p, shard in iter_shards(store_dir, "stoch10"):
        summ = shard["summ"].to(torch.float32)
        for row, (ci, ri) in enumerate(shard["index"]):
            acc[ci] += summ[row]
            cnt[ci] += 1.0
            if ri == 0:
                s1new[ci] = summ[row]
            if bool(shard["response_empty"][row]):
                empty_any[ci] = True
    assert torch.all(cnt > 0), "stoch10 store has contexts with zero rollouts"
    vbar10 = acc / cnt[:, None, None]

    vg = torch.zeros((n_ctx, n_layers, hidden), dtype=torch.float32)
    g_empty = torch.zeros(n_ctx, dtype=torch.bool)
    for _p, shard in iter_shards(store_dir, "greedy"):
        summ = shard["summ"].to(torch.float32)
        for row, (ci, _ri) in enumerate(shard["index"]):
            vg[ci] = summ[row]
            g_empty[ci] = bool(shard["response_empty"][row])
    meta = I.reproducibility_metadata({"script": "issue1073_capture", "artifact": "reductions"})
    for name, tensor in (("vbar10", vbar10), ("v_greedy", vg), ("stoch1_new", s1new)):
        tmp = red_dir / f"{name}.pt.tmp"
        torch.save({"tensor": tensor, "layers": layers, "metadata": meta}, tmp)
        tmp.replace(red_dir / f"{name}.pt")
    torch.save(
        {
            "stoch_rollout_counts": cnt,
            "stoch_any_empty": empty_any,
            "greedy_empty": g_empty,
            "metadata": meta,
        },
        red_dir / "coverage.pt",
    )
    logger.info(
        "[reductions] vbar10/v_greedy/stoch1_new written (N=%d; stoch-empty ctx=%d, "
        "greedy-empty ctx=%d)",
        n_ctx,
        int(empty_any.sum()),
        int(g_empty.sum()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #1073 P3 batched span-mean capture.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--batch-size", type=int, default=CAPTURE_BATCH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["greedy", "stoch10"])
    args = parser.parse_args()

    I.phase("p3")
    root = I.out_root(args.smoke, args.out_root)
    in_dir = I.inputs_dir(root)
    bundle_path = in_dir / I.BUNDLE_PATH_IN_REPO
    assert bundle_path.exists(), f"bundle missing at {bundle_path} — run P0 first"

    model, tokenizer = I.load_model_and_tokenizer(args.model, smoke=args.smoke)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    layers = list(range(n_layers))

    bundle = I.load_bundle(
        bundle_path,
        expected_layers=n_layers,
        expected_hidden=hidden,
        min_n=2 if args.smoke else 4900,
    )
    prompts = bundle["prompts"]
    n_ctx = len(prompts)

    gen_dir = root / "raw_completions"
    store_dir = root / "v_store"
    arm_items = {}
    for arm in args.arms:
        records = I.read_text_shards(gen_dir / arm, arm)
        arm_items[arm] = build_items(prompts, records, arm)
        expected = n_ctx * (I.N_ROLLOUTS if arm == "stoch10" else 1)
        assert len(arm_items[arm]) == expected, (arm, len(arm_items[arm]), expected)

    total = sum(len(v) for v in arm_items.values())
    t0 = time.time()
    done_holder = [0]
    for arm, items in arm_items.items():
        run_arm(
            model,
            tokenizer,
            layers,
            arm,
            items,
            n_ctx,
            store_dir,
            args.batch_size,
            t0,
            total,
            done_holder,
        )

    if {"greedy", "stoch10"} <= set(args.arms):
        write_reductions(store_dir, root / "reductions", n_ctx, layers, hidden)

    if not args.no_upload:
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        upload_dir_sharded(
            store_dir,
            I.HF_DATA_REPO,
            f"{I.HF_PREFIX}/analysis_tensors/v_store",
            shard_glob="*_v_shard*.pt",
            delete_local=False,  # P4 consumes the local store
        )
        upload_dir_sharded(
            root / "reductions",
            I.HF_DATA_REPO,
            f"{I.HF_PREFIX}/analysis_tensors/reductions",
            shard_glob="*.pt",
            delete_local=False,
        )

    elapsed_h = (time.time() - t0) / 3600.0
    summary = {
        "arms": args.arms,
        "n_contexts": n_ctx,
        "n_rollouts": total,
        "gpu_hours_wall": round(elapsed_h, 3),
        "smoke": args.smoke,
    }
    I.write_json_atomic(root / "capture_summary.json", summary)
    logger.info("P3 DONE: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
