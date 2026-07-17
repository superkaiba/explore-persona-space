"""Issue #1310: teacher-forced 28-layer span-summary capture, per model.

One ITEM per SCENE (a generated `<LABEL>:` script): the input is prefix_ids +
story_ids, where prefix is the few-shot prime (base) or the rendered chat
template (instruct); token ids are CONCATENATED (never re-tokenize the join —
#1092 BPE-seam trap) and each turn's story-local C/T spans are shifted by
len(prefix_ids). A scene carries MANY per-turn pairs, so ONE forward captures
every turn's summaries. Per pair (= per target turn) it stores ONLY the reduced
summaries — x_spanmean (mean over C = context before that turn's line),
x_last (the C boundary token — parent-matched single-position X), y (mean over
that turn's dialogue content) — never per-token grids (#666/#772 stream-reduce).
Each record also carries turn_index (the persona's Nth spoken line — the matched
scene position the swap control pairs on). bf16 storage, finiteness asserted.
Right-padded batches + attention mask (causal mask => pads cannot influence real
positions); a batched-vs-batch-1 two-bar equivalence gate (#779) via
--equivalence-check. Shards flush at BATCH boundaries (whole scenes, never
split), so --resume skips whole scenes cleanly.

CLI:
  uv run python scripts/issue1310_extract_store.py --model base \
      [--data-dir data/issue_1310] [--store-dir data/issue_1310/store]
      [--batch-size 8] [--tiny-model-dir <dir>] [--resume] [--equivalence-check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps must bind before torch import

import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue931_extract_store as ext931  # noqa: E402
import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1310_extract_store.py"
SHARD_PAIRS = 512

EXPECTED_LAYERS = c1310.EXPECTED_LAYERS
EXPECTED_HIDDEN = c1310.EXPECTED_HIDDEN


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=c1310.MODEL_KINDS, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1310"))
    ap.add_argument(
        "--store-dir", type=Path, default=None, help="default <data-dir>/<store-subdir>"
    )
    ap.add_argument(
        "--flavor",
        choices=("perturn", "onpolicy", "tf"),
        default="perturn",
        help=(
            "onpolicy = prefill records (v_C at end-of-prefix ending in the label, "
            "v_A over the generated turn; base n>0 by construction). tf = matched "
            "teacher-forced cross-check on --tf-source-model scenes (no prefix, both "
            "models on the SAME body). perturn = the run-2 parsed-scene path."
        ),
    )
    ap.add_argument(
        "--store-subdir",
        type=str,
        default="store",
        help="store lives at <data-dir>/<store-subdir>/<model> (e.g. store_onpolicy, store_tf)",
    )
    ap.add_argument(
        "--tf-source-model",
        choices=c1310.MODEL_KINDS,
        default="instruct",
        help="tf flavor: which model's stories+pairs to capture BOTH models on (matched body)",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tiny-model-dir", type=str, default=None, help="CPU smoke model dir")
    ap.add_argument("--resume", action="store_true", help="skip rows already in shards")
    ap.add_argument("--equivalence-check", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all (smoke slicing)")
    ap.add_argument(
        "--make-tiny-model",
        type=str,
        default=None,
        help="SMOKE: write a tiny random-init Qwen2 (real tokenizer) to this dir and exit",
    )
    return ap.parse_args()


def load_model(model_id: str, tiny_model_dir: str | None):
    """bf16 GPU model (device pinned; off-GPU params fail loud) or CPU tiny fp32."""
    global EXPECTED_LAYERS, EXPECTED_HIDDEN
    from transformers import AutoModelForCausalLM

    if tiny_model_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
        model.eval()
        EXPECTED_LAYERS = int(model.config.num_hidden_layers)
        EXPECTED_HIDDEN = int(model.config.hidden_size)
        print(f"[i1310-p2] TINY model: L={EXPECTED_LAYERS} D={EXPECTED_HIDDEN}")
        return model
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, f"{len(off_gpu)} params not on CUDA (e.g. {off_gpu[:3]})"
    assert model.config.num_hidden_layers == EXPECTED_LAYERS
    assert model.config.hidden_size == EXPECTED_HIDDEN
    return model


def render_prefix_ids(tokenizer, prompt: str, model_kind: str) -> list[int]:
    """Prefix token ids: raw prompt (base) or rendered chat template (instruct)."""
    if model_kind == "instruct":
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        prefix = prompt
    return list(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def _load_items_onpolicy(model_kind: str, data_dir: Path) -> tuple[list[dict], dict]:
    """One item per PREFILL RECORD (scenario, persona, slot): input = the stored
    prompt_token_ids + completion_token_ids (concatenated ids, NEVER re-tokenized
    at the join — #1092 BPE-seam rule). c_span = the last CONTEXT_CAP_TOKENS of
    the prompt (which ends in the label cue; x_last = the boundary token, v_C);
    t_span = the generated turn (v_A). Degenerate (empty/short) turns drop."""
    recs = ext931._read_jsonl(
        data_dir / "prefill" / f"{model_kind}_prefill_seed{common.GEN_SEED}.jsonl"
    )
    items: list[dict] = []
    counters = {
        "records": 0,
        "kept": 0,
        "dropped_short_dialogue": 0,
        "dropped_short_context": 0,
    }
    for r in recs:
        counters["records"] += 1
        prompt_ids = list(r["prompt_token_ids"])
        comp_ids = list(r["completion_token_ids"])
        n_prompt, n_comp = len(prompt_ids), len(comp_ids)
        if n_comp < c1310.DIALOGUE_MIN_TOKENS:
            counters["dropped_short_dialogue"] += 1
            continue
        c_lo = max(0, n_prompt - c1310.CONTEXT_CAP_TOKENS)
        if n_prompt - c_lo < c1310.CONTEXT_MIN_TOKENS:
            counters["dropped_short_context"] += 1
            continue
        input_ids = prompt_ids + comp_ids
        n_tok = len(input_ids)
        pair = common.PairSpec(
            row_id=r["row_id"],
            group_id=r["scenario_id"],
            char_id=r["persona"],
            c_span=(c_lo, n_prompt),
            t_spans=[(n_prompt, n_prompt + n_comp)],
            ctx_span=(c_lo, n_prompt),
            meta={
                "turn_index": int(r["slot"]),
                "scene_row_id": r["scene_row_id"],
                "prefix_len": n_prompt,
            },
        )
        pair.validate(n_tok, min_c=c1310.CONTEXT_MIN_TOKENS, min_t=c1310.DIALOGUE_MIN_TOKENS)
        items.append(
            {
                "item_id": r["row_id"],
                "group_id": r["scenario_id"],
                "char_id": r["persona"],
                "input_ids": input_ids,
                "pairs": [pair],
            }
        )
        counters["kept"] += 1
    return items, counters


def load_items(
    model_kind: str,
    data_dir: Path,
    tokenizer,
    *,
    flavor: str = "perturn",
    source_model: str | None = None,
) -> tuple[list[dict], dict]:
    """Flavor-aware item assembly. Returns (items, capture_drop_counters).

    onpolicy -> prefill records (one item per (scenario, persona, slot)).
    perturn / tf -> parsed-scene stories+pairs grouped by scene. tf uses NO
    model-specific prefix (prefix_ids=[]) so BOTH models see the byte-identical
    body (matched teacher-forced cross-check) and reads ``source_model``'s
    stories+pairs; perturn uses each model's own prefix + own scenes.
    """
    if flavor == "onpolicy":
        return _load_items_onpolicy(model_kind, data_dir)

    src = source_model or model_kind
    use_prefix = flavor != "tf"  # tf: no prefix (matched body across models)
    stories = {
        s["row_id"]: s
        for s in ext931._read_jsonl(
            data_dir / "stories" / f"{src}_stories_seed{common.GEN_SEED}.jsonl"
        )
    }
    pairs = [
        common.PairSpec.from_dict(d)
        for d in ext931._read_jsonl(data_dir / "pairs" / f"{src}_pairs.jsonl")
    ]
    by_scene: dict[str, list[common.PairSpec]] = {}
    for p in pairs:
        by_scene.setdefault(p.meta["scene_row_id"], []).append(p)

    items = []
    for scene_row_id, scene_pairs in by_scene.items():
        story = stories[scene_row_id]
        prefix_ids = render_prefix_ids(tokenizer, story["prompt"], model_kind) if use_prefix else []
        story_ids = list(tokenizer(story["story"], add_special_tokens=False)["input_ids"])
        off = len(prefix_ids)
        n_tok = off + len(story_ids)
        shifted = []
        for p in scene_pairs:
            q = common.PairSpec(
                row_id=p.row_id,
                group_id=p.group_id,
                char_id=p.char_id,
                c_span=(p.c_span[0] + off, p.c_span[1] + off),
                t_spans=[(lo + off, hi + off) for lo, hi in p.t_spans],
                ctx_span=(p.ctx_span[0] + off, p.ctx_span[1] + off),
                meta={**p.meta, "prefix_len": off},
            )
            q.validate(n_tok, min_c=c1310.CONTEXT_MIN_TOKENS, min_t=c1310.DIALOGUE_MIN_TOKENS)
            shifted.append(q)
        items.append(
            {
                "item_id": scene_row_id,
                "group_id": scene_pairs[0].group_id,
                "char_id": scene_pairs[0].char_id,
                "input_ids": prefix_ids + story_ids,
                "pairs": shifted,
            }
        )
    return items, {"records": len(items), "kept": len(items)}


def process_batch(model, batch: list[dict], pad_id: int) -> list[dict]:
    """One right-padded batched forward; per-pair reduced summaries to CPU bf16."""
    lengths = [len(it["input_ids"]) for it in batch]
    bsz, max_len = len(batch), max(lengths)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, it in enumerate(batch):
        input_ids[i, : lengths[i]] = torch.tensor(it["input_ids"], dtype=torch.long)
        attention_mask[i, : lengths[i]] = 1
    device = next(model.parameters()).device
    captured = extract_layer_activations(
        model,
        input_ids.to(device),
        layers=range(EXPECTED_LAYERS),
        return_logits=False,
        attention_mask=attention_mask.to(device),
        detach_to_cpu=False,
    )
    assert set(captured) == set(range(EXPECTED_LAYERS)), "missing layers in capture"
    acts = torch.stack([captured[layer] for layer in range(EXPECTED_LAYERS)], dim=0)
    assert acts.shape == (EXPECTED_LAYERS, bsz, max_len, EXPECTED_HIDDEN), acts.shape

    records = []
    for i, it in enumerate(batch):
        true_len = lengths[i]
        for p in it["pairs"]:
            cs, ce = p.c_span
            assert 0 <= cs < ce <= true_len, (p.row_id, "c_span", cs, ce, true_len)
            rec: dict = {
                "row_id": p.row_id,
                "group_id": p.group_id,
                "char_id": p.char_id,
                "turn_index": int(p.meta["turn_index"]),
            }
            rec["x_spanmean"] = acts[:, i, cs:ce, :].float().mean(dim=1)
            rec["x_last"] = acts[:, i, ce - 1, :].float()
            total = torch.zeros(
                EXPECTED_LAYERS, EXPECTED_HIDDEN, dtype=torch.float32, device=acts.device
            )
            n_t = 0
            for lo, hi in p.t_spans:
                assert 0 <= lo < hi <= true_len, (p.row_id, "t_span", lo, hi, true_len)
                total += acts[:, i, lo:hi, :].float().sum(dim=1)
                n_t += hi - lo
            rec["y"] = total / n_t
            for k in list(rec.keys()):
                if isinstance(rec[k], torch.Tensor):
                    rec[k] = ext931._finite(
                        rec[k].to(device="cpu", dtype=torch.bfloat16), k, p.row_id
                    )
            records.append(rec)
    del captured, acts
    return records


def run_extraction(model, items: list[dict], pad_id: int, batch_size: int):
    """Length-grouped batching with OOM-halving (floor 1); yields per-batch recs."""
    order = sorted(range(len(items)), key=lambda j: len(items[j]["input_ids"]))
    bs, pos, done = batch_size, 0, 0
    while pos < len(order):
        chunk = [items[order[j]] for j in range(pos, min(pos + bs, len(order)))]
        try:
            recs = process_batch(model, chunk, pad_id)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"[oom] CUDA OOM — halving batch size to {bs}")
            continue
        pos += len(chunk)
        done += 1
        if done % 10 == 0 or pos >= len(order):
            print(f"[i1310-p2] {pos}/{len(order)} items done (batch size {bs})", flush=True)
        yield recs


def write_shard(records: list[dict], out_dir: Path, shard_idx: int, model_kind: str) -> None:
    """One .pt shard (stacked arrays) + JSON sidecar; atomic-ish via tmp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in records[0] if isinstance(records[0][k], torch.Tensor)]
    payload = {
        "row_ids": [r["row_id"] for r in records],
        "group_ids": [r["group_id"] for r in records],
        "char_ids": [r["char_id"] for r in records],
        "turn_indices": [int(r["turn_index"]) for r in records],
        "arrays": {k: torch.stack([r[k] for r in records]) for k in keys},
    }
    for k, v in payload["arrays"].items():
        assert v.shape == (len(records), EXPECTED_LAYERS, EXPECTED_HIDDEN), (k, v.shape)
    pt_path = out_dir / f"{model_kind}_shard{shard_idx:03d}.pt"
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(pt_path)
    sidecar = {
        "model_kind": model_kind,
        "shard_index": shard_idx,
        "n_rows": len(records),
        "row_ids": payload["row_ids"],
        "group_ids": payload["group_ids"],
        "char_ids": payload["char_ids"],
        "turn_indices": payload["turn_indices"],
        "keys": keys,
        "shape_per_row": [EXPECTED_LAYERS, EXPECTED_HIDDEN],
        "metadata": common.metadata(SCRIPT, common.GEN_SEED, len(records)),
    }
    (out_dir / f"{model_kind}_shard{shard_idx:03d}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[i1310-p2] wrote {pt_path} ({len(records)} rows)")


def equivalence_check(model, items: list[dict], pad_id: int) -> dict:
    """Batched (B=3) vs batch-1 capture equivalence, two-bar #779 gate.

    Early-layer (first 4) per-layer cosine >= 0.999 AND flattened all-layer
    cosine >= 0.995 over every stored summary. Right-pad + causal mask means
    pads cannot influence real positions; bf16 deep-layer jitter is the residual.
    """
    take = items[:3]
    if len(take) < 2:
        take = items[:1] * 2
    batched = process_batch(model, take, pad_id)
    serial = []
    for it in take:
        serial.extend(process_batch(model, [it], pad_id))
    assert len(batched) == len(serial)
    early_min, flat_min = 1.0, 1.0
    n_early = min(4, EXPECTED_LAYERS)
    for rb, rs in zip(batched, serial, strict=True):
        for k in rb:
            if not isinstance(rb[k], torch.Tensor):
                continue
            a, b = rb[k].float(), rs[k].float()
            flat = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            flat_min = min(flat_min, flat)
            per_layer = torch.nn.functional.cosine_similarity(a, b, dim=1)
            early_min = min(early_min, float(per_layer[:n_early].min()))
    result = {"early_cos_min": early_min, "flat_cos_min": flat_min, "n_items": len(take)}
    assert early_min >= 0.999 and flat_min >= 0.995, f"equivalence gate FAIL: {result}"
    print(f"[i1310-p2] equivalence gate PASS: {result}")
    return result


def main() -> int:
    args = parse_args()
    if args.make_tiny_model:
        ext931.make_tiny_model(Path(args.make_tiny_model))
        return 0
    assert args.model, "--model is required unless --make-tiny-model"
    model_kind = args.model
    model_id = c1310.MODEL_IDS[model_kind]
    store_dir = (args.store_dir or (args.data_dir / args.store_subdir)) / model_kind
    src_model = args.tf_source_model if args.flavor == "tf" else None
    print(
        f"[phase=p2_extract_{model_kind}] span-summary capture "
        f"({model_id}, flavor={args.flavor}"
        + (f", tf-source={src_model}" if src_model else "")
        + ")"
    )
    tokenizer = common.get_tokenizer(model_id)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    items, drops = load_items(
        model_kind, args.data_dir, tokenizer, flavor=args.flavor, source_model=src_model
    )
    store_dir.mkdir(parents=True, exist_ok=True)
    c1310.write_json(
        store_dir / f"{model_kind}_capture_drops.json", {"flavor": args.flavor, **drops}
    )
    if args.max_items:
        items = items[: args.max_items]
    print(
        f"[i1310-p2] {len(items)} items (model={model_kind}, flavor={args.flavor}, drops={drops})"
    )

    # onpolicy items are independent per (scenario, persona, slot), so resume
    # skips DONE ROW IDS; perturn/tf scenes flush whole (batch-aligned), so a
    # turn row_id present means its whole scene is written -> skip the scene.
    by_row = args.flavor == "onpolicy"
    done: set[str] = set()
    shard_idx = 0
    if args.resume:
        for sc in sorted(store_dir.glob(f"{model_kind}_shard*.json")):
            side = json.loads(sc.read_text())
            if by_row:
                done.update(side["row_ids"])
            else:
                done.update(rid.rsplit(":t", 1)[0] for rid in side["row_ids"])
            shard_idx = max(shard_idx, side["shard_index"] + 1)
        if done:
            items = [it for it in items if it["item_id"] not in done]
            unit = "rows" if by_row else "scenes"
            print(f"[i1310-p2] resume: {len(done)} {unit} done; {len(items)} {unit} left")

    if not items:
        print(f"[i1310-p2] no items to capture (model={model_kind})")
        return 0

    model = load_model(model_id, args.tiny_model_dir)
    if args.equivalence_check:
        eq = equivalence_check(model, items, pad_id)
        c1310.write_json(store_dir / f"{model_kind}_equivalence.json", eq)

    buf: list[dict] = []
    for recs in run_extraction(model, items, pad_id, args.batch_size):
        buf.extend(recs)
        # Flush at BATCH boundaries (a batch = whole scenes), so a scene's turns
        # are never split across shards — makes scene-level --resume exact.
        if len(buf) >= SHARD_PAIRS:
            write_shard(buf, store_dir, shard_idx, model_kind)
            buf = []
            shard_idx += 1
    if buf:
        write_shard(buf, store_dir, shard_idx, model_kind)
    print(f"[i1310-p2] done (model={model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
