"""Issue #931 P2: teacher-forced 28-layer span-summary extraction (3 regimes).

Adapts issue825_extract_turnstore's length-grouped OOM-halving batched capture
to arbitrary named spans: per pair it stores ONLY the reduced summaries —
x_spanmean (mean over C), x_last (the intro-span boundary token — the
parent-matched single-position X), x_ctxmean (B1 whole-window read), x_sep
(Arm C anchor position), y (mean over the target spans) — never per-token
grids (#666/#772 stream-reduce). bf16 storage (Qwen residual outliers exceed
fp16 range), finiteness asserted before every write. Right-padded batches +
attention mask (causal mask => pads cannot influence real positions); a
batched-vs-batch-1 equivalence gate is available via --equivalence-check.

Regimes:
  armA  windows_armA.jsonl + pairs_armA.jsonl (raw novel text, no template)
  armB  stories_seed42.jsonl + pairs_armB.jsonl (chat template; spans are
        story-local and get the rendered-prefix offset added here)
  armC  articles_armC.jsonl + pairs_armC.jsonl (raw text; anchor single pos)

CLI:
  uv run python scripts/issue931_extract_store.py --regime armA \
      [--data-dir data/issue_931] [--store-dir data/issue_931/store] \
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

SCRIPT = "scripts/issue931_extract_store.py"
SHARD_PAIRS = 512

EXPECTED_LAYERS = common.EXPECTED_LAYERS
EXPECTED_HIDDEN = common.EXPECTED_HIDDEN


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", default=None, choices=("armA", "armB", "armC"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--store-dir", type=Path, default=None, help="default <data-dir>/store")
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


def make_tiny_model(dest: Path, *, layers: int = 4, hidden: int = 64) -> None:
    """Tiny random-init Qwen2 with the REAL Qwen tokenizer (CPU smoke model)."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(common.MODEL_ID)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    print(f"[i931-p2] tiny model written to {dest} (L={layers}, D={hidden})")


def load_model(tiny_model_dir: str | None):
    """bf16 GPU model (device_map pinned; off-GPU params fail loud) or CPU tiny."""
    global EXPECTED_LAYERS, EXPECTED_HIDDEN
    from transformers import AutoModelForCausalLM

    if tiny_model_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
        model.eval()
        EXPECTED_LAYERS = int(model.config.num_hidden_layers)
        EXPECTED_HIDDEN = int(model.config.hidden_size)
        print(f"[i931-p2] TINY model: L={EXPECTED_LAYERS} D={EXPECTED_HIDDEN}")
        return model
    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, f"{len(off_gpu)} params not on CUDA (e.g. {off_gpu[:3]})"
    assert model.config.num_hidden_layers == EXPECTED_LAYERS
    assert model.config.hidden_size == EXPECTED_HIDDEN
    return model


# ---------------------------------------------------------------------------
# Item assembly (item = one window / story / article + its pairs)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file (newline-split; asserts the path exists).

    split("\\n"), not splitlines(): JSON string content may carry U+2028/NEL-class
    separators that splitlines() shreds mid-row (gotchas.md JSONL rule).
    """
    assert path.exists(), f"missing input: {path}"
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def load_items(regime: str, data_dir: Path) -> list[dict]:
    """[{item_id, group_id, input_ids, pairs: [PairSpec (item-local spans)]}]"""
    pairs_dir = data_dir / "pairs"
    if regime in ("armA", "armC"):
        src = "windows_armA.jsonl" if regime == "armA" else "articles_armC.jsonl"
        psrc = "pairs_armA.jsonl" if regime == "armA" else "pairs_armC.jsonl"
        windows = {w["window_id"]: w for w in _read_jsonl(pairs_dir / src)}
        pairs = [common.PairSpec.from_dict(d) for d in _read_jsonl(pairs_dir / psrc)]
        by_item: dict[str, list[common.PairSpec]] = {}
        for p in pairs:
            by_item.setdefault(p.meta["window_id"], []).append(p)
        items = []
        for wid, plist in by_item.items():
            w = windows[wid]
            items.append(
                {
                    "item_id": wid,
                    "group_id": plist[0].group_id,
                    "input_ids": list(w["input_ids"]),
                    "pairs": plist,
                }
            )
        return items

    # armB: render the chat-template prefix per prompt; offset story-local spans.
    tokenizer = common.get_tokenizer()
    stories = {
        s["prompt_id"]: s
        for s in _read_jsonl(data_dir / "stories" / f"stories_seed{common.GEN_SEED}.jsonl")
    }
    pairs = [common.PairSpec.from_dict(d) for d in _read_jsonl(pairs_dir / "pairs_armB.jsonl")]
    by_item = {}
    for p in pairs:
        by_item.setdefault(p.meta["window_id"], []).append(p)
    items = []
    for pid, plist in by_item.items():
        story = stories[pid]
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": story["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        story_ids = tokenizer(story["story"], add_special_tokens=False)["input_ids"]
        off = len(prefix_ids)
        shifted = []
        for p in plist:
            q = common.PairSpec(
                row_id=p.row_id,
                group_id=p.group_id,
                char_id=p.char_id,
                c_span=(p.c_span[0] + off, p.c_span[1] + off),
                t_spans=[(lo + off, hi + off) for lo, hi in p.t_spans],
                # B1 for armB: story tokens preceding min(T) — template excluded.
                ctx_span=(off, p.ctx_span[1] + off),
                meta={**p.meta, "prefix_len": off},
            )
            q.validate(off + len(story_ids))
            shifted.append(q)
        items.append(
            {
                "item_id": pid,
                "group_id": pid,
                "input_ids": list(prefix_ids) + list(story_ids),
                "pairs": shifted,
            }
        )
    return items


# ---------------------------------------------------------------------------
# Batched capture + span reductions
# ---------------------------------------------------------------------------


def _finite(t: torch.Tensor, name: str, row_id: str) -> torch.Tensor:
    assert torch.isfinite(t.float()).all(), f"{row_id}: non-finite {name}"
    return t


def process_batch(model, batch: list[dict], pad_id: int, regime: str) -> list[dict]:
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
        detach_to_cpu=False,  # reductions stay device-side; only summaries move
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
            }
            rec["x_spanmean"] = acts[:, i, cs:ce, :].float().mean(dim=1)
            rec["x_last"] = acts[:, i, ce - 1, :].float()
            if regime == "armC":
                a = int(p.meta["anchor_pos"])
                assert 0 <= a < true_len, (p.row_id, "anchor", a, true_len)
                rec["x_sep"] = acts[:, i, a, :].float()
            else:
                xs, xe = p.ctx_span
                assert 0 <= xs < xe <= true_len, (p.row_id, "ctx_span", xs, xe, true_len)
                rec["x_ctxmean"] = acts[:, i, xs:xe, :].float().mean(dim=1)
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
                    rec[k] = _finite(rec[k].to(device="cpu", dtype=torch.bfloat16), k, p.row_id)
            records.append(rec)
    del captured, acts
    return records


def run_extraction(model, items: list[dict], pad_id: int, batch_size: int, regime: str):
    """Length-grouped batching with OOM-halving (floor 1); yields per-batch recs."""
    order = sorted(range(len(items)), key=lambda j: len(items[j]["input_ids"]))
    bs, pos, done = batch_size, 0, 0
    while pos < len(order):
        chunk = [items[order[j]] for j in range(pos, min(pos + bs, len(order)))]
        try:
            recs = process_batch(model, chunk, pad_id, regime)
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
            print(f"[i931-p2] {pos}/{len(order)} items done (batch size {bs})", flush=True)
        yield recs


def write_shard(records: list[dict], out_dir: Path, shard_idx: int, regime: str) -> None:
    """One .pt shard (stacked arrays) + JSON sidecar; atomic-ish via tmp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in records[0] if isinstance(records[0][k], torch.Tensor)]
    payload = {
        "row_ids": [r["row_id"] for r in records],
        "group_ids": [r["group_id"] for r in records],
        "char_ids": [r["char_id"] for r in records],
        "arrays": {k: torch.stack([r[k] for r in records]) for k in keys},
    }
    for k, v in payload["arrays"].items():
        assert v.shape == (len(records), EXPECTED_LAYERS, EXPECTED_HIDDEN), (k, v.shape)
    pt_path = out_dir / f"{regime}_shard{shard_idx:03d}.pt"
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(pt_path)
    sidecar = {
        "regime": regime,
        "shard_index": shard_idx,
        "n_rows": len(records),
        "row_ids": payload["row_ids"],
        "group_ids": payload["group_ids"],
        "keys": keys,
        "shape_per_row": [EXPECTED_LAYERS, EXPECTED_HIDDEN],
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(records)),
    }
    (out_dir / f"{regime}_shard{shard_idx:03d}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[i931-p2] wrote {pt_path} ({len(records)} rows)")


def equivalence_check(model, items: list[dict], pad_id: int, regime: str) -> dict:
    """Batched (B=3) vs batch-1 capture equivalence on real items.

    Two-bar gate per the #779 calibration: early-layer (first 4) per-layer
    cosine >= 0.999 AND flattened all-layer cosine >= 0.995, computed over
    every stored summary of every pair. Right-pad + causal mask means pads
    cannot influence real positions; bf16 deep-layer jitter is the residual.
    """
    take = items[:3]
    if len(take) < 2:
        take = items[:1] * 2  # degenerate smoke fallback (still exercises pad)
    batched = process_batch(model, take, pad_id, regime)
    serial = []
    for it in take:
        serial.extend(process_batch(model, [it], pad_id, regime))
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
            per_layer = torch.nn.functional.cosine_similarity(a, b, dim=1)  # (L,)
            early_min = min(early_min, float(per_layer[:n_early].min()))
    result = {"early_cos_min": early_min, "flat_cos_min": flat_min, "n_items": len(take)}
    assert early_min >= 0.999 and flat_min >= 0.995, f"equivalence gate FAIL: {result}"
    print(f"[i931-p2] equivalence gate PASS: {result}")
    return result


def main() -> int:
    args = parse_args()
    if args.make_tiny_model:
        make_tiny_model(Path(args.make_tiny_model))
        return 0
    assert args.regime, "--regime is required unless --make-tiny-model"
    store_dir = (args.store_dir or (args.data_dir / "store")) / args.regime
    print(f"[phase=p2_extract_{args.regime.lower()}] span-summary extraction")
    tokenizer = common.get_tokenizer()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    items = load_items(args.regime, args.data_dir)
    if args.max_items:
        items = items[: args.max_items]
    n_pairs = sum(len(it["pairs"]) for it in items)
    print(f"[i931-p2] {len(items)} items / {n_pairs} pairs (regime={args.regime})")

    done_rows: set[str] = set()
    shard_idx = 0
    if args.resume:
        for sc in sorted(store_dir.glob(f"{args.regime}_shard*.json")):
            side = json.loads(sc.read_text())
            done_rows.update(side["row_ids"])
            shard_idx = max(shard_idx, side["shard_index"] + 1)
        if done_rows:
            for it in items:
                it["pairs"] = [p for p in it["pairs"] if p.row_id not in done_rows]
            items = [it for it in items if it["pairs"]]
            print(f"[i931-p2] resume: {len(done_rows)} rows done; {len(items)} items left")

    model = load_model(args.tiny_model_dir)
    if args.equivalence_check and items:
        eq = equivalence_check(model, items, pad_id, args.regime)
        common.write_json(store_dir / f"{args.regime}_equivalence.json", eq)

    buf: list[dict] = []
    for recs in run_extraction(model, items, pad_id, args.batch_size, args.regime):
        buf.extend(recs)
        while len(buf) >= SHARD_PAIRS:
            write_shard(buf[:SHARD_PAIRS], store_dir, shard_idx, args.regime)
            buf = buf[SHARD_PAIRS:]
            shard_idx += 1
    if buf:
        write_shard(buf, store_dir, shard_idx, args.regime)
    print(f"[i931-p2] done (regime={args.regime})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
