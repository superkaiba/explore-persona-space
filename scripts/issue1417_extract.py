"""Issue #1417 — thin teacher-forced capture driver per (model, cell) unit.

Consumes the gen JSONL's EXACT token ids (prompt ids + completion ids
concatenated — never re-tokenization of concatenated strings; spans were
validated at gen time) and writes the #825 turnstore storage schema MINUS
perpos (plan §2 divergence 3): per shard, ``slots`` = per-row (2, 28, 3584)
bf16 tensors [x_ctx @ last prompt token, x_prefix @ last pre-query token],
``profiles`` = per-row (1, 28, 3584) bf16 [y = mean over generated tokens],
``conv_ids`` — loadable straight through ``issue825_fit_cells._load_bundle_any``
with cells {model_key, format_key=<cell slug>, track="s", slot_index in
{0 (ctx), 1 (prefix)}, target_turn_index=0}.

Batched right-padded forwards via ``analysis.extraction.extract_layer_activations``
(hook capture, no full-logits materialization), batch 8 with OOM-halving,
bf16 model, ALL 28 layers. A batched-vs-batch-1 equivalence gate (two-bar
#779 calibration: early layers >= 0.999, flattened >= 0.995) runs under
``--equivalence-check``.

CLI:
  uv run python scripts/issue1417_extract.py --model instruct --cell c2_rude \
      --data-dir data/issue_1417 [--batch-size 8] [--equivalence-check] \
      [--tiny-model-dir /tmp/tiny]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common931  # noqa: E402
import issue1417_gen as g1417  # noqa: E402
import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_extract.py"
SHARD_CONVS = 500  # plan §4: 500 convs/shard
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# Two-bar equivalence calibration (#779; gotchas "bf16 padded-batch"):
EQUIV_EARLY_LAYERS = 4
EQUIV_EARLY_MIN = 0.999
EQUIV_FLAT_MIN = 0.995


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--model", required=True, choices=list(r1417.MODELS))
    ap.add_argument("--cell", required=True, choices=list(r1417.CELLS))
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--equivalence-check", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--tiny-model-dir", default=None, help="SMOKE ONLY: tiny random-init Qwen2 dir (CPU)"
    )
    return ap.parse_args()


def store_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "store"


def shard_stem(model: str, cell: str) -> str:
    # {model_key}_{format_key}_{track}* — the fit825._load_bundle_pt glob.
    return f"{model}_{cell}_s"


def load_model(model_key: str, tiny_model_dir: str | None):
    """bf16 CUDA load (GPU-pinned, fail-loud) or the tiny CPU smoke model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global EXPECTED_LAYERS, EXPECTED_HIDDEN
    if tiny_model_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, torch_dtype=torch.float32)
        model.eval()
        EXPECTED_LAYERS = int(model.config.num_hidden_layers)
        EXPECTED_HIDDEN = int(model.config.hidden_size)
        return model, tokenizer
    model_id = r1417.MODEL_IDS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, f"{model_id}: {len(off_gpu)} params off-CUDA — refusing CPU offload"
    assert model.config.num_hidden_layers == EXPECTED_LAYERS
    assert model.config.hidden_size == EXPECTED_HIDDEN
    return model, tokenizer


def build_items(records: list[dict]) -> tuple[list[dict], dict]:
    """Gen records -> capture items (token-id concat; span asserts)."""
    items: list[dict] = []
    counters = {"records": 0, "kept": 0, "dropped_empty_completion": 0, "dropped_zero_prefix": 0}
    for r in records:
        counters["records"] += 1
        prompt_ids = list(r["prompt_token_ids"])
        comp_ids = list(r["completion_token_ids"])
        n_prompt, n_comp = len(prompt_ids), len(comp_ids)
        if n_comp == 0:
            counters["dropped_empty_completion"] += 1
            continue
        n_prefix = int(r["n_prefix_tokens"])
        if n_prefix < 1:
            counters["dropped_zero_prefix"] += 1
            continue
        assert n_prefix <= n_prompt, (r["conv_id"], n_prefix, n_prompt)
        assert r["n_prompt_tokens"] == n_prompt, (r["conv_id"], "prompt-count drift")
        items.append(
            {
                "conv_id": r["conv_id"],
                "input_ids": prompt_ids + comp_ids,
                "n_prompt": n_prompt,
                "n_prefix": n_prefix,
                "prefix_seam": bool(r.get("prefix_seam", False)),
            }
        )
        counters["kept"] += 1
    return items, counters


def process_batch(model, batch: list[dict], pad_id: int) -> list[dict]:
    """One right-padded batched forward; per-row slot + profile summaries."""
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

    records: list[dict] = []
    for i, it in enumerate(batch):
        n_prompt, n_prefix, true_len = it["n_prompt"], it["n_prefix"], lengths[i]
        assert 0 < n_prefix <= n_prompt < true_len, (it["conv_id"], n_prefix, n_prompt, true_len)
        x_ctx = acts[:, i, n_prompt - 1, :].float()  # (L, D) last prompt token
        x_prefix = acts[:, i, n_prefix - 1, :].float()  # (L, D) last pre-query token
        y = acts[:, i, n_prompt:true_len, :].float().mean(dim=1)  # (L, D) answer mean
        slots = torch.stack([x_ctx, x_prefix], dim=0)  # (2, L, D)
        profiles = y.unsqueeze(0)  # (1, L, D)
        for name, t in (("slots", slots), ("profiles", profiles)):
            assert torch.isfinite(t).all(), (it["conv_id"], name, "non-finite capture")
        records.append(
            {
                "conv_id": it["conv_id"],
                "slots": slots.to(torch.bfloat16).cpu(),
                "profiles": profiles.to(torch.bfloat16).cpu(),
                "n_prompt": n_prompt,
                "n_prefix": n_prefix,
                "n_total": true_len,
                "prefix_seam": it["prefix_seam"],
            }
        )
    del captured, acts
    return records


def run_extraction(model, items: list[dict], pad_id: int, batch_size: int):
    """Length-grouped batching with OOM-halving (floor 1); yields batches."""
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
            print(f"[i1417-extract] CUDA OOM — halving batch size to {bs}")
            continue
        pos += len(chunk)
        done += 1
        if done % 10 == 0 or pos >= len(order):
            print(f"[i1417-extract] {pos}/{len(order)} items (batch {bs})", flush=True)
        yield recs


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def equivalence_check(model, items: list[dict], pad_id: int) -> dict:
    """Two-bar batched-vs-batch-1 gate over slots+profiles (#779 calibration)."""
    probe = items[: min(4, len(items))]
    assert len(probe) >= 2, "equivalence check needs >=2 rows"
    batched = process_batch(model, probe, pad_id)
    singles = [process_batch(model, [it], pad_id)[0] for it in probe]
    early_min, flat_min = 1.0, 1.0
    for b, s in zip(batched, singles, strict=True):
        for key in ("slots", "profiles"):
            tb, ts = b[key].float(), s[key].float()
            for li in range(min(EQUIV_EARLY_LAYERS, tb.shape[1])):
                early_min = min(early_min, _cos(tb[:, li, :], ts[:, li, :]))
            flat_min = min(flat_min, _cos(tb, ts))
    result = {
        "n_probe": len(probe),
        "early_cos_min": early_min,
        "flat_cos_min": flat_min,
        "early_bar": EQUIV_EARLY_MIN,
        "flat_bar": EQUIV_FLAT_MIN,
        "pass": bool(early_min >= EQUIV_EARLY_MIN and flat_min >= EQUIV_FLAT_MIN),
    }
    print(f"[i1417-extract] equivalence gate: {result}")
    assert result["pass"], f"batched-vs-batch-1 equivalence FAILED: {result}"
    return result


def write_shard(records: list[dict], out_dir: Path, stem: str, shard_idx: int, meta: dict) -> None:
    """One .pt shard (per-row tensor lists, the turnstore contract) + sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "conv_ids": [r["conv_id"] for r in records],
        "slots": [r["slots"] for r in records],
        "profiles": [r["profiles"] for r in records],
    }
    pt_path = out_dir / f"{stem}_shard{shard_idx:03d}.pt"
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(pt_path)
    sidecar = {
        **meta,
        "shard_index": shard_idx,
        "n_rows": len(records),
        "conv_ids": payload["conv_ids"],
        "n_prompt": [r["n_prompt"] for r in records],
        "n_prefix": [r["n_prefix"] for r in records],
        "n_total": [r["n_total"] for r in records],
        "prefix_seam": [r["prefix_seam"] for r in records],
        "slot_names": ["x_ctx", "x_prefix"],
        "turn_names": ["y"],
        "shape_slots": [2, EXPECTED_LAYERS, EXPECTED_HIDDEN],
        "shape_profiles": [1, EXPECTED_LAYERS, EXPECTED_HIDDEN],
    }
    (out_dir / f"{stem}_shard{shard_idx:03d}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[i1417-extract] wrote {pt_path} ({len(records)} rows)")


def main() -> int:
    args = parse_args()
    gen_file = g1417.gen_path(args.data_dir, args.model, args.cell)
    assert gen_file.exists(), f"gen JSONL missing: {gen_file} — run issue1417_gen.py first"
    records = g1417._read_jsonl(gen_file)
    assert records, f"empty gen JSONL: {gen_file}"
    assert r1417.fingerprint_matches(records[0]), (
        f"{gen_file}: render fingerprint mismatch — regenerate before capture"
    )

    stem = shard_stem(args.model, args.cell)
    out_dir = store_dir(args.data_dir)
    meta = {
        **r1417.fingerprint(),
        "cell": args.cell,
        "model": args.model,
        "metadata": common931.metadata(SCRIPT, r1417.GEN_SEED, len(records)),
    }
    items, counters = build_items(records)
    print(f"[i1417-extract] {args.model}/{args.cell}: {counters}")
    n_shards_expected = (len(items) + SHARD_CONVS - 1) // SHARD_CONVS
    if args.resume:
        existing = sorted(out_dir.glob(f"{stem}_shard*.pt"))
        sidecars = sorted(out_dir.glob(f"{stem}_shard*.json"))
        if len(existing) == n_shards_expected and len(sidecars) == n_shards_expected:
            side0 = json.loads(sidecars[0].read_text())
            if r1417.fingerprint_matches(side0):
                print(f"[i1417-extract] resume: {stem} complete ({len(existing)} shards) — skipped")
                return 0

    model, tokenizer = load_model(args.model, args.tiny_model_dir)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if args.equivalence_check:
        eq = equivalence_check(model, items, pad_id)
        eq_path = out_dir / f"{stem}_equivalence.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        eq_path.write_text(json.dumps({**meta, **eq}, indent=2))

    buf: list[dict] = []
    shard_idx = 0
    for recs in run_extraction(model, items, pad_id, args.batch_size):
        buf.extend(recs)
        while len(buf) >= SHARD_CONVS:
            write_shard(buf[:SHARD_CONVS], out_dir, stem, shard_idx, meta)
            buf = buf[SHARD_CONVS:]
            shard_idx += 1
    if buf:
        write_shard(buf, out_dir, stem, shard_idx, meta)
        shard_idx += 1
    print(f"[i1417-extract] {args.model}/{args.cell}: {shard_idx} shards complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
