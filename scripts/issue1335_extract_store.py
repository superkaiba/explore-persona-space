"""Issue #1335: teacher-forced 28-layer bf16 span-summary capture per (rung, model).

Generalizes ``issue1310_extract_store.py``'s onpolicy flavor to the ladder's
uniform gen-record schema. One ITEM per gen record: input_ids = STORED
prompt_token_ids + completion_token_ids (token-id join — #1092 BPE-seam rule;
never re-tokenized). Per row it stores ONLY reduced summaries (fp32-computed,
bf16-cast — the #1310 store recipe; finiteness asserted):

  x_spanmean   v_C: mean over the last <=512 context tokens ending at the cue
  x_prefixmean v_P: mean over tokens ENDING inside the prefix text (header +
               earlier turns; the stored n_prefix_tokens boundary). Empty
               prefix (r0/r1/r2 — structurally degenerate arm) falls back to
               the FIRST context token, flagged `prefix_fallback_first_token`
               in the sidecar (the fit reports it as the degenerate control).
  x_last       the cue boundary token (single-position companion)
  y            v_A: mean over the generated completion tokens
  y96 / x_spanmean_nocap   r0-only extras (plan §4.1: v_A96 sub-read + the
               no-cap context companion verifying the 512 cap is inert on Q&A)

Row filters (#825/#1310): completion >= 4 tokens, context >= 8 tokens, total
row <= 2048 tokens — drops COUNTED in the sidecar, never padded.

Shard sidecars carry the c24 fingerprint {rung_slug, render_config_hash,
code_sha}; --resume skips a row ONLY when its shard sidecar fingerprint
matches the CURRENT config + SHA (issue1310's row-id-presence-only resume is
deliberately extended per plan §8).

--equivalence-check: the two-bar batched-vs-batch-1 gate (#779 calibration:
early-layer per-layer cosine >= 0.999, flattened >= 0.995).
--wiring-check: own-context vs derangement-shuffled-context teacher-forced
NLL over n rows (the #825 round-4 wiring gate; plan §5).

CLI:
  uv run python scripts/issue1335_extract_store.py --rung r1_qa_oneline --model base \
      [--data-dir data/issue_1335] [--batch-size 8] [--tiny-model-dir D] \
      [--resume] [--equivalence-check] [--wiring-check N] [--max-items 0]
  uv run python scripts/issue1335_extract_store.py --make-tiny-model /tmp/tiny
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
import issue1310_extract_store as ext1310  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_extract_store.py"
SHARD_PAIRS = 512

EXPECTED_LAYERS = c1310.EXPECTED_LAYERS
EXPECTED_HIDDEN = c1310.EXPECTED_HIDDEN


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rung", choices=list(r1335.RUNGS), default=None)
    ap.add_argument("--model", choices=list(r1335.MODEL_KINDS), default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--store-dir", type=Path, default=None, help="default <data-dir>/store")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_results/issue_1335"),
        help="wiring-check JSON destination (smoke passes a scratch dir)",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tiny-model-dir", type=str, default=None, help="CPU smoke model dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--equivalence-check", action="store_true")
    ap.add_argument("--wiring-check", type=int, default=0, help="N rows for the NLL wiring gate")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all (smoke slicing)")
    ap.add_argument(
        "--make-tiny-model",
        type=str,
        default=None,
        help="SMOKE: write a tiny random-init Qwen2 (real tokenizer) to this dir and exit",
    )
    return ap.parse_args()


def load_model(model_id: str, tiny_model_dir: str | None):
    """bf16 GPU model (device pinned, fail-loud) or tiny CPU fp32 (smoke)."""
    global EXPECTED_LAYERS, EXPECTED_HIDDEN
    model = ext1310.load_model(model_id, tiny_model_dir)
    EXPECTED_LAYERS = ext1310.EXPECTED_LAYERS
    EXPECTED_HIDDEN = ext1310.EXPECTED_HIDDEN
    return model


def build_items(slug: str, records: list[dict]) -> tuple[list[dict], dict]:
    """Gen records -> capture items with per-summary spans; drops counted."""
    cfg = r1335.RUNGS[slug]
    extras = cfg.get("extra_summaries", ())
    items: list[dict] = []
    counters = {
        "records": 0,
        "kept": 0,
        "dropped_short_dialogue": 0,
        "dropped_short_context": 0,
        "dropped_row_too_long": 0,
        "prefix_fallback_first_token": 0,
    }
    for r in records:
        counters["records"] += 1
        prompt_ids = list(r["prompt_token_ids"])
        comp_ids = list(r["completion_token_ids"])
        n_prompt, n_comp = len(prompt_ids), len(comp_ids)
        n_tok = n_prompt + n_comp
        if n_comp < r1335.DIALOGUE_MIN_TOKENS:
            counters["dropped_short_dialogue"] += 1
            continue
        c_lo = max(0, n_prompt - r1335.CONTEXT_CAP_TOKENS)
        if n_prompt - c_lo < r1335.CONTEXT_MIN_TOKENS:
            counters["dropped_short_context"] += 1
            continue
        if n_tok > r1335.ROW_MAX_TOKENS:
            counters["dropped_row_too_long"] += 1
            continue
        n_prefix = int(r["n_prefix_tokens"])
        assert 0 <= n_prefix <= n_prompt, (r["row_id"], n_prefix, n_prompt)
        if n_prefix == 0:
            counters["prefix_fallback_first_token"] += 1
            p_span = (0, 1)  # degenerate arm: constant first-token fallback
        else:
            p_span = (0, n_prefix)
        spans = {
            "x_spanmean": (c_lo, n_prompt),
            "x_prefixmean": p_span,
            "y": (n_prompt, n_tok),
        }
        if "y96" in extras:
            spans["y96"] = (n_prompt, n_prompt + min(96, n_comp))
        if "x_spanmean_nocap" in extras:
            spans["x_spanmean_nocap"] = (0, n_prompt)
        for k, (lo, hi) in spans.items():
            assert 0 <= lo < hi <= n_tok, (r["row_id"], k, lo, hi, n_tok)
        items.append(
            {
                "item_id": r["row_id"],
                "row_id": r["row_id"],
                "group_id": r["group_id"],
                "char_id": r["persona"],
                "turn_index": int(r.get("slot", 0)),
                "input_ids": prompt_ids + comp_ids,
                "n_prompt": n_prompt,
                "spans": spans,
            }
        )
        counters["kept"] += 1
    return items, counters


def process_batch(model, batch: list[dict], pad_id: int) -> list[dict]:
    """One right-padded batched forward; per-row reduced summaries to CPU bf16."""
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
        rec: dict = {
            "row_id": it["row_id"],
            "group_id": it["group_id"],
            "char_id": it["char_id"],
            "turn_index": it["turn_index"],
        }
        for key, (lo, hi) in it["spans"].items():
            assert 0 <= lo < hi <= true_len, (it["row_id"], key, lo, hi, true_len)
            rec[key] = acts[:, i, lo:hi, :].float().mean(dim=1)
        ce = it["spans"]["x_spanmean"][1]
        rec["x_last"] = acts[:, i, ce - 1, :].float()
        for k in list(rec.keys()):
            if isinstance(rec[k], torch.Tensor):
                rec[k] = ext931._finite(
                    rec[k].to(device="cpu", dtype=torch.bfloat16), k, it["row_id"]
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
            print(f"[i1335-p2] {pos}/{len(order)} items done (batch size {bs})", flush=True)
        yield recs


def write_shard(
    records: list[dict], out_dir: Path, shard_idx: int, model_kind: str, fp: dict, flags: dict
) -> None:
    """One .pt shard (stacked arrays) + fingerprinted JSON sidecar."""
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
        **fp,
        "model_kind": model_kind,
        "shard_index": shard_idx,
        "n_rows": len(records),
        "row_ids": payload["row_ids"],
        "group_ids": payload["group_ids"],
        "char_ids": payload["char_ids"],
        "turn_indices": payload["turn_indices"],
        "keys": keys,
        "shape_per_row": [EXPECTED_LAYERS, EXPECTED_HIDDEN],
        "capture_flags": flags,
        "metadata": common.metadata(SCRIPT, c1310.GEN_SEED, len(records)),
    }
    (out_dir / f"{model_kind}_shard{shard_idx:03d}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[i1335-p2] wrote {pt_path} ({len(records)} rows)")


def equivalence_check(model, items: list[dict], pad_id: int) -> dict:
    """Two-bar batched-vs-batch-1 gate over every stored summary (#779 calibration)."""
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
    print(f"[i1335-p2] equivalence gate PASS: {result}")
    return result


def wiring_check(model, items: list[dict], pad_id: int, n_rows: int, batch_size: int) -> dict:
    """Own-context vs derangement-shuffled-context teacher-forced NLL (#825 r4 gate).

    Pairs row i's completion with row (i+1)'s prompt (a cyclic derangement) and
    compares mean per-token completion NLL; own must beat shuffled.
    """
    import numpy as np

    take = items[: min(n_rows, len(items))]
    assert len(take) >= 2, "wiring check needs >= 2 rows"
    device = next(model.parameters()).device

    def _mean_nll(prompt_ids_list, comp_ids_list) -> float:
        total_nll, total_tok = 0.0, 0
        for ci in range(0, len(prompt_ids_list), batch_size):
            pr = prompt_ids_list[ci : ci + batch_size]
            co = comp_ids_list[ci : ci + batch_size]
            seqs = [p + c for p, c in zip(pr, co, strict=True)]
            lengths = [len(s) for s in seqs]
            max_len = max(lengths)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : lengths[i]] = torch.tensor(s, dtype=torch.long)
                attention_mask[i, : lengths[i]] = 1
            with torch.no_grad():
                logits = model(
                    input_ids.to(device), attention_mask=attention_mask.to(device)
                ).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for i, (p, c) in enumerate(zip(pr, co, strict=True)):
                pos = torch.arange(len(p) - 1, len(p) + len(c) - 1)
                tgt = torch.tensor(c, dtype=torch.long)
                lp = logprobs[i, pos, :].gather(1, tgt.unsqueeze(1).to(device)).squeeze(1)
                total_nll += float(-lp.sum().item())
                total_tok += len(c)
        return total_nll / max(total_tok, 1)

    prompts = [it["input_ids"][: it["n_prompt"]] for it in take]
    comps = [it["input_ids"][it["n_prompt"] :] for it in take]
    own = _mean_nll(prompts, comps)
    shuf_prompts = prompts[1:] + prompts[:1]  # cyclic derangement
    shuf = _mean_nll(shuf_prompts, comps)
    result = {
        "n_rows": len(take),
        "nll_own": own,
        "nll_shuffled": shuf,
        "own_beats_shuffled": bool(own < shuf),
        "delta": float(shuf - own),
    }
    assert np.isfinite(own) and np.isfinite(shuf), result
    print(f"[i1335-p2] wiring check: {result}")
    return result


def main() -> int:
    args = parse_args()
    if args.make_tiny_model:
        ext931.make_tiny_model(Path(args.make_tiny_model))
        return 0
    assert args.rung and args.model, "--rung and --model are required"
    slug, model_kind = args.rung, args.model
    model_id = r1335.MODEL_IDS[model_kind]
    fp = r1335.fingerprint(slug)
    store_dir = (args.store_dir or (args.data_dir / "store")) / slug / model_kind
    print(f"[phase=p2_extract_{slug}_{model_kind}] span-summary capture ({model_id})")

    records = r1335._read_jsonl(r1335.gen_path(args.data_dir, slug, model_kind))
    items, drops = build_items(slug, records)
    store_dir.mkdir(parents=True, exist_ok=True)
    flags = {"prefix_fallback_first_token": drops["prefix_fallback_first_token"] > 0}
    c1310.write_json(store_dir / f"{model_kind}_capture_drops.json", {**fp, **drops})
    if args.max_items:
        items = items[: args.max_items]
    print(f"[i1335-p2] {len(items)} items (rung={slug}, model={model_kind}, drops={drops})")

    # c24 resume: skip rows already in shards whose sidecar fingerprint matches
    # the CURRENT render config + code SHA; a mismatched sidecar fails loud
    # (stale store for a changed render — wipe or re-key before resuming).
    done: set[str] = set()
    shard_idx = 0
    if args.resume:
        for sc in sorted(store_dir.glob(f"{model_kind}_shard*.json")):
            side = json.loads(sc.read_text())
            assert r1335.fingerprint_matches(side, slug), (
                f"resume fingerprint mismatch for {sc} — the persisted shard was "
                "captured under a different render config / code SHA; quarantine "
                "the stale store before resuming (c24 guard)"
            )
            done.update(side["row_ids"])
            shard_idx = max(shard_idx, side["shard_index"] + 1)
        if done:
            items = [it for it in items if it["item_id"] not in done]
            print(f"[i1335-p2] resume: {len(done)} rows done; {len(items)} rows left")

    if not items:
        print(f"[i1335-p2] no items to capture (rung={slug}, model={model_kind})")
        return 0

    model = load_model(model_id, args.tiny_model_dir)
    tokenizer = common.get_tokenizer(model_id)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if args.equivalence_check:
        eq = equivalence_check(model, items, pad_id)
        c1310.write_json(store_dir / f"{model_kind}_equivalence.json", {**fp, **eq})
    if args.wiring_check:
        w = wiring_check(model, items, pad_id, args.wiring_check, args.batch_size)
        c1310.write_json(args.out_dir / f"wiring_{slug}_{model_kind}.json", {**fp, **w})
        assert w["own_beats_shuffled"] or args.tiny_model_dir, (
            "wiring gate FAIL: own-context NLL does not beat shuffled context"
        )

    buf: list[dict] = []
    for recs in run_extraction(model, items, pad_id, args.batch_size):
        buf.extend(recs)
        if len(buf) >= SHARD_PAIRS:
            write_shard(buf, store_dir, shard_idx, model_kind, fp, flags)
            buf = []
            shard_idx += 1
    if buf:
        write_shard(buf, store_dir, shard_idx, model_kind, fp, flags)
    print(f"[i1335-p2] done (rung={slug}, model={model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
