#!/usr/bin/env python
"""Issue #825 — teacher-forced turn-store extraction (W2b).

One teacher-forced forward per conversation batch yields, per conversation:
  - slot vectors: residual activation at each role-header slot index, all 28
    layers -> (n_slots, 28, 3584)
  - per-turn profiles: mean residual activation over each turn's content span,
    all 28 layers -> (n_turns, 28, 3584)
  - per-position activations at PEAK layers only, first POSITIONS_CAP positions
    of each turn -> (n_turns, POSITIONS_CAP, n_peak, 3584) + coverage mask
  - per-turn mean teacher-forced NLL from the SAME logits (shift-by-one:
    position t's prediction lives at logits index t-1)

Storage: fp16 sharded .pt per (model, format, track) + JSON sidecar. Shards
every 500 conversations. Works CPU-only (slow); no hard CUDA requirement.

Track M renders the full multi-turn conversation; track S routes a single-turn
{u1, a1} dict (the conversation's last complete pair) through the SAME render
path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.experiments.issue_825.common import (  # noqa: E402
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    FROZEN_LAYERS,
    MODEL_INSTRUCT,
    MODEL_PRETRAINED,
    POSITIONS_CAP,
    Rendered,
)

SHARD_SIZE = 500
_TURN_KEY_RE = re.compile(r"^([ua])(\d+)$")


def _render_module():
    """Lazy import of the sibling render module (same scripts/ directory)."""
    import issue825_render_formats

    return issue825_render_formats


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("instruct", "pretrained"), required=True)
    parser.add_argument("--format", choices=("chat", "naturalistic"), required=True)
    parser.add_argument("--conversations", type=Path, required=True, help="input JSONL")
    parser.add_argument("--track", choices=("m", "s"), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/issue_825/turnstore"))
    parser.add_argument(
        "--peak-layers",
        default=",".join(str(x) for x in sorted(FROZEN_LAYERS)),
        help="comma-separated block indices for per-position capture",
    )
    parser.add_argument(
        "--batch-size", default="auto", help='"auto" (start 8, halve on OOM) or int'
    )
    parser.add_argument("--assert-causal", action="store_true", help="prefix-vs-full slot check")
    parser.add_argument("--smoke", action="store_true", help="first 8 convs; causal check ON")
    parser.add_argument(
        "--tiny-model-dir",
        default=None,
        help=(
            "SMOKE ONLY: load the model from this local dir (a tiny random-init "
            "Qwen2 with the real tokenizer) and derive the expected layer/hidden "
            "dims from ITS config instead of the 7B constants. Plumbing/shape "
            "validation on a GPU-less VM; production runs NEVER pass this."
        ),
    )
    return parser.parse_args()


def _load_conversations(path: Path) -> list[dict]:
    rf = _render_module()
    loader = getattr(rf, "load_conversations", None)
    if loader is not None:
        convs = list(loader(path))
    else:
        convs = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    convs.append(json.loads(line))
    assert convs, f"no conversations found in {path}"
    return convs


def to_single_turn(conv: dict) -> dict:
    """Track S: reduce a row to a single (u1, a1) pair.

    Accepts BOTH shapes: the gen script's Track-S rows
    ``{prompt_idx, prompt, response}`` (mapped directly to u1/a1) and flat
    ``{u1..uK, a1..aK}`` conversations (reduced to the last complete pair).
    """
    if "prompt" in conv and "response" in conv:
        cid = conv.get("conv_id") or f"s{conv.get('prompt_idx', 'x')}"
        return {"conv_id": str(cid), "u1": conv["prompt"], "a1": conv["response"]}
    pairs: dict[int, dict[str, str]] = {}
    for key, val in conv.items():
        m = _TURN_KEY_RE.match(key)
        if m:
            pairs.setdefault(int(m.group(2)), {})[m.group(1)] = val
    complete = [k for k in sorted(pairs) if "u" in pairs[k] and "a" in pairs[k]]
    if not complete:
        raise ValueError(f"conversation {conv.get('conv_id')!r} has no complete (u_k, a_k) pair")
    k = complete[-1]
    single = {key: val for key, val in conv.items() if not _TURN_KEY_RE.match(key)}
    cid = conv.get("conv_id") or conv.get("id") or "conv"
    single["conv_id"] = f"{cid}__s{k}"
    single["u1"] = pairs[k]["u"]
    single["a1"] = pairs[k]["a"]
    return single


def render_conv(conv: dict, tokenizer, fmt: str) -> Rendered:
    rf = _render_module()
    if fmt == "chat":
        return rf.render_chat(conv, tokenizer)
    if fmt == "naturalistic":
        return rf.render_naturalistic(conv, tokenizer)
    raise ValueError(f"unknown format: {fmt!r}")


def load_model(model_key: str, tiny_model_dir: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tiny_model_dir is not None:
        # SMOKE ONLY: tiny random-init Qwen2 with the real tokenizer. Expected
        # dims are rebound to ITS config so the downstream shape asserts stay
        # active (they validate internal consistency, not the 7B constants).
        model_id = f"TINY::{tiny_model_dir}"
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, torch_dtype=torch.float32)
        model.eval()
        cfg = model.config
        globals()["EXPECTED_LAYERS"] = int(cfg.num_hidden_layers)
        globals()["EXPECTED_HIDDEN"] = int(cfg.hidden_size)
        return model, tokenizer, model_id

    model_id = MODEL_INSTRUCT if model_key == "instruct" else MODEL_PRETRAINED
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    cfg = model.config
    assert cfg.num_hidden_layers == EXPECTED_LAYERS, (
        f"{model_id}: num_hidden_layers={cfg.num_hidden_layers} != {EXPECTED_LAYERS}"
    )
    assert cfg.hidden_size == EXPECTED_HIDDEN, (
        f"{model_id}: hidden_size={cfg.hidden_size} != {EXPECTED_HIDDEN}"
    )
    return model, tokenizer, model_id


def _ordered_slots(r: Rendered) -> list[tuple[str, int]]:
    assert r.slot_idx, f"{r.conv_id}: empty slot_idx"
    return sorted(r.slot_idx.items(), key=lambda kv: kv[1])


def _ordered_turns(r: Rendered) -> list[tuple[str, tuple[int, int]]]:
    assert r.spans, f"{r.conv_id}: empty spans"
    return sorted(r.spans.items(), key=lambda kv: kv[1][0])


def _turn_nll(
    row_logits: torch.Tensor,
    row_ids: torch.Tensor,
    true_len: int,
    turns: list[tuple[str, tuple[int, int]]],
    conv_id: str,
    align_state: dict,
) -> torch.Tensor:
    """Mean teacher-forced NLL per turn span from the SAME forward's logits.

    Shift-by-one: the prediction for the token at position t lives at logits
    index t-1, so a content span (s, e) reads token log-probs at [s-1, e-1).
    """
    logprobs = torch.log_softmax(row_logits[: true_len - 1].float(), dim=-1)
    targets = row_ids[1:true_len].to(logprobs.device)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert token_lp.shape == (true_len - 1,), f"{conv_id}: token_lp {tuple(token_lp.shape)}"
    out = torch.empty(len(turns), dtype=torch.float32)
    for t, (name, (s, e)) in enumerate(turns):
        assert s >= 1, f"{conv_id}: span {name} starts at 0 — no teacher-forced target exists"
        if not align_state.get("done", False):
            gathered = targets[s - 1 : e - 1].cpu()
            expected = row_ids[s:e].cpu()
            assert torch.equal(gathered, expected), (
                f"{conv_id}: shift-by-one misalignment on span {name}"
            )
            align_state["done"] = True
            print(f"[align] shift-by-one target alignment verified on {conv_id}:{name}")
        out[t] = -token_lp[s - 1 : e - 1].mean().float().cpu()
    return out


def process_batch(
    model,
    batch: list[Rendered],
    peak_layers: list[int],
    pad_id: int,
    align_state: dict,
) -> list[dict]:
    lengths = [len(r.input_ids) for r in batch]
    bsz, max_len = len(batch), max(lengths)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, r in enumerate(batch):
        input_ids[i, : lengths[i]] = torch.tensor(r.input_ids, dtype=torch.long)
        attention_mask[i, : lengths[i]] = 1
    device = model.device
    captured, logits = extract_layer_activations(
        model,
        input_ids.to(device),
        layers=range(EXPECTED_LAYERS),
        return_logits=True,
        attention_mask=attention_mask.to(device),
        detach_to_cpu=True,
    )
    assert set(captured) == set(range(EXPECTED_LAYERS)), "missing layers in capture"
    acts = torch.stack([captured[layer] for layer in range(EXPECTED_LAYERS)], dim=0)
    assert acts.shape == (EXPECTED_LAYERS, bsz, max_len, EXPECTED_HIDDEN), (
        f"acts shape {tuple(acts.shape)}"
    )
    peak_t = torch.tensor(peak_layers, dtype=torch.long)
    n_peak = len(peak_layers)
    records: list[dict] = []
    for i, r in enumerate(batch):
        true_len = lengths[i]
        slots = _ordered_slots(r)
        turns = _ordered_turns(r)
        for name, idx in slots:
            assert 0 <= idx < true_len, f"{r.conv_id}: slot {name}={idx} beyond len {true_len}"
        for name, (s, e) in turns:
            assert 1 <= s < e <= true_len, (
                f"{r.conv_id}: span {name}=({s},{e}) invalid for unpadded len {true_len}"
            )
        slot_pos = torch.tensor([idx for _, idx in slots], dtype=torch.long)
        slot_vecs = acts[:, i, slot_pos, :].permute(1, 0, 2).contiguous()
        assert slot_vecs.shape == (len(slots), EXPECTED_LAYERS, EXPECTED_HIDDEN)
        profiles = torch.stack(
            [acts[:, i, s:e, :].float().mean(dim=1) for _, (s, e) in turns], dim=0
        )
        assert profiles.shape == (len(turns), EXPECTED_LAYERS, EXPECTED_HIDDEN)
        perpos = torch.zeros(
            (len(turns), POSITIONS_CAP, n_peak, EXPECTED_HIDDEN), dtype=torch.float16
        )
        perpos_mask = torch.zeros((len(turns), POSITIONS_CAP), dtype=torch.bool)
        for t, (_, (s, e)) in enumerate(turns):
            take = min(POSITIONS_CAP, e - s)
            window = acts[peak_t][:, i, s : s + take, :].permute(1, 0, 2)
            assert window.shape == (take, n_peak, EXPECTED_HIDDEN)
            perpos[t, :take] = window.to(torch.float16)
            perpos_mask[t, :take] = True
        nll = _turn_nll(logits[i], input_ids[i], true_len, turns, r.conv_id, align_state)
        assert nll.shape == (len(turns),)
        records.append(
            {
                "conv_id": r.conv_id,
                "slots": slot_vecs.to(torch.float16),
                "profiles": profiles.to(torch.float16),
                "perpos": perpos,
                "perpos_mask": perpos_mask,
                "nll": nll,
                "spans_meta": {
                    "conv_id": r.conv_id,
                    "format": r.format,
                    "seq_len": true_len,
                    "slot_names": [n for n, _ in slots],
                    "slot_idx": {n: int(v) for n, v in slots},
                    "turn_names": [n for n, _ in turns],
                    "spans": {n: [int(s), int(e)] for n, (s, e) in turns},
                    "meta": r.meta,
                },
            }
        )
    del captured, acts, logits
    return records


def causal_check(
    model, rendered: list[Rendered], atol: float = 1e-2, n_conversations: int = 3
) -> float:
    """Re-forward the prefix ending at each slot; slot activation must match full-seq."""
    device = model.device
    max_diff = 0.0
    n_checked = min(n_conversations, len(rendered))
    for r in rendered[:n_checked]:
        ids = torch.tensor(r.input_ids, dtype=torch.long).unsqueeze(0).to(device)
        full = extract_layer_activations(
            model, ids, layers=range(EXPECTED_LAYERS), detach_to_cpu=True
        )
        for name, idx in _ordered_slots(r):
            pre = extract_layer_activations(
                model, ids[:, : idx + 1], layers=range(EXPECTED_LAYERS), detach_to_cpu=True
            )
            for layer in range(EXPECTED_LAYERS):
                a = pre[layer][0, idx].float()
                b = full[layer][0, idx].float()
                diff = float((a - b).abs().max())
                max_diff = max(max_diff, diff)
                assert torch.allclose(a, b, atol=atol), (
                    f"causal-slot mismatch {r.conv_id}:{name} layer {layer}: "
                    f"max|diff|={diff:.4g} > atol={atol}"
                )
    print(
        f"[causal] slot-prefix equality OK on {n_checked} conversations; max|diff|={max_diff:.4g}"
    )
    return max_diff


def run_extraction(
    model,
    rendered: list[Rendered],
    peak_layers: list[int],
    pad_id: int,
    batch_size: int,
) -> list[dict]:
    """Length-grouped batching with OOM-halving (floor 1); restores input order."""
    order = sorted(range(len(rendered)), key=lambda j: len(rendered[j].input_ids))
    align_state: dict = {}
    results: dict[int, dict] = {}
    bs = batch_size
    pos = 0
    batches_done = 0
    while pos < len(order):
        chunk_idx = order[pos : pos + bs]
        chunk = [rendered[j] for j in chunk_idx]
        try:
            recs = process_batch(model, chunk, peak_layers, pad_id, align_state)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"[oom] CUDA OOM — halving batch size to {bs}")
            continue
        for j, rec in zip(chunk_idx, recs, strict=True):
            results[j] = rec
        pos += len(chunk_idx)
        batches_done += 1
        if batches_done % 10 == 0 or pos >= len(order):
            print(f"[extract] {pos}/{len(order)} conversations done (batch size {bs})")
    return [results[j] for j in range(len(rendered))]


def write_shards(records: list[dict], out_dir: Path, stem: str, sidecar_base: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k in range(0, len(records), SHARD_SIZE):
        shard = records[k : k + SHARD_SIZE]
        shard_idx = k // SHARD_SIZE
        payload = {
            "conv_ids": [r["conv_id"] for r in shard],
            "slots": [r["slots"] for r in shard],
            "profiles": [r["profiles"] for r in shard],
            "perpos": [r["perpos"] for r in shard],
            "perpos_mask": [r["perpos_mask"] for r in shard],
            "nll": [r["nll"] for r in shard],
            "spans_meta": [r["spans_meta"] for r in shard],
        }
        pt_path = out_dir / f"{stem}_shard{shard_idx:03d}.pt"
        torch.save(payload, pt_path)
        coverage = float(torch.cat([r["perpos_mask"].reshape(-1) for r in shard]).float().mean())
        sidecar = dict(sidecar_base)
        sidecar.update(
            {
                "shard_index": shard_idx,
                "n_conversations": len(shard),
                "conv_ids": payload["conv_ids"],
                "shapes": {
                    "slots": [list(r["slots"].shape) for r in shard],
                    "profiles": [list(r["profiles"].shape) for r in shard],
                    "perpos": [list(r["perpos"].shape) for r in shard],
                    "perpos_mask": [list(r["perpos_mask"].shape) for r in shard],
                    "nll": [list(r["nll"].shape) for r in shard],
                },
                "perpos_coverage": coverage,
            }
        )
        json_path = out_dir / f"{stem}_shard{shard_idx:03d}.json"
        json_path.write_text(json.dumps(sidecar, indent=2))
        paths.append(pt_path)
        print(f"[write] {pt_path} ({len(shard)} conversations; perpos coverage {coverage:.3f})")
    return paths


def main() -> None:
    args = parse_args()
    peak_layers = [int(x) for x in str(args.peak_layers).split(",") if x.strip()]
    assert peak_layers, "--peak-layers parsed to an empty list"
    assert all(0 <= p < EXPECTED_LAYERS for p in peak_layers), (
        f"peak layers {peak_layers} out of range"
    )
    convs = _load_conversations(args.conversations)
    if args.smoke:
        convs = convs[:8]
        print(f"[smoke] limiting to {len(convs)} conversations")
    if args.track == "s":
        convs = [to_single_turn(c) for c in convs]
    model, tokenizer, model_id = load_model(args.model, tiny_model_dir=args.tiny_model_dir)
    rendered = [render_conv(c, tokenizer, args.format) for c in convs]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    do_causal = args.assert_causal or args.smoke
    causal_max_diff = causal_check(model, rendered) if do_causal else None
    if args.batch_size == "auto":
        bs = 8
    else:
        bs = int(args.batch_size)
        assert bs >= 1, f"--batch-size must be >= 1, got {bs}"
    if args.smoke:
        bs = min(bs, 2)
    print(
        f"[run] model={args.model} ({model_id}) format={args.format} track={args.track} "
        f"n={len(rendered)} batch_size={bs} peak_layers={peak_layers}"
    )
    records = run_extraction(model, rendered, peak_layers, pad_id, bs)
    stem = f"{args.model}_{args.format}_{args.track}"
    sidecar_base = {
        "model": args.model,
        "model_id": model_id,
        "format": args.format,
        "track": args.track,
        "peak_layers": peak_layers,
        "positions_cap": POSITIONS_CAP,
        "expected_layers": EXPECTED_LAYERS,
        "expected_hidden": EXPECTED_HIDDEN,
        "shard_size": SHARD_SIZE,
        "git_commit": _git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "causal_check_max_abs_diff": causal_max_diff,
        "smoke": bool(args.smoke),
    }
    paths = write_shards(records, args.out_dir, stem, sidecar_base)
    print(f"[done] {len(records)} conversations -> {len(paths)} shard(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
