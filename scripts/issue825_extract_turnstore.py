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
import contextlib
import ctypes
import gc
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

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch import

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
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=(
            "conversations per extraction block == per shard file; each block is "
            "extracted then flushed to disk before the next starts, bounding host "
            "RAM to ~one shard (run-4 rc=137: accumulating all 2000 records "
            "~8 MB each OOM-killed the 16 GB host). Non-default is SMOKE ONLY "
            "(exercises multi-shard flushing on a tiny slice)."
        ),
    )
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
    parser.add_argument(
        "--validate-spans-only",
        action="store_true",
        help=(
            "OFFLINE full-corpus span validation: render EVERY conversation in "
            "--format with the real tokenizer (NO model, GPU-free), report the "
            "zero-width-span drop count + rate, and assert NO kept row trips the "
            "residual hard span/slot checks. Exits 0. Cheap over the full "
            "5,000-row track_s.jsonl — the pre-GPU gate the 8-conv smoke misses."
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
    # device_map={"": 0} pins ALL weights to the GPU: if VRAM is unavailable
    # (e.g. a lingering VLLM::EngineCore from the gen phases still holds it),
    # this raises CUDA OOM at load time instead of device_map="auto" silently
    # offloading ~15 GB of layers to the 16 GB host and getting the process
    # kernel-OOM-killed minutes later (runs 3-4, rc=137).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, (
        f"{model_id}: {len(off_gpu)} params not on CUDA (e.g. {off_gpu[:3]}) — "
        "refusing to run with CPU-offloaded weights (host-RAM OOM risk)"
    )
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


def degenerate_content_turns(r: Rendered) -> list[str]:
    """Required content turns whose span has width < 1 (``s >= e``).

    A very short single-turn row whose whole answer text BPE-merges into the
    naturalistic ``User: ``/``\\n\\n`` plain-text delimiters collapses to a
    zero-width ``(anchor, anchor)`` span (see ``_tokenize_segments_offsets``);
    the profile mean over an empty span is NaN and the downstream
    ``1 <= s < e`` assert crashes. This is the ONE tolerated drop (#825
    crash-fix). Genuinely-impossible cases — a slot index beyond the sequence,
    or a non-zero-width span starting at 0 / out of range — are NOT reported
    here; they stay hard errors at the ``process_batch`` / ``_turn_nll``
    asserts (the last-resort guard). Chat renders never hit this (special-token
    delimiters bracket even a 1-token answer)."""
    return [name for name, (s, e) in r.spans.items() if s >= e]


def partition_rendered(rendered: list[Rendered]) -> tuple[list[Rendered], list[dict]]:
    """Split rendered rows into (kept, dropped).

    A row is DROPPED iff any content turn has a zero-width span
    (``degenerate_content_turns``); everything else is kept. Prints one
    ``[drop] conv_id=<id> reason=zero_width_span:<turns>`` line per drop
    (conv_id + turn names only — never corpus text). Returns the kept rows and
    a list of ``{"conv_id", "turns"}`` drop records for the shard sidecar."""
    kept: list[Rendered] = []
    drops: list[dict] = []
    for r in rendered:
        bad = degenerate_content_turns(r)
        if bad:
            drops.append({"conv_id": r.conv_id, "turns": bad})
            print(f"[drop] conv_id={r.conv_id} reason=zero_width_span:{','.join(bad)}")
        else:
            kept.append(r)
    return kept, drops


def assert_residual_span_integrity(kept: list[Rendered]) -> None:
    """Mirror the ``process_batch`` / ``_turn_nll`` hard span/slot asserts on
    kept rows (no model needed). Zero-width content spans are already filtered
    out by ``partition_rendered``, so anything that trips HERE is a
    genuinely-impossible case (bad slot index, span starting at 0, span out of
    range) — a hard error, never a tolerated drop. Fail-fast before any GPU
    forward and the offline validation gate both call this."""
    for r in kept:
        true_len = len(r.input_ids)
        for name, idx in r.slot_idx.items():
            assert 0 <= idx < true_len, f"{r.conv_id}: slot {name}={idx} beyond len {true_len}"
        for name, (s, e) in r.spans.items():
            assert 1 <= s < e <= true_len, (
                f"{r.conv_id}: span {name}=({s},{e}) invalid for unpadded len {true_len}"
            )


def _finite(t: torch.Tensor, name: str, conv_id: str) -> torch.Tensor:
    """Assert finiteness before storage — silent inf corrupts downstream fits."""
    assert torch.isfinite(t.float()).all(), f"{conv_id}: non-finite values in {name}"
    return t


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
            # Structural slice-bounds check ONLY. (The former targets-vs-
            # row_ids compare was an identity — targets derive from row_ids —
            # and validated nothing; code-review round-1.) Semantic
            # logits<->position alignment rests on extract_layer_activations
            # returning logits from the same forward, an architectural fact.
            assert 1 <= s < e <= true_len, (conv_id, name, s, e, true_len)
            assert 0 <= s - 1 < e - 1 <= token_lp.shape[0], (conv_id, name, s, e)
            align_state["done"] = True
            print(f"[align] NLL slice bounds checked on {conv_id}:{name}")
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
        # Keep activations ON DEVICE: all reductions below (slot gathers,
        # span means, capped per-position windows, NLL) run device-side and
        # only the REDUCED tensors move to CPU — never the full (L,B,T,H)
        # grid (round-3 review: PCIe/CPU bottleneck; code-style rule).
        detach_to_cpu=False,
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
        slot_vecs = acts[:, i, slot_pos.to(acts.device), :].permute(1, 0, 2).contiguous().cpu()
        assert slot_vecs.shape == (len(slots), EXPECTED_LAYERS, EXPECTED_HIDDEN)
        profiles = torch.stack(
            [acts[:, i, s:e, :].float().mean(dim=1) for _, (s, e) in turns], dim=0
        ).cpu()
        assert profiles.shape == (len(turns), EXPECTED_LAYERS, EXPECTED_HIDDEN)
        perpos = torch.zeros(
            (len(turns), POSITIONS_CAP, n_peak, EXPECTED_HIDDEN), dtype=torch.bfloat16
        )
        perpos_mask = torch.zeros((len(turns), POSITIONS_CAP), dtype=torch.bool)
        for t, (_, (s, e)) in enumerate(turns):
            take = min(POSITIONS_CAP, e - s)
            window = acts[peak_t.to(acts.device)][:, i, s : s + take, :].permute(1, 0, 2)
            assert window.shape == (take, n_peak, EXPECTED_HIDDEN)
            perpos[t, :take] = window.to(device="cpu", dtype=torch.bfloat16)
            perpos_mask[t, :take] = True
        nll = _turn_nll(logits[i], input_ids[i], true_len, turns, r.conv_id, align_state)
        assert nll.shape == (len(turns),)
        records.append(
            {
                "conv_id": r.conv_id,
                # bf16, NOT fp16: Qwen-class residual outlier dims can
                # exceed fp16's 65504 max and silently become inf
                # (code-review round-1); bf16 shares fp32's range at the
                # same 2 bytes. Finiteness asserted before storage.
                "slots": _finite(slot_vecs.to(torch.bfloat16), "slots", r.conv_id),
                "profiles": _finite(profiles.to(torch.bfloat16), "profiles", r.conv_id),
                "perpos": _finite(perpos, "perpos", r.conv_id),
                "perpos_mask": perpos_mask,
                "nll": _finite(nll, "nll", r.conv_id),
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


# Cosine-mode bars for causal_check (#1345 crash-fix, att-20260715-151246).
# Calibration source: the #779 r12 bf16 single-position equivalence-gate
# measurement on this exact model family (Qwen-2.5-7B, 28 layers, bf16 —
# gotchas.md "bf16 padded-batch equivalence gates"): bug-free bf16 kernel
# jitter reads per-layer cos >= 0.999995 at layer 0 and >= 0.996907 at the
# worst deep layer (flattened worst 0.998770), while a REAL wrong-position /
# mask / row-mapping bug reads flattened cos 0.39-0.62 and layer-0 cos
# 0.43-0.84. Bars: early per-layer 0.999 (the sharp bug catcher — position
# bugs corrupt layer 0 immediately, where jitter is ~1e-6), flattened 0.995
# (>=4x headroom over the measured worst bf16 deviation, ~0.35 above the
# real-bug regime). The norm-ratio guard closes cosine's scale blind spot
# (a doubled vector has cos 1.0): bf16 jitter norm-ratio is ~1e-3, a scale
# bug is O(1).
CAUSAL_COS_EARLY_LAYERS = 4
CAUSAL_COS_EARLY_MIN = 0.999
CAUSAL_COS_FLAT_MIN = 0.995
CAUSAL_NORM_REL_MAX = 0.05


def _causal_cosine_stats(pre_by_layer: list, full_by_layer: list) -> dict:
    """fp32 cosine/norm stats for ONE slot's prefix-vs-full comparison.

    Returns early_cos_min (per-layer cosine over the first
    CAUSAL_COS_EARLY_LAYERS), flat_cos (all layers concatenated), norm_rel
    (flattened norm ratio abs(norm(pre) - norm(full)) / norm(full)), and max_abs_diff.
    """
    pre = torch.stack([v.float() for v in pre_by_layer])
    full = torch.stack([v.float() for v in full_by_layer])
    per_layer_cos = torch.nn.functional.cosine_similarity(pre, full, dim=1)
    n_early = min(CAUSAL_COS_EARLY_LAYERS, per_layer_cos.shape[0])
    flat_cos = torch.nn.functional.cosine_similarity(
        pre.reshape(1, -1), full.reshape(1, -1), dim=1
    )[0]
    norm_rel = float((pre.norm() - full.norm()).abs() / full.norm().clamp_min(1e-12))
    return {
        "early_cos_min": float(per_layer_cos[:n_early].min()),
        "flat_cos": float(flat_cos),
        "norm_rel": norm_rel,
        "max_abs_diff": float((pre - full).abs().max()),
    }


def causal_check(
    model,
    rendered: list[Rendered],
    atol: float = 1e-2,
    n_conversations: int = 3,
    *,
    mode: str = "abs",
) -> float:
    """Re-forward the prefix ending at each slot; slot activation must match full-seq.

    ``mode="abs"`` (default — byte-identical #825 behavior): per-layer
    ``torch.allclose(atol)``, calibrated on the #825 header slots (mid/late
    positions, small prefix-vs-full length disparity). ``mode="cosine"``: the
    #779-calibrated two-bar cosine gate + norm-ratio guard, for slot sets where
    a flat atol has NO bf16 headroom — early-position slots (#1345's ``prefix``
    at token ~2: a 3-token prefix forward meets the full-length forward's
    different GEMM shapes, and a SINGLE bf16 ULP at the large-magnitude
    early-token dims reads 0.03125/0.0625 at layer 0; incident
    att-20260715-151246, benign-numerics-verified by fp32 re-probe). Both
    forwards are batch-1, unpadded, same slot index — the compare is pure
    kernel numerics, so a wrong-position bug reads cos ~0.4-0.6, far below
    either bar.
    """
    assert mode in ("abs", "cosine"), f"unknown causal_check mode: {mode!r}"
    device = model.device
    max_diff = 0.0
    worst = {"early_cos_min": 1.0, "flat_cos": 1.0, "norm_rel": 0.0}
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
            if mode == "abs":
                for layer in range(EXPECTED_LAYERS):
                    a = pre[layer][0, idx].float()
                    b = full[layer][0, idx].float()
                    diff = float((a - b).abs().max())
                    max_diff = max(max_diff, diff)
                    assert torch.allclose(a, b, atol=atol), (
                        f"causal-slot mismatch {r.conv_id}:{name} layer {layer}: "
                        f"max|diff|={diff:.4g} > atol={atol}"
                    )
            else:
                stats = _causal_cosine_stats(
                    [pre[layer][0, idx] for layer in range(EXPECTED_LAYERS)],
                    [full[layer][0, idx] for layer in range(EXPECTED_LAYERS)],
                )
                max_diff = max(max_diff, stats["max_abs_diff"])
                worst["early_cos_min"] = min(worst["early_cos_min"], stats["early_cos_min"])
                worst["flat_cos"] = min(worst["flat_cos"], stats["flat_cos"])
                worst["norm_rel"] = max(worst["norm_rel"], stats["norm_rel"])
                assert (
                    stats["early_cos_min"] >= CAUSAL_COS_EARLY_MIN
                    and stats["flat_cos"] >= CAUSAL_COS_FLAT_MIN
                    and stats["norm_rel"] <= CAUSAL_NORM_REL_MAX
                ), (
                    f"causal-slot mismatch {r.conv_id}:{name} (cosine mode): "
                    f"early_cos_min={stats['early_cos_min']:.6f} (min {CAUSAL_COS_EARLY_MIN}) "
                    f"flat_cos={stats['flat_cos']:.6f} (min {CAUSAL_COS_FLAT_MIN}) "
                    f"norm_rel={stats['norm_rel']:.4g} (max {CAUSAL_NORM_REL_MAX}) "
                    f"max|diff|={stats['max_abs_diff']:.4g}"
                )
    if mode == "abs":
        print(
            f"[causal] slot-prefix equality OK on {n_checked} conversations; "
            f"max|diff|={max_diff:.4g}"
        )
    else:
        print(
            f"[causal] mode=cosine slot-prefix consistency OK on {n_checked} conversations; "
            f"early_cos_min={worst['early_cos_min']:.6f} flat_cos_min={worst['flat_cos']:.6f} "
            f"norm_rel_max={worst['norm_rel']:.4g} max|diff|={max_diff:.4g}"
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


def write_shards(
    records: list[dict],
    out_dir: Path,
    stem: str,
    sidecar_base: dict,
    shard_offset: int = 0,
    shard_size: int = SHARD_SIZE,
) -> list[Path]:
    """Write records as fp .pt shard(s) + JSON sidecars, starting at shard_offset.

    Called once per extraction block (block == shard), so records never
    accumulate across blocks in host RAM. An empty records list writes nothing
    (main never passes one — every block is non-empty by construction).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k in range(0, len(records), shard_size):
        shard = records[k : k + shard_size]
        shard_idx = shard_offset + k // shard_size
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


def _validate_spans_only(args: argparse.Namespace) -> None:
    """OFFLINE full-corpus span validation (no model). Renders every row in the
    requested format with the real tokenizer, reports the zero-width-span drop
    count + rate, and asserts every kept row passes the residual hard span/slot
    checks. Prints conv_ids + counts only — never corpus text."""
    from transformers import AutoTokenizer

    if args.tiny_model_dir:
        tok_src = args.tiny_model_dir
    else:
        tok_src = MODEL_INSTRUCT if args.model == "instruct" else MODEL_PRETRAINED
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    convs = _load_conversations(args.conversations)
    if args.track == "s":
        convs = [to_single_turn(c) for c in convs]
    rendered_all = [render_conv(c, tokenizer, args.format) for c in convs]
    kept, drops = partition_rendered(rendered_all)
    rate = len(drops) / len(rendered_all) if rendered_all else 0.0
    # The load-bearing gate: no kept row may trip the consumer's hard asserts.
    assert_residual_span_integrity(kept)
    print(
        f"[validate-spans] format={args.format} track={args.track} "
        f"tokenizer={tok_src} n={len(rendered_all)} kept={len(kept)} "
        f"dropped={len(drops)} rate={rate:.4f} residual_hard_assert=PASS"
    )


def main() -> None:
    args = parse_args()
    if args.validate_spans_only:
        _validate_spans_only(args)
        return
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
    rendered_all = [render_conv(c, tokenizer, args.format) for c in convs]
    # Drop degenerate zero-width-span rows (short single-turn answers that
    # BPE-merge entirely into the naturalistic plain-text delimiters, #825) and
    # report the rate; the remaining rows must pass the residual hard span/slot
    # checks (fail-fast before any GPU forward). Chat renders never drop.
    rendered, drops = partition_rendered(rendered_all)
    if drops:
        rate = len(drops) / len(rendered_all)
        print(
            f"[drops] {len(drops)} of {len(rendered_all)} rows dropped "
            f"(zero-width {args.format} spans); rate={rate:.4f}"
        )
    assert rendered, (
        f"all {len(rendered_all)} rendered rows dropped as zero-width — a "
        f"systematic {args.format} render bug, not a handful of degenerate rows"
    )
    assert_residual_span_integrity(rendered)
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
    shard_size = int(args.shard_size)
    assert shard_size >= 1, f"--shard-size must be >= 1, got {shard_size}"
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
        "shard_size": shard_size,
        "git_commit": _git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "causal_check_max_abs_diff": causal_max_diff,
        "smoke": bool(args.smoke),
        # Degenerate zero-width-span rows dropped from extraction (#825). Recorded
        # in EVERY shard sidecar (sidecar_base is spread into each). The contrast
        # script intersects by conv_id, so dropped naturalistic rows stay paired.
        "n_rendered_pre_filter": len(rendered_all),
        "n_dropped_zero_width": len(drops),
        "dropped_conv_ids": [d["conv_id"] for d in drops],
        "dropped_turns": {d["conv_id"]: d["turns"] for d in drops},
    }
    # Block-wise extract -> flush: one input-order block == one shard file,
    # written the moment its block completes, so host RAM holds at most ~one
    # shard of records (~4 GB at the production 500) instead of the full set
    # (2000 x ~8 MB ~= 16 GB, which kernel-OOM-killed the 16 GB g2-standard-4
    # in run 4 with zero shards flushed). Shard indices/naming/order are
    # identical to the old write-at-end path; the fit_cells sorted-glob loader
    # is unchanged.
    paths: list[Path] = []
    n_done = 0
    for block_idx, block_start in enumerate(range(0, len(rendered), shard_size)):
        block = rendered[block_start : block_start + shard_size]
        records = run_extraction(model, block, peak_layers, pad_id, bs)
        assert len(records) == len(block), (block_idx, len(records), len(block))
        paths += write_shards(
            records,
            args.out_dir,
            stem,
            sidecar_base,
            shard_offset=block_idx,
            shard_size=shard_size,
        )
        n_done += len(records)
        del records, block
        gc.collect()
        # Return freed arena pages to the OS. The per-conv ~7 MB CPU tensors sit
        # under glibc's dynamic mmap threshold, so without an explicit trim the
        # freed blocks stay in the arena free lists and RSS climbs monotonically
        # across blocks (run 5: 14.9 GiB anon RSS ~= 2 fully-retained flushed
        # blocks + a partial third, despite `del`). Portability no-op guard only
        # (non-glibc); this is an allocator hint, not a correctness path.
        with contextlib.suppress(OSError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    print(f"[done] {n_done} conversations -> {len(paths)} shard(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
