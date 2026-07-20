#!/usr/bin/env python
"""Issue #825 ``kresample-user`` Phase 2: capture v(u2) for orig + K fresh draws.

For each context (the onpolicy instruct-allowlist, n=1914) capture the per-turn
u2 profile v(u2) for FIVE draws: draw 0 = the ORIGINAL parent Haiku u2 (from
conversations.jsonl), draws 1..K = the fresh Haiku redraws (from draws.jsonl).
v(u2) = per-layer mean residual over the u2 content span at the frozen layers,
byte-compatible with the parent turnstore convention (render_chat span rule +
process_batch's ``acts[:, i, s:e, :].float().mean(dim=1).to(bfloat16)`` pooling,
FROZEN_LAYERS, bf16 storage). Reused verbatim: ``render_chat`` (span rule),
``load_model`` (device/asserts), ``extract_layer_activations`` (hidden states).

Draw 0 (recaptured original, MY rig) enables the #1482-G2 exchangeability read
(orig vs fresh distribution) WITHOUT a cross-rig confound. A separate optional
``--parity-shard`` cross-check streams the parent's STORED v(u2) for the shard's
contexts and asserts my-rig-orig ~= parent-stored (bf16 kernel jitter), validating
the rig reproduces the producer.

GPU-bound: teacher-forced Qwen-2.5-7B bf16 forwards on 1 GPU. CPU smoke via
``--tiny-model-dir`` (built by ``--build-tiny``) exercises render/extract/pool/
store on a 2-layer from-config Qwen2 with the real tokenizer.

TEXT DISCIPLINE: never prints turn text — only counts, shapes, cosines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.extraction import extract_layer_activations
from explore_persona_space.experiments.issue_825.common import (
    FROZEN_LAYERS,
    HF_DATA_REPO,
)
from explore_persona_space.orchestrate.env import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue825_extract_turnstore import load_model  # noqa: E402
from issue825_render_formats import render_chat  # noqa: E402

HF_PREFIX = "issue825_kresample_user"
MAX_CAPTURE_LEN = 4096  # renders above this are flagged + skipped (expect ~0)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _load_convs(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["conv_id"])] = r
    return out


def _load_draws(path: Path) -> dict[int, dict[int, str]]:
    """conv_id -> {draw_k: u2} for non-error fresh draws."""
    out: dict[int, dict[int, str]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("error") and isinstance(r.get("u2"), str) and r["u2"].strip():
                out[int(r["conv_id"])] = out.get(int(r["conv_id"]), {})
                out[int(r["conv_id"])][int(r["draw"])] = r["u2"]
    return out


def _render_and_span(tokenizer, conv_id: int, u1: str, a1: str, u2: str):
    """Return (rendered, (s,e)) or (None, None) if the u2 content span is degenerate
    or the render exceeds MAX_CAPTURE_LEN. Mirrors the extractor's span asserts."""
    if not (isinstance(u2, str) and u2.strip()):
        return None, None
    r = render_chat({"conv_id": conv_id, "u1": u1, "a1": a1, "u2": u2}, tokenizer)
    n = len(r.input_ids)
    s, e = r.spans.get("u2", (0, 0))
    if not (1 <= s < e <= n) or n > MAX_CAPTURE_LEN:
        return None, None
    return r, (s, e)


def capture_reader(model, tokenizer, contexts, draws_by_ctx, frozen, batch_size, k):
    """Capture V (n_ctx, k+1, len(frozen), H) bf16 + valid mask (n_ctx, k+1).

    draw 0 = original u2; draws 1..k = fresh. Batched teacher-forced forwards
    (sorted by length); only the requested frozen layers are captured.
    """
    n_ctx = len(contexts)
    n_draws = k + 1
    H = int(model.config.hidden_size)  # real model hidden (3584 for 7B; small for the tiny stub)
    V = torch.zeros((n_ctx, n_draws, len(frozen), H), dtype=torch.bfloat16)
    mask = torch.zeros((n_ctx, n_draws), dtype=torch.bool)
    device = model.device

    # Build the flat work list of renderable (ctx_idx, draw_idx, rendered, span).
    work = []
    for ci, (cid, u1, a1, orig_u2, fresh) in enumerate(contexts):
        u2s = {0: orig_u2}
        u2s.update({dk: fresh.get(dk) for dk in range(1, k + 1)})
        for di, u2 in u2s.items():
            r, span = _render_and_span(tokenizer, cid, u1, a1, u2) if u2 else (None, None)
            if r is not None:
                work.append((ci, di, r, span))
    work.sort(key=lambda w: len(w[2].input_ids))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for pos in range(0, len(work), batch_size):
        chunk = work[pos : pos + batch_size]
        lengths = [len(w[2].input_ids) for w in chunk]
        max_len = max(lengths)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, w in enumerate(chunk):
            input_ids[i, : lengths[i]] = torch.tensor(w[2].input_ids, dtype=torch.long)
            attn[i, : lengths[i]] = 1
        with torch.no_grad():
            captured = extract_layer_activations(
                model,
                input_ids.to(device),
                layers=list(frozen),
                attention_mask=attn.to(device),
                return_logits=False,
                detach_to_cpu=False,
            )
        for i, (ci, di, r, (s, e)) in enumerate(chunk):
            # Verbatim producer pooling: per-layer mean over the u2 content span,
            # fp32 accumulate then round to bf16 (process_batch parity).
            prof = torch.stack(
                [captured[L][i, s:e, :].float().mean(dim=0) for L in frozen], dim=0
            ).to(torch.bfloat16)
            V[ci, di] = prof.cpu()
            mask[ci, di] = True
        del captured
    return V, mask


def _stream_parent_shard(shard_pt: Path, shard_json: Path, u2_turn_index: int, frozen):
    """Load the parent turnstore shard's STORED v(u2) at frozen layers.

    Returns {conv_id: tensor(len(frozen), H) bf16}. The shard stores profiles for
    all 28 layers over turns [u1,a1,u2,a2]; profiles[u2_turn_index] is v(u2)."""
    side = json.loads(shard_json.read_text())
    conv_ids = [int(c) for c in side["conv_ids"]]
    records = torch.load(shard_pt, map_location="cpu", weights_only=False)
    out = {}
    for cid, rec in zip(conv_ids, records, strict=True):
        prof = rec["profiles"]  # (n_turns, 28, H)
        out[cid] = prof[u2_turn_index][list(frozen), :].to(torch.bfloat16)
    return out


def run_parity_check(args, contexts, frozen):
    """Cross-check: my-rig recaptured original v(u2) vs parent STORED v(u2)."""
    from huggingface_hub import hf_hub_download

    model, tokenizer, model_id = load_model(args.reader, tiny_model_dir=args.tiny_model_dir)
    idx = {
        cid: (u1, a1, orig)
        for cid, u1, a1, orig, _ in [(c[0], c[1], c[2], c[3], c[4]) for c in contexts]
    }
    stored = {}
    base = (
        f"issue825_userbase_map/analysis_tensors/{args.reader}_chat_m_shard{args.parity_shard:03d}"
    )
    from explore_persona_space.orchestrate import hub

    pt = Path(
        hub.retry_transient(
            lambda: hf_hub_download(HF_DATA_REPO, base + ".pt", repo_type="dataset"),
            what=f"parity shard {base}.pt",
        )
    )
    sj = Path(
        hub.retry_transient(
            lambda: hf_hub_download(HF_DATA_REPO, base + ".json", repo_type="dataset"),
            what=f"parity shard {base}.json",
        )
    )
    stored = _stream_parent_shard(pt, sj, u2_turn_index=2, frozen=frozen)
    cids = [c for c in stored if c in idx]
    cosines = []
    for cid in cids:
        u1, a1, orig = idx[cid]
        r, span = _render_and_span(tokenizer, cid, u1, a1, orig)
        if r is None:
            continue
        s, e = span
        with torch.no_grad():
            cap = extract_layer_activations(
                model,
                torch.tensor([r.input_ids]).to(model.device),
                layers=list(frozen),
                return_logits=False,
                detach_to_cpu=False,
            )
        mine = torch.stack([cap[L][0, s:e, :].float().mean(0) for L in frozen], 0)
        theirs = stored[cid].float()
        # layer-19 headline cosine (frozen index of 19)
        li = list(frozen).index(19)
        cos = torch.nn.functional.cosine_similarity(mine[li], theirs[li], dim=0).item()
        cosines.append(cos)
    cosines = np.array(cosines, dtype=np.float64)
    res = {
        "reader": args.reader,
        "parity_shard": args.parity_shard,
        "n_compared": int(len(cosines)),
        "layer19_cosine_min": float(cosines.min()) if len(cosines) else None,
        "layer19_cosine_mean": float(cosines.mean()) if len(cosines) else None,
        "layer19_cosine_p05": float(np.percentile(cosines, 5)) if len(cosines) else None,
    }
    print(
        f"[parity] {args.reader} shard{args.parity_shard}: n={res['n_compared']} "
        f"L19 cos min={res['layer19_cosine_min']} mean={res['layer19_cosine_mean']}"
    )
    return res


def build_tiny(dst: str) -> None:
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # 28 layers (== EXPECTED_LAYERS) so the frozen read-out layers 14/18/19/26
    # exist; hidden 64 keeps it a tiny CPU stub.
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    )
    m = Qwen2ForCausalLM(cfg)
    Path(dst).mkdir(parents=True, exist_ok=True)
    m.save_pretrained(dst)
    tok.save_pretrained(dst)
    print(f"[tiny] built 28-layer hidden-64 Qwen2 stub -> {dst}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conversations",
        type=Path,
        default=Path("data/issue_825/kresample_user/inputs/conversations.jsonl"),
    )
    ap.add_argument(
        "--allowlists",
        type=Path,
        default=Path("data/issue_825/kresample_user/inputs/row_allowlists.json"),
    )
    ap.add_argument("--draws", type=Path, default=Path("data/issue_825/kresample_user/draws.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_825/kresample_user"))
    ap.add_argument("--reader", choices=("instruct", "pretrained"), default="instruct")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--allowlist-key", default="M_instruct_user_chat")
    ap.add_argument("--tiny-model-dir", default=None)
    ap.add_argument("--build-tiny", default=None, help="build a tiny Qwen2 at this dir and exit")
    ap.add_argument("--smoke-n", type=int, default=0, help="cap contexts (0=all)")
    ap.add_argument(
        "--parity-shard",
        type=int,
        default=None,
        help="run only the parent-parity cross-check on this shard",
    )
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    if args.build_tiny:
        build_tiny(args.build_tiny)
        return

    frozen = tuple(FROZEN_LAYERS)
    convs = _load_convs(args.conversations)
    allow = json.loads(args.allowlists.read_text())[args.allowlist_key]
    conv_ids = [int(c) for c in allow]
    if args.smoke_n:
        conv_ids = conv_ids[: args.smoke_n]
    draws = _load_draws(args.draws) if args.draws.exists() else {}
    contexts = []
    for cid in conv_ids:
        r = convs[cid]
        contexts.append((cid, r["u1"], r["a1"], r.get("u2"), draws.get(cid, {})))
    print(f"[cap] reader={args.reader} contexts={len(contexts)} k={args.k} frozen={frozen}")

    if args.parity_shard is not None:
        res = run_parity_check(args, contexts, frozen)
        (args.out_dir / f"parity_{args.reader}_shard{args.parity_shard:03d}.json").write_text(
            json.dumps(res, indent=2) + "\n"
        )
        return

    model, tokenizer, model_id = load_model(args.reader, tiny_model_dir=args.tiny_model_dir)
    t0 = time.time()
    V, mask = capture_reader(model, tokenizer, contexts, draws, frozen, args.batch_size, args.k)
    wall = time.time() - t0

    n_full_fresh = int((mask[:, 1:].sum(1) == args.k).sum())
    n_all = int((mask.sum(1) == args.k + 1).sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / f"vu2_{args.reader}.pt"
    torch.save(
        {
            "V": V,
            "valid_mask": mask,
            "conv_ids": conv_ids,
            "draw_labels": ["orig"] + [f"d{i}" for i in range(1, args.k + 1)],
            "frozen_layers": list(frozen),
            "reader": args.reader,
            "model_id": model_id,
        },
        out_pt,
    )
    meta = {
        "followup_label": "kresample-user",
        "phase": "capture",
        "reader": args.reader,
        "model_id": model_id,
        "n_contexts": len(contexts),
        "k_draws": args.k,
        "frozen_layers": list(frozen),
        "V_shape": list(V.shape),
        "dtype": "bfloat16",
        "n_contexts_all_fresh_valid": n_full_fresh,
        "n_contexts_all_5_valid": n_all,
        "pooling": "per-layer mean over u2 content span, fp32 accumulate -> bf16 (process_batch parity)",
        "render": "render_chat 3-turn (u1,a1,u2); v(u2)=profiles[u2] span mean",
        "wall_seconds": round(wall, 1),
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiny": bool(args.tiny_model_dir),
        "smoke_n": args.smoke_n,
    }
    meta_path = args.out_dir / f"vu2_{args.reader}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[cap] {args.reader}: V{tuple(V.shape)} all_fresh_valid={n_full_fresh}/{len(contexts)} "
        f"all5={n_all} wall={wall:.1f}s -> {out_pt}"
    )

    if args.tiny_model_dir or args.no_upload:
        print("[cap] skipping HF upload")
        return
    from explore_persona_space.orchestrate import hub

    hub._upload(
        out_pt,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/analysis_tensors/vu2_{args.reader}.pt",
        upload_as_file=True,
    )
    hub._upload(
        meta_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/analysis_tensors/vu2_{args.reader}_meta.json",
        upload_as_file=True,
    )
    print(f"[cap] uploaded -> {HF_DATA_REPO}/{HF_PREFIX}/analysis_tensors/")


if __name__ == "__main__":
    main()
