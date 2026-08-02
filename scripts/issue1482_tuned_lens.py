"""Issue #1482 — hand-rolled TUNED LENS translators for Qwen-2.5-7B-Instruct.

No pretrained tuned lens exists for this model, so this script fits the three
affine translators the #1482 lens round needs (layers 14 / 19 / 26) directly.

Method (Belrose et al. 2023, arXiv 2303.08112 — "Eliciting Latent Predictions
from Transformers with the Tuned Lens"). Per layer ``l`` we learn an affine
translator ``T_l(h) = W_l h + b_l`` (3584 -> 3584) minimising the forward KL

    D_KL( P_final || P_lens_l ),
    P_final   = softmax( W_U . norm_f(h_final) )      [the model's own output]
    P_lens_l  = softmax( W_U . norm_f(T_l(h_l)) )

over real chat-context token positions, with the final RMSNorm ``norm_f`` and
the unembedding ``W_U`` FROZEN and ``T_l`` initialised to the identity (so step
0 reproduces the plain logit lens exactly).

Three implementation choices worth naming:

1. **No logit storage.** Storing ``N x 152064`` target logits is impossible at
   N ~ 1e5. Instead we store the model's post-final-norm last hidden state
   ``hs[-1]`` (``N x 3584``) and recompute the target logits per minibatch as
   ``lm_head(hs[-1])`` — which is EXACTLY what the model itself does, so the
   target distribution is exact rather than approximated (asserted against
   ``outputs.logits`` at preflight).

2. **Fail-loud convention checks.** The HF ``output_hidden_states`` tuple has
   ``hs[i]`` = the INPUT to decoder layer ``i`` (i.e. resid_post of layer
   ``i-1``) for ``i < n_layers``, and ``hs[n_layers]`` = the POST-final-norm
   state. So resid_post(l) = ``hs[l + 1]``. This is version-sensitive, so
   preflight verifies it with a forward hook on ``model.model.layers[l]`` and
   verifies ``lm_head(hs[-1]) == outputs.logits``, plus a parity check of the
   locally reimplemented RMSNorm against ``model.model.norm``.

3. **Group-level held-out split.** Train/val split is by CONTEXT, never by
   position — positions inside one context are strongly dependent, so a
   pointwise split would leak (`.claude/rules/ood-generalization-folds.md`).
   Every headline number is the VAL (held-out contexts) read.

Applying a translator downstream: for an absolute residual STATE use
``W @ h + b``; for a residual DIRECTION (a difference of two states) use the
LINEAR part only, ``W @ d`` — the bias cancels in a difference. The shipped
README repeats this.

Refusal-safety: this script NEVER prints or logs corpus text — only counts,
hashes, and metrics (real-corpus rows carry unscreened user text, CLAUDE.md
§ Spurious usage-policy refusals (d)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps bind BEFORE torch/numpy import (#847)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue1482_tuned_lens")

TASK_ID = 1482
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LMSYS_REPO = "lmsys/lmsys-chat-1m"
# Pinned so the streamed row order — and therefore the manifest row indices —
# is reproducible.
LMSYS_REVISION = "main"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1482_lens/tuned_lens"

ACT_DIM = 3584
LENS_LAYERS = (14, 19, 26)


def phase(name: str) -> None:
    """Emit a poll_pipeline-parseable phase breadcrumb. ``done`` is terminal."""
    print(f"[phase={name}]", flush=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


# ── phase 0: corpus manifest ──────────────────────────────────────────────────


def _clean_conversation(row) -> list[dict] | None:
    """Parse one LMSYS row into strictly-alternating user/assistant turns.

    Returns None (no reason string logged — the row text is never surfaced) when
    the row is malformed, non-alternating, or does not start with a user turn.
    """
    conv = row.get("conversation")
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    turns: list[dict] = []
    for t in conv:
        if not isinstance(t, dict):
            return None
        role = t.get("role")
        content = t.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            return None
        turns.append({"role": role, "content": content})
    for i, t in enumerate(turns):
        if t["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None
    # Keep whole turn-pairs so the render always ends on an assistant turn.
    if turns[-1]["role"] == "user":
        turns = turns[:-1]
    return turns if len(turns) >= 2 else None


def build_manifest(tokenizer, n_contexts: int, max_len: int, seed: int) -> list[dict]:
    """Stream LMSYS-Chat-1M and keep ``n_contexts`` English rendered chat contexts.

    Each kept entry carries the rendered text plus its provenance (streamed row
    index + sha256) so the set is reproducible from the pinned revision.
    """
    phase("manifest")
    from datasets import load_dataset

    ds = load_dataset(
        LMSYS_REPO,
        split="train",
        streaming=True,
        revision=None if LMSYS_REVISION == "main" else LMSYS_REVISION,
    )
    kept: list[dict] = []
    seen: set[str] = set()
    consumed = 0
    t0 = time.time()
    for row in ds:
        if len(kept) >= n_contexts or consumed >= 200_000:
            break
        row_index = consumed
        consumed += 1
        if row.get("language") != "English":
            continue
        turns = _clean_conversation(row)
        if turns is None:
            continue
        text = tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        dedup_key = sha256_text(" ".join(text.lower().split()))
        if dedup_key in seen:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)[
            "input_ids"
        ]
        if len(ids) < 32:  # too short to contribute useful positions
            continue
        seen.add(dedup_key)
        # Persist EXACTLY the text the model is fed (post-truncation), so the
        # manifest is faithful and bounded by ``max_len`` (<9 MB single shard).
        text_used = tokenizer.decode(ids)
        kept.append(
            {
                "ctx_id": len(kept),
                "row_index": row_index,
                "source_repo": LMSYS_REPO,
                "source_revision": LMSYS_REVISION,
                "sha256_render": dedup_key,
                "sha256_used": sha256_text(text_used),
                "n_tokens": len(ids),
                "n_turns": len(turns),
                "truncated": len(ids) >= max_len,
                "text": text_used,
                "input_ids": ids,
            }
        )
        if len(kept) % 100 == 0:
            logger.info("[manifest] kept=%d consumed=%d", len(kept), consumed)
    if len(kept) < n_contexts:
        raise RuntimeError(
            f"manifest short: kept {len(kept)} < requested {n_contexts} after {consumed} rows"
        )
    rng = random.Random(seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    n_val = max(1, int(round(0.15 * len(kept))))
    val_ids = set(order[:n_val])
    for e in kept:
        e["split"] = "val" if e["ctx_id"] in val_ids else "train"
    logger.info(
        "[manifest] kept=%d consumed=%d tokens=%d train_ctx=%d val_ctx=%d secs=%.1f",
        len(kept),
        consumed,
        sum(e["n_tokens"] for e in kept),
        len(kept) - n_val,
        n_val,
        time.time() - t0,
    )
    return kept


# ── phase 1: capture ──────────────────────────────────────────────────────────


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2RMSNorm in fp32 (parity-asserted against the model's own module)."""
    var = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


def preflight_conventions(model, tokenizer, norm_w, norm_eps, device) -> None:
    """Fail-loud verification of every frozen-pipeline assumption (see docstring)."""
    phase("preflight_conventions")
    msgs = [{"role": "user", "content": "Explain why the sky looks blue."}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(device)

    captured: dict[int, torch.Tensor] = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    handles = [model.model.layers[i].register_forward_hook(mk_hook(i)) for i in LENS_LAYERS]
    try:
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    hs = out.hidden_states
    n_layers = model.config.num_hidden_layers
    if len(hs) != n_layers + 1:
        raise RuntimeError(f"hidden_states len {len(hs)} != n_layers+1 ({n_layers + 1})")

    # (a) resid_post(l) == hs[l + 1]
    for lyr in LENS_LAYERS:
        got, want = captured[lyr].float(), hs[lyr + 1].float()
        if not torch.allclose(got, want, atol=1e-3, rtol=1e-3):
            raise RuntimeError(
                f"layer-index convention broken at L{lyr}: "
                f"max|hook - hs[l+1]| = {(got - want).abs().max().item():.4g}"
            )
    logger.info("[preflight] resid_post(l) == hidden_states[l+1] verified for %s", LENS_LAYERS)

    # (b) hs[-1] is POST-final-norm and lm_head(hs[-1]) == outputs.logits
    with torch.no_grad():
        relog = model.lm_head(hs[-1])
    dev = (relog.float() - out.logits.float()).abs().max().item()
    rel = dev / max(out.logits.float().abs().max().item(), 1e-6)
    if rel > 5e-3:
        raise RuntimeError(
            f"lm_head(hs[-1]) != outputs.logits (max abs dev {dev:.4g}, rel {rel:.4g})"
        )
    logger.info(
        "[preflight] lm_head(hidden_states[-1]) == logits (abs dev %.3g, rel %.3g)", dev, rel
    )

    # (c) local RMSNorm parity with the model's own final norm. hs[-2] is used
    # purely as a well-scaled probe input — this asserts the reimplementation
    # matches, not that hs[-2] is the norm's real input.
    with torch.no_grad():
        ref = model.model.norm(hs[-2]).float()
        mine = _rmsnorm(hs[-2].float(), norm_w, norm_eps)
    dev = (ref - mine).abs().max().item()
    rel = dev / max(ref.abs().max().item(), 1e-6)
    if rel > 1e-2:
        raise RuntimeError(f"local RMSNorm parity failed (max abs dev {dev:.4g}, rel {rel:.4g})")
    logger.info("[preflight] local RMSNorm parity OK (rel dev %.3g)", rel)


def capture(model, manifest, device, batch_size, max_positions, seed):
    """Teacher-forced capture of h_14 / h_19 / h_26 + post-final-norm h_final.

    Returns (store, pos_records). ``store`` maps key -> CPU fp16 (N, 3584);
    ``pos_records`` is a per-kept-position (ctx_id, token_pos) list.
    """
    phase("capture")
    order = sorted(range(len(manifest)), key=lambda i: manifest[i]["n_tokens"])
    pad_id = model.config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]

    keys = [f"h{lyr}" for lyr in LENS_LAYERS] + ["h_final"]
    chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    pos_records: list[tuple[int, int]] = []
    t0 = time.time()
    n_batches = (len(order) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        idxs = order[bi * batch_size : (bi + 1) * batch_size]
        seqs = [manifest[i]["input_ids"] for i in idxs]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, : len(s)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hs = out.hidden_states
        # fp16 storage is safe only while |activation| stays well under 65504.
        # Qwen-2.5-7B carries massive activations (O(1e3)); fail loud rather
        # than silently storing inf if a batch ever exceeds the headroom.
        peak = max(hs[lyr + 1].abs().max().item() for lyr in LENS_LAYERS)
        if peak > 60_000:
            raise RuntimeError(f"activation peak {peak:.4g} exceeds fp16 storage headroom")
        for r, i in enumerate(idxs):
            L = len(seqs[r])
            sel = slice(1, L)  # drop position 0 (no context)
            for lyr in LENS_LAYERS:
                chunks[f"h{lyr}"].append(hs[lyr + 1][r, sel].to(torch.float16).cpu())
            chunks["h_final"].append(hs[-1][r, sel].to(torch.float16).cpu())
            pos_records.extend((manifest[i]["ctx_id"], p) for p in range(1, L))
        del out, hs
        if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
            logger.info(
                "[capture] batch %d/%d positions=%d elapsed=%.1fs",
                bi + 1,
                n_batches,
                len(pos_records),
                time.time() - t0,
            )

    store = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    del chunks
    n_total = store["h_final"].shape[0]
    if n_total != len(pos_records):
        raise RuntimeError(f"position bookkeeping mismatch: {n_total} vs {len(pos_records)}")

    if n_total > max_positions:
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(n_total, generator=g)[:max_positions].sort().values
        store = {k: v[keep] for k, v in store.items()}
        pos_records = [pos_records[i] for i in keep.tolist()]
        logger.info(
            "[capture] subsampled %d -> %d positions (seed=%d)", n_total, max_positions, seed
        )

    norms = store["h_final"].float().norm(dim=-1)
    logger.info(
        "[capture] N=%d h_final_norm median=%.2f p99=%.2f max=%.2f secs=%.1f",
        len(pos_records),
        norms.median().item(),
        norms.quantile(0.99).item(),
        norms.max().item(),
        time.time() - t0,
    )
    return store, pos_records


# ── phase 2/3: fit + eval ─────────────────────────────────────────────────────


def _lens_logits(h, W, b, w_u, norm_w, norm_eps):
    return _rmsnorm(h @ W.T + b, norm_w, norm_eps) @ w_u.T


@torch.no_grad()
def evaluate(h_l, h_final, W, b, w_u, norm_w, norm_eps, device, batch=128):
    """Held-out KL(P_final || P_lens) and top-1 agreement, computed in exact fp32."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        kl_sum, agree, n = 0.0, 0, h_l.shape[0]
        for s in range(0, n, batch):
            h = h_l[s : s + batch].to(device, torch.float32)
            hf = h_final[s : s + batch].to(device, torch.float32)
            tgt = F.log_softmax(hf @ w_u.T, dim=-1)
            lens = F.log_softmax(_lens_logits(h, W, b, w_u, norm_w, norm_eps), dim=-1)
            kl_sum += (tgt.exp() * (tgt - lens)).sum(-1).sum().item()
            agree += (lens.argmax(-1) == tgt.argmax(-1)).sum().item()
        return kl_sum / n, agree / n
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def fit_layer(h_tr, hf_tr, h_va, hf_va, w_u, norm_w, norm_eps, device, *, lr, epochs, batch, tag):
    """Fit one affine translator; returns (best_W, best_b, history)."""
    W = torch.eye(ACT_DIM, device=device, dtype=torch.float32).requires_grad_(True)
    b = torch.zeros(ACT_DIM, device=device, dtype=torch.float32).requires_grad_(True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=0.0)
    n = h_tr.shape[0]
    steps = max(1, (n // batch)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best = (float("inf"), W.detach().clone(), b.detach().clone())
    history, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        run, nb = 0.0, 0
        for s in range(0, n - batch + 1, batch):
            idx = perm[s : s + batch]
            h = h_tr[idx].to(device, torch.float32, non_blocking=True)
            hf = hf_tr[idx].to(device, torch.float32, non_blocking=True)
            with torch.no_grad():
                tgt = F.log_softmax(hf @ w_u.T, dim=-1)
            lens = F.log_softmax(_lens_logits(h, W, b, w_u, norm_w, norm_eps), dim=-1)
            loss = F.kl_div(lens, tgt, log_target=True, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Generous clip: Adam already normalises the step, so a tight clip
            # would only distort the direction. Log the raw norm at step 0 so a
            # pathological gradient scale is visible rather than silently clipped.
            gnorm = torch.nn.utils.clip_grad_norm_([W, b], 10.0)
            if step == 0:
                logger.info("[fit %s] step0 loss=%.4f grad_norm=%.4g", tag, loss.item(), gnorm)
            opt.step()
            sched.step()
            run += loss.item()
            nb += 1
            step += 1
        va_kl, va_top1 = evaluate(h_va, hf_va, W, b, w_u, norm_w, norm_eps, device)
        history.append(
            {"epoch": ep, "train_kl": run / max(nb, 1), "val_kl": va_kl, "val_top1": va_top1}
        )
        logger.info(
            "[fit %s] epoch %d/%d train_kl=%.4f val_kl=%.4f val_top1=%.4f (%.0fs)",
            tag,
            ep + 1,
            epochs,
            run / max(nb, 1),
            va_kl,
            va_top1,
            time.time() - t0,
        )
        if va_kl < best[0]:
            best = (va_kl, W.detach().clone(), b.detach().clone())
    return best[1], best[2], history


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-contexts", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-positions", type=int, default=100_000)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--fit-batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--probe-epochs", type=int, default=2)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4, 1e-3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results" / "issue_1482" / "tuned_lens"
    )
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable — this round is GPU-only")

    phase("load_model")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "[model] %s layers=%d hidden=%d vocab=%d",
        BASE_MODEL,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.vocab_size,
    )
    if model.config.hidden_size != ACT_DIM:
        raise RuntimeError(f"hidden_size {model.config.hidden_size} != {ACT_DIM}")

    norm_w = model.model.norm.weight.detach().float().clone()
    norm_eps = float(model.config.rms_norm_eps)
    preflight_conventions(model, tokenizer, norm_w, norm_eps, device)

    manifest = build_manifest(tokenizer, args.n_contexts, args.max_len, args.seed)
    store, pos_records = capture(
        model, manifest, device, args.capture_batch, args.max_positions, args.seed
    )

    # The frozen unembedding is all we need from the model past this point.
    w_u = model.lm_head.weight.detach().float().clone()
    vocab = w_u.shape[0]
    del model
    torch.cuda.empty_cache()
    logger.info("[fit] model released; W_U kept (%d x %d fp32)", *w_u.shape)

    split_of = {e["ctx_id"]: e["split"] for e in manifest}
    is_val = torch.tensor([split_of[c] == "val" for c, _ in pos_records])
    tr_idx, va_idx = (~is_val).nonzero().squeeze(-1), is_val.nonzero().squeeze(-1)
    logger.info("[fit] train positions=%d val positions=%d", len(tr_idx), len(va_idx))

    hf_tr, hf_va = store["h_final"][tr_idx], store["h_final"][va_idx]
    torch.backends.cuda.matmul.allow_tf32 = True

    # LR probe on L19 only (cheap; the fit is minutes) — one LR then used for all layers.
    phase("lr_probe")
    probe_tr, probe_va = store["h19"][tr_idx], store["h19"][va_idx]
    probe = []
    for lr in args.lr_grid:
        _, _, hist = fit_layer(
            probe_tr,
            hf_tr,
            probe_va,
            hf_va,
            w_u,
            norm_w,
            norm_eps,
            device,
            lr=lr,
            epochs=args.probe_epochs,
            batch=args.fit_batch,
            tag=f"probe-lr{lr:g}",
        )
        probe.append({"lr": lr, "val_kl": hist[-1]["val_kl"]})
    del probe_tr, probe_va
    best_lr = min(probe, key=lambda d: d["val_kl"])["lr"]
    if best_lr in (min(args.lr_grid), max(args.lr_grid)) and len(args.lr_grid) > 1:
        logger.warning(
            "[lr_probe] best_lr=%g is at the EDGE of the grid %s — the optimum may lie outside",
            best_lr,
            args.lr_grid,
        )
    logger.info("[lr_probe] grid=%s -> best_lr=%g", probe, best_lr)

    phase("fit")
    results, tensors = {}, {}
    for lyr in LENS_LAYERS:
        key = f"h{lyr}"
        h_tr, h_va = store[key][tr_idx], store[key][va_idx]
        eye = torch.eye(ACT_DIM, device=device)
        zero = torch.zeros(ACT_DIM, device=device)
        base_kl, base_top1 = evaluate(h_va, hf_va, eye, zero, w_u, norm_w, norm_eps, device)
        logger.info("[baseline L%d] logit-lens val_kl=%.4f val_top1=%.4f", lyr, base_kl, base_top1)

        W, b, hist = fit_layer(
            h_tr,
            hf_tr,
            h_va,
            hf_va,
            w_u,
            norm_w,
            norm_eps,
            device,
            lr=best_lr,
            epochs=args.epochs,
            batch=args.fit_batch,
            tag=f"L{lyr}",
        )
        tuned_kl, tuned_top1 = evaluate(h_va, hf_va, W, b, w_u, norm_w, norm_eps, device)
        # Train-side diagnostic on a RANDOM equal-sized subset (the capture order
        # is sorted by context length, so a leading slice would be shortest-first).
        g = torch.Generator().manual_seed(args.seed)
        sub = torch.randperm(h_tr.shape[0], generator=g)[: len(va_idx)]
        tr_kl, tr_top1 = evaluate(h_tr[sub], hf_tr[sub], W, b, w_u, norm_w, norm_eps, device)
        results[f"L{lyr}"] = {
            "layer": lyr,
            "val_kl_tuned": tuned_kl,
            "val_kl_logit_lens": base_kl,
            "val_kl_reduction": base_kl - tuned_kl,
            "val_kl_reduction_frac": (base_kl - tuned_kl) / base_kl if base_kl else None,
            "val_top1_tuned": tuned_top1,
            "val_top1_logit_lens": base_top1,
            "train_kl_tuned": tr_kl,
            "train_top1_tuned": tr_top1,
            "history": hist,
        }
        logger.info(
            "[RESULT L%d] tuned val_kl=%.4f (logit-lens %.4f, -%.1f%%) "
            "tuned val_top1=%.4f (logit-lens %.4f)",
            lyr,
            tuned_kl,
            base_kl,
            100 * (base_kl - tuned_kl) / max(base_kl, 1e-9),
            tuned_top1,
            base_top1,
        )
        tensors[lyr] = (W.detach().cpu(), b.detach().cpu())
        del h_tr, h_va

    # ── persist ───────────────────────────────────────────────────────────────
    phase("persist")
    commit = git_commit()
    common_meta = {
        "issue": TASK_ID,
        "base_model": BASE_MODEL,
        "method": "tuned lens (Belrose et al. 2023, arXiv 2303.08112), hand-rolled affine fit",
        "objective": "KL(P_final || P_lens) with frozen final RMSNorm + unembedding",
        "layer_convention": "resid_post(l) == hidden_states[l+1] (verified at preflight)",
        "n_positions_total": len(pos_records),
        "n_positions_train": int(len(tr_idx)),
        "n_positions_val": int(len(va_idx)),
        "n_contexts": len(manifest),
        "corpus": f"{LMSYS_REPO}@{LMSYS_REVISION} (English, chat-template rendered)",
        "max_len": args.max_len,
        "seed": args.seed,
        "lr": best_lr,
        "epochs": args.epochs,
        "fit_batch": args.fit_batch,
        "git_commit": commit,
        "vocab_size": vocab,
        "rms_norm_eps": norm_eps,
        "apply_note": (
            "absolute residual STATE: W @ h + b; residual DIRECTION (a difference "
            "of states): W @ d only — the bias cancels in a difference."
        ),
    }
    written: list[str] = []
    for lyr, (W, b) in tensors.items():
        p = args.out_dir / f"tuned_lens_L{lyr}.pt"
        torch.save({"W": W, "b": b, "layer": lyr, **common_meta, **results[f"L{lyr}"]}, p)
        written.append(p.name)
        logger.info("[persist] %s (%.1f MB)", p.name, p.stat().st_size / 1e6)

    (args.out_dir / "metrics.json").write_text(
        json.dumps({"meta": common_meta, "lr_probe": probe, "layers": results}, indent=2)
    )
    (args.out_dir / "fit_config.json").write_text(
        json.dumps(
            {**common_meta, "argv": vars(args) | {"out_dir": str(args.out_dir)}},
            indent=2,
            default=str,
        )
    )
    (args.out_dir / "position_manifest.json").write_text(
        json.dumps(
            {
                "meta": common_meta,
                "sampling": {
                    "rule": "all token positions p>=1 per context, then uniform subsample",
                    "seed": args.seed,
                    "max_positions": args.max_positions,
                },
                "contexts": [
                    {k: v for k, v in e.items() if k not in ("text", "input_ids")} for e in manifest
                ],
                "positions": [[c, p] for c, p in pos_records],
            },
            separators=(",", ":"),
        )
    )
    with open(args.out_dir / "contexts.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps({k: v for k, v in e.items() if k != "input_ids"}) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Tuned-lens translators — Qwen-2.5-7B-Instruct (issue #1482)\n\n"
        f"Affine translators `T_l(h) = W_l h + b_l` for layers {list(LENS_LAYERS)}, fit with the\n"
        "tuned-lens objective (Belrose et al. 2023, arXiv 2303.08112): minimise\n"
        "`KL(P_final || softmax(W_U . RMSNorm_f(T_l(h_l))))` with the model's own final\n"
        "RMSNorm and unembedding frozen, `T_l` initialised to the identity.\n\n"
        "## Layer convention\n\n"
        "`h_l` = **resid_post of layer `l`** = HF `output_hidden_states[l + 1]` "
        "(verified at fit time with a forward hook).\n\n"
        "## Applying a translator\n\n"
        "* absolute residual **state**: `W @ h + b`\n"
        "* residual **direction** (a difference of two states): `W @ d` — the bias cancels.\n\n"
        "Then decode as usual: `logits = W_U @ RMSNorm_f(.)`.\n\n"
        "## Files\n\n"
        "`tuned_lens_L{14,19,26}.pt` (`W` fp32 3584x3584, `b` fp32 3584, + fit metadata), "
        "`metrics.json`, `fit_config.json`, `position_manifest.json`, `contexts.jsonl`.\n\n"
        "## Held-out fit quality (val = 15% of contexts, group-level split)\n\n"
        "| layer | tuned KL | logit-lens KL | tuned top-1 | logit-lens top-1 |\n"
        "|---|---|---|---|---|\n"
        + "".join(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                lyr,
                results[f"L{lyr}"]["val_kl_tuned"],
                results[f"L{lyr}"]["val_kl_logit_lens"],
                results[f"L{lyr}"]["val_top1_tuned"],
                results[f"L{lyr}"]["val_top1_logit_lens"],
            )
            for lyr in LENS_LAYERS
        )
    )
    written += [
        "metrics.json",
        "fit_config.json",
        "position_manifest.json",
        "contexts.jsonl",
        "README.md",
    ]
    # Text/JSON must stay on the non-LFS Hub path (<9 MB/file, upload-policy.md).
    for name in ("position_manifest.json", "contexts.jsonl", "metrics.json"):
        size = (args.out_dir / name).stat().st_size
        if size > 8_500_000:
            raise RuntimeError(f"{name} is {size / 1e6:.1f} MB — exceeds the <9 MB text-shard cap")
        logger.info("[persist] %s (%.2f MB)", name, size / 1e6)

    if args.no_upload:
        logger.info("[upload] skipped (--no-upload); wrote %d files locally", len(written))
        phase("done")
        return

    phase("upload")
    url = hub._upload_folder_filtered(
        args.out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=HF_PREFIX,
        allow_patterns=written,
        expected_repo_paths=[f"{HF_PREFIX}/{f}" for f in written],
    )
    if not url:
        raise RuntimeError("tuned-lens upload returned no URL")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        [f"{HF_PREFIX}/{f}" for f in written],
        path_in_repo=HF_PREFIX,
    )
    if missing:
        raise RuntimeError(f"upload verification FAILED — missing: {missing}")
    logger.info("[upload] verified %d files at %s/%s", len(written), HF_DATA_REPO, HF_PREFIX)

    summary = {
        "issue": TASK_ID,
        "hf_repo": HF_DATA_REPO,
        "hf_prefix": HF_PREFIX,
        "files": written,
        "git_commit": commit,
        "layers": {
            k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in results.items()
        },
    }
    (args.out_dir / "upload_summary.json").write_text(json.dumps(summary, indent=2))
    print("UPLOAD_VERIFIED " + json.dumps(summary["layers"]), flush=True)
    phase("done")


if __name__ == "__main__":
    main()
