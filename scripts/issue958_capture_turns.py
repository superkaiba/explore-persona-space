#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #958 per-turn capture: 5-position fp16 states from persisted rollouts.

Per unit (conversation c, turn k) the model teacher-forces
``ctx(c,k) + answer(c,k) + [<|im_end|>, \\n]`` through ``model.model`` (bare
decoder — lm_head never materialized) with forward hooks on ``embed_tokens`` +
every decoder block (#922 machinery: ``_right_pad_batch`` / ``_resolve_rows``
reused verbatim), and stores a ``(5, 29, H)`` fp16 tensor per unit:

    rows = [prefix_end, ctx_last−1, ctx_last, answer_mean, answer_last]

``answer_mean`` is computed ON-GPU over the answer span INCLUDING the trailing
``<|im_end|>`` + ``\\n`` (#779 parity, plan §4.2; span recorded in metadata) —
the full per-position grid is never materialized off-GPU (stream-reduce,
#666/#772). Boundary asserts: token-level prefix property
(``full_ids[:len(prefix_ids)] == prefix_ids``), the generation-prompt suffix,
and the 7,168-token input cap.

Batched-vs-batch-1 equivalence gate (3 real corpus items) runs once at start
with the #779 r12 two-bar calibration: EARLY rows (0–4) per-row cos ≥ 0.999 +
flattened all-row cos ≥ 0.995 (bf16 single-position deep-layer jitter breaches
a flat 0.999 with a bug-free path); ``max_rel`` reported, never asserted.

``--stub-model`` (VM smoke): a tiny random Qwen2 on CPU with the REAL Qwen
tokenizer — same tokenization, hooks, batching, gate, shard IO code path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_capture_positions as CAP  # noqa: E402  (#922 internals, reused)
import issue958_common as C  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_capture")

TOKEN_BUDGET = 65536  # B × max_T packing budget: bounds hook memory ≈ 29·budget·H·2B


# ── tokenization (plan §4.2) ──────────────────────────────────────────────────


def tokenize_unit(
    tokenizer, unit: dict, corpora: dict, rollouts: dict[str, dict], token_cap: int
) -> dict:
    """Token ids + capture positions for one unit, with boundary asserts.

    Returns the unit extended with ``full_ids``, ``prefix_end``, ``ctx_end``,
    ``ans_lo``/``ans_hi`` (span INCLUDES <|im_end|> + \\n), and token counts.
    """
    if unit["set"] == "onpol":
        k1 = rollouts[C.unit_id("main", unit["ci"], 1)]["text"]
        msgs = C.onpol_prompt_messages(corpora["main"][unit["ci"]], k1)
    else:
        msgs = C.unit_prompt_messages(unit, corpora)
    ctx_txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if len(msgs) == 1:  # k=1: default system block only (constant; plan §4.2)
        cut = ctx_txt.find("<|im_start|>user")
        assert cut > 0, "k=1 ctx lacks a user-turn header"
        prefix_txt = ctx_txt[:cut]
    else:
        prefix_txt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=False
        )
    assert ctx_txt.startswith(prefix_txt), f"prefix not a text-prefix of ctx ({unit['uid']})"
    ctx_ids = tokenizer(ctx_txt, padding=False)["input_ids"]
    prefix_ids = tokenizer(prefix_txt, padding=False)["input_ids"]
    # token-level prefix property (special-token boundaries never BPE-merge;
    # assert anyway — the zero-width-span trap family, fail loud)
    assert ctx_ids[: len(prefix_ids)] == prefix_ids, f"token-prefix drift ({unit['uid']})"
    suffix = tokenizer.decode(ctx_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"ctx suffix {suffix!r} != generation prompt"
    assert len(ctx_ids) <= token_cap, (unit["uid"], len(ctx_ids), token_cap)
    answer = rollouts[unit["uid"]]["text"]
    resp_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    full_ids = list(ctx_ids) + list(resp_ids) + [C.IM_END_ID, C.NL_ID]
    prefix_end = len(prefix_ids) - 1
    ctx_end = len(ctx_ids) - 1
    assert prefix_end >= 1 and ctx_end - 1 > prefix_end, (unit["uid"], prefix_end, ctx_end)
    return {
        **unit,
        "full_ids": full_ids,
        "prefix_end": prefix_end,
        "ctx_end": ctx_end,
        "ans_lo": ctx_end + 1,
        "ans_hi": len(full_ids),  # exclusive; INCLUDES <|im_end|> + \n (#779 parity)
        "prefix_tokens": len(prefix_ids),
        "query_tokens": len(ctx_ids) - len(prefix_ids),
        "ans_tokens": len(resp_ids),
        "finish_reason": rollouts[unit["uid"]].get("finish_reason"),
    }


# ── batched capture (hooked forward + on-GPU 5-position reduce) ───────────────


def _pack_batches(order: list[int], items: list[dict], batch_size: int) -> list[list[int]]:
    """Greedy length-sorted packing under B ≤ batch_size and B·max_T ≤ budget."""
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        t = len(items[i]["full_ids"])
        trial_max = max([t] + [len(items[j]["full_ids"]) for j in cur]) if cur else t
        if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * trial_max > TOKEN_BUDGET):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def capture_units_batched(model, tokenizer, items: list[dict], batch_size: int) -> list[dict]:
    """Per-item ``(5, R, H)`` fp16 CPU tensors; right-padded packed batches.

    Positions gathered GPU-side per row; ``answer_mean`` reduced ON-GPU via a
    span mask before the single thin fp16 CPU move. Returns input order.
    """
    mods, _labels = CAP._resolve_rows(model)
    pad_id = CAP._pad_id_for(tokenizer)
    out: list[dict | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["full_ids"]))
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(ri: int):
        def _hook(_m, _i, output):
            captured[ri] = output[0] if isinstance(output, tuple) else output

        return _hook

    def _forward(sel: list[int]) -> None:
        batch = [items[i] for i in sel]
        ids_b, mask_b = CAP._right_pad_batch([b["full_ids"] for b in batch], pad_id, model.device)
        B, T = ids_b.shape
        pos = torch.tensor(
            [[b["prefix_end"], b["ctx_end"] - 1, b["ctx_end"], b["ans_hi"] - 1] for b in batch],
            dtype=torch.long,
        ).to(model.device)  # (B, 4)
        arange = torch.arange(T, device=model.device).unsqueeze(0)
        lo = torch.tensor([b["ans_lo"] for b in batch], device=model.device).unsqueeze(1)
        hi = torch.tensor([b["ans_hi"] for b in batch], device=model.device).unsqueeze(1)
        span = ((arange >= lo) & (arange < hi)).to(model.dtype)  # (B, T)
        span_len = span.sum(1, keepdim=True)  # (B, 1)
        assert bool((span_len.squeeze(1) >= 3).all()), "answer span < 3 positions"
        captured.clear()
        handles = [m.register_forward_hook(_make_hook(ri)) for ri, m in enumerate(mods)]
        try:
            model.model(input_ids=ids_b, attention_mask=mask_b)
        finally:
            for h in handles:
                h.remove()
        row_slices = []
        for ri in range(len(mods)):
            hs = captured[ri]  # (B, T, H) on device
            H = hs.shape[-1]
            gidx = pos.unsqueeze(-1).expand(B, 4, H)
            sel4 = torch.gather(hs, 1, gidx)  # (B, 4, H)
            amean = (hs * span.unsqueeze(-1)).sum(1) / span_len  # (B, H) on-GPU reduce
            five = torch.cat([sel4[:, :3], amean.unsqueeze(1), sel4[:, 3:4]], dim=1)
            row_slices.append(five.to(torch.float16))  # (B, 5, H)
        captured.clear()
        stacked = torch.stack(row_slices, dim=2).cpu()  # (B, 5, R, H)
        del row_slices
        for bi, gi in enumerate(sel):
            it = batch[bi]
            h5 = stacked[bi].clone()
            assert torch.isfinite(h5.float()).all(), f"non-finite capture ({it['uid']})"
            out[gi] = {
                "h": h5,  # (5, R, H) fp16
                "prefix_tokens": it["prefix_tokens"],
                "query_tokens": it["query_tokens"],
                "ans_tokens": it["ans_tokens"],
                "finish_reason": it["finish_reason"],
                "ans_span_convention": "includes_im_end_and_newline",
            }

    for sel in _pack_batches(order, items, batch_size):
        try:
            _forward(sel)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("[oom] batch of %d OOMed — retrying halved", len(sel))
            mid = max(1, len(sel) // 2)
            _forward(sel[:mid])
            _forward(sel[mid:])
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def equivalence_gate(model, tokenizer, items: list[dict]) -> dict:
    """Batched(3) vs batch-1 gate, two-bar calibration (#779 r12; gotchas.md).

    EARLY rows (store rows 0–4) per-row cos ≥ 0.999 (the sharp bug catcher —
    mask/RoPE/pad bugs corrupt layer 0 immediately); flattened all-row cos ≥
    0.995 (headroom over deep-layer bf16 padded-batch jitter on
    single-position states). ``max_rel`` reported, never asserted.
    """
    probe = items[:3]
    assert len(probe) == 3, "equivalence gate needs 3 units"
    bat = capture_units_batched(model, tokenizer, probe, batch_size=3)
    ser = [capture_units_batched(model, tokenizer, [it], batch_size=1)[0] for it in probe]
    early_cos_min, flat_cos_min, max_rel = 1.0, 1.0, 0.0
    for s, b in zip(ser, bat, strict=True):
        a, c = s["h"].double(), b["h"].double()  # (5, R, H)
        for r in range(min(5, a.shape[1])):
            ar, cr = a[:, r].flatten(), c[:, r].flatten()
            cos = float(torch.dot(ar, cr) / (ar.norm() * cr.norm() + 1e-12))
            early_cos_min = min(early_cos_min, cos)
        af, cf = a.flatten(), c.flatten()
        flat_cos_min = min(flat_cos_min, float(torch.dot(af, cf) / (af.norm() * cf.norm() + 1e-12)))
        max_rel = max(max_rel, float((af - cf).abs().max()) / (float(af.abs().max()) + 1e-12))
    assert early_cos_min >= 0.999, (early_cos_min, flat_cos_min, max_rel)
    assert flat_cos_min >= 0.995, (early_cos_min, flat_cos_min, max_rel)
    logger.info(
        "[gate] batched-vs-batch-1 PASS (early_cos_min=%.6f flat_cos_min=%.6f max_rel=%.2e)",
        early_cos_min,
        flat_cos_min,
        max_rel,
    )
    return {"early_cos_min": early_cos_min, "flat_cos_min": flat_cos_min, "max_rel": max_rel}


# ── shard validation (resume; mirrors #922 validate_shard) ────────────────────


def validate_shard(path: Path, expected_uids: set[str], n_rows: int, hidden: int, fingerprint: str):
    """(blob, 'ok') when an existing shard matches the CURRENT regime, else (None, why).

    The regime includes the CORPUS FINGERPRINT (r2 fix): a shard captured
    under a different corpus build fails validation and is recaptured —
    never silently consumed against a rebuilt corpus.
    """
    try:
        blob = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        return None, f"unloadable ({type(e).__name__})"
    if blob.get("corpus_fingerprint") != fingerprint:
        return None, (
            f"corpus fingerprint mismatch ({str(blob.get('corpus_fingerprint'))[:12]} "
            f"vs {fingerprint[:12]})"
        )
    units = blob.get("units") or {}
    if set(units) != expected_uids:
        return None, f"uid set mismatch ({len(units)} vs {len(expected_uids)})"
    for uid, rec in units.items():
        hh = rec.get("h")
        if hh is None or hh.dtype != torch.float16 or tuple(hh.shape) != (5, n_rows, hidden):
            return None, f"{uid} h invalid (want fp16 (5, {n_rows}, {hidden}))"
    return blob, "ok"


def _build_model(args):
    """Real Qwen (bf16 cuda / fp32 cpu) or the --stub-model tiny random Qwen2."""
    if args.stub_model:
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or C.DEFAULT_MODEL)
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=args.stub_hidden,
            intermediate_size=4 * args.stub_hidden,
            num_hidden_layers=args.stub_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=C.TOKEN_CAP + C.ROLLOUT_MAX_TOKENS,
        )
        model = Qwen2ForCausalLM(cfg).to(torch.float32)
        logger.warning(
            "[stub] tiny random Qwen2 (layers=%d hidden=%d) on CPU — VM smoke only",
            args.stub_layers,
            args.stub_hidden,
        )
        return model.eval(), tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    return model.eval(), tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #958 per-turn 5-position capture.")
    ap.add_argument("--corpus", type=Path, default=Path("data/issue_958/corpus"))
    ap.add_argument("--rollouts", type=Path, default=Path("data/issue_958/rollouts"))
    ap.add_argument("--out", type=Path, default=Path("data/issue_958/store"))
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--token-cap", type=int, default=C.TOKEN_CAP)
    ap.add_argument("--stub-model", action="store_true", help="VM smoke: tiny random Qwen2")
    ap.add_argument("--stub-layers", type=int, default=4)
    ap.add_argument("--stub-hidden", type=int, default=64)
    ap.add_argument(
        "--unit-sets", default="main,long,graft,onpol", help="comma list (dispatch threads all)"
    )
    args = ap.parse_args()

    model, tokenizer = _build_model(args)
    CAP.assert_template_tail(tokenizer)
    n_rows = len(model.model.layers) + 1
    hidden = model.config.hidden_size

    corpus_fp = C.corpus_fingerprint(args.corpus)
    corpora = {
        "main": C.load_corpus(args.corpus, "main"),
        "long": C.load_corpus(args.corpus, "long"),
    }
    units_all = C.enumerate_units(args.corpus)
    rollouts_by_set = {s: C.load_rollouts(args.rollouts, s) for s in units_all}
    dropped_by_set = {s: C.load_dropped(args.rollouts, s) for s in units_all}
    # onpol prefix needs the main k1 rollouts too
    rollouts_flat: dict[str, dict] = {}
    for d in rollouts_by_set.values():
        rollouts_flat.update(d)

    # equivalence gate on 3 real corpus items (first main units WITH rollouts)
    gate_units = [u for u in units_all["main"] if u["uid"] in rollouts_flat][:3]
    gate_items = [
        tokenize_unit(tokenizer, u, corpora, rollouts_flat, args.token_cap) for u in gate_units
    ]
    gate = equivalence_gate(model, tokenizer, gate_items)

    t0 = time.time()
    summary_sets: dict[str, dict] = {}
    dropped_uids: dict[str, list[str]] = {}
    for unit_set in [s for s in args.unit_sets.split(",") if s]:
        # dropped-with-record units are SKIPPED (never crash capture); a gap
        # NOT recorded dropped stays fail-loud (row-coverage coherence, r2)
        units, missing_not_dropped, dropped_here = [], [], []
        for u in units_all[unit_set]:
            if u["uid"] in rollouts_flat:
                units.append(u)
            elif u["uid"] in dropped_by_set[unit_set]:
                dropped_here.append(u["uid"])
            else:
                missing_not_dropped.append(u["uid"])
        assert not missing_not_dropped, (
            f"{unit_set}: rollouts missing (NOT recorded dropped) {missing_not_dropped[:3]}"
        )
        dropped_uids[unit_set] = sorted(dropped_here)
        (args.out / unit_set).mkdir(parents=True, exist_ok=True)
        n_shards = (len(units) + C.SHARD_UNITS - 1) // C.SHARD_UNITS
        shard_uid_index: dict[str, list[str]] = {}
        for s in range(n_shards):
            path = C.store_shard_path(args.out, unit_set, s)
            shard_units = units[s * C.SHARD_UNITS : (s + 1) * C.SHARD_UNITS]
            uids = {u["uid"] for u in shard_units}
            shard_uid_index[str(s)] = sorted(uids)
            if path.exists():
                blob, why = validate_shard(path, uids, n_rows, hidden, corpus_fp)
                if blob is not None:
                    logger.info("[%s shard %d/%d] valid — skip", unit_set, s + 1, n_shards)
                    del blob
                    continue
                logger.warning("[%s shard %d] FAILS validation (%s) — recapture", unit_set, s, why)
            tok_items = [
                tokenize_unit(tokenizer, u, corpora, rollouts_flat, args.token_cap)
                for u in shard_units
            ]
            rows = capture_units_batched(model, tokenizer, tok_items, args.batch)
            units_blob = {u["uid"]: r for u, r in zip(shard_units, rows, strict=True)}
            torch.save(
                {
                    "unit_set": unit_set,
                    "positions": C.POS_NAMES,
                    "corpus_fingerprint": corpus_fp,
                    "units": units_blob,
                    "metadata": C.reproducibility_metadata(
                        {"script": "issue958_capture_turns", "set": unit_set, "shard": s}
                    ),
                },
                path,
            )
            blob2, why2 = validate_shard(path, uids, n_rows, hidden, corpus_fp)
            if blob2 is None:
                raise RuntimeError(f"freshly-written shard {path} fails validation: {why2}")
            del blob2
            logger.info(
                "[%s shard %d/%d] saved %d units (%.1fs elapsed)",
                unit_set,
                s + 1,
                n_shards,
                len(units_blob),
                time.time() - t0,
            )
        # sidecar index: uid → shard lookup without loading tensor blobs
        C.write_json_atomic(
            C.store_index_path(args.out, unit_set),
            {
                "unit_set": unit_set,
                "corpus_fingerprint": corpus_fp,
                "n_rows": n_rows,
                "hidden": hidden,
                "shards": shard_uid_index,
            },
        )
        summary_sets[unit_set] = {"n_units": len(units), "n_dropped": len(dropped_here)}

    # capture/rollout-yield kill (plan §7): the k=1..4 main chain must be intact
    # for >= 90% of main conversations (a captured unit == present in a shard).
    idx = (
        C.load_store_index(args.out, "main", expect_fingerprint=corpus_fp)
        if "main" in summary_sets
        else {}
    )
    n_main = len(corpora["main"])
    intact = sum(
        1
        for ci in range(n_main)
        if all(C.unit_id("main", ci, k) in idx for k in range(1, C.K_MAIN + 1))
    )
    if n_main and "main" in summary_sets:
        frac = intact / n_main
        assert frac >= 0.9, f"CHAIN-YIELD KILL (plan §7): only {frac:.1%} main chains intact"

    realized = sum(
        p.stat().st_size for s in summary_sets for p in (args.out / s).glob("shard_*.pt")
    )
    summary = {
        "unit_sets": summary_sets,
        "corpus_fingerprint": corpus_fp,
        "dropped_uids": dropped_uids,
        "n_rows": n_rows,
        "hidden": hidden,
        "equivalence_gate": gate,
        "main_chain_intact_frac": (intact / n_main) if n_main else None,
        "realized_store_bytes": realized,
        "stub_model": bool(args.stub_model),
        "wall_seconds": time.time() - t0,
        "metadata": C.reproducibility_metadata({"script": "issue958_capture_turns"}),
    }
    C.write_json_atomic(args.out / "capture_summary.json", summary)
    logger.info("DONE: %s", json.dumps({k: v for k, v in summary.items() if k != "metadata"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
