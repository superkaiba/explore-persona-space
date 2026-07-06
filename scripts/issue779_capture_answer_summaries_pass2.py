#!/usr/bin/env python3
"""Issue #779 follow-up: answer-summary capture PASS 2 — next-turn template positions.

Pass 1 (``issue779_capture_answer_summaries.py``) captured summaries of the
assistant turn up to its final formatted token (the ``"\\n"`` after
``<|im_end|>``). Pass 2 EXTENDS each teacher-forced sequence with the
next-USER-turn template prefix ``<|im_start|>user\\n`` (exact token ids
verified at runtime and recorded in shard metadata) and captures, per rollout x
all 28 layers, fp16:

  - ``v_im_end``        — state at the turn-final ``<|im_end|>`` (this position
    was NOT a pass-1 summary; pass-1 ``v_last_turn`` is the ``"\\n"`` AFTER it,
    i.e. the addendum's (e) — reused from pass 1, not recaptured here);
  - ``v_im_start``      — state at the next-turn ``<|im_start|>``;
  - ``v_user``          — state at the ``user`` role token;
  - ``v_nl_after_user`` — state at the ``"\\n"`` after ``user`` (the true final
    pre-next-user-content position);
  - ``v_tmpl_mean`` / ``v_tmpl_max`` — mean / element-wise max over the 5
    template tokens {<|im_end|>, \\n, <|im_start|>, user, \\n};
  - ``v_full_mean`` / ``v_full_max`` — template-INCLUSIVE mean / element-wise
    max over the FULL span (response content + the 5 template tokens).

Because attention is causal, the states at the pass-1 positions are unchanged
by the extension; only the 3 appended positions are new. Same batching /
sharding / upload / hygiene as pass 1 (shards -> HF
``final_token_capture/pass2/``, bulk ``upload_folder`` per tag, verified).
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE torch freezes its pool.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_capture_answer_summaries as P1  # noqa: E402
import issue779_common as C  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_capture_answer_summaries_pass2")

CAPTURE_SUBDIR_P2 = f"{P1.CAPTURE_SUBDIR}/pass2"
SUMMARIES2 = (
    "v_im_end",
    "v_im_start",
    "v_user",
    "v_nl_after_user",
    "v_tmpl_mean",
    "v_tmpl_max",
    "v_full_mean",
    "v_full_max",
)
IM_START_ID = 151644
NEXT_USER_SUFFIX = "<|im_start|>user\n"

# Equivalence-gate bars — measured calibration (pod-77902 H100, 2026-07-03;
# diagnostic logs /workspace/logs/issue779_gate_diag_{bf16,fp32}.log):
#   - fp32 batch-3 vs batch-1 agrees at cos=1.000000 on EVERY (item, summary)
#     cell (per-layer min 1.000000): the capture path has NO padding/indexing
#     bug (right-pad + causal mask => pads cannot influence real positions).
#   - bf16 batch-3 vs batch-1 differences are depth-amplified kernel numerics:
#     layer-0 cos >= 0.999995 everywhere; the worst flattened cell was
#     0.998770 (item 1 v_im_start, driven by layer 27 ALONE at 0.996907) —
#     which tripped the old flat 0.999 bar. Pass-1's realized gate on the
#     same rig was 0.999748: the 0.999 bar had no headroom for pass-2's
#     single-position template-token states (span means smooth the jitter;
#     single positions do not).
#   - a REAL off-by-one / pad / row-mapping bug reads cos_flat 0.39-0.62 and
#     layer-0 cos 0.43-0.84 (measured between adjacent-position states,
#     identical in both dtypes) — far below either bar.
# Gate design: (a) EARLY-layer per-layer cosine (first GATE_EARLY_LAYERS
# layers) >= 0.999 is the sharp bug catcher — mask/RoPE/pad/row bugs corrupt
# layer 0 immediately, where bf16 batched-kernel jitter is ~1e-6; (b) the
# flattened cosine >= 0.995 bounds depth-amplified bf16 jitter with >=4x the
# measured worst deviation as headroom while staying ~0.35 above the ~0.6
# bug regime. max_rel stays reported-never-asserted (same rationale as
# pass 1: bf16 one-ULP diffs read ~2-3% of the global max).
GATE_COS_FLAT_MIN = 0.995
GATE_COS_EARLY_MIN = 0.999
GATE_EARLY_LAYERS = 4


def _next_user_suffix_ids(tokenizer) -> list[int]:
    """Token ids of the next-user-turn template prefix, cross-checked.

    Verifies the direct encode of ``<|im_start|>user\\n`` matches the SUFFIX
    the chat template itself appends when a next user turn follows the
    assistant turn (fail-loud on any drift).
    """
    ids = tokenizer(NEXT_USER_SUFFIX, add_special_tokens=False)["input_ids"]
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    base = tokenizer(
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
        add_special_tokens=False,
    )["input_ids"]
    withu = tokenizer(
        tokenizer.apply_chat_template(
            [*msgs, {"role": "user", "content": "X"}],
            tokenize=False,
            add_generation_prompt=False,
        ),
        add_special_tokens=False,
    )["input_ids"]
    assert withu[: len(base)] == base, "template prefix drift"
    assert withu[len(base) : len(base) + len(ids)] == ids, (
        f"next-user suffix mismatch: template continues {withu[len(base) : len(base) + 4]} "
        f"but direct encode gives {ids}"
    )
    assert ids[0] == IM_START_ID and ids[-1] == 198, ids
    return ids


@torch.no_grad()
def capture_pass2_batched(
    model, tokenizer, items: list[dict], layers: list[int], batch_size: int, suffix_ids: list[int]
) -> list[dict]:
    """Capture the 8 pass-2 summaries per item x layer over EXTENDED sequences.

    Same forward path as pass 1 (``model.model`` + block hooks, right-padded,
    sorted by length). Positions (0-based, ext_ids = full_ids + suffix_ids):
    im_end = full_len-2, nl1 = full_len-1, im_start = full_len, user =
    full_len+1, nl2 = full_len+2; template group = [full_len-2, full_len+3);
    full span = [prompt_len, full_len+3).
    """
    blocks = model.model.layers
    pad_id = P1._pad_id_for(tokenizer)
    out: list[dict | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i]["full_len"])
    n_layers = len(layers)
    hidden = model.config.hidden_size

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(L: int):
        def _hook(_m, _i, output):
            captured[L] = output[0] if isinstance(output, tuple) else output

        return _hook

    for start in range(0, len(order), batch_size):
        sel = order[start : start + batch_size]
        batch = [items[i] for i in sel]
        ext_lists = [[*b["full_ids"], *suffix_ids] for b in batch]
        ids_b, mask_b, _ = P1._right_pad_batch(ext_lists, pad_id, model.device)
        captured.clear()
        handles = [blocks[L].register_forward_hook(_make_hook(L)) for L in layers]
        try:
            model.model(input_ids=ids_b, attention_mask=mask_b)
        finally:
            for h in handles:
                h.remove()
        for bi, gi in enumerate(sel):
            it = batch[bi]
            pl, fl = it["prompt_len"], it["full_len"]
            ext_ids = ext_lists[bi]
            assert ext_ids[fl - 2] == P1.IM_END_ID and ext_ids[fl] == IM_START_ID, (
                ext_ids[fl - 2 : fl + 1],
            )
            summ = torch.full((len(SUMMARIES2), n_layers, hidden), 0.0, dtype=torch.float16)
            for li_pos, L in enumerate(layers):
                hs = captured[L][bi]  # (T, H) bf16, right-padded
                tmpl = hs[fl - 2 : fl + 3]  # 5 template tokens
                full = hs[pl : fl + 3]  # content + 5 template tokens
                summ[0, li_pos] = hs[fl - 2].to(torch.float16).cpu()  # v_im_end
                summ[1, li_pos] = hs[fl].to(torch.float16).cpu()  # v_im_start
                summ[2, li_pos] = hs[fl + 1].to(torch.float16).cpu()  # v_user
                summ[3, li_pos] = hs[fl + 2].to(torch.float16).cpu()  # v_nl_after_user
                summ[4, li_pos] = tmpl.mean(dim=0).to(torch.float16).cpu()  # v_tmpl_mean
                summ[5, li_pos] = tmpl.max(dim=0).values.to(torch.float16).cpu()  # v_tmpl_max
                summ[6, li_pos] = full.mean(dim=0).to(torch.float16).cpu()  # v_full_mean
                summ[7, li_pos] = full.max(dim=0).values.to(torch.float16).cpu()  # v_full_max
            out[gi] = {
                "summ": summ,
                "valid": torch.ones(len(SUMMARIES2), dtype=torch.bool),
                "last_turn_token_id": int(ext_ids[fl + 2]),
                "prompt_len": pl,
                "span_len": fl - pl,
                "content_len": it["content_end"] - pl,
            }
        captured.clear()
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def equivalence_gate_p2(model, tokenizer, layers: list[int], suffix_ids: list[int]) -> dict:
    """Batched (padded) vs batch-1 equivalence for the pass-2 capture path.

    Two asserts (calibration comment at GATE_COS_FLAT_MIN): early-layer
    per-layer cosine >= GATE_COS_EARLY_MIN catches real padding / indexing /
    mask bugs (they corrupt layer 0, where bf16 jitter is ~1e-6); the
    flattened all-layer cosine >= GATE_COS_FLAT_MIN bounds depth-amplified
    bf16 padded-batch kernel jitter.
    """
    msgs = [
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi."}],
        [
            {"role": "system", "content": "You are a careful, verbose assistant."},
            {"role": "user", "content": "Explain in detail why the sky appears blue at noon."},
        ],
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
    bat = capture_pass2_batched(model, tokenizer, items, layers, 3, suffix_ids)
    ser = [capture_pass2_batched(model, tokenizer, [it], layers, 1, suffix_ids)[0] for it in items]
    max_rel, cos_min, early_cos_min = 0.0, 1.0, 1.0
    for s, b in zip(ser, bat, strict=True):
        for k in range(len(SUMMARIES2)):
            a = s["summ"][k].double()  # (L, H)
            c = b["summ"][k].double()
            scale = float(a.abs().max()) + 1e-12
            max_rel = max(max_rel, float((a - c).abs().max()) / scale)
            af, cf = a.flatten(), c.flatten()
            cos_min = min(cos_min, float(torch.dot(af, cf) / (af.norm() * cf.norm() + 1e-12)))
            per_layer = torch.nn.functional.cosine_similarity(a, c, dim=1)  # (L,)
            early_cos_min = min(early_cos_min, float(per_layer[:GATE_EARLY_LAYERS].min()))
    # Two-bar gate (calibration + rationale at GATE_COS_FLAT_MIN definition).
    assert early_cos_min >= GATE_COS_EARLY_MIN, (early_cos_min, cos_min, max_rel)
    assert cos_min >= GATE_COS_FLAT_MIN, (cos_min, early_cos_min, max_rel)
    logger.info(
        "[gate-p2] equivalence PASS (cos_min=%.6f early_cos_min=%.6f max_rel=%.4f)",
        cos_min,
        early_cos_min,
        max_rel,
    )
    return {"cos_min": cos_min, "early_cos_min": early_cos_min, "max_rel": max_rel}


def _save_shard_p2(
    path: Path,
    tag: str,
    layers: list[int],
    ctx_range: tuple[int, int],
    index: list[tuple[int, int]],
    rows: list[dict],
    suffix_ids: list[int],
) -> None:
    torch.save(
        {
            "tag": tag,
            "summaries": list(SUMMARIES2),
            "layers": layers,
            "context_range": list(ctx_range),
            "index": index,
            "summ": torch.stack([r["summ"] for r in rows]),  # (n, 8, L, H) fp16
            "valid": torch.stack([r["valid"] for r in rows]),  # (n, 8) bool (all True)
            "last_turn_token_ids": torch.tensor(
                [r["last_turn_token_id"] for r in rows], dtype=torch.long
            ),
            "prompt_lens": torch.tensor([r["prompt_len"] for r in rows], dtype=torch.long),
            "span_lens": torch.tensor([r["span_len"] for r in rows], dtype=torch.long),
            "content_lens": torch.tensor([r["content_len"] for r in rows], dtype=torch.long),
            "metadata": C.reproducibility_metadata(
                {
                    "script": "issue779_capture_answer_summaries_pass2",
                    "tag": tag,
                    "next_user_suffix_ids": suffix_ids,
                    "position_convention": (
                        "ext_ids = pass1 full_ids + <|im_start|>user\\n; im_end=fl-2, "
                        "nl1=fl-1 (pass-1 v_last_turn), im_start=fl, user=fl+1, "
                        "nl2=fl+2; tmpl group=[fl-2,fl+3); full span=[prompt_len,fl+3)"
                    ),
                }
            ),
        },
        path,
    )
    logger.info("[shard-p2] %s: %d rows", path.name, len(rows))


def _hf_p2_files() -> set[str]:
    from huggingface_hub import list_repo_files

    prefix = f"{P1.HF_ROUND_PREFIX}/{CAPTURE_SUBDIR_P2}/"
    return {f for f in list_repo_files(C.HF_DATA_REPO, repo_type="dataset") if f.startswith(prefix)}


def _upload_p2(local_dir: Path, names: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{P1.HF_ROUND_PREFIX}/{CAPTURE_SUBDIR_P2}"
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=prefix,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=names,
        commit_message=f"issue779 answer-summary capture pass2: {len(names)} shard(s)",
    )
    repo = _hf_p2_files()
    missing = [n for n in names if f"{prefix}/{n}" not in repo]
    if missing:
        raise RuntimeError(f"pass2 upload verification FAILED: missing {missing}")
    logger.info("[upload-p2] verified %d shard(s) under %s", len(names), prefix)


def run_tag_p2(
    model,
    tokenizer,
    layers: list[int],
    tag: str,
    items: list[dict],
    n_ctx: int,
    out_dir: Path,
    hf_done: set[str],
    batch_size: int,
    suffix_ids: list[int],
    t0: float,
    total: int,
    done_holder: list[int],
    smoke: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ci: dict[int, list[dict]] = {}
    for it in items:
        by_ci.setdefault(it["ci"], []).append(it)
    n_shards = (n_ctx + P1.SHARD_CTX - 1) // P1.SHARD_CTX
    new_names: list[str] = []
    prefix = f"{P1.HF_ROUND_PREFIX}/{CAPTURE_SUBDIR_P2}"
    for k in range(n_shards):
        name = P1._shard_name(tag, k)
        path = out_dir / name
        lo, hi = k * P1.SHARD_CTX, min((k + 1) * P1.SHARD_CTX, n_ctx)
        shard_items = [it for ci in range(lo, hi) for it in by_ci.get(ci, [])]
        if path.exists() or f"{prefix}/{name}" in hf_done:
            logger.info("[%s-p2] shard %d/%d already done; skip", tag, k + 1, n_shards)
            done_holder[0] += len(shard_items)
            continue
        logger.info(
            "[%s-p2] shard %d/%d: contexts [%d,%d) -> %d rollouts (tokenizing)",
            tag,
            k + 1,
            n_shards,
            lo,
            hi,
            len(shard_items),
        )
        tok_items = [P1._tokenize_item(tokenizer, it) for it in shard_items]
        rows = capture_pass2_batched(model, tokenizer, tok_items, layers, batch_size, suffix_ids)
        _save_shard_p2(
            path,
            tag,
            layers,
            (lo, hi),
            [(it["ci"], it["ri"]) for it in shard_items],
            rows,
            suffix_ids,
        )
        new_names.append(name)
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
    if new_names and not smoke:
        _upload_p2(out_dir, new_names)
    elif new_names:
        logger.info("[%s-p2] SMOKE: %d shard(s) kept local-only", tag, len(new_names))


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 answer-summary capture PASS 2.")
    parser.add_argument("--model", default=C.DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/issue779_capture"))
    parser.add_argument("--in-dir", type=Path, default=Path("/workspace/issue779_capture/inputs"))
    parser.add_argument("--batch-size", type=int, default=P1.CAPTURE_BATCH)
    parser.add_argument("--traits", nargs="+", default=[*C.TRAITS, "lmsys"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=C.EXPECTED_HIDDEN)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    n_layers = len(model.model.layers)
    assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
    assert model.config.hidden_size == args.expected_hidden
    layers = list(range(n_layers))

    suffix_ids = _next_user_suffix_ids(tokenizer)
    logger.info("[p2] next-user suffix ids: %s", suffix_ids)
    gate = equivalence_gate_p2(model, tokenizer, layers, suffix_ids)
    out_dir = args.out_dir / ("shards_pass2_smoke" if args.smoke else "shards_pass2")
    hf_done = set() if args.smoke else _hf_p2_files()

    tag_items: dict[str, tuple[list[dict], int]] = {}
    for tag in args.traits:
        if tag == "lmsys":
            items = P1.build_lmsys_items(args.in_dir)
            n_ctx = P1.N_LMSYS_EXPECTED
        else:
            items = P1.build_corpus_items(tag, args.in_dir)
            n_ctx = P1.N_CORPUS_CTX_EXPECTED
        if args.smoke:
            items = [it for it in items if it["ci"] < 2]
            n_ctx = 2
        tag_items[tag] = (items, n_ctx)
        logger.info("[inputs] %s: %d rollouts over %d contexts", tag, len(items), n_ctx)

    total = sum(len(v[0]) for v in tag_items.values())
    t0 = time.time()
    done_holder = [0]
    for tag, (items, n_ctx) in tag_items.items():
        run_tag_p2(
            model,
            tokenizer,
            layers,
            tag,
            items,
            n_ctx,
            out_dir,
            hf_done,
            args.batch_size,
            suffix_ids,
            t0,
            total,
            done_holder,
            args.smoke,
        )

    elapsed_h = (time.time() - t0) / 3600.0
    summary = {
        "tags": list(tag_items),
        "n_rollouts": total,
        "gpu_hours_wall": round(elapsed_h, 3),
        "equivalence_gate": gate,
        "next_user_suffix_ids": suffix_ids,
        "smoke": args.smoke,
    }
    C.write_json_atomic(out_dir / "capture_summary_pass2.json", summary)
    logger.info("DONE: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
