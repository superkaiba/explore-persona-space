#!/usr/bin/env python
"""#825 onpolicy-turn-depth-map pod worker: on-policy turn-k answers + capture.

Per model (instruct | pretrained), on the #1092 dynamics panel (497 logged
WildChat/LMSYS multi-turn conversations, 2,572 assistant-turn pairs/model):

  gen      — for each (conv, turn k) pair, prompt = render(turns[0..k-1]) with
             the generation scaffold; ONE chunked vLLM ``LLM.generate()`` call
             (plan §4.2 sampling: n=1, temperature=1.0, top_p=0.95,
             max_tokens=1024, seed=42; stop tokens per model from
             issue1092_gpu_phase). Writes the rollout JSONL checkpoint
             immediately (checkpoint-per-phase).
  capture  — truncated render turns[0..k] with turn-k assistant content := the
             generated text; batched HF teacher-forced forward with hidden
             states; writes prefix_k / context_k / answer_own_t1 at layers
             {14,18,19}, fp16 npy + row_index_own.jsonl + drop_report.json.
  smoke_bank — (smoke only) fabricates the "banked store" by capturing
             context_k / answer_k_t1 from the FULL logged renders with the
             tiny model, via the verbatim #1092 capture helper — so the fit
             script's G1/G2 gates run against a real, causally-consistent
             reference.

G3 (set-typed pair alignment) runs at panel-build in EVERY phase: the rebuilt
(conv_id, turn_index) pair SET must equal the SET derived from the downloaded
store row index (``_build_pairing``), with the expected cardinality DERIVED
from the banked turn_depth_map results.json (never hard-coded).

Content hygiene: the corpus is REAL-USER text. Conversation/generation text is
never printed/logged — only written to the rollout JSONL artifact. Drop
reports carry (conv_id, turn_index, reason) only; caught span-assert messages
are logged by exception TYPE only (their str() embeds content).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from issue825_turn_depth_map import _build_pairing  # noqa: E402
from issue1092_gpu_phase import (  # noqa: E402
    DEFAULT_VLLM_CHUNK_SIZE,
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    MAX_MODEL_LEN,
    PRETRAINED_MODEL,
    PRETRAINED_REVISION,
    STOP_TOKENS_INSTRUCT,
    STOP_TOKENS_PRETRAINED,
    _call_model_with_hidden_states,
    _capture_dynamics_loaded_model,
    _dynamics_cut_plan,
    _dynamics_panel,
    _filter_dynamics_panel_by_rendered_length,
    _prefix_turns,
    _render_full_conversation,
    _token_len,
    _write_jsonl_rows,
    _write_layer_stack,
    load_store,
)

logger = logging.getLogger("i825_onpolicy_td")

CAPTURE_LAYERS = (14, 18, 19)
# Plan §11: sampling recipe = the #825 single-turn anchor construction
# (Source: scripts/issue825_gen_conversations.py:521).
GEN_N = 1
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 1024
GEN_SEED = 42
# Plan §11: engine window = capture window + generation budget (9216); the
# CAPTURE window stays MAX_MODEL_LEN (8192, store parity) — overflow pairs are
# dropped pair-wise + counted, never truncated.
ENGINE_MAX_MODEL_LEN = MAX_MODEL_LEN + GEN_MAX_TOKENS
CAPTURE_TOKEN_BUDGET = int(os.environ.get("EPM_CAPTURE_TOKEN_BUDGET", "16384"))
GEN_SCAFFOLD_PRETRAINED = "\n\nAssistant:"

MODEL_SPEC = {
    "instruct": (INSTRUCT_MODEL, INSTRUCT_REVISION, STOP_TOKENS_INSTRUCT),
    "pretrained": (PRETRAINED_MODEL, PRETRAINED_REVISION, STOP_TOKENS_PRETRAINED),
}
DROP_REASONS = (
    "empty_completion",
    "zero_width_span",
    "span_assert",
    "window_overflow",
    "no_user_turn_before",
)


def _load_tokenizer(model_type: str, args: argparse.Namespace):
    from transformers import AutoTokenizer

    if args.smoke:
        return AutoTokenizer.from_pretrained(args.tiny_model_dir, trust_remote_code=True)
    name, revision, _stops = MODEL_SPEC[model_type]
    return AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)


def _seed_instruct_render_tokenizer(args: argparse.Namespace) -> None:
    """Smoke hermeticity: pre-seed issue1092_gpu_phase._get_tokenizer's lazy cache.

    ``_render_full_conversation("instruct")`` lazily loads the PINNED instruct
    tokenizer from the Hub; in smoke the tiny dir carries the SAME real Qwen
    tokenizer files, so seeding the documented cache keeps the smoke off the
    network without stubbing any function body.
    """
    if not args.smoke:
        return
    import issue1092_gpu_phase as gp

    gp._get_tokenizer._tok = _load_tokenizer("instruct", args)


def _panel_and_pairs(
    args: argparse.Namespace, model_type: str, tokenizers: dict, *, check_g3: bool = True
) -> tuple[dict[str, list[dict]], list[tuple[str, int]], dict]:
    """Rebuild the #1092 dynamics panel and G3-verify against the banked row index.

    ``check_g3=False`` only for the smoke_bank phase, which CREATES the
    reference row index the other phases G3-verify against.

    Returns (turns_by_conv, ordered pair list [(conv_id, turn_idx)], filter digest).
    """
    store = load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    panel = _dynamics_panel(store)
    panel, filt_digest = _filter_dynamics_panel_by_rendered_length(
        panel, tokenizers, max_tokens=MAX_MODEL_LEN
    )
    turns_by_conv: dict[str, list[dict]] = {}
    rebuilt: list[tuple[str, int]] = []
    for item in panel:
        conv_id = str(item.get("conv_id") or item.get("prefix_id") or item.get("id"))
        turns = _prefix_turns(item)
        turns_by_conv[conv_id] = turns
        for turn_idx, turn in enumerate(turns):
            if turn.get("role") == "assistant":
                rebuilt.append((conv_id, turn_idx))
    if not check_g3:
        return turns_by_conv, rebuilt, filt_digest

    # Banked pair set from the downloaded row index (verbatim _build_pairing).
    paired = _build_pairing(Path(args.row_index_dir), model_type)
    banked_pairs = {(conv, turn) for _ci, _aj, conv, turn in paired}
    rebuilt_set = set(rebuilt)

    # Expected cardinality DERIVED from the banked turn_depth_map JSON (plan §4.1).
    with open(args.banked_json) as f:
        banked = json.load(f)
    expected_n = sum(int(v) for v in banked["n_per_turn"][model_type].values())

    # --- G3 (exact, set-typed, fail-loud) ---
    if rebuilt_set != banked_pairs:
        missing = sorted(banked_pairs - rebuilt_set)[:20]
        extra = sorted(rebuilt_set - banked_pairs)[:20]
        raise AssertionError(
            f"[G3] pair-set mismatch for {model_type}: rebuilt={len(rebuilt_set)} "
            f"banked={len(banked_pairs)}; banked-only (first 20)={missing}; "
            f"rebuilt-only (first 20)={extra}"
        )
    if len(banked_pairs) != expected_n:
        raise AssertionError(
            f"[G3] cardinality mismatch for {model_type}: row-index pair set "
            f"{len(banked_pairs)} != banked n_per_turn sum {expected_n}"
        )
    logger.info(
        "[G3] PASS %s: %d pairs (set equality + derived cardinality)",
        model_type,
        len(banked_pairs),
    )
    return turns_by_conv, rebuilt, filt_digest


def _render_gen_prompt(turns_before: list[dict], model_type: str, tokenizer) -> str:
    if model_type == "instruct":
        msgs = [{"role": t["role"], "content": t["content"]} for t in turns_before]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return _render_full_conversation(turns_before, "pretrained") + GEN_SCAFFOLD_PRETRAINED


def _generate(prompts: list[str], model_type: str, args: argparse.Namespace) -> list[dict]:
    """Return [{text, finish_reason, n_prompt_tokens, n_gen_tokens}] per prompt.

    Smoke swaps ONLY the engine call for deterministic canned completions (the
    surrounding code path — pair build, prompt render, rollout write, drop
    rules — is identical); production runs chunked vLLM per the #1092 recipe
    (large-batch deadlock gotcha: chunked generate + per-chunk INFO logs;
    use_tqdm=False per the vLLM-0.11 tqdm ZeroDivision gotcha).
    """
    if args.smoke:
        out = []
        for i, _p in enumerate(prompts):
            text = (
                ""
                if i == 1
                else (
                    f"Canned smoke answer {i}: a short deterministic reply the tiny "
                    f"model teacher-forces during capture."
                )
            )
            out.append(
                {
                    "text": text,
                    "finish_reason": "smoke_canned",
                    "n_prompt_tokens": None,
                    "n_gen_tokens": None,
                }
            )
        return out

    name, revision, stop_tokens = MODEL_SPEC[model_type]
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=name,
        revision=revision,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=GEN_SEED,
        gpu_memory_utilization=0.85,
        max_model_len=ENGINE_MAX_MODEL_LEN,
    )
    params = SamplingParams(
        n=GEN_N,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS,
        seed=GEN_SEED,
        stop=list(stop_tokens),
    )
    results: list[dict] = []
    chunk_size = max(1, int(args.chunk_size))
    n_chunks = math.ceil(len(prompts) / chunk_size)
    for start in range(0, len(prompts), chunk_size):
        chunk = prompts[start : start + chunk_size]
        logger.info(
            "[gen] %s vLLM chunk %d/%d (%d prompts)",
            model_type,
            start // chunk_size + 1,
            n_chunks,
            len(chunk),
        )
        outputs = llm.generate(chunk, params, use_tqdm=False)
        for out in outputs:
            top = out.outputs[0] if out.outputs else None
            results.append(
                {
                    "text": top.text if top is not None else "",
                    "finish_reason": (top.finish_reason if top is not None else "no_output"),
                    "n_prompt_tokens": len(out.prompt_token_ids or []),
                    "n_gen_tokens": len(top.token_ids) if top is not None else 0,
                }
            )
    # Best-effort in-process teardown; the dispatch-level gpu-guard between the
    # gen and capture phases is the authoritative cleaner (vLLM worker
    # subprocesses are NOT reliably reaped in-process — gotchas.md).
    del llm
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        logger.info("[gen] in-process teardown raised; dispatch gpu-guard will reap")
    return results


def _rollout_path(args: argparse.Namespace, model_type: str) -> Path:
    return Path(args.rollout_dir) / f"{model_type}_own_turn_answers.jsonl"


def run_gen(args: argparse.Namespace, model_type: str, tokenizers: dict) -> None:
    turns_by_conv, pairs, _digest = _panel_and_pairs(args, model_type, tokenizers)
    tok = tokenizers[model_type]
    prompts: list[str] = []
    for conv_id, k in pairs:
        turns = turns_by_conv[conv_id]
        assert turns[k].get("role") == "assistant", (conv_id, k)
        prompts.append(_render_gen_prompt(turns[:k], model_type, tok))
    t0 = time.time()
    gens = _generate(prompts, model_type, args)
    assert len(gens) == len(pairs), (len(gens), len(pairs))
    rows = []
    for (conv_id, k), g in zip(pairs, gens, strict=True):
        rows.append(
            {
                "conv_id": conv_id,
                "turn_index": k,
                "model_type": model_type,
                "text": g["text"],
                "finish_reason": g["finish_reason"],
                "n_prompt_tokens": g["n_prompt_tokens"],
                "n_gen_tokens": g["n_gen_tokens"],
            }
        )
    out = _rollout_path(args, model_type)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=True (json default): raw U+2028/NEL in real-user text shreds
    # splitlines-style consumers of JSONL (gotchas.md); ASCII-escape on write.
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_empty = sum(1 for r in rows if not r["text"].strip())
    logger.info(
        "[gen] %s: %d completions in %.1fs -> %s (%d empty)",
        model_type,
        len(rows),
        time.time() - t0,
        out,
        n_empty,
    )


def _load_rollout(args: argparse.Namespace, model_type: str) -> dict[tuple[str, int], dict]:
    path = _rollout_path(args, model_type)
    rows: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:  # file iteration, never read().splitlines() (gotchas.md)
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[(str(r["conv_id"]), int(r["turn_index"]))] = r
    return rows


def _load_hf_model(model_type: str, args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM

    if args.smoke:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.tiny_model_dir, torch_dtype=torch.float32, trust_remote_code=True
        )
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "capture requires CUDA in production (silent CPU fallback burned "
                "#667); check the launcher CUDA_VISIBLE_DEVICES pin"
            )
        device = "cuda:0"
        name, revision, _stops = MODEL_SPEC[model_type]
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # explicit single visible device, no auto-offload (#825 rc=137 lesson)
            device_map={"": device},
        )
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    if max(CAPTURE_LAYERS) >= n_layers:
        raise ValueError(f"model has {n_layers} layers; need > {max(CAPTURE_LAYERS)}")
    return model, device, n_layers, hidden_dim


def _turn_k_cuts(cuts: dict, k: int, u_idx: int) -> tuple[int, int, int, int]:
    """(prefix_pos, context_pos, answer_start, answer_end) for assistant turn k."""
    ctx = [(s, e, t) for s, e, t in cuts["context_k"] if t == k]
    ans = [(s, e, t) for s, e, t in cuts["answer_k_t1"] if t == k]
    u1 = [(s, e, t) for s, e, t in cuts["u1"] if t == u_idx]
    if len(ctx) != 1 or len(ans) != 1 or len(u1) != 1:
        raise AssertionError(f"cut-plan cardinality at turn {k}: {len(ctx)}/{len(ans)}/{len(u1)}")
    context_pos = ctx[0][0]
    answer_start, answer_end = ans[0][0], ans[0][1]
    prefix_pos = max(0, u1[0][0] - 1)
    return prefix_pos, context_pos, answer_start, answer_end


def _capture_prepass(
    pairs: list[tuple[str, int]],
    rollout: dict[tuple[str, int], dict],
    turns_by_conv: dict[str, list[dict]],
    tok,
    model_type: str,
) -> tuple[list[dict], list[dict], dict[int, int]]:
    """Truncated renders + the counted pair-drop rules. Returns (kept, drops, per_turn)."""
    kept: list[dict] = []
    drops: list[dict] = []
    per_turn_total: dict[int, int] = {}

    def _drop(conv_id: str, k: int, reason: str) -> None:
        drops.append({"conv_id": conv_id, "turn_index": k, "reason": reason})

    for conv_id, k in pairs:
        per_turn_total[k] = per_turn_total.get(k, 0) + 1
        row = rollout[(conv_id, k)]
        content = str(row["text"]).strip()
        if not content:
            _drop(conv_id, k, "empty_completion")
            continue
        turns = turns_by_conv[conv_id]
        u_idx = max((i for i in range(k) if turns[i].get("role") == "user"), default=-1)
        if u_idx < 0:
            _drop(conv_id, k, "no_user_turn_before")
            continue
        turns_trunc = [*turns[:k], {"role": "assistant", "content": content}]
        render = _render_full_conversation(turns_trunc, model_type)
        n_tok = _token_len(tok, render)
        if n_tok > MAX_MODEL_LEN:
            _drop(conv_id, k, "window_overflow")
            continue
        try:
            cuts = _dynamics_cut_plan(turns_trunc, tok, model_type, n_tok)
            _prefix_pos, _context_pos, a_start, a_end = _turn_k_cuts(cuts, k, u_idx)
        except AssertionError:
            # NEVER log str(e): span-assert messages embed real-user content.
            _drop(conv_id, k, "span_assert")
            continue
        if a_end <= a_start:
            _drop(conv_id, k, "zero_width_span")
            continue
        kept.append(
            {
                "conv_id": conv_id,
                "turn_index": k,
                "u_idx": u_idx,
                "turns_trunc": turns_trunc,
                "render": render,
                "n_tok": n_tok,
                "finish_reason": row.get("finish_reason"),
                "n_gen_tokens": row.get("n_gen_tokens"),
            }
        )
    return kept, drops, per_turn_total


def run_capture(args: argparse.Namespace, model_type: str, tokenizers: dict) -> None:
    import torch

    turns_by_conv, pairs, filt_digest = _panel_and_pairs(args, model_type, tokenizers)
    rollout = _load_rollout(args, model_type)
    missing = [p for p in pairs if p not in rollout]
    if missing:
        raise AssertionError(
            f"[capture] {model_type}: {len(missing)} pairs missing from rollout "
            f"(first 5: {missing[:5]}) — gen phase incomplete?"
        )
    tok = tokenizers[model_type]
    kept, drops, per_turn_total = _capture_prepass(pairs, rollout, turns_by_conv, tok, model_type)
    n_total, n_kept = len(pairs), len(kept)
    drop_rate = (n_total - n_kept) / max(1, n_total)
    drop_counts = {r: sum(1 for d in drops if d["reason"] == r) for r in DROP_REASONS}
    logger.info(
        "[capture] %s: %d/%d pairs kept (drop rate %.1f%%): %s",
        model_type,
        n_kept,
        n_total,
        100 * drop_rate,
        drop_counts,
    )

    # ---- batched teacher-forced forward, token-budget batching ----
    model, device, n_layers, hidden_dim = _load_hf_model(model_type, args)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # positions computed from sequence start (store parity)

    states = {kind: [] for kind in ("prefix_k", "context_k", "answer_own_t1")}
    index_rows: list[dict] = []
    batch: list[dict] = []

    def _flush(batch_rows: list[dict]) -> None:
        if not batch_rows:
            return
        texts = [r["render"] for r in batch_rows]
        inputs = tok(
            texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model,
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
            )
        hs = outputs.hidden_states[1:]
        if len(hs) != n_layers or hs[0].shape[-1] != hidden_dim:
            raise ValueError(f"hidden-state shape mismatch: {len(hs)} x {hs[0].shape[-1]}")
        for i, r in enumerate(batch_rows):
            n_tok = int(inputs["attention_mask"][i].sum().item())
            if n_tok != r["n_tok"]:
                raise AssertionError(
                    f"tokenization drift {r['conv_id']}/t{r['turn_index']}: "
                    f"forward {n_tok} != pre-pass {r['n_tok']}"
                )
            # Definitive cut plan against the forward's OWN token ids (verbatim
            # consumer asserts — the store-capture pattern).
            full_ids = inputs["input_ids"][i, :n_tok].tolist()
            cuts = _dynamics_cut_plan(
                r["turns_trunc"], tok, model_type, n_tok, full_token_ids=full_ids
            )
            prefix_pos, context_pos, a_start, a_end = _turn_k_cuts(
                cuts, r["turn_index"], r["u_idx"]
            )
            for kind, sl in (
                ("prefix_k", slice(prefix_pos, prefix_pos + 1)),
                ("context_k", slice(context_pos, context_pos + 1)),
                ("answer_own_t1", slice(a_start, a_end)),
            ):
                vec = np.stack(
                    [
                        hs[layer][i, sl, :].mean(dim=0).float().cpu().numpy()
                        for layer in CAPTURE_LAYERS
                    ],
                    axis=0,
                ).astype(np.float16)
                states[kind].append(vec)
            index_rows.append(
                {
                    "conv_id": r["conv_id"],
                    "turn_index": r["turn_index"],
                    "prefix_pos": prefix_pos,
                    "context_pos": context_pos,
                    "answer_start": a_start,
                    "answer_end": a_end,
                    "n_tokens": n_tok,
                    "finish_reason": r["finish_reason"],
                    "n_gen_tokens": r["n_gen_tokens"],
                }
            )
        del outputs, hs, inputs

    max_len = 0
    t0 = time.time()
    for r in kept:
        cand = max(max_len, r["n_tok"])
        if batch and cand * (len(batch) + 1) > CAPTURE_TOKEN_BUDGET:
            _flush(batch)
            if len(index_rows) % 256 < len(batch):
                logger.info(
                    "[capture] %s: %d/%d rows (%.1fs)",
                    model_type,
                    len(index_rows),
                    n_kept,
                    time.time() - t0,
                )
            batch, max_len = [], 0
        batch.append(r)
        max_len = max(max_len, r["n_tok"])
    _flush(batch)
    assert len(index_rows) == n_kept, (len(index_rows), n_kept)

    out_root = Path(args.out_dir) / model_type
    out_root.mkdir(parents=True, exist_ok=True)
    for kind, vals in states.items():
        arr = (
            np.stack(vals, axis=0)
            if vals
            else np.empty((0, len(CAPTURE_LAYERS), hidden_dim), dtype=np.float16)
        )
        # store-parity file naming: {kind}_L{layer:02d}.npy, loadable by
        # issue1092_fit_grid._load_summary's unsharded fallback.
        for li, layer in enumerate(CAPTURE_LAYERS):
            np.save(out_root / f"{kind}_L{layer:02d}.npy", arr[:, li, :].astype(np.float16))
    _write_jsonl_rows(out_root / "row_index_own.jsonl", index_rows)
    drop_report = {
        "model_type": model_type,
        "n_total_pairs": n_total,
        "n_kept": n_kept,
        "drop_rate": drop_rate,
        "drop_counts": drop_counts,
        "per_turn_total": {str(t): c for t, c in sorted(per_turn_total.items())},
        "per_turn_kept": {
            str(t): sum(1 for r in index_rows if r["turn_index"] == t)
            for t in sorted(per_turn_total)
        },
        "drops": drops,
        "length_filter_digest": filt_digest,
        "capture_window": MAX_MODEL_LEN,
        "layers": list(CAPTURE_LAYERS),
        "hidden_dim": hidden_dim,
    }
    with open(out_root / "drop_report.json", "w") as f:
        json.dump(drop_report, f, indent=1)
    logger.info(
        "[capture] %s: wrote %d rows x %d layers -> %s (%.1fs)",
        model_type,
        n_kept,
        len(CAPTURE_LAYERS),
        out_root,
        time.time() - t0,
    )
    if drop_rate > 0.20:
        # Kill criterion (plan §6): surfaced loudly here; the fit script is the
        # binding halt (do not silently fit the survivor subset).
        logger.warning(
            "[capture] %s: drop rate %.1f%% EXCEEDS the 20%% kill line — the fit "
            "script will HALT; generation-recipe problem, not a finding",
            model_type,
            100 * drop_rate,
        )


def run_smoke_bank(args: argparse.Namespace, model_type: str, tokenizers: dict) -> None:
    """Smoke-only: fabricate the banked store from FULL logged renders (tiny model)."""
    assert args.smoke, "smoke_bank is a smoke-only phase"
    turns_by_conv, pairs, _digest = _panel_and_pairs(args, model_type, tokenizers, check_g3=False)
    conv_ids = sorted(turns_by_conv)
    prompts = [_render_full_conversation(turns_by_conv[c], model_type) for c in conv_ids]
    model, device, n_layers, hidden_dim = _load_hf_model(model_type, args)
    arrays, index_rows = _capture_dynamics_loaded_model(
        prompts=prompts,
        turns_by_prompt=[turns_by_conv[c] for c in conv_ids],
        conv_ids=conv_ids,
        model=model,
        tokenizer=tokenizers[model_type],
        model_type=model_type,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
    )
    bank_root = Path(args.bank_dir) / f"dynamics_{model_type}"
    for kind in ("context_k", "answer_k_t1"):
        _write_layer_stack(bank_root, kind, arrays[kind])
        _write_jsonl_rows(bank_root / f"row_index_{kind}.jsonl", index_rows[kind])
    n_pairs_banked = len(index_rows["context_k"])
    logger.info(
        "[smoke_bank] %s: %d banked rows (%d expected pairs) -> %s",
        model_type,
        n_pairs_banked,
        len(pairs),
        bank_root,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("instruct", "pretrained"))
    ap.add_argument("--phases", default="gen,capture", help="comma list: gen,capture,smoke_bank")
    ap.add_argument("--corpus-dir", required=True, help="dir containing prefix_store.jsonl")
    ap.add_argument(
        "--row-index-dir",
        required=True,
        help="root containing dynamics_{model}/row_index_{context_k,answer_k_t1}*.jsonl",
    )
    ap.add_argument("--out-dir", required=True, help="capture output root (per-model subdir)")
    ap.add_argument("--rollout-dir", required=True, help="rollout JSONL output dir")
    ap.add_argument(
        "--banked-json",
        default=str(REPO_ROOT / "eval_results/issue_825/turn_depth_map/results.json"),
        help="banked turn_depth_map results.json (G3 cardinality source)",
    )
    ap.add_argument("--bank-dir", default="", help="smoke_bank output root (smoke only)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-model-dir", default="", help="tiny model dir (smoke)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_VLLM_CHUNK_SIZE)
    args = ap.parse_args()
    if args.smoke and not args.tiny_model_dir:
        ap.error("--smoke requires --tiny-model-dir")

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in ("gen", "capture", "smoke_bank")]
    if unknown:
        ap.error(f"unknown phases: {unknown}")
    if "smoke_bank" in phases and not args.bank_dir:
        ap.error("smoke_bank requires --bank-dir")

    _seed_instruct_render_tokenizer(args)
    # The panel length filter is a JOINT pair-drop across BOTH models' renders
    # (store parity — issue1092 filtered with both tokenizers).
    tokenizers = {
        "instruct": _load_tokenizer("instruct", args),
        "pretrained": _load_tokenizer("pretrained", args),
    }
    for phase in phases:
        logger.info("[worker] model=%s phase=%s start", args.model, phase)
        if phase == "smoke_bank":
            run_smoke_bank(args, args.model, tokenizers)
        elif phase == "gen":
            run_gen(args, args.model, tokenizers)
        else:
            run_capture(args, args.model, tokenizers)
        logger.info("[worker] model=%s phase=%s done", args.model, phase)


if __name__ == "__main__":
    main()
