#!/usr/bin/env python
"""#825 turn-dynamics-allturns-5000 P2/P3 GPU worker (plan v24 §4).

Phases (one per invocation; the dispatcher's work-conserving queue composes
them across 8 GPUs):

  gen            arm-R forced-context generation: every (conv, assistant turn
                 t<=max_turn) prompt is INDEPENDENT (real logged context incl.
                 logged prior answers; the model writes only turn-k's answer —
                 round-10 parity). Chunked vLLM (use_tqdm=False, per-chunk INFO,
                 per-chunk JSONL checkpoints + fingerprint-gated resume).
  capture_own    arm-R own-answer capture: per-(conv, turn) teacher-forced
                 forward over context_k + own answer (window 8,192 — round-10
                 store parity), batched by token budget. Kinds: prefix_k /
                 context_k / answer_own_t1.
  capture_logged arm-R logged capture: ONE forward per conversation over the
                 FULL logged render (causal attention => all per-turn states in
                 one pass; includes the t>K_real logged tail). Window 8,192.
                 Runs the assumption-5 spot-check (one-forward == per-pair on
                 20 pairs, two-bar cosine gate).
  capture_armG   arm-G rollout capture: ONE forward per conversation over the
                 full depth-K_gen rollout (budget 15,872 / engine 16,384 —
                 declared divergence 5). Panel = first --panel-n alive seeds by
                 seed_rank (uniform across depths).

All captures store layers {14,18,19} at full n (fp16 npy, round-10 shard
format) PLUS all 28 layers for a fixed conv subsample (same forwards, more
layers read — plan §4 P3), with per-group shard checkpoints + resume.

Content hygiene: real-user corpora — conversation/rollout text is never
printed or logged; only counts, ids, hashes, and paths.
"""

from __future__ import annotations

import argparse
import hashlib
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

# vLLM v1 forks its EngineCore by default; parent CUDA init kills the child —
# spawn BEFORE any vllm import (round-10 crash att-20260715-141100).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from issue825_onpolicy_turn_depth_gpu import (  # noqa: E402
    GEN_MAX_TOKENS,
    GEN_N,
    GEN_SEED,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    MODEL_SPEC,
    _load_hf_model,
    _load_tokenizer,
    _render_gen_prompt,
    _seed_instruct_render_tokenizer,
)
from issue825_turndyn_harvest import read_jsonl_stem  # noqa: E402
from issue1092_gpu_phase import (  # noqa: E402
    DEFAULT_VLLM_CHUNK_SIZE,
    MAX_MODEL_LEN,
    _call_model_with_hidden_states,
    _dynamics_cut_plan,
    _render_full_conversation,
    _token_len,
    _write_jsonl_rows,
)

logger = logging.getLogger("i825_turndyn_gpu")

BASE_LAYERS = (14, 18, 19)  # round-10 parity (common.py FROZEN_LAYERS lineage)
ENGINE_MAX_MODEL_LEN = MAX_MODEL_LEN + GEN_MAX_TOKENS  # 9216 arm-R gen parity
CAPTURE_TOKEN_BUDGET = int(os.environ.get("EPM_CAPTURE_TOKEN_BUDGET", "16384"))
SPOTCHECK_N = 20  # assumption-5 one-forward-vs-per-pair equality pairs
# Two-bar cosine gate (bf16 single-position calibration — gotchas.md): early
# layer (14) per-vec >= 0.999; flattened all-checked >= 0.995.
SPOT_EARLY_COS_MIN = 0.999
SPOT_FLAT_COS_MIN = 0.995
DROP_REASONS = (
    "empty_completion",
    "zero_width_span",
    "span_assert",
    "window_overflow",
    "no_user_turn_before",
)


# ---------------------------------------------------------------------------
# panel loading
# ---------------------------------------------------------------------------


def _load_panel(args: argparse.Namespace) -> tuple[list[dict], int]:
    """Load the harvest panel stem -> ([{conv_id, turns}], max_turn).

    ``--max-turn 0`` reads K_real from harvest_report.json; ``-1`` = all
    assistant turns (the gc_panel round-10-parity shape).
    """
    rows = read_jsonl_stem(Path(args.panel_dir), args.panel_stem)
    panel = []
    for r in rows:
        cid = str(r.get("conv_id") or r.get("id"))
        panel.append({"conv_id": cid, "turns": r["turns"]})
    panel.sort(key=lambda r: r["conv_id"])
    max_turn = int(args.max_turn)
    if max_turn == 0:
        with open(Path(args.panel_dir) / "harvest_report.json") as f:
            max_turn = int(json.load(f)["K_real"])
    return panel, max_turn


def _assistant_pairs(turns: list[dict], max_turn: int) -> list[tuple[int, int]]:
    """[(turns-list index k, 1-based assistant-turn number t)] for a conversation."""
    out = []
    t = 0
    for k, turn in enumerate(turns):
        if turn.get("role") == "assistant":
            t += 1
            if max_turn >= 0 and t > max_turn:
                break
            out.append((k, t))
    return out


def _shard_select(items: list, shard: str) -> list:
    si, sn = (int(v) for v in shard.split("/"))
    assert 0 <= si < sn, shard
    return [x for i, x in enumerate(items) if i % sn == si]


# ---------------------------------------------------------------------------
# gen (P2): forced-context turn-k answers, chunk-checkpointed
# ---------------------------------------------------------------------------


def run_gen(args: argparse.Namespace) -> None:
    """Arm-R generation: chunked vLLM over independent forced-context prompts."""
    panel, max_turn = _load_panel(args)
    panel = _shard_select(panel, args.shard)
    tok = _load_tokenizer(args.model, args)
    jobs: list[dict] = []  # one per (conv, turn) prompt
    n_overflow = 0
    for row in panel:
        turns = row["turns"]
        for k, t in _assistant_pairs(turns, max_turn):
            prompt = _render_gen_prompt(turns[:k], args.model, tok)
            if _token_len(tok, prompt) + GEN_MAX_TOKENS > ENGINE_MAX_MODEL_LEN:
                n_overflow += 1
                continue
            jobs.append({"conv_id": row["conv_id"], "turn_index": k, "turn": t, "prompt": prompt})
    logger.info(
        "[gen] %s %s shard %s: %d prompts (%d prompt-overflow skipped)",
        args.model,
        args.panel_stem,
        args.shard,
        len(jobs),
        n_overflow,
    )

    out_root = Path(args.out_dir) / args.tag / args.model / f"shard{args.shard.replace('/', 'of')}"
    out_root.mkdir(parents=True, exist_ok=True)
    fp = {
        "phase": "gen",
        "model": args.model,
        "panel_stem": args.panel_stem,
        "max_turn": max_turn,
        "shard": args.shard,
        "n_jobs": len(jobs),
        "jobs_sha256": hashlib.sha256(
            "\n".join(f"{j['conv_id']}:{j['turn_index']}" for j in jobs).encode()
        ).hexdigest(),
        "sampling": {
            "n": GEN_N,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "max_tokens": GEN_MAX_TOKENS,
            "seed": GEN_SEED,
        },
        "engine_max_len": ENGINE_MAX_MODEL_LEN,
        "smoke": bool(args.smoke),
    }
    _assert_fingerprint(out_root / "gen_fingerprint.json", fp)

    chunk_size = max(1, int(args.chunk_size))
    n_chunks = math.ceil(len(jobs) / chunk_size)
    llm = None
    params = None
    if not args.smoke:
        from vllm import LLM, SamplingParams

        name, revision, stop_tokens = MODEL_SPEC[args.model]
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
    t0 = time.time()
    for ci in range(n_chunks):
        chunk_path = out_root / f"gen_chunk{ci:05d}.jsonl"
        if chunk_path.exists():
            continue  # chunk-checkpointed resume (fingerprint asserted above)
        chunk = jobs[ci * chunk_size : (ci + 1) * chunk_size]
        logger.info(
            "[gen] %s vLLM chunk %d/%d (%d prompts, %.0fs)",
            args.model,
            ci + 1,
            n_chunks,
            len(chunk),
            time.time() - t0,
        )
        if args.smoke:
            gens = [
                {
                    "text": f"Canned smoke answer for chunk {ci} item {i}: short reply.",
                    "finish_reason": "smoke_canned",
                    "n_gen_tokens": None,
                }
                for i in range(len(chunk))
            ]
        else:
            outputs = llm.generate([j["prompt"] for j in chunk], params, use_tqdm=False)
            gens = []
            for out in outputs:
                top = out.outputs[0] if out.outputs else None
                gens.append(
                    {
                        "text": top.text if top is not None else "",
                        "finish_reason": (top.finish_reason if top is not None else "no_output"),
                        "n_gen_tokens": len(top.token_ids) if top is not None else 0,
                    }
                )
        rows = []
        for j, g in zip(chunk, gens, strict=True):
            rows.append(
                {
                    "conv_id": j["conv_id"],
                    "turn_index": j["turn_index"],
                    "turn": j["turn"],
                    "model_type": args.model,
                    "text": g["text"],
                    "finish_reason": g["finish_reason"],
                    "n_gen_tokens": g["n_gen_tokens"],
                }
            )
        tmp = chunk_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")  # ensure_ascii: U+2028/NEL-safe
        os.replace(tmp, chunk_path)
    summary = {
        "model": args.model,
        "panel_stem": args.panel_stem,
        "shard": args.shard,
        "n_jobs": len(jobs),
        "n_chunks": n_chunks,
        "n_prompt_overflow": n_overflow,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_root / "gen_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    logger.info(
        "[gen] %s DONE: %d prompts in %d chunks -> %s", args.model, len(jobs), n_chunks, out_root
    )


def _assert_fingerprint(path: Path, fp: dict) -> None:
    """Regime-keyed resume guard (#722 r3): refuse resume on any key change."""
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        if old != fp:
            diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
            raise SystemExit(
                f"[fingerprint] MISMATCH on keys {diff} at {path}; refusing to resume — "
                f"move the output dir aside or fix the flags"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(fp, f, indent=1)


def _load_gen_rollout(args: argparse.Namespace) -> dict[tuple[str, int], dict]:
    """All gen chunks for (tag, model), across ALL shards -> {(conv, k): row}."""
    root = Path(args.gen_dir) / args.tag / args.model
    shard_dirs = sorted(p for p in root.glob("shard*") if p.is_dir())
    assert shard_dirs, f"[capture] no gen shards under {root}"
    rows: dict[tuple[str, int], dict] = {}
    for sd in shard_dirs:
        chunks = sorted(sd.glob("gen_chunk*.jsonl"))
        assert chunks, f"[capture] no gen chunks under {sd}"
        for cp in chunks:
            with cp.open(encoding="utf-8") as f:
                for line in f:  # file iteration, never splitlines (gotchas.md)
                    line = line.strip("\n")
                    if line:
                        r = json.loads(line)
                        rows[(str(r["conv_id"]), int(r["turn_index"]))] = r
    return rows


# ---------------------------------------------------------------------------
# capture core: batched teacher-forced forwards + per-turn cut extraction
# ---------------------------------------------------------------------------


def _subsample_ids(conv_ids: list[str], n: int) -> set[str]:
    """Deterministic layer-sweep subsample: first n conv_ids in sorted order."""
    return set(sorted(set(conv_ids))[: max(0, n)])


class _CaptureWriter:
    """Sharded fp16 store writer with per-group checkpoints + resume.

    Files: {kind}_L{layer:02d}_shard{gi:05d}.npy per completed group + one
    row_index_{tag}_shard{gi:05d}.jsonl (loadable via _load_summary /
    read shards in order). A group is the atomic resume unit.
    """

    def __init__(self, root: Path, kinds: list[str], base_layers: tuple[int, ...]):
        self.root = root
        self.kinds = kinds
        self.base_layers = base_layers
        root.mkdir(parents=True, exist_ok=True)

    def group_done(self, gi: int) -> bool:
        return (self.root / f"group{gi:05d}.done.json").exists()

    def write_group(
        self,
        gi: int,
        index_rows: list[dict],
        base_vecs: dict[str, list[np.ndarray]],
        sweep_rows: list[dict],
        sweep_vecs: dict[str, list[np.ndarray]],
        n_all_layers: int,
        hidden_dim: int,
    ) -> None:
        for kind in self.kinds:
            vals = base_vecs[kind]
            arr = (
                np.stack(vals, axis=0)
                if vals
                else np.empty((0, len(self.base_layers), hidden_dim), dtype=np.float16)
            )
            for li, layer in enumerate(self.base_layers):
                np.save(
                    self.root / f"{kind}_L{layer:02d}_shard{gi:05d}.npy",
                    arr[:, li, :].astype(np.float16),
                )
            svals = sweep_vecs[kind]
            if svals:
                sarr = np.stack(svals, axis=0)
                for layer in range(n_all_layers):
                    np.save(
                        self.root / f"sweep_{kind}_L{layer:02d}_shard{gi:05d}.npy",
                        sarr[:, layer, :].astype(np.float16),
                    )
        _write_jsonl_rows(self.root / f"row_index_shard{gi:05d}.jsonl", index_rows)
        if sweep_rows:
            _write_jsonl_rows(self.root / f"row_index_sweep_shard{gi:05d}.jsonl", sweep_rows)
        done = self.root / f"group{gi:05d}.done.json"
        tmp = done.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"gi": gi, "n_rows": len(index_rows)}))
        os.replace(tmp, done)


def _extract_turn_states(
    hs: list,
    local_i: int,
    cuts: dict,
    pairs: list[tuple[int, int]],
    layers: tuple[int, ...] | None,
    answer_key: str,
) -> list[tuple[int, int, dict[str, np.ndarray]]]:
    """Per assistant turn (k, t): prefix_k / context_k / answer mean vecs.

    ``layers=None`` -> all layers (the sweep read). Returns
    [(k, t, {kind: (L, H) fp16})] — turns whose cut cardinality fails are
    raised by the caller's cut construction, never silently skipped here.
    """
    ctx_by_turn = {tk: (s, e) for s, e, tk in cuts["context_k"]}
    ans_by_turn = {tk: (s, e) for s, e, tk in cuts["answer_k_t1"]}
    u1_by_turn = {tk: (s, e) for s, e, tk in cuts["u1"]}
    user_indices = sorted(u1_by_turn)
    layer_sel = list(range(len(hs))) if layers is None else list(layers)
    out = []
    for k, t in pairs:
        if k not in ctx_by_turn or k not in ans_by_turn:
            raise AssertionError(f"cut-plan missing assistant turn index {k}")
        u_prev = [u for u in user_indices if u < k]
        if not u_prev:
            raise AssertionError(f"no user turn before assistant turn index {k}")
        u_start = u1_by_turn[u_prev[-1]][0]
        prefix_pos = max(0, u_start - 1)
        ctx_pos = ctx_by_turn[k][0]
        a_start, a_end = ans_by_turn[k]
        if a_end <= a_start:
            raise AssertionError(f"zero-width answer span at turn index {k}")
        vecs: dict[str, np.ndarray] = {}
        for kind, sl in (
            ("prefix_k", slice(prefix_pos, prefix_pos + 1)),
            ("context_k", slice(ctx_pos, ctx_pos + 1)),
            (answer_key, slice(a_start, a_end)),
        ):
            vecs[kind] = np.stack(
                [
                    hs[layer][local_i, sl, :].mean(dim=0).float().cpu().numpy()
                    for layer in layer_sel
                ],
                axis=0,
            ).astype(np.float16)
        out.append((k, t, vecs))
    return out


def _capture_conversations(  # noqa: C901 — linear batched-capture driver
    args: argparse.Namespace,
    convs: list[dict],
    *,
    window: int,
    answer_key: str,
    tag: str,
    sweep_ids: set[str],
    spotcheck: bool = False,
) -> None:
    """ONE forward per conversation; extract every assistant turn's states.

    ``convs``: [{conv_id, turns, [extra index fields]}] (shard-selected).
    ``sweep_ids``: the GLOBAL (pre-shard) 28-layer subsample conv set — the
    caller selects it from the FULL panel so the union across shards is
    exactly --sweep-n convs. Overflow (> window) conversations are dropped +
    counted per depth (feeds the H4 asymmetry read). Group-checkpointed.
    """
    import torch

    tok = _load_tokenizer(args.model, args)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model, device, n_layers, hidden_dim = _load_hf_model(args.model, args)
    kinds = ["prefix_k", "context_k", answer_key]
    out_root = Path(args.out_dir) / tag / args.model / f"shard{args.shard.replace('/', 'of')}"
    writer = _CaptureWriter(out_root, kinds, BASE_LAYERS)
    fp = {
        "phase": tag,
        "model": args.model,
        "window": window,
        "n_convs": len(convs),
        "conv_sha256": hashlib.sha256(
            "\n".join(sorted(c["conv_id"] for c in convs)).encode()
        ).hexdigest(),
        "base_layers": list(BASE_LAYERS),
        "sweep_n": int(args.sweep_n),
        "smoke": bool(args.smoke),
    }
    _assert_fingerprint(out_root / "capture_fingerprint.json", fp)

    # pre-pass: renders + window drops (counted per depth)
    kept: list[dict] = []
    drops: list[dict] = []
    for c in convs:
        turns = c["turns"]
        render = _render_full_conversation(turns, args.model)
        n_tok = _token_len(tok, render)
        n_ass = sum(1 for t in turns if t.get("role") == "assistant")
        if n_tok > window:
            drops.append(
                {
                    "conv_id": c["conv_id"],
                    "n_tok": n_tok,
                    "depth": n_ass,
                    "reason": "window_overflow",
                }
            )
            continue
        kept.append({**c, "render": render, "n_tok": n_tok})
    logger.info(
        "[%s] %s: %d/%d conversations kept (window %d; %d overflow-dropped)",
        tag,
        args.model,
        len(kept),
        len(convs),
        window,
        len(drops),
    )

    group_size = max(1, int(args.group_size))
    n_groups = math.ceil(len(kept) / group_size)
    spot_pairs: list[dict] = []
    t0 = time.time()

    def _flush(batch_rows: list[dict], buf: dict) -> None:
        """One batched forward; extract per-turn states into the group buffers."""
        if not batch_rows:
            return
        texts = [r["render"] for r in batch_rows]
        inputs = tok(
            texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model, inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
            )
        hs = outputs.hidden_states[1:]
        assert len(hs) == n_layers and hs[0].shape[-1] == hidden_dim, (len(hs), hs[0].shape)
        for i, r in enumerate(batch_rows):
            n_tok = int(inputs["attention_mask"][i].sum().item())
            if n_tok != r["n_tok"]:
                raise AssertionError(
                    f"tokenization drift {r['conv_id']}: forward {n_tok} != pre-pass {r['n_tok']}"
                )
            full_ids = inputs["input_ids"][i, :n_tok].tolist()
            cuts = _dynamics_cut_plan(r["turns"], tok, args.model, n_tok, full_token_ids=full_ids)
            pairs = _assistant_pairs(r["turns"], -1)
            per_turn = _extract_turn_states(hs, i, cuts, pairs, BASE_LAYERS, answer_key)
            sweep_turn = (
                _extract_turn_states(hs, i, cuts, pairs, None, answer_key)
                if r["conv_id"] in sweep_ids
                else [(k2, t2, None) for k2, t2 in pairs]
            )
            for (k, t, vecs), (_, _, sw) in zip(per_turn, sweep_turn, strict=True):
                row = {
                    "conv_id": r["conv_id"],
                    "turn_index": k,
                    "turn": t,
                    "n_tokens": n_tok,
                    **{kx: r[kx] for kx in ("seed_rank", "brief_id") if kx in r},
                }
                buf["index_rows"].append(row)
                for kind in kinds:
                    buf["base_vecs"][kind].append(vecs[kind])
                if sw is not None:
                    buf["sweep_rows"].append(row)
                    for kind in kinds:
                        buf["sweep_vecs"][kind].append(sw[kind])
            if spotcheck and len(spot_pairs) < SPOTCHECK_N and per_turn:
                k, t, vecs = per_turn[-1]
                spot_pairs.append(
                    {"conv_id": r["conv_id"], "turns": r["turns"], "k": k, "vecs": vecs}
                )
        del outputs, hs, inputs

    for gi in range(n_groups):
        if writer.group_done(gi):
            continue
        grp = kept[gi * group_size : (gi + 1) * group_size]
        buf: dict = {
            "index_rows": [],
            "base_vecs": {k: [] for k in kinds},
            "sweep_rows": [],
            "sweep_vecs": {k: [] for k in kinds},
        }
        batch: list[dict] = []
        max_len = 0
        for r in grp:
            cand = max(max_len, r["n_tok"])
            if batch and cand * (len(batch) + 1) > CAPTURE_TOKEN_BUDGET:
                _flush(batch, buf)
                batch, max_len = [], 0
            batch.append(r)
            max_len = max(max_len, r["n_tok"])
        _flush(batch, buf)
        writer.write_group(
            gi,
            buf["index_rows"],
            buf["base_vecs"],
            buf["sweep_rows"],
            buf["sweep_vecs"],
            n_layers,
            hidden_dim,
        )
        logger.info(
            "[%s] %s group %d/%d: %d rows (%.0fs)",
            tag,
            args.model,
            gi + 1,
            n_groups,
            len(buf["index_rows"]),
            time.time() - t0,
        )

    per_depth_drop = {}
    for d in drops:
        per_depth_drop[str(d["depth"])] = per_depth_drop.get(str(d["depth"]), 0) + 1
    report = {
        "model_type": args.model,
        "tag": tag,
        "n_convs_in": len(convs),
        "n_convs_kept": len(kept),
        "n_window_overflow": len(drops),
        "per_depth_overflow": per_depth_drop,
        "overflow_conv_ids": sorted(d["conv_id"] for d in drops),
        "capture_window": window,
        "base_layers": list(BASE_LAYERS),
        "n_all_layers": n_layers,
        "hidden_dim": hidden_dim,
        "sweep_n": int(args.sweep_n),
        "n_sweep_convs": len(sweep_ids & {c["conv_id"] for c in kept}),
    }
    with open(out_root / "capture_report.json", "w") as f:
        json.dump(report, f, indent=1)
    if spotcheck:
        _run_spotcheck(
            args, spot_pairs, model, tok, device, n_layers, hidden_dim, out_root, answer_key
        )
    logger.info("[%s] %s DONE -> %s", tag, args.model, out_root)


def _run_spotcheck(
    args, spot_pairs, model, tok, device, n_layers, hidden_dim, out_root, answer_key
) -> None:
    """Assumption-5 verify: one-forward capture == per-pair truncated capture.

    For each sampled (conv, assistant turn k): re-run a TRUNCATED forward over
    turns[:k+1] and compare context_k + answer means against the full-forward
    slices. Two-bar gate: layer-14 cosine >= 0.999 per vec; flattened >= 0.995
    (bf16 single-position calibration, gotchas.md).
    """
    import torch

    if not spot_pairs:
        logger.info("[spotcheck] no pairs sampled — skipping")
        return
    early_li = BASE_LAYERS.index(14)
    flat_cos: list[float] = []
    early_cos: list[float] = []
    for sp in spot_pairs:
        turns_trunc = sp["turns"][: sp["k"] + 1]
        render = _render_full_conversation(turns_trunc, args.model)
        n_tok = _token_len(tok, render)
        inputs = tok(
            [render], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model, inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
            )
        hs = outputs.hidden_states[1:]
        full_ids = inputs["input_ids"][0, :n_tok].tolist()
        cuts = _dynamics_cut_plan(turns_trunc, tok, args.model, n_tok, full_token_ids=full_ids)
        pairs = _assistant_pairs(turns_trunc, -1)
        per_turn = _extract_turn_states(hs, 0, cuts, pairs[-1:], BASE_LAYERS, answer_key)
        _, _, trunc_vecs = per_turn[0]
        for kind in ("context_k", answer_key):
            a = sp["vecs"][kind].astype(np.float64)
            b = trunc_vecs[kind].astype(np.float64)
            for li in range(a.shape[0]):
                num = float(a[li] @ b[li])
                den = float(np.linalg.norm(a[li]) * np.linalg.norm(b[li])) or 1e-12
                c = num / den
                flat_cos.append(c)
                if li == early_li:
                    early_cos.append(c)
        del outputs, hs, inputs
    rec = {
        "n_pairs": len(spot_pairs),
        "early_layer": 14,
        "early_cos_min": float(min(early_cos)),
        "flat_cos_min": float(min(flat_cos)),
        "bars": {"early": SPOT_EARLY_COS_MIN, "flat": SPOT_FLAT_COS_MIN},
    }
    with open(out_root / "spotcheck.json", "w") as f:
        json.dump(rec, f, indent=1)
    if rec["early_cos_min"] < SPOT_EARLY_COS_MIN or rec["flat_cos_min"] < SPOT_FLAT_COS_MIN:
        raise SystemExit(
            f"[spotcheck] FAIL: one-forward != per-pair capture (early {rec['early_cos_min']:.6f} "
            f"/ flat {rec['flat_cos_min']:.6f} vs bars {SPOT_EARLY_COS_MIN}/{SPOT_FLAT_COS_MIN}) — "
            f"render/offset drift, halt before fitting (assumption 5)"
        )
    logger.info(
        "[spotcheck] PASS: %d pairs, early_cos_min %.6f flat_cos_min %.6f",
        rec["n_pairs"],
        rec["early_cos_min"],
        rec["flat_cos_min"],
    )


# ---------------------------------------------------------------------------
# capture_own (P3a): per-(conv, turn) forwards over context + own answer
# ---------------------------------------------------------------------------


def run_capture_own(args: argparse.Namespace) -> None:  # noqa: C901 — linear driver
    """Arm-R own capture at window 8,192 — the round-10 run_capture shape at scale."""
    import torch

    panel, max_turn = _load_panel(args)
    sweep_ids = _subsample_ids([c["conv_id"] for c in panel], args.sweep_n)
    panel = _shard_select(panel, args.shard)
    rollout = _load_gen_rollout(args)
    tok = _load_tokenizer(args.model, args)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kept: list[dict] = []
    drops: list[dict] = []
    per_turn_total: dict[int, int] = {}
    for c in panel:
        for k, t in _assistant_pairs(c["turns"], max_turn):
            per_turn_total[t] = per_turn_total.get(t, 0) + 1
            row = rollout.get((c["conv_id"], k))
            if row is None:
                # prompt-overflow at gen time (counted there) — count here too
                drops.append({"conv_id": c["conv_id"], "turn": t, "reason": "window_overflow"})
                continue
            content = str(row["text"]).strip()
            if not content:
                drops.append({"conv_id": c["conv_id"], "turn": t, "reason": "empty_completion"})
                continue
            turns_trunc = [*c["turns"][:k], {"role": "assistant", "content": content}]
            render = _render_full_conversation(turns_trunc, args.model)
            n_tok = _token_len(tok, render)
            if n_tok > MAX_MODEL_LEN:
                drops.append({"conv_id": c["conv_id"], "turn": t, "reason": "window_overflow"})
                continue
            kept.append(
                {
                    "conv_id": c["conv_id"],
                    "turn_index": k,
                    "turn": t,
                    "turns_trunc": turns_trunc,
                    "render": render,
                    "n_tok": n_tok,
                }
            )
    drop_counts = {r: sum(1 for d in drops if d["reason"] == r) for r in DROP_REASONS}
    logger.info(
        "[capture_own] %s shard %s: %d kept / %d dropped %s",
        args.model,
        args.shard,
        len(kept),
        len(drops),
        drop_counts,
    )

    model, device, n_layers, hidden_dim = _load_hf_model(args.model, args)
    kinds = ["prefix_k", "context_k", "answer_own_t1"]
    out_root = Path(args.out_dir) / args.tag / args.model / f"shard{args.shard.replace('/', 'of')}"
    writer = _CaptureWriter(out_root, kinds, BASE_LAYERS)
    fp = {
        "phase": "capture_own",
        "model": args.model,
        "panel_stem": args.panel_stem,
        "max_turn": max_turn,
        "shard": args.shard,
        "n_kept": len(kept),
        "kept_sha256": hashlib.sha256(
            "\n".join(f"{r['conv_id']}:{r['turn_index']}" for r in kept).encode()
        ).hexdigest(),
        "window": MAX_MODEL_LEN,
        "sweep_n": int(args.sweep_n),
        "smoke": bool(args.smoke),
    }
    _assert_fingerprint(out_root / "capture_fingerprint.json", fp)

    group_size = max(1, int(args.group_size)) * 4  # per-pair rows are smaller units
    n_groups = math.ceil(len(kept) / group_size)
    t0 = time.time()
    span_drop_count = [0]

    def _flush(batch_rows: list[dict], buf: dict) -> None:
        """One batched forward over per-pair truncated renders."""
        if not batch_rows:
            return
        texts = [r["render"] for r in batch_rows]
        inputs = tok(
            texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model, inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
            )
        hs = outputs.hidden_states[1:]
        for i, r in enumerate(batch_rows):
            n_tok = int(inputs["attention_mask"][i].sum().item())
            if n_tok != r["n_tok"]:
                raise AssertionError(
                    f"tokenization drift {r['conv_id']}/t{r['turn']}: {n_tok} != {r['n_tok']}"
                )
            full_ids = inputs["input_ids"][i, :n_tok].tolist()
            try:
                cuts = _dynamics_cut_plan(
                    r["turns_trunc"], tok, args.model, n_tok, full_token_ids=full_ids
                )
                per_turn = _extract_turn_states(
                    hs, i, cuts, [(r["turn_index"], r["turn"])], BASE_LAYERS, "answer_own_t1"
                )
            except AssertionError:
                # NEVER log the assert text: it can embed real-user content.
                span_drop_count[0] += 1
                drops.append({"conv_id": r["conv_id"], "turn": r["turn"], "reason": "span_assert"})
                continue
            _, _, vecs = per_turn[0]
            row = {
                "conv_id": r["conv_id"],
                "turn_index": r["turn_index"],
                "turn": r["turn"],
                "n_tokens": n_tok,
            }
            buf["index_rows"].append(row)
            for kind in kinds:
                buf["base_vecs"][kind].append(vecs[kind])
            if r["conv_id"] in sweep_ids:
                sw = _extract_turn_states(
                    hs, i, cuts, [(r["turn_index"], r["turn"])], None, "answer_own_t1"
                )[0][2]
                buf["sweep_rows"].append(row)
                for kind in kinds:
                    buf["sweep_vecs"][kind].append(sw[kind])
        del outputs, hs, inputs

    for gi in range(n_groups):
        if writer.group_done(gi):
            continue
        grp = kept[gi * group_size : (gi + 1) * group_size]
        buf: dict = {
            "index_rows": [],
            "base_vecs": {k: [] for k in kinds},
            "sweep_rows": [],
            "sweep_vecs": {k: [] for k in kinds},
        }
        batch: list[dict] = []
        max_len = 0
        for r in grp:
            cand = max(max_len, r["n_tok"])
            if batch and cand * (len(batch) + 1) > CAPTURE_TOKEN_BUDGET:
                _flush(batch, buf)
                batch, max_len = [], 0
            batch.append(r)
            max_len = max(max_len, r["n_tok"])
        _flush(batch, buf)
        writer.write_group(
            gi,
            buf["index_rows"],
            buf["base_vecs"],
            buf["sweep_rows"],
            buf["sweep_vecs"],
            n_layers,
            hidden_dim,
        )
        logger.info(
            "[capture_own] %s shard %s group %d/%d (%.0fs)",
            args.model,
            args.shard,
            gi + 1,
            n_groups,
            time.time() - t0,
        )
    n_span_drop = span_drop_count[0]

    drop_counts = {r: sum(1 for d in drops if d["reason"] == r) for r in DROP_REASONS}
    n_total = sum(per_turn_total.values())
    report = {
        "model_type": args.model,
        "tag": args.tag,
        "shard": args.shard,
        "n_total_pairs": n_total,
        "n_kept": len(kept) - n_span_drop,
        "drop_counts": drop_counts,
        "drop_rate": (n_total - (len(kept) - n_span_drop)) / max(1, n_total),
        "per_turn_total": {str(t): c for t, c in sorted(per_turn_total.items())},
        "capture_window": MAX_MODEL_LEN,
        "base_layers": list(BASE_LAYERS),
        "sweep_n": int(args.sweep_n),
    }
    with open(out_root / "capture_report.json", "w") as f:
        json.dump(report, f, indent=1)
    if report["drop_rate"] > 0.20:
        logger.warning(
            "[capture_own] %s: drop rate %.1f%% EXCEEDS the 20%% kill line — the fit "
            "script HALTs on this (generation-recipe problem, not a finding)",
            args.model,
            100 * report["drop_rate"],
        )
    logger.info("[capture_own] %s shard %s DONE -> %s", args.model, args.shard, out_root)


# ---------------------------------------------------------------------------
# capture_logged / capture_armG: one-forward-per-conversation captures
# ---------------------------------------------------------------------------


def run_capture_logged(args: argparse.Namespace) -> None:
    panel, _mt = _load_panel(args)
    sweep_ids = _subsample_ids([c["conv_id"] for c in panel], args.sweep_n)
    panel = _shard_select(panel, args.shard)
    _capture_conversations(
        args,
        panel,
        window=MAX_MODEL_LEN,
        answer_key="answer_logged_t1",
        tag=args.tag,
        sweep_ids=sweep_ids,
        spotcheck=(args.shard.split("/")[0] == "0"),  # once per model, shard 0
    )


def run_capture_armg(args: argparse.Namespace) -> None:
    """Arm-G capture from rollout_final: first --panel-n alive seeds by seed_rank."""
    model_root = Path(args.rollout_dir) / args.model
    shard_dirs = sorted(p for p in model_root.glob("shard*") if p.is_dir())
    assert shard_dirs, f"[capture_armG] no rollout shards under {model_root}"
    convs: list[dict] = []
    for sd in shard_dirs:
        convs.extend(read_jsonl_stem(sd, "rollout_final"))
    alive = sorted((c for c in convs if c["alive"]), key=lambda c: int(c["seed_rank"]))
    sel = alive[: args.panel_n] if args.panel_n else alive
    panel_ids = sorted(c["conv_id"] for c in sel)
    logger.info(
        "[capture_armG] %s: %d alive of %d rolled; panel = first %d by seed_rank",
        args.model,
        len(alive),
        len(convs),
        len(sel),
    )
    out_root = Path(args.out_dir) / args.tag / args.model
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "armG_panel_ids.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_panel": len(sel),
                "panel_ids_sha256": hashlib.sha256("\n".join(panel_ids).encode()).hexdigest(),
                "panel_ids": panel_ids,
            },
            f,
            indent=1,
        )
    sweep_ids = _subsample_ids(panel_ids, args.sweep_n)
    sel_shard = _shard_select(sel, args.shard)
    _capture_conversations(
        args,
        [
            {
                "conv_id": c["conv_id"],
                "turns": c["turns"],
                "seed_rank": c["seed_rank"],
                "brief_id": c["brief_id"],
            }
            for c in sel_shard
        ],
        window=args.capture_budget,
        answer_key="answer_own_t1",
        tag=args.tag,
        sweep_ids=sweep_ids,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("instruct", "pretrained"))
    ap.add_argument(
        "--phase",
        required=True,
        choices=("gen", "capture_own", "capture_logged", "capture_armG"),
    )
    ap.add_argument("--panel-dir", default="", help="harvest output dir (panel/seed shards)")
    ap.add_argument("--panel-stem", default="panel_armR", help="panel_armR | gc_panel")
    ap.add_argument("--max-turn", type=int, default=0, help="0=K_real from report; -1=all turns")
    ap.add_argument("--shard", default="0/1", help="i/n shard of the panel / rollout convs")
    ap.add_argument("--tag", default="armR_own", help="output subdir tag (arm identity)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gen-dir", default="", help="gen output root (capture_own reads it)")
    ap.add_argument("--rollout-dir", default="", help="P1 rollout root (capture_armG)")
    ap.add_argument("--panel-n", type=int, default=5000, help="arm-G capture panel size")
    ap.add_argument("--capture-budget", type=int, default=15872)
    ap.add_argument("--sweep-n", type=int, default=0, help="28-layer subsample conv count")
    ap.add_argument("--group-size", type=int, default=250, help="convs per checkpoint group")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_VLLM_CHUNK_SIZE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-model-dir", default="")
    args = ap.parse_args()
    if args.smoke and not args.tiny_model_dir:
        ap.error("--smoke requires --tiny-model-dir")
    if args.phase in ("gen", "capture_own", "capture_logged") and not args.panel_dir:
        ap.error(f"--phase {args.phase} requires --panel-dir")
    if args.phase == "capture_own" and not args.gen_dir:
        ap.error("--phase capture_own requires --gen-dir")
    if args.phase == "capture_armG" and not args.rollout_dir:
        ap.error("--phase capture_armG requires --rollout-dir")

    _seed_instruct_render_tokenizer(args)
    logger.info(
        "[worker] model=%s phase=%s tag=%s shard=%s", args.model, args.phase, args.tag, args.shard
    )
    if args.phase == "gen":
        run_gen(args)
    elif args.phase == "capture_own":
        run_capture_own(args)
    elif args.phase == "capture_logged":
        run_capture_logged(args)
    else:
        run_capture_armg(args)
    logger.info("[worker] model=%s phase=%s done", args.model, args.phase)


if __name__ == "__main__":
    main()
