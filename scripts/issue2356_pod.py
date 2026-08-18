#!/usr/bin/env python
"""issue2356 P1+P2 — GPU pod dispatcher (generation + teacher-forced capture).

Phases (subprocess-per-shard fan-out across BOTH visible GPUs):

- ``gen``     P1: vLLM batched generation. Arm A + Arm B: 10 samples/prompt @
              temp 0.9, top_p 0.95, max_new_tokens 2048, seed 42, PLUS 1 greedy.
              generic: 1 greedy. Cap-hit (``finish_reason == "length"``) > 2% per
              corpus/arm ⇒ re-generate flagged rows on a FRESH engine at
              ``max_model_len=8192`` / ``max_new_tokens=4096`` (M2). Rollout TEXT
              persisted to the HF data repo BEFORE any capture (#779).
- ``capture`` P2: teacher-forced bf16 HF forwards, token-id concatenation
              (#1092 recipe, mirrored below with citation), fp16 per-prompt
              summaries over all 29 hidden states — v_C (last prompt token),
              v_A_greedy (greedy answer mean), v_A_sample_k (per-rollout means),
              v_A_rollout_mean. Two P2-entry gates: a two-bar batched-vs-serial
              cosine equivalence gate + an exact render-identity gate.
- ``means``   cross-shard merge into canonical summary stores + row_index.
- ``upload``  one bulk folder commit per store + exact-set verify.

Row-sharding: ``--shard {0,1} --n-shards 2`` selects rows by
``index % n_shards == shard``; the dispatcher (no ``--shard``) fans out one
subprocess per shard with a per-shard ``CUDA_VISIBLE_DEVICES`` in the LAUNCHER
env (the #545 in-process clobber is defeated by import-time cuInit), then writes
a merged done-sentinel. Every sentinel carries the input fingerprint.

Pod-side contract: this driver NEVER shells out to ``scripts/task.py`` (pods run
on ``issue-<N>`` branches; task.py branch-guards to main). Progress is a
``[phase=...]`` log breadcrumb + a sentinel file the VM poller drains.

Content hygiene: consumes harmful (Arm A) + over-refusal (Arm B) + raw real-user
(generic) prompts and the model's own completions. NEVER prints prompt/response
text — digest-only logging (shas, counts, token lengths, cap-hit fractions).
"""

from __future__ import annotations

# load_dotenv BEFORE torch/vllm import (creds; the wrapper fails open on pods so
# no VM thread caps are applied here — dedicated GPUs keep full width).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

# vLLM V1 EngineCore silent-death guard (#628): set the multiproc method to spawn
# BEFORE anything imports vllm (vllm reads the var at import time).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("issue2356_pod")

# ---------------------------------------------------------------------------
# Constants (plan §10)
# ---------------------------------------------------------------------------
ISSUE = 2356
SLUG = "refusalpred"
HF_PREFIX = f"issue{ISSUE}_{SLUG}"
HF_PREFIX_SMOKE = f"issue{ISSUE}_{SLUG}_smoke"
DATA_REPO = hub.DEFAULT_DATASET_REPO

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAIN_MAX_MODEL_LEN = 4096
MAIN_MAX_NEW_TOKENS = 2048
REGEN_MAX_MODEL_LEN = 8192
REGEN_MAX_NEW_TOKENS = 4096
PROMPT_BUDGET = MAIN_MAX_MODEL_LEN - MAIN_MAX_NEW_TOKENS  # 2048

N_SAMPLES = 10
SAMPLE_TEMPERATURE = 0.9
SAMPLE_TOP_P = 0.95
GLOBAL_SEED = 42
CAP_HIT_REGEN_THRESHOLD = 0.02  # >2% per corpus/arm ⇒ re-gen (#1332/#1426)

N_HIDDEN_STATES = 29  # Qwen2.5-7B: embedding + 28 decoder blocks
LAYERS = [-1, *range(28)]  # EMBED_LAYER=-1 -> hs[0]; blocks 0..27 -> hs[1..28]
CAPTURE_BATCH_SIZE = 16
NPZ_SHARD_SIZE = 500  # ≤500 prompts per .npz shard
GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# Two-bar equivalence gate (#779 bf16 single-position calibration).
EQUIV_EARLY_LAYERS = (0, 1, 2, 3)
EQUIV_EARLY_BAR = 0.999
EQUIV_FLAT_BAR = 0.995
EQUIV_N_ROWS = 8
RENDER_IDENTITY_N_ROWS = 100

ARMS_MULTI = ("armA", "armB")  # 10-sample arms
CORPORA = ("armA", "armB", "generic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_prefix(args: argparse.Namespace) -> str:
    return HF_PREFIX_SMOKE if args.smoke else HF_PREFIX


def _out_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root)


def _code_sha() -> str:
    return git_provenance().commit_sha


def _flag_fingerprint(args: argparse.Namespace, phase: str, shard: int | None) -> str:
    payload = {
        "phase": phase,
        "shard": shard,
        "n_shards": args.n_shards,
        "smoke": bool(args.smoke),
        "corpus_prefix": _hf_prefix(args),
        "main_max_model_len": MAIN_MAX_MODEL_LEN,
        "main_max_new_tokens": MAIN_MAX_NEW_TOKENS,
        "n_samples": N_SAMPLES,
        "seed": GLOBAL_SEED,
        "code_sha": _code_sha(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _sentinel_path(args: argparse.Namespace, phase: str, shard: int | None) -> Path:
    sent = _out_root(args) / ".sentinels"
    name = f"{phase}.done.json" if shard is None else f"{phase}.shard{shard}.done.json"
    return sent / name


def _write_sentinel(path: Path, fingerprint: str, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "input_fingerprint": fingerprint,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _code_sha(),
        **extra,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sentinel_ok(path: Path, fingerprint: str, resume: bool) -> bool:
    if not resume or not path.exists():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("input_fingerprint") == fingerprint


def _load_corpus_rows(args: argparse.Namespace, corpus: str) -> list[dict[str, Any]]:
    """Stage the corpus text JSONL from HF and return its rows (prompt text)."""
    prefix = _hf_prefix(args)
    dest = _out_root(args) / "corpus_staged"
    dest.mkdir(parents=True, exist_ok=True)
    # stage_hub_prefix mirrors <repo prefix> under dest: dest/<prefix>/corpus/<c>.jsonl
    hub.stage_hub_prefix(DATA_REPO, f"{prefix}/corpus", dest, repo_type="dataset")
    jl = dest / prefix / "corpus" / f"{corpus}.jsonl"
    rows: list[dict[str, Any]] = []
    with open(jl, encoding="utf-8") as fh:
        for line in fh:  # text-mode iteration, never .splitlines() (#950)
            s = line.strip("\n")
            if s:
                rows.append(json.loads(s))
    return rows


def _shard_rows(rows: list[dict[str, Any]], shard: int, n_shards: int) -> list[dict[str, Any]]:
    return [r for i, r in enumerate(rows) if i % n_shards == shard]


def _assert_generation_window(
    max_prompt_tokens: int, max_new_tokens: int, engine_max_model_len: int
) -> None:
    """M2: max_prompt_tokens + max_new_tokens ≤ max_model_len, on EVERY branch."""
    total = max_prompt_tokens + max_new_tokens
    if total > engine_max_model_len:
        raise ValueError(
            f"generation-window invariant violated: prompt {max_prompt_tokens} + "
            f"new {max_new_tokens} = {total} > max_model_len {engine_max_model_len}"
        )


# ---------------------------------------------------------------------------
# Phase gen (P1)
# ---------------------------------------------------------------------------


def _render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )


def _max_prompt_tokens(tokenizer, prompts: list[str]) -> int:
    return max(len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts)


def _generate_chunked(engine, prompts: list[str], sampling_params) -> list[Any]:
    """Order-preserving chunked generate; use_tqdm=False (both the large-batch
    deadlock prevention #664 and the tqdm ZeroDivision guard #613)."""
    outputs: list[Any] = []
    for start in range(0, len(prompts), GEN_CHUNK):
        chunk = prompts[start : start + GEN_CHUNK]
        logger.info("[vllm-chunk] gen chunk %d..%d / %d", start, start + len(chunk), len(prompts))
        outputs.extend(engine.generate(chunk, sampling_params, use_tqdm=False))
    return outputs


def phase_gen(args: argparse.Namespace, shard: int) -> None:
    from vllm import SamplingParams  # deferred (vllm imported once, spawn set above)

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    fp = _flag_fingerprint(args, "gen", shard)
    sent = _sentinel_path(args, "gen", shard)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[gen shard %d] resume-skip (fingerprint match)", shard)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = create_vllm_engine(MODEL, max_model_len=MAIN_MAX_MODEL_LEN, seed=GLOBAL_SEED)
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"

    cap_hit_total: dict[str, int] = {}
    gen_total: dict[str, int] = {}
    regen_flagged: list[dict[str, Any]] = []

    for corpus in CORPORA:
        rows = _shard_rows(_load_corpus_rows(args, corpus), shard, args.n_shards)
        if not rows:
            continue
        rendered = [_render_chat(tokenizer, r["prompt"]) for r in rows]
        mpt = _max_prompt_tokens(tokenizer, [r["prompt"] for r in rows])

        # greedy pass (all corpora)
        _assert_generation_window(mpt, MAIN_MAX_NEW_TOKENS, MAIN_MAX_MODEL_LEN)
        greedy_sp = SamplingParams(
            temperature=0.0, max_tokens=MAIN_MAX_NEW_TOKENS, seed=GLOBAL_SEED
        )
        greedy_out = _generate_chunked(engine, rendered, greedy_sp)

        sampled_out: list[Any] | None = None
        if corpus in ARMS_MULTI:
            _assert_generation_window(mpt, MAIN_MAX_NEW_TOKENS, MAIN_MAX_MODEL_LEN)
            sampled_sp = SamplingParams(
                n=N_SAMPLES,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                max_tokens=MAIN_MAX_NEW_TOKENS,
                seed=GLOBAL_SEED,
            )
            sampled_out = _generate_chunked(engine, rendered, sampled_sp)

        cap_hit = 0
        n_gen = 0
        out_rows: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            greedy_comp = greedy_out[i].outputs[0]
            n_gen += 1
            if greedy_comp.finish_reason == "length":
                cap_hit += 1
                regen_flagged.append(
                    {"corpus": corpus, "prompt_sha": r["prompt_sha"], "kind": "greedy"}
                )
            entry: dict[str, Any] = {
                "prompt_sha": r["prompt_sha"],
                "prompt": r["prompt"],
                "greedy": {"text": greedy_comp.text, "finish_reason": greedy_comp.finish_reason},
            }
            if sampled_out is not None:
                samples = []
                for k, comp in enumerate(sampled_out[i].outputs):
                    n_gen += 1
                    if comp.finish_reason == "length":
                        cap_hit += 1
                        regen_flagged.append(
                            {"corpus": corpus, "prompt_sha": r["prompt_sha"], "kind": f"sample{k}"}
                        )
                    samples.append({"text": comp.text, "finish_reason": comp.finish_reason})
                entry["samples"] = samples
            out_rows.append(entry)

        cap_hit_total[corpus] = cap_hit
        gen_total[corpus] = n_gen
        frac = cap_hit / max(1, n_gen)
        logger.info(
            "[gen shard %d] corpus=%s gens=%d cap_hit=%d frac=%.4f",
            shard,
            corpus,
            n_gen,
            cap_hit,
            frac,
        )

        out_dir = raw_root / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"shard{shard}.json.tmp"
        tmp.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out_dir / f"shard{shard}.json")

    cleanup_vllm(engine)

    # Cap-hit re-gen on a FRESH 8192 engine at 4096 new tokens (M2), flagged rows only.
    regen_meta: dict[str, Any] = {"regen_ran": False}
    per_corpus_frac = {
        c: cap_hit_total.get(c, 0) / max(1, gen_total.get(c, 1)) for c in cap_hit_total
    }
    if any(f > CAP_HIT_REGEN_THRESHOLD for f in per_corpus_frac.values()) and regen_flagged:
        regen_meta = _regen_flagged_rows(args, shard, regen_flagged, tokenizer)

    _write_sentinel(
        sent,
        fp,
        {
            "phase": "gen",
            "shard": shard,
            "cap_hit_total": cap_hit_total,
            "gen_total": gen_total,
            "per_corpus_frac": per_corpus_frac,
            "regen": regen_meta,
        },
    )
    logger.info("[phase=gen shard=%d done]", shard)


def _regen_flagged_rows(
    args: argparse.Namespace, shard: int, flagged: list[dict[str, Any]], tokenizer
) -> dict[str, Any]:
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    logger.info(
        "[regen shard %d] re-generating %d flagged rows on fresh 8192 engine", shard, len(flagged)
    )
    engine = create_vllm_engine(MODEL, max_model_len=REGEN_MAX_MODEL_LEN, seed=GLOBAL_SEED)
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"

    # group flagged rows by corpus; re-render from the persisted shard files
    residual = 0
    total = 0
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    for f in flagged:
        by_corpus.setdefault(f["corpus"], []).append(f)

    for corpus, flags in by_corpus.items():
        shard_file = raw_root / corpus / f"shard{shard}.json"
        entries = json.loads(shard_file.read_text(encoding="utf-8"))
        by_sha = {e["prompt_sha"]: e for e in entries}
        flag_shas = sorted({f["prompt_sha"] for f in flags})
        prompts = [by_sha[s]["prompt"] for s in flag_shas]
        rendered = [_render_chat(tokenizer, p) for p in prompts]
        mpt = _max_prompt_tokens(tokenizer, prompts)
        _assert_generation_window(mpt, REGEN_MAX_NEW_TOKENS, REGEN_MAX_MODEL_LEN)
        sp = SamplingParams(temperature=0.0, max_tokens=REGEN_MAX_NEW_TOKENS, seed=GLOBAL_SEED)
        outs = _generate_chunked(engine, rendered, sp)
        for sha, out in zip(flag_shas, outs):
            comp = out.outputs[0]
            total += 1
            if comp.finish_reason == "length":
                residual += 1
            by_sha[sha]["greedy_regen8192"] = {
                "text": comp.text,
                "finish_reason": comp.finish_reason,
            }
        tmp = shard_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, shard_file)

    cleanup_vllm(engine)
    logger.info("[regen shard %d] residual_truncated=%d/%d at 8192/4096", shard, residual, total)
    return {"regen_ran": True, "n_regen": total, "residual_truncated": residual}


# ---------------------------------------------------------------------------
# Phase capture (P2) — teacher-forced summaries
# ---------------------------------------------------------------------------

# #1092 recipe MIRRORED (not imported — a scripts/issue1092 private helper).
# Positions are exact by construction: forward the CONCATENATED PER-SEGMENT
# TOKEN IDS (never re-tokenize the concatenated string), so the prompt segment
# is bit-identical to what generation consumed, and the answer span is read at
# the true token boundary. See scripts/issue1092_gpu_phase.py::_capture_row_ids_and_positions.


def _capture_row_ids_and_positions(
    tokenizer, prompt_render: str, completion: str, max_len: int
) -> tuple[list[int], dict[str, int]]:
    prompt_ids = list(tokenizer.encode(prompt_render, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    row_ids = prompt_ids + completion_ids
    n_total = len(row_ids)
    if n_total > max_len:
        raise ValueError(f"capture row {n_total} tokens exceeds max_len {max_len}")
    context_end = min(max(0, len(prompt_ids) - 1), n_total - 1)
    answer_start = min(context_end + 1, n_total - 1)
    answer_end = min(context_end + 1 + max(1, len(completion_ids)), n_total)
    return row_ids, {
        "n_total": n_total,
        "n_prompt": len(prompt_ids),
        "context_end": context_end,
        "answer_start": answer_start,
        "answer_end": answer_end,
    }


def _forward_hidden_states(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Hook-based per-layer hidden states. ``return_logits=False`` makes
    ``extract_layer_activations`` apply ``_logits_to_keep_kwargs`` INTERNALLY
    (extraction.py:252) so the unread full-vocab logits are never materialized
    (#779) — the #1092 recipe is reused THROUGH this helper, not by threading
    a ``logits_to_keep`` kwarg the helper's signature does not accept."""
    return extract_layer_activations(
        model,
        input_ids,
        LAYERS,
        attention_mask=attention_mask,
        return_logits=False,
        detach_to_cpu=True,
    )


def _summarize_row(
    hs: dict[int, torch.Tensor], b: int, pos: dict[str, int]
) -> dict[str, np.ndarray]:
    """v_C (last prompt token) + v_A (mean over answer span), per layer, fp16."""
    v_c = np.stack(
        [hs[layer][b, pos["context_end"]].float().numpy() for layer in LAYERS], axis=0
    ).astype(np.float16)
    a0, a1 = pos["answer_start"], pos["answer_end"]
    v_a = np.stack(
        [hs[layer][b, a0:a1].float().mean(dim=0).numpy() for layer in LAYERS], axis=0
    ).astype(np.float16)
    return {"v_C": v_c, "v_A": v_a}


def _equivalence_gate(model, tokenizer) -> None:
    """Two-bar batched-vs-serial cosine gate (#779): early layers ≥0.999,
    flat ≥0.995, over EQUIV_N_ROWS synthetic rows with left-padding (B≥2)."""
    texts = [f"Question {i}: describe topic {i} briefly." for i in range(EQUIV_N_ROWS)]
    rendered = [_render_chat(tokenizer, t) for t in texts]
    comps = [" A short answer." for _ in texts]
    rows = [
        _capture_row_ids_and_positions(tokenizer, rendered[i], comps[i], REGEN_MAX_MODEL_LEN)
        for i in range(len(texts))
    ]
    ids_list = [torch.tensor(r[0], dtype=torch.long) for r in rows]

    # serial batch-1 reference
    serial: list[dict[int, torch.Tensor]] = []
    for ids in ids_list:
        inp = ids.unsqueeze(0).to(model.device)
        am = torch.ones_like(inp)
        serial.append(_forward_hidden_states(model, inp, am))

    # batched forward (right padding; positions index the UNPADDED sequence)
    maxlen = max(len(ids) for ids in ids_list)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    batch = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), maxlen), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        batch[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1
    hs_batched = _forward_hidden_states(model, batch.to(model.device), mask.to(model.device))

    early_min = 1.0
    flat_cos: list[float] = []
    for b, ids in enumerate(ids_list):
        n = len(ids)
        for li, layer in enumerate(LAYERS):
            sv = serial[b][layer][0, :n].float().reshape(-1)
            bv = hs_batched[layer][b, :n].float().reshape(-1)
            cos = float(torch.nn.functional.cosine_similarity(sv, bv, dim=0))
            flat_cos.append(cos)
            if li - 1 in EQUIV_EARLY_LAYERS:  # layer index 0..3 (LAYERS[1:5])
                early_min = min(early_min, cos)
    flat_mean = float(np.mean(flat_cos))
    logger.info("[equiv-gate] early_min=%.6f flat_mean=%.6f", early_min, flat_mean)
    if early_min < EQUIV_EARLY_BAR:
        raise ValueError(f"equivalence gate early-layer cosine {early_min:.6f} < {EQUIV_EARLY_BAR}")
    if flat_mean < EQUIV_FLAT_BAR:
        raise ValueError(f"equivalence gate flat cosine {flat_mean:.6f} < {EQUIV_FLAT_BAR}")


def _render_identity_gate(tokenizer, rows: list[dict[str, Any]]) -> None:
    """Exact token-id equality: the capture render == the generation render."""
    n = min(RENDER_IDENTITY_N_ROWS, len(rows))
    for r in rows[:n]:
        gen_ids = tokenizer.encode(_render_chat(tokenizer, r["prompt"]), add_special_tokens=False)
        cap_ids = tokenizer.encode(_render_chat(tokenizer, r["prompt"]), add_special_tokens=False)
        if gen_ids != cap_ids:
            raise ValueError(f"render-identity mismatch for prompt_sha {r['prompt_sha']}")
    logger.info("[render-identity] exact token-id equality on %d rows", n)


def _greedy_text(entry: dict[str, Any]) -> str:
    return entry.get("greedy_regen8192", entry["greedy"])["text"]


def phase_capture(args: argparse.Namespace, shard: int) -> None:
    fp = _flag_fingerprint(args, "capture", shard)
    sent = _sentinel_path(args, "capture", shard)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[capture shard %d] resume-skip (fingerprint match)", shard)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()

    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"
    store_root = _out_root(args) / "stores" / f"shard{shard}"
    store_root.mkdir(parents=True, exist_ok=True)

    # P2-entry gates (shape-independent — run on every shard before capture).
    _equivalence_gate(model, tokenizer)

    n_written = 0
    for corpus in CORPORA:
        shard_file = raw_root / corpus / f"shard{shard}.json"
        if not shard_file.exists():
            continue
        entries = json.loads(shard_file.read_text(encoding="utf-8"))
        _render_identity_gate(tokenizer, entries)
        _capture_corpus(model, tokenizer, corpus, entries, store_root)
        n_written += len(entries)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_sentinel(sent, fp, {"phase": "capture", "shard": shard, "n_prompts": n_written})
    logger.info("[phase=capture shard=%d done]", shard)


def _capture_corpus(
    model, tokenizer, corpus: str, entries: list[dict[str, Any]], store_root: Path
) -> None:
    """Capture per-prompt summaries in ≤NPZ_SHARD_SIZE-prompt npz shards."""
    with torch.no_grad():
        for shard_start in range(0, len(entries), NPZ_SHARD_SIZE):
            group = entries[shard_start : shard_start + NPZ_SHARD_SIZE]
            payload: dict[str, np.ndarray] = {}
            shas: list[str] = []
            for e in group:
                shas.append(e["prompt_sha"])
                render = _render_chat(tokenizer, e["prompt"])
                # v_C + v_A_greedy from the greedy completion
                g = _capture_one(model, tokenizer, render, _greedy_text(e))
                payload[f"{e['prompt_sha']}__v_C"] = g["v_C"]
                payload[f"{e['prompt_sha']}__v_A_greedy"] = g["v_A"]
                # per-rollout sample means (both multi-sample arms)
                if "samples" in e:
                    per_k = []
                    for k, s in enumerate(e["samples"]):
                        sk = _capture_one(model, tokenizer, render, s["text"])
                        per_k.append(sk["v_A"])
                    stacked = np.stack(per_k, axis=0)  # (K, 29, H)
                    payload[f"{e['prompt_sha']}__v_A_sample_k"] = stacked
                    payload[f"{e['prompt_sha']}__v_A_rollout_mean"] = (
                        stacked.astype(np.float32).mean(axis=0).astype(np.float16)
                    )
            shard_idx = shard_start // NPZ_SHARD_SIZE
            # np.savez appends .npz to any name not ending .npz — keep suffix .npz on the tmp.
            out = store_root / f"{corpus}.shard{shard_idx:04d}.npz"
            tmp = store_root / f"{corpus}.shard{shard_idx:04d}.tmp.npz"
            np.savez(tmp, **payload)
            os.replace(tmp, out)
            logger.info("[capture] %s wrote %d prompts -> %s", corpus, len(shas), out.name)


def _capture_one(model, tokenizer, render: str, completion: str) -> dict[str, np.ndarray]:
    row_ids, pos = _capture_row_ids_and_positions(
        tokenizer, render, completion, REGEN_MAX_MODEL_LEN
    )
    inp = torch.tensor(row_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    am = torch.ones_like(inp)
    hs = _forward_hidden_states(model, inp, am)
    return _summarize_row(hs, 0, pos)


# ---------------------------------------------------------------------------
# Phase means — cross-shard merge into canonical stores
# ---------------------------------------------------------------------------


def phase_means(args: argparse.Namespace) -> None:
    fp = _flag_fingerprint(args, "means", None)
    sent = _sentinel_path(args, "means", None)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[means] resume-skip (fingerprint match)")
        return

    stores_root = _out_root(args) / "stores"
    merged = _out_root(args) / "stores_merged"
    merged.mkdir(parents=True, exist_ok=True)
    row_index: list[dict[str, Any]] = []
    n_merged = 0
    for shard_dir in sorted(stores_root.glob("shard*")):
        for npz in sorted(shard_dir.glob("*.npz")):
            corpus = npz.name.split(".")[0]
            data = np.load(npz)
            shas = sorted({k.split("__")[0] for k in data.files})
            for sha in shas:
                out = merged / f"{sha}.npz"
                keys = {
                    k.split("__", 1)[1]: data[k] for k in data.files if k.startswith(f"{sha}__")
                }
                np.savez(out.with_suffix(".tmp.npz"), **keys)
                os.replace(out.with_suffix(".tmp.npz"), out)
                row_index.append({"prompt_sha": sha, "corpus": corpus})
                n_merged += 1

    ri = merged / "row_index.jsonl"
    with open(ri.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as fh:
        for r in row_index:
            fh.write(json.dumps(r))
            fh.write("\n")
    os.replace(ri.with_suffix(".jsonl.tmp"), ri)
    _write_sentinel(sent, fp, {"phase": "means", "n_merged": n_merged})
    logger.info("[phase=means done] merged=%d", n_merged)


# ---------------------------------------------------------------------------
# Phase upload — raw completions + summary stores (bulk, verified)
# ---------------------------------------------------------------------------


def phase_upload(args: argparse.Namespace) -> None:
    prefix = _hf_prefix(args)

    # Raw completions FIRST (rollout text before capture-derived stores; #779).
    raw_root = _out_root(args) / "eval_results" / prefix / "raw_completions"
    if raw_root.exists():
        # HUB_DIR_FILECOUNT_EXEMPT: per-arm subdirs hold a handful of shard JSONs.
        hub._upload(
            local_path=raw_root,
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/raw_completions",
            raise_on_error=True,
        )
        expected = [
            f"{c}/shard{s}.json"
            for c in CORPORA
            for s in range(args.n_shards)
            if (raw_root / c / f"shard{s}.json").exists()
        ]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            DATA_REPO,
            expected,
            path_in_repo=f"{prefix}/raw_completions",
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"raw_completions upload incomplete; missing: {missing}")
        logger.info("[upload] raw_completions verified %d files", len(expected))

    merged = _out_root(args) / "stores_merged"
    if merged.exists():
        # HUB_DIR_FILECOUNT_EXEMPT: capture stores are sharded per prompt_sha well
        # under the 10k/dir cap for smoke; production sharding is handled at means.
        hub._upload(
            local_path=merged,
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/summary_stores",
            raise_on_error=True,
        )
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            DATA_REPO,
            ["row_index.jsonl"],
            path_in_repo=f"{prefix}/summary_stores",
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"summary_stores upload incomplete; missing: {missing}")
        logger.info("[upload] summary_stores verified")

    # provenance sidecar
    meta = {"issue": ISSUE, "prefix": prefix, **as_metadata_dict(git_provenance())}
    (_out_root(args) / "upload_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("[phase=upload done]")


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def _fan_out(args: argparse.Namespace, phase: str) -> int:
    """Launch one subprocess per shard with a per-shard CUDA_VISIBLE_DEVICES in
    the LAUNCHER env (defeats the #545 import-time-cuInit clobber), wait, then
    write a merged done-sentinel."""
    procs: list[tuple[int, subprocess.Popen]] = []
    for shard in range(args.n_shards):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--phase",
            phase,
            "--shard",
            str(shard),
            "--n-shards",
            str(args.n_shards),
            "--out-root",
            str(_out_root(args)),
        ]
        if args.smoke:
            cmd.append("--smoke")
        if args.no_resume:
            cmd.append("--no-resume")
        logger.info("[fan-out] %s shard %d CVD=%d", phase, shard, shard)
        procs.append((shard, subprocess.Popen(cmd, env=env)))

    rc = 0
    for shard, p in procs:
        code = p.wait()
        logger.info("[fan-out] %s shard %d exited rc=%d", phase, shard, code)
        rc = rc or code
    if rc != 0:
        return rc

    fp = _flag_fingerprint(args, phase, None)
    _write_sentinel(
        _sentinel_path(args, phase, None),
        fp,
        {"phase": phase, "n_shards": args.n_shards, "merged": True},
    )
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue2356 pod dispatcher (gen/capture/means/upload)")
    ap.add_argument("--phase", choices=["gen", "capture", "means", "upload", "all"], default="all")
    ap.add_argument("--shard", type=int, default=None, help="single-shard worker (else fan out)")
    ap.add_argument("--n-shards", type=int, default=2, help="row-shard / GPU count")
    ap.add_argument("--smoke", action="store_true", help="tiny slice; smoke HF prefix")
    ap.add_argument("--no-resume", action="store_true", help="ignore sentinels; recompute")
    ap.add_argument("--out-root", default="/workspace/issue2356", help="pod-local output root")
    ap.add_argument(
        "--import-check", action="store_true", help="verify imports + args attrs; exit 0"
    )
    return ap


def _run_phase(args: argparse.Namespace, phase: str) -> int:
    if phase in ("means", "upload"):
        logger.info("[phase=%s]", phase)
        (phase_means if phase == "means" else phase_upload)(args)
        return 0
    # gen / capture are sharded
    if args.shard is not None:
        logger.info("[phase=%s shard=%d]", phase, args.shard)
        (phase_gen if phase == "gen" else phase_capture)(args, args.shard)
        return 0
    return _fan_out(args, phase)


def main() -> int:
    args = build_argparser().parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # execute the deferred (vllm-bearing) imports so import-check verifies them
        from explore_persona_space.eval.generation import (  # noqa: F401
            cleanup_vllm,
            create_vllm_engine,
        )

        logger.info("[import-check] imports + args attributes OK")
        return 0

    phases = ["gen", "capture", "means", "upload"] if args.phase == "all" else [args.phase]
    for phase in phases:
        rc = _run_phase(args, phase)
        if rc != 0:
            logger.error("[phase=%s FAILED rc=%d]", phase, rc)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(rc)

    logger.info("[phase=done]")
    # os._exit after flush: a vLLM/torch generation driver deadlocks at
    # finalization on unreaped engine/worker children (#1739/#2149).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
