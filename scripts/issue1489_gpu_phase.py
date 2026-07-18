#!/usr/bin/env python3
"""Issue #1489 GPU phases: generation, capture, distillation, dose probes, FT eval.

Thin adaptation of ``scripts/issue1092_gpu_phase.py`` (plan §4.2/§4.3): the
work-conserving per-GPU worker queue, sentinel files
(``/workspace/logs/issue-1489-*.json``), per-shard checkpoint resume and the
stream-reduce teacher-forced capture are inherited; the #1489 deltas are
(a) 17 augmentation cells rendered from the P0 conditions manifest,
(b) SEVEN capture kinds (prefix_end, prefix_mean, context_end, context_mean,
t1, t1_odd, t1_even) written in the parent summaries schema
(``summaries/<cell>/{kind}_L{layer:02d}_shard*.npy`` + row_index shards) so
``issue1092_fit_grid._load_summary`` consumes them unchanged, and
(c) the P3 context-distillation LoRA runs + P4/P4b checkpoint phases.

Phases (each fans out over all visible GPUs; cells always enumerate FROM the
conditions manifest, so ``--smoke`` P0 output threads the canary subset
through every phase — PASS_UNIFIED):

    gen           P1  vLLM greedy generation over all manifest rows
    capture       P2  teacher-forced HF capture (7 kinds x 28 layers, fp16)
    distill       P3  LoRA context-distillation runs (2/family, plan §4.3)
    dose_probes   P4  per-checkpoint generation on the probe rows (multi-LoRA)
    ft_gen        P4b generation under the selected checkpoints (plain eval rows)
    ft_capture    P4b capture for cells cell_ft_<slug> (+ cell_plain re-captures)
    upload        upload raw completions + summaries to the issue HF prefix

Decoding is byte-matched to #1092 (greedy, seed 42, max_tokens 1024,
max_model_len 8192, stop <|im_end|>). Upload prefixes are ALWAYS
``issue1489_ctx_aug/...`` (never the parent's — #1005 clobber class).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# vLLM v1 EngineCore fork-poisoning guard (#628): must be set before any
# `import vllm` anywhere in this process tree.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

import issue1092_gpu_phase as parent  # noqa: E402
from issue1489_common import (  # noqa: E402
    DISTILL_BATCH_SIZE,
    DISTILL_EPOCHS,
    DISTILL_GRAD_ACCUM,
    DISTILL_LORA_ALPHA,
    DISTILL_LORA_R,
    DISTILL_LR,
    DISTILL_RUNS,
    DISTILL_SEED,
    GEN_SEED,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    HF_PREFIX,
    MAX_GEN_TOKENS,
    MAX_MODEL_LEN,
    N_LAYERS,
    HIDDEN_DIM,
    augmented_turns,
    load_conditions_manifest,
    rows_for_cell,
    verify_default_system_prompt,
)

logger = logging.getLogger("issue1489_gpu_phase")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SUMMARY_KINDS_1489 = [
    "prefix_end",
    "prefix_mean",
    "context_end",
    "context_mean",
    "t1",
    "t1_odd",
    "t1_even",
]
GEN_SHARD_SIZE = 250
CAPTURE_BATCH_SIZE = 8
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
RECAP_B_ROWS = 100  # cell_plain re-captures on provision B (cross-provision drift bound)

# Parent contract asserts: the reused capture-position helper reads these
# module constants — a drift there silently changes the row budget here.
assert parent.MAX_MODEL_LEN == MAX_MODEL_LEN, (parent.MAX_MODEL_LEN, MAX_MODEL_LEN)


# ---------------------------------------------------------------------------
# Rendering (conditions manifest -> prefix/prompt text)
# ---------------------------------------------------------------------------


def render_row_1489(
    row: dict, prefix_store: dict[str, dict], query_store: dict[str, dict]
) -> tuple[str, str]:
    """(prefix_text, prompt_text) for one conditions-manifest row (instruct format).

    The augmentation text (when present) is appended as the final system-prompt
    paragraph via ``augmented_turns`` BEFORE rendering, so the prefix arm
    carries the augmentation and the context arm adds the user query on top —
    both mapping arms by construction (plan §4.2).
    """
    prefix_item = prefix_store.get(str(row["prefix_id"]))
    query_item = query_store.get(str(row["query_id"]))
    if prefix_item is None or query_item is None:
        raise KeyError(f"row {row['row_id']} missing prefix/query store entry")
    turns = parent._prefix_turns(prefix_item)
    query = parent._query_text(query_item)
    aug_text = row.get("augment_text") or ""
    if aug_text:
        turns = augmented_turns(turns, aug_text)
    return parent._render_prompt_parts(turns, query, "instruct")


def plain_prompt_messages(
    row: dict, prefix_store: dict[str, dict], query_store: dict[str, dict]
) -> list[dict]:
    """PLAIN-context chat messages for a base row (the P3 distillation prompt)."""
    prefix_item = prefix_store[str(row["prefix_id"])]
    query_item = query_store[str(row["query_id"])]
    turns = [dict(t) for t in parent._prefix_turns(prefix_item)]
    turns.append({"role": "user", "content": parent._query_text(query_item)})
    return turns


# ---------------------------------------------------------------------------
# Shard planning + IO helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def shard_plan(rows: list[dict], shard_size: int = GEN_SHARD_SIZE) -> list[list[dict]]:
    """Deterministic manifest-order shards (the canonical row order per cell)."""
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


def gen_shard_path(out_dir: Path, cell: str, idx: int) -> Path:
    return out_dir / "raw_completions" / "generation" / cell / f"shard{idx:03d}.json"


def _gen_shard_done(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return len(payload.get("rows", [])) == expected_rows


def capture_done_path(out_dir: Path, cell: str, idx: int) -> Path:
    return out_dir / "summaries" / cell / f".done_shard{idx:03d}"


def load_gen_completions(out_dir: Path, cell: str, idx: int, rows: list[dict]) -> list[str]:
    """Completions for one shard, aligned to `rows` order; fail-loud on any gap."""
    path = gen_shard_path(out_dir, cell, idx)
    payload = json.loads(path.read_text())
    by_id = {r["row_id"]: r["completion"] for r in payload["rows"]}
    missing = [r["row_id"] for r in rows if r["row_id"] not in by_id]
    if missing:
        raise KeyError(f"{path} missing completions for {len(missing)} rows: {missing[:3]}")
    return [by_id[r["row_id"]] for r in rows]


# ---------------------------------------------------------------------------
# Capture (7 kinds; parent recipe: per-segment token-id concat + offsets)
# ---------------------------------------------------------------------------


def capture_batch_1489(
    *,
    prefix_texts: list[str],
    prompts: list[str],
    completions: list[str],
    model,
    tokenizer,
    device: str,
    log_label: str,
    batch_size: int = CAPTURE_BATCH_SIZE,
    n_layers: int = N_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
) -> tuple[list[dict[str, np.ndarray]], list[dict]]:
    """Teacher-forced capture -> per-row {kind: (n_layers, hidden) fp16} + flags.

    Mirrors ``issue1092_gpu_phase._capture_batch_loaded_model`` (token-id
    concatenation + offset-derived positions via the parent's
    ``_capture_row_ids_and_positions`` — the round-8.4 BPE-seam fix) with the
    #1489 kind set. ``t1_odd``/``t1_even`` mean the odd/even LOCAL answer-token
    indices (the per-row Δv split-half instrument + the Q4 disjoint-baseline
    legs); a 1-token answer is DEGENERATE (both halves = t1) and is flagged in
    the returned per-row dict so the fit driver can exclude it from
    disjoint-baseline statistics.
    """
    import torch

    if len({len(prefix_texts), len(prompts), len(completions)}) != 1:
        raise ValueError("prefix/prompt/completion lists must be equal length")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError(
            "capture positions index the UNPADDED sequence and require RIGHT padding; "
            f"tokenizer.padding_side={tokenizer.padding_side!r}"
        )
    boundary = parent._boundary_suffix("instruct")
    summaries: list[dict[str, np.ndarray]] = []
    flags: list[dict] = []
    n_total_rows = len(prompts)
    for batch_start in range(0, n_total_rows, max(1, batch_size)):
        batch_end = min(batch_start + max(1, batch_size), n_total_rows)
        if batch_start % (max(1, batch_size) * 5) == 0:
            logger.info(
                "[%s] capture rows %d:%d/%d", log_label, batch_start, batch_end, n_total_rows
            )
        batch_ids: list[list[int]] = []
        positions: list[dict[str, int]] = []
        for local_i in range(batch_start, batch_end):
            row_ids, pos = parent._capture_row_ids_and_positions(
                tokenizer,
                prefix_texts[local_i],
                prompts[local_i],
                completions[local_i],
                boundary,
                row_label=str(local_i),
            )
            batch_ids.append(row_ids)
            positions.append(pos)
        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = parent._call_model_with_hidden_states(model, input_ids, attention_mask)
        hidden_states = outputs.hidden_states[1:]
        if len(hidden_states) != n_layers:
            raise ValueError(f"model returned {len(hidden_states)} layers != {n_layers}")
        if hidden_states[0].shape[-1] != hidden_dim:
            raise ValueError(f"hidden dim {hidden_states[0].shape[-1]} != {hidden_dim}")

        for local_i, pos in enumerate(positions):

            def pos_state(p: int, *, row_i: int = local_i, hs_layers=hidden_states) -> np.ndarray:
                return np.stack(
                    [hs[row_i, p, :].to(torch.float16).cpu().numpy() for hs in hs_layers],
                    axis=0,
                )

            def span_mean(
                idx: list[int], *, row_i: int = local_i, hs_layers=hidden_states
            ) -> np.ndarray:
                sel = torch.tensor(idx, device=hs_layers[0].device, dtype=torch.long)
                return np.stack(
                    [
                        hs[row_i, sel, :].mean(dim=0).to(torch.float16).cpu().numpy()
                        for hs in hs_layers
                    ],
                    axis=0,
                )

            n_total = pos["n_total"]
            a0 = min(max(0, pos["answer_start"]), n_total - 1)
            a1 = min(max(a0 + 1, pos["answer_end"]), n_total)
            answer_idx = list(range(a0, a1))
            even_idx = answer_idx[0::2]
            odd_idx = answer_idx[1::2]
            degenerate = len(odd_idx) == 0
            if degenerate:
                odd_idx = even_idx  # flagged below; excluded from disjoint stats downstream
            row_summary = {
                "prefix_end": pos_state(pos["prefix_end"]),
                "prefix_mean": span_mean(list(range(0, pos["prefix_end"] + 1))),
                "context_end": pos_state(pos["context_end"]),
                "context_mean": span_mean(list(range(0, pos["context_end"] + 1))),
                "t1": span_mean(answer_idx),
                "t1_even": span_mean(even_idx),
                "t1_odd": span_mean(odd_idx),
            }
            summaries.append(row_summary)
            flags.append(
                {
                    "n_total": n_total,
                    "n_prompt": pos["n_prompt"],
                    "n_answer_tokens": len(answer_idx),
                    "t1_halves_degenerate": degenerate,
                }
            )
        del outputs, hidden_states, input_ids, attention_mask
    return summaries, flags


def write_capture_shard(
    out_dir: Path,
    cell: str,
    idx: int,
    rows: list[dict],
    summaries: list[dict[str, np.ndarray]],
    flags: list[dict],
) -> None:
    """Write one capture shard in the #1092 summaries schema + row_index shard."""
    cell_dir = out_dir / "summaries" / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) != len(summaries):
        raise ValueError(f"{cell} shard{idx}: {len(rows)} rows vs {len(summaries)} summaries")
    for kind in SUMMARY_KINDS_1489:
        stacked = np.stack([s[kind] for s in summaries], axis=0)  # (n, layers, hidden) fp16
        for layer in range(N_LAYERS):
            path = cell_dir / f"{kind}_L{layer:02d}_shard{idx:03d}.npy"
            tmp = path.with_name(path.name + ".tmp.npy")
            np.save(tmp, stacked[:, layer, :])
            os.replace(tmp, path)
    index_path = cell_dir / f"row_index_shard{idx:03d}.jsonl"
    lines = []
    for row, flag in zip(rows, flags, strict=True):
        rec = dict(row)
        rec.update(flag)
        lines.append(json.dumps(rec, ensure_ascii=False))
    _atomic_write(index_path, "\n".join(lines) + "\n")
    _atomic_write(capture_done_path(out_dir, cell, idx), datetime.datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def _build_vllm_engine(args: argparse.Namespace, *, enable_lora: bool = False):
    from vllm import LLM

    kwargs: dict = {
        "model": MODEL_ID,
        "max_model_len": MAX_MODEL_LEN,
        "seed": GEN_SEED,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
    }
    if args.enforce_eager or os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        kwargs["enforce_eager"] = True
    if args.no_prefix_caching or os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        kwargs["enable_prefix_caching"] = False
    if enable_lora:
        kwargs.update({"enable_lora": True, "max_lora_rank": DISTILL_LORA_R, "max_loras": 8})
    return LLM(**kwargs)


def _greedy(llm, prompts: list[str], *, lora_request=None) -> list[str]:
    """Chunked greedy generation (deadlock prevention, per-chunk INFO log)."""
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_GEN_TOKENS,
        stop=["<|im_end|>"],
    )
    out: list[str] = []
    n_chunks = (len(prompts) + VLLM_CHUNK - 1) // VLLM_CHUNK
    for i in range(0, len(prompts), VLLM_CHUNK):
        chunk = prompts[i : i + VLLM_CHUNK]
        logger.info(
            "[vllm-chunk] _greedy chunk %d/%d (%d prompts)",
            i // VLLM_CHUNK + 1,
            n_chunks,
            len(chunk),
        )
        kwargs = {"use_tqdm": False}
        if lora_request is not None:
            kwargs["lora_request"] = lora_request
        chunk_out = llm.generate(chunk, sp, **kwargs)
        out.extend(o.outputs[0].text for o in chunk_out)
    return out


# ---------------------------------------------------------------------------
# Workers (spawned one per GPU; CVD pinned in-child before torch/vLLM import)
# ---------------------------------------------------------------------------


def _gen_worker(gpu_id: int, task_q, result_q, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = argparse.Namespace(**args_dict)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s gen-w{gpu_id} %(message)s")
    out_dir = Path(args.out)
    prefix_store = parent.load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    query_store = parent.load_store(Path(args.corpus_dir), "query_store.jsonl")
    llm = None
    while True:
        task = task_q.get()
        if task is None:
            break
        cell, idx, rows = task
        try:
            path = gen_shard_path(out_dir, cell, idx)
            if _gen_shard_done(path, len(rows)):
                result_q.put(("ok", cell, idx, "resume-skip"))
                continue
            if llm is None:
                llm = _build_vllm_engine(args)
            rendered = [render_row_1489(r, prefix_store, query_store) for r in rows]
            prompts = [p for _pfx, p in rendered]
            completions = _greedy(llm, prompts)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "issue": 1489,
                "cell": cell,
                "shard": idx,
                "model": MODEL_ID,
                "decode": {
                    "temperature": 0.0,
                    "seed": GEN_SEED,
                    "max_tokens": MAX_GEN_TOKENS,
                    "max_model_len": MAX_MODEL_LEN,
                    "stop": ["<|im_end|>"],
                },
                "git_sha": args_dict.get("git_sha", "unknown"),
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                "rows": [
                    {
                        "row_id": r["row_id"],
                        "base_row_id": r["base_row_id"],
                        "cell_id": r["cell_id"],
                        "split": r["split"],
                        "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()[:16],
                        "completion": c,
                    }
                    for r, p, c in zip(rows, prompts, completions, strict=True)
                ],
            }
            _atomic_write(path, json.dumps(payload, ensure_ascii=False))
            result_q.put(("ok", cell, idx, f"{len(rows)} rows"))
        except Exception as exc:  # noqa: BLE001 — reported to parent, which fails loud
            logging.exception("gen worker failed on %s shard %d", cell, idx)
            result_q.put(("error", cell, idx, f"{type(exc).__name__}: {exc}"))


def _capture_worker(gpu_id: int, task_q, result_q, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s cap-w{gpu_id} %(message)s")
    import torch
    from transformers import AutoModelForCausalLM

    out_dir = Path(args.out)
    prefix_store = parent.load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    query_store = parent.load_store(Path(args.corpus_dir), "query_store.jsonl")
    tokenizer = parent._get_tokenizer()
    verify_default_system_prompt(tokenizer)
    model = None
    adapter_loaded: str | None = None
    while True:
        task = task_q.get()
        if task is None:
            break
        cell, idx, rows, adapter_path = task
        try:
            if capture_done_path(out_dir, cell, idx).exists():
                result_q.put(("ok", cell, idx, "resume-skip"))
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
                )
                model.eval()
            if adapter_path != adapter_loaded:
                raise RuntimeError(
                    f"capture worker got adapter {adapter_path!r} but loaded "
                    f"{adapter_loaded!r}; ft_capture assigns ONE adapter per worker run"
                )
            rendered = [render_row_1489(r, prefix_store, query_store) for r in rows]
            prefixes = [pfx for pfx, _p in rendered]
            prompts = [p for _pfx, p in rendered]
            completions = load_gen_completions(
                out_dir, args_dict.get("gen_cell_map", {}).get(cell, cell), idx, rows
            )
            summaries, flags = capture_batch_1489(
                prefix_texts=prefixes,
                prompts=prompts,
                completions=completions,
                model=model,
                tokenizer=tokenizer,
                device="cuda:0",
                log_label=f"{cell}/s{idx}",
            )
            write_capture_shard(out_dir, cell, idx, rows, summaries, flags)
            result_q.put(("ok", cell, idx, f"{len(rows)} rows"))
        except Exception as exc:  # noqa: BLE001 — reported to parent, which fails loud
            logging.exception("capture worker failed on %s shard %d", cell, idx)
            result_q.put(("error", cell, idx, f"{type(exc).__name__}: {exc}"))


def _ft_worker(gpu_id: int, task_q, result_q, args_dict: dict) -> None:
    """Provision-B worker: generation + capture under ONE selected checkpoint.

    Each task = (slug, adapter_path, rows). Generation applies the LoRA via
    vLLM ``LoRARequest``; capture applies the SAME adapter via PEFT on the HF
    model (teacher-forced), then the worker moves to the next run. vLLM engine
    and HF model never co-reside above gpu_memory_utilization 0.5.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = argparse.Namespace(**args_dict)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s ft-w{gpu_id} %(message)s")
    out_dir = Path(args.out)
    prefix_store = parent.load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    query_store = parent.load_store(Path(args.corpus_dir), "query_store.jsonl")

    while True:
        task = task_q.get()
        if task is None:
            break
        slug, adapter_path, rows = task
        cell = f"cell_ft_{slug}" if slug != "__plain_recap__" else "cell_plain_recap_b"
        try:
            _run_ft_cell(
                args,
                out_dir,
                prefix_store,
                query_store,
                slug,
                adapter_path,
                rows,
                cell,
            )
            result_q.put(("ok", cell, 0, f"{len(rows)} rows"))
        except Exception as exc:  # noqa: BLE001
            logging.exception("ft worker failed on %s", cell)
            result_q.put(("error", cell, 0, f"{type(exc).__name__}: {exc}"))


def _run_ft_cell(
    args, out_dir: Path, prefix_store, query_store, slug: str, adapter_path: str, rows, cell: str
) -> None:
    """Generate + capture one FT eval cell (or the plain re-capture cell)."""
    import gc

    import torch

    shards = shard_plan(rows)
    pending_gen = [
        (i, srows)
        for i, srows in enumerate(shards)
        if not _gen_shard_done(gen_shard_path(out_dir, cell, i), len(srows))
    ]
    if pending_gen:
        llm = _build_vllm_engine(args, enable_lora=adapter_path != "")
        lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(slug, abs(hash(slug)) % 100_000 + 1, adapter_path)
        for i, srows in pending_gen:
            rendered = [render_row_1489(r, prefix_store, query_store) for r in srows]
            prompts = [p for _pfx, p in rendered]
            completions = _greedy(llm, prompts, lora_request=lora_request)
            path = gen_shard_path(out_dir, cell, i)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "issue": 1489,
                "cell": cell,
                "shard": i,
                "adapter": adapter_path,
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                "rows": [
                    {
                        "row_id": r["row_id"],
                        "base_row_id": r["base_row_id"],
                        "cell_id": cell,
                        "split": r["split"],
                        "completion": c,
                    }
                    for r, c in zip(srows, completions, strict=True)
                ],
            }
            _atomic_write(path, json.dumps(payload, ensure_ascii=False))
        # Reap the engine before the HF capture load (#653 r9 recipe).
        try:
            llm.llm_engine.engine_core.shutdown()
        except AttributeError:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(1.0)

    pending_cap = [
        (i, srows)
        for i, srows in enumerate(shards)
        if not capture_done_path(out_dir, cell, i).exists()
    ]
    if not pending_cap:
        return
    from transformers import AutoModelForCausalLM

    tokenizer = parent._get_tokenizer()
    verify_default_system_prompt(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    for i, srows in pending_cap:
        rendered = [render_row_1489(r, prefix_store, query_store) for r in srows]
        prefixes = [pfx for pfx, _p in rendered]
        prompts = [p for _pfx, p in rendered]
        completions = load_gen_completions(out_dir, cell, i, srows)
        summaries, flags = capture_batch_1489(
            prefix_texts=prefixes,
            prompts=prompts,
            completions=completions,
            model=model,
            tokenizer=tokenizer,
            device="cuda:0",
            log_label=f"{cell}/s{i}",
        )
        write_capture_shard(out_dir, cell, i, srows, summaries, flags)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _distill_worker(gpu_id: int, task_q, result_q, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s distill-w{gpu_id} %(message)s")
    while True:
        task = task_q.get()
        if task is None:
            break
        slug, train_jsonl, run_out = task
        try:
            _train_distill_run(args, slug, Path(train_jsonl), Path(run_out))
            result_q.put(("ok", slug, 0, "trained"))
        except Exception as exc:  # noqa: BLE001
            logging.exception("distill worker failed on %s", slug)
            result_q.put(("error", slug, 0, f"{type(exc).__name__}: {exc}"))


def _train_distill_run(args, slug: str, train_jsonl: Path, run_out: Path) -> None:
    """One context-distillation LoRA run (plan §4.3 recipe, Source: #778)."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    n_rows = sum(1 for line in open(train_jsonl, encoding="utf-8") if line.strip())
    steps_per_epoch = max(1, math.ceil(n_rows / (DISTILL_BATCH_SIZE * DISTILL_GRAD_ACCUM)))
    if args.smoke:
        epochs = 1
        save_steps = 1  # 2 optimizer steps @ 30 rows -> 2 checkpoints (multi-LoRA floor)
    else:
        epochs = DISTILL_EPOCHS
        save_steps = max(1, math.ceil(steps_per_epoch / 2))  # every 0.5 epoch -> 8 ckpts
    cfg = TrainLoraConfig(
        gpu_id=0,  # CVD pinned by the launcher env; inherited single-GPU pin is authoritative
        epochs=epochs,
        lr=DISTILL_LR,
        lora_r=DISTILL_LORA_R,
        lora_alpha=DISTILL_LORA_ALPHA,
        batch_size=DISTILL_BATCH_SIZE,
        grad_accum=DISTILL_GRAD_ACCUM,
        max_length=MAX_MODEL_LEN,
        seed=DISTILL_SEED,
        run_name=f"issue1489_distill_{slug}",
        report_to="wandb",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,  # keep the full 8-ckpt dose ladder (uploaded below)
        save_only_model=True,
        hf_upload=False,  # checkpoints uploaded explicitly below (all rungs)
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    out_dir, loss = train_lora(MODEL_ID, str(train_jsonl), str(run_out), cfg=cfg)
    logger.info("[distill] %s trained: loss=%.4f out=%s", slug, loss, out_dir)
    (run_out / "train_meta.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "n_rows": n_rows,
                "epochs": epochs,
                "save_steps": save_steps,
                "steps_per_epoch": steps_per_epoch,
                "final_loss": loss,
                "recipe": {
                    "lora_r": DISTILL_LORA_R,
                    "lora_alpha": DISTILL_LORA_ALPHA,
                    "use_rslora": True,
                    "lr": DISTILL_LR,
                    "batch_size": DISTILL_BATCH_SIZE,
                    "grad_accum": DISTILL_GRAD_ACCUM,
                    "seed": DISTILL_SEED,
                },
                "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            },
            indent=2,
        )
    )


def _dose_probe_worker(gpu_id: int, task_q, result_q, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = argparse.Namespace(**args_dict)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s probe-w{gpu_id} %(message)s")
    out_dir = Path(args.out)
    prefix_store = parent.load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    query_store = parent.load_store(Path(args.corpus_dir), "query_store.jsonl")
    llm = None
    while True:
        task = task_q.get()
        if task is None:
            break
        slug, ckpt_paths, rows = task
        try:
            probe_dir = out_dir / "raw_completions" / "dose_probes" / slug
            probe_dir.mkdir(parents=True, exist_ok=True)
            rendered = [render_row_1489(r, prefix_store, query_store) for r in rows]
            prompts = [p for _pfx, p in rendered]  # PLAIN context (rows are cell_plain rows)
            if llm is None:
                llm = _build_vllm_engine(args, enable_lora=True)
            from vllm.lora.request import LoRARequest

            for k, ckpt in enumerate(ckpt_paths, start=1):
                out_path = probe_dir / f"ckpt{k}_completions.json"
                if out_path.exists():
                    continue
                if args.sequential_lora:
                    # fallback path: fresh engine per checkpoint (plan §8 risk row)
                    try:
                        llm.llm_engine.engine_core.shutdown()
                    except AttributeError:
                        pass
                    del llm
                    import gc

                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                    llm = _build_vllm_engine(args, enable_lora=True)
                req = LoRARequest(f"{slug}_ck{k}", (abs(hash(slug)) % 10_000) * 10 + k, ckpt)
                completions = _greedy(llm, prompts, lora_request=req)
                payload = {
                    "issue": 1489,
                    "slug": slug,
                    "ckpt_index": k,
                    "ckpt_path": ckpt,
                    "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                    "rows": [
                        {
                            "row_id": r["row_id"],
                            "base_row_id": r["base_row_id"],
                            "split": r["split"],
                            "completion": c,
                        }
                        for r, c in zip(rows, completions, strict=True)
                    ],
                }
                _atomic_write(out_path, json.dumps(payload, ensure_ascii=False))
                logger.info("[probe] %s ckpt%d: %d completions", slug, k, len(rows))
            result_q.put(("ok", slug, 0, f"{len(ckpt_paths)} ckpts"))
        except Exception as exc:  # noqa: BLE001
            logging.exception("dose-probe worker failed on %s", slug)
            result_q.put(("error", slug, 0, f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Dispatch (work-conserving queue; one worker per visible GPU)
# ---------------------------------------------------------------------------


def _run_workers(worker_fn, tasks: list, args: argparse.Namespace, phase: str) -> None:
    import multiprocessing as mp

    n_gpus = parent._detect_gpu_count()
    if n_gpus < 1:
        raise RuntimeError(f"phase {phase}: no visible GPUs")
    n_workers = min(n_gpus, max(1, len(tasks)))
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for t in tasks:
        task_q.put(t)
    args_dict = vars(args).copy()
    args_dict["git_sha"] = _git_sha()
    for _ in range(n_workers):
        task_q.put(None)
    procs = []
    for gpu_id in range(n_workers):
        p = ctx.Process(target=worker_fn, args=(gpu_id, task_q, result_q, args_dict))
        p.start()
        procs.append(p)
    errors = []
    for _ in range(len(tasks)):
        status, key, idx, note = result_q.get()
        logger.info("[%s] %s %s/%s: %s", phase, status, key, idx, note)
        if status == "error":
            errors.append((key, idx, note))
    for p in procs:
        p.join()
    for p in procs:
        if p.exitcode not in (0, None):
            errors.append(("worker", p.pid, f"exitcode={p.exitcode}"))
    if errors:
        raise RuntimeError(f"phase {phase}: {len(errors)} task failures: {errors[:5]}")


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _write_sentinel(args: argparse.Namespace, phase: str, note: str = "") -> None:
    """Pod-side progress sentinel for poll_pipeline.py (no task.py shellouts ever).

    Conforms to ``poll_pipeline._SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version
    / kind / version) — a bare ``{phase, note}`` payload is skipped WITHOUT rename
    and warn-spams every poller tick. One-way write-once channel: each write gets
    a fresh epoch-stamped filename (never rewritten in place; the drain renames
    to ``.processed``). The dispatcher's terminal ``epm:results`` sentinel is
    owned by issue1489_dispatch.sh, not this helper.
    """
    sentinel_dir = Path("/workspace/logs")
    try:
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        (sentinel_dir / ".probe").write_text("")
    except OSError:
        sentinel_dir = Path(args.out) / "logs"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "issue": 1489,
        "phase": phase,
        "out": args.out,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "note": note,
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": 1489,
        "by": "issue1489_gpu_phase",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(body, default=str),
    }
    path = sentinel_dir / f"issue-1489-epm_progress-{int(time.time() * 1000)}.json"
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
    except OSError:
        fallback = Path(args.out) / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        path = fallback / path.name
        path.write_text(json.dumps(payload, indent=2))
    logger.info("[sentinel] wrote %s (phase=%s)", path, phase)


def _headroom(out_dir: Path, need_gb: float, phase: str) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    free = assert_out_root_headroom(out_dir, need_gb, phase=phase)
    logger.info("[%s] out-root headroom OK: %.1f GB free (floor %.1f)", phase, free, need_gb)


# ---------------------------------------------------------------------------
# Phase entrypoints
# ---------------------------------------------------------------------------


def _manifest_cells(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        seen.setdefault(r["cell_id"])
    return list(seen)


def phase_gen(args: argparse.Namespace, manifest: list[dict]) -> None:
    _headroom(Path(args.out), 5.0, "gen")
    tasks = []
    for cell in _manifest_cells(manifest):
        rows = rows_for_cell(manifest, cell)
        for idx, srows in enumerate(shard_plan(rows)):
            tasks.append((cell, idx, srows))
    logger.info("[gen] %d shard tasks across %d cells", len(tasks), len(_manifest_cells(manifest)))
    _run_workers(_gen_worker, tasks, args, "gen")
    _write_sentinel(args, "gen_done", f"{len(tasks)} shards")


def phase_capture(args: argparse.Namespace, manifest: list[dict]) -> None:
    _headroom(Path(args.out), 70.0, "capture")
    tasks = []
    for cell in _manifest_cells(manifest):
        rows = rows_for_cell(manifest, cell)
        for idx, srows in enumerate(shard_plan(rows)):
            tasks.append((cell, idx, srows, None))
    _run_workers(_capture_worker, tasks, args, "capture")
    _write_sentinel(args, "capture_done", f"{len(tasks)} shards")


def build_distill_jsonl(
    args: argparse.Namespace, manifest: list[dict], slug: str, dest: Path
) -> Path:
    """(plain-context prompt, augmented-context answer) rows for one distill run.

    Targets are the P1 generations of cell_<slug> on the TRAIN-split rows —
    on-policy instruct-and-strip by construction (plan §4.3): the augmentation
    is present at generation time and stripped from the training prompt.
    """
    prefix_store = parent.load_store(Path(args.corpus_dir), "prefix_store.jsonl")
    query_store = parent.load_store(Path(args.corpus_dir), "query_store.jsonl")
    cell = f"cell_{slug}"
    rows = rows_for_cell(manifest, cell)
    train_rows = [r for r in rows if r["split"] == "train"]
    if not train_rows:
        raise ValueError(f"distill {slug}: no train-split rows in manifest")
    out_dir = Path(args.out)
    shards = shard_plan(rows)
    comp_by_base: dict[str, str] = {}
    for idx, srows in enumerate(shards):
        payload = json.loads(gen_shard_path(out_dir, cell, idx).read_text())
        for rec in payload["rows"]:
            comp_by_base[rec["base_row_id"]] = rec["completion"]
    lines = []
    n_empty = 0
    for r in train_rows:
        comp = comp_by_base.get(r["base_row_id"])
        if comp is None:
            raise KeyError(f"distill {slug}: no generation for base row {r['base_row_id']}")
        if not comp.strip():
            n_empty += 1
            continue
        messages = plain_prompt_messages(r, prefix_store, query_store)
        lines.append(json.dumps({"prompt": messages, "completion": comp}, ensure_ascii=False))
    if n_empty > max(1, len(train_rows) // 20):
        raise ValueError(f"distill {slug}: {n_empty}/{len(train_rows)} empty generations")
    if n_empty:
        logger.warning("[distill] %s: dropped %d empty-generation rows", slug, n_empty)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(dest, "\n".join(lines) + "\n")
    logger.info("[distill] %s: %d train rows -> %s", slug, len(lines), dest)
    return dest


def _distill_slugs(args: argparse.Namespace, manifest: list[dict]) -> list[str]:
    cells = set(_manifest_cells(manifest))
    slugs = [s for s in DISTILL_RUNS if f"cell_{s}" in cells]
    if args.smoke:
        slugs = slugs[:1]  # fact_veg — the plan's smoke distill canary
    if args.runs:
        requested = [s.strip() for s in args.runs.split(",") if s.strip()]
        unknown = [s for s in requested if f"cell_{s}" not in cells]
        if unknown:
            raise ValueError(f"--runs slugs not in manifest cells: {unknown}")
        slugs = requested
    if not slugs:
        raise ValueError("no distillation runs resolvable from the manifest")
    return slugs


def phase_distill(args: argparse.Namespace, manifest: list[dict]) -> None:
    _headroom(Path(args.out), 30.0, "distill")
    out_dir = Path(args.out)
    tasks = []
    for slug in _distill_slugs(args, manifest):
        run_out = out_dir / "distill" / slug
        if (run_out / "train_meta.json").exists():
            logger.info("[distill] %s already trained (resume-skip)", slug)
            continue
        jsonl = out_dir / "distill" / f"{slug}_train.jsonl"
        rows = build_distill_jsonl(args, manifest, slug, jsonl)
        if args.smoke:
            # tiny run: 30 rows (plan smoke block)
            lines = [line for line in jsonl.read_text().split("\n") if line.strip()][:30]
            _atomic_write(jsonl, "\n".join(lines) + "\n")
        tasks.append((slug, str(jsonl), str(run_out)))
    if tasks:
        _run_workers(_distill_worker, tasks, args, "distill")
    _upload_distill_checkpoints(args)
    _write_sentinel(args, "distill_done", f"{len(tasks)} runs")


def _checkpoint_dirs(run_out: Path) -> list[Path]:
    ckpts = sorted(
        (p for p in run_out.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint-* dirs under {run_out}")
    return ckpts


def _upload_distill_checkpoints(args: argparse.Namespace) -> None:
    """Upload EVERY dose-ladder checkpoint to the model repo (plan §4.3)."""
    if args.skip_upload:
        logger.warning("[distill] --skip-upload set; checkpoints NOT uploaded")
        return
    from explore_persona_space.orchestrate import hub

    out_dir = Path(args.out)
    cards: dict[str, list[str]] = {}
    for run_out in sorted((out_dir / "distill").glob("*")):
        if not (run_out / "train_meta.json").exists():
            continue
        slug = run_out.name
        adapter_paths: list[str] = []
        for k, ckpt in enumerate(_checkpoint_dirs(run_out), start=1):
            url = hub._upload(
                ckpt,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                path_in_repo=f"issue1489_distill/{slug}/ckpt{k}",
            )
            if not url:
                raise RuntimeError(f"checkpoint upload returned no path: {slug} ckpt{k}")
            adapter_paths.append(f"issue1489_distill/{slug}/ckpt{k}")
        cards[slug] = adapter_paths
        logger.info("[distill] uploaded %s checkpoint ladder", slug)
    _write_reproducibility_card(out_dir, cards)


def _write_reproducibility_card(out_dir: Path, cards: dict[str, list[str]]) -> None:
    """Persist the epm:results reproducibility_card (pod-side-reporting.md).

    The dispatcher's terminal results sentinel embeds this file so the
    upload-verifier resolves adapter + WandB rows mechanically. wandb_entity is
    read from the live SDK (never hand-typed; the #597 stale-literal trap).
    """
    entity = ""
    try:
        import wandb

        entity = wandb.Api().default_entity or ""
    except Exception as exc:  # noqa: BLE001 — card stays usable without entity
        logger.warning("[card] wandb entity unresolved (%s); field omitted", exc)
    card = {
        "hf_model_repo": HF_MODEL_REPO,
        "adapter_paths": sorted(p for paths in cards.values() for p in paths),
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue1489"),
        "wandb_run_names": [f"issue1489_distill_{slug}" for slug in sorted(cards)],
        **({"wandb_entity": entity} if entity else {}),
    }
    path = out_dir / "distill" / "reproducibility_card.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2))
    logger.info("[card] wrote %s (%d adapters)", path, len(card["adapter_paths"]))


def phase_dose_probes(args: argparse.Namespace, manifest: list[dict]) -> None:
    out_dir = Path(args.out)
    plain_rows = rows_for_cell(manifest, "cell_plain")
    probe_rows = [r for r in plain_rows if r["split"] == "probe"]
    if not probe_rows:
        raise ValueError("no probe-split rows in the conditions manifest")
    tasks = []
    for slug in _distill_slugs(args, manifest):
        run_out = out_dir / "distill" / slug
        ckpts = [str(p) for p in _checkpoint_dirs(run_out)]
        tasks.append((slug, ckpts, probe_rows))
    _run_workers(_dose_probe_worker, tasks, args, "dose_probes")
    _write_sentinel(args, "dose_probes_done", f"{len(tasks)} runs x {len(probe_rows)} rows")


def _load_selection(args: argparse.Namespace) -> dict[str, dict]:
    sel_path = Path(args.selection)
    sel = json.loads(sel_path.read_text())
    runs = sel.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError(f"{sel_path} has no runs mapping")
    for slug, spec in runs.items():
        if "ckpt_index" not in spec:
            raise ValueError(f"selection for {slug} missing ckpt_index")
    return runs


def phase_ft(args: argparse.Namespace, manifest: list[dict]) -> None:
    """P4b: FT eval generation + capture for the selected checkpoints."""
    _headroom(Path(args.out), 40.0, "ft")
    out_dir = Path(args.out)
    runs = _load_selection(args)
    plain_rows = rows_for_cell(manifest, "cell_plain")
    eval_rows = [r for r in plain_rows if r["split"] == "eval"]
    if not eval_rows:
        raise ValueError("no eval-split rows in the conditions manifest")
    tasks = []
    for slug, spec in runs.items():
        adapter = spec.get("ckpt_path")
        if not adapter:
            run_out = out_dir / "distill" / slug
            ckpts = _checkpoint_dirs(run_out)
            adapter = str(ckpts[int(spec["ckpt_index"]) - 1])
        if not Path(adapter).exists():
            raise FileNotFoundError(f"selected checkpoint missing for {slug}: {adapter}")
        tasks.append((slug, adapter, eval_rows))
    # cell_plain re-captures (cross-provision drift bound): first RECAP_B_ROWS eval rows
    recap_rows = eval_rows[: min(RECAP_B_ROWS, len(eval_rows))]
    tasks.append(("__plain_recap__", "", recap_rows))
    _run_workers(_ft_worker, tasks, args, "ft")
    _write_sentinel(args, "ft_done", f"{len(tasks)} cells")


def phase_upload(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Upload rollout text (unconditional) + capture summaries to the issue prefix."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    out_dir = Path(args.out)
    api = HfApi()
    raw_root = out_dir / "raw_completions"
    if not raw_root.exists():
        raise FileNotFoundError(f"nothing to upload: {raw_root} missing")
    url = hub._upload(
        raw_root,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/raw_completions",
    )
    if not url:
        raise RuntimeError("raw_completions upload returned no path")
    expected = [
        f"{HF_PREFIX}/raw_completions/{p.relative_to(raw_root)}"
        for p in raw_root.rglob("*")
        if p.is_file()
    ]
    missing = hub.verify_repo_paths_uploaded(
        api,
        HF_DATA_REPO,
        expected,
        path_in_repo=f"{HF_PREFIX}/raw_completions",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"raw_completions verify missing {len(missing)}: {sorted(missing)[:5]}")
    logger.info("[upload] raw_completions verified: %d files", len(expected))

    summaries_root = out_dir / "summaries"
    if summaries_root.exists():
        for cell_dir in sorted(p for p in summaries_root.iterdir() if p.is_dir()):
            prefix = f"{HF_PREFIX}/analysis_tensors/summaries/{cell_dir.name}"
            url = hub._upload(
                cell_dir, repo_id=HF_DATA_REPO, repo_type="dataset", path_in_repo=prefix
            )
            if not url:
                raise RuntimeError(f"summaries upload returned no path: {cell_dir.name}")
            expected = [
                f"{prefix}/{p.relative_to(cell_dir)}" for p in cell_dir.rglob("*") if p.is_file()
            ]
            missing = hub.verify_repo_paths_uploaded(
                api, HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
            )
            if missing:
                raise RuntimeError(
                    f"summaries verify missing {len(missing)} for {cell_dir.name}: "
                    f"{sorted(missing)[:5]}"
                )
            logger.info("[upload] summaries/%s verified: %d files", cell_dir.name, len(expected))
    _write_sentinel(args, "upload_done", "raw_completions + summaries verified")


# ---------------------------------------------------------------------------
# Batched-vs-serial capture equivalence smoke (two-bar recipe, gotchas.md)
# ---------------------------------------------------------------------------


def verify_capture_equivalence(args: argparse.Namespace) -> None:
    """cosine(batched, serial) gate on a tiny CPU model (B=3 so padding fires)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    tokenizer = parent._get_tokenizer()
    verify_default_system_prompt(tokenizer)
    # 2-layer same-arch tiny model over the REAL vocab space (tiny-real recipe)
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    prefixes, prompts, comps = [], [], []
    for aug, q, c in [
        ("", "What is 2+2?", "4."),
        ("The user's name is Sarah.", "Suggest a dinner.", "A hearty lentil stew.\nEnjoy!"),
        ("Respond only with JSON.", "Name a color.", '{"answer": "blue"}'),
    ]:
        turns = augmented_turns([], aug) if aug else []
        pfx, prm = parent._render_prompt_parts(turns, q, "instruct")
        prefixes.append(pfx)
        prompts.append(prm)
        comps.append(c)

    def run(batch_size: int):
        return capture_batch_1489(
            prefix_texts=prefixes,
            prompts=prompts,
            completions=comps,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            log_label="equiv",
            batch_size=batch_size,
            n_layers=2,
            hidden_dim=64,
        )

    batched, bflags = run(3)
    serial, _sflags = run(1)
    if not any(f["n_answer_tokens"] >= 2 for f in bflags):
        raise AssertionError("equivalence smoke needs >=1 multi-token answer row")
    worst_early, worst_flat = 1.0, 1.0
    for b, s in zip(batched, serial, strict=True):
        for kind in SUMMARY_KINDS_1489:
            vb, vs = b[kind].astype(np.float64), s[kind].astype(np.float64)
            for layer in range(vb.shape[0]):
                cos = float(
                    np.dot(vb[layer], vs[layer])
                    / (np.linalg.norm(vb[layer]) * np.linalg.norm(vs[layer]) + 1e-12)
                )
                if layer < 2:
                    worst_early = min(worst_early, cos)
            flat = float(
                np.dot(vb.ravel(), vs.ravel())
                / (np.linalg.norm(vb.ravel()) * np.linalg.norm(vs.ravel()) + 1e-12)
            )
            worst_flat = min(worst_flat, flat)
    if worst_early < 0.999 or worst_flat < 0.995:
        raise AssertionError(
            f"capture equivalence gate failed: early={worst_early:.6f} flat={worst_flat:.6f}"
        )
    print(f"CAPTURE-EQUIV PASS: early_cos_min={worst_early:.6f} flat_cos_min={worst_flat:.6f}")


def verify_imports() -> None:
    """AST-walk this file + issue1489_common and execute every deferred import."""
    import ast
    import importlib

    failures = []
    for mod_path in [Path(__file__), REPO_ROOT / "scripts" / "issue1489_common.py"]:
        tree = ast.parse(mod_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                try:
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        if alias.name == "*" or hasattr(mod, alias.name):
                            continue
                        try:  # `from pkg import submodule` — import the submodule
                            importlib.import_module(f"{node.module}.{alias.name}")
                        except ImportError:
                            failures.append(f"{mod_path.name}: {node.module}.{alias.name}")
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{mod_path.name}: {node.module} ({exc})")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{mod_path.name}: {alias.name} ({exc})")
    if failures:
        raise ImportError(f"deferred-import verification failed: {failures}")
    print("VERIFY-IMPORTS PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        required=True,
        choices=[
            "gen",
            "capture",
            "distill",
            "dose_probes",
            "ft",
            "upload",
            "verify_capture",
            "verify_imports",
        ],
    )
    p.add_argument("--conditions-dir", default="data/issue_1489/conditions")
    p.add_argument("--corpus-dir", default="data/issue_1489/hf_dl/corpus")
    p.add_argument("--out", default="data/issue_1489")
    p.add_argument("--smoke", action="store_true", help="tiny-real slice (SAME code path)")
    p.add_argument("--runs", default="", help="comma-separated distill slugs override")
    p.add_argument("--selection", default="", help="checkpoint selection JSON (phase ft)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--sequential-lora", action="store_true", help="multi-LoRA fallback path")
    p.add_argument("--skip-upload", action="store_true")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # One WandB project for every issue-1489 training phase (distill + ft) so the
    # reproducibility card's wandb_project matches the live runs (#608 convention).
    os.environ.setdefault("WANDB_PROJECT", "issue1489")
    args = build_argparser().parse_args()
    if args.phase == "verify_imports":
        verify_imports()
        return 0
    if args.phase == "verify_capture":
        verify_capture_equivalence(args)
        return 0
    manifest = load_conditions_manifest(Path(args.conditions_dir))
    logger.info(
        "[phase=%s] manifest rows=%d cells=%d smoke=%s",
        args.phase,
        len(manifest),
        len(_manifest_cells(manifest)),
        args.smoke,
    )
    _write_sentinel(args, f"{args.phase}_start")
    try:
        if args.phase == "gen":
            phase_gen(args, manifest)
        elif args.phase == "capture":
            phase_capture(args, manifest)
        elif args.phase == "distill":
            phase_distill(args, manifest)
        elif args.phase == "dose_probes":
            phase_dose_probes(args, manifest)
        elif args.phase == "ft":
            if not args.selection:
                raise ValueError("--selection is required for phase ft")
            phase_ft(args, manifest)
        elif args.phase == "upload":
            phase_upload(args, manifest)
    except Exception as exc:
        _write_sentinel(args, f"{args.phase}_failed", f"{type(exc).__name__}: {exc}")
        raise
    logger.info(
        "[phase-complete] %s", args.phase
    )  # reserved [phase=done] stays the DISPATCHER terminal line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
