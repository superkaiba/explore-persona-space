"""Batched teacher-forced activation capture for issue #1739 (round B).

Mirrors #1092's span/template conventions EXACTLY so the new captures join the
same representation space as the reused #1092 summary store:

- ``capture_row_ids_and_positions`` mirrors
  ``scripts/issue1092_gpu_phase.py::_capture_row_ids_and_positions`` (round-8.4):
  per-segment TOKEN-ID concatenation (never re-tokenize concatenated text —
  BPE seam merges shift every position, the #825/#1092-G2 class) + OFFSET-based
  prefix_end (last prompt token ending inside prefix_text).
- ``capture_batch`` mirrors
  ``scripts/issue1092_gpu_phase.py::_capture_batch_loaded_model``: right-pad
  guard, padded batch forwards with ``output_hidden_states=True``,
  ``hidden_states[1:]`` (post-block states, 28 layers), fp16 summaries;
  ``prefix_end``/``context_end`` at positions, ``t1`` = answer-span mean.
- The teacher-forcing boundary suffix mirrors
  ``scripts/issue1092_gpu_phase.py::_boundary_suffix`` (instruct).

Output shards use the SAME layout ``store_io.load_summaries`` reads:
``{kind}_L{layer:02d}_shard{NN:02d}.npy`` + ``row_index_shard{NN:02d}.jsonl``.

torch/transformers are imported LAZILY inside the model-facing functions so
tests import this module (and its span arithmetic) without GPU deps.

CONTENT HYGIENE: logs carry ids, counts, shapes — never row text.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.constants import (
    HIDDEN_DIM,
    MODEL_NAME,
    N_LAYERS,
    SUMMARY_KINDS,
)
from explore_persona_space.experiments.issue_1739.generation import (
    INSTRUCT_REVISION,
    MAX_MODEL_LEN,
)

logger = logging.getLogger(__name__)

# issue1092_gpu_phase.py:63 parity (prompt-side budget within the 8192 window).
MAX_FORMATTED_TOKENS = 7168
# issue1092_gpu_phase.py::_boundary_suffix("instruct") — the teacher-forcing
# boundary appended after the completion (special-token boundary: never
# BPE-merges into the completion tail).
BOUNDARY_INSTRUCT = "<|im_end|>\n<|im_start|>user\n"
DEFAULT_CAPTURE_BATCH_SIZE = int(os.environ.get("EPM_CAPTURE_BATCH_SIZE", "8"))
DEFAULT_SHARD_ROWS = 512


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def capture_row_ids_and_positions(
    tokenizer,
    prefix_text: str,
    prompt: str,
    completion: str,
    boundary: str = BOUNDARY_INSTRUCT,
    row_label: str = "?",
    *,
    max_model_len: int = MAX_MODEL_LEN,
    max_formatted_tokens: int = MAX_FORMATTED_TOKENS,
) -> tuple[list[int], dict[str, int]]:
    """Teacher-forcing input ids + capture positions for one row.

    Mirror of ``issue1092_gpu_phase._capture_row_ids_and_positions`` (round-8.4
    fix): the forwarded sequence is built by CONCATENATING PER-SEGMENT TOKEN
    IDS (the prompt segment is bit-identical to what generation consumed), and
    ``prefix_end`` is derived from the prompt's OFFSET MAPPING (last token
    ending within ``prefix_text``) — positions are exact by construction.
    Fails loud on over-budget rows (the loader must filter them).
    """
    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(prompt_enc["input_ids"])
    offsets = prompt_enc["offset_mapping"]
    completion_ids = _token_ids(tokenizer, completion)
    boundary_ids = _token_ids(tokenizer, boundary)
    row_ids = prompt_ids + completion_ids + boundary_ids
    n_total_tokens = len(row_ids)
    if n_total_tokens > max_model_len:
        raise ValueError(
            f"capture row {row_label} has {n_total_tokens} tokens, exceeding "
            f"max_model_len={max_model_len}; loader must filter it"
        )
    if len(prompt_ids) > max_formatted_tokens:
        raise ValueError(
            f"capture row {row_label} prompt has {len(prompt_ids)} tokens, "
            f"exceeding prompt budget {max_formatted_tokens}"
        )

    # prefix_end: last prompt token that ends INSIDE prefix_text (offset-based;
    # a token BPE-merging across the prefix boundary ends beyond
    # len(prefix_text) and is correctly excluded). Empty prefix -> clamped 0.
    n_prefix_chars = len(prefix_text)
    n_prefix_tokens = sum(1 for start, end in offsets if end <= n_prefix_chars and end > start)

    prefix_end_pos = min(max(0, n_prefix_tokens - 1), n_total_tokens - 1)
    context_end_pos = min(max(0, len(prompt_ids) - 1), n_total_tokens - 1)
    answer_start = min(context_end_pos + 1, n_total_tokens - 1)
    answer_end = min(context_end_pos + 1 + max(1, len(completion_ids)), n_total_tokens)
    return row_ids, {
        "n_total": n_total_tokens,
        "n_prompt": len(prompt_ids),
        "prefix_end": prefix_end_pos,
        "context_end": context_end_pos,
        "answer_start": answer_start,
        "answer_end": answer_end,
    }


def capture_batch(
    prefix_texts: list[str],
    prompts: list[str],
    completions: list[str],
    *,
    model,
    tokenizer,
    n_layers: int = N_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
    device: str = "cuda",
    batch_size: int = DEFAULT_CAPTURE_BATCH_SIZE,
    log_label: str = "capture",
    boundary: str = BOUNDARY_INSTRUCT,
    max_model_len: int = MAX_MODEL_LEN,
    max_formatted_tokens: int = MAX_FORMATTED_TOKENS,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, int]]]:
    """Teacher-forced capture with padded batch forwards (fp16 summaries).

    Returns ``(summaries, positions)``: per row a dict of
    ``kind -> (n_layers, hidden_dim) float16`` for kinds ``prefix_end`` /
    ``context_end`` (single positions) and ``t1`` (answer-span mean — the
    #1092 t1 = response-avg answer summary), plus the per-row position dicts.
    Mirror of ``issue1092_gpu_phase._capture_batch_loaded_model`` (chunked for
    VRAM via ``batch_size``; never truncates — over-budget rows fail loud).
    """
    import torch  # lazy: GPU dep

    if len({len(prefix_texts), len(prompts), len(completions)}) != 1:
        raise ValueError("prefix_texts, prompts, and completions must have equal length")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError(
            "capture positions index the UNPADDED sequence and require RIGHT padding; "
            f"tokenizer.padding_side={tokenizer.padding_side!r} (#1092 round-8.4 guard)"
        )

    summaries: list[dict[str, np.ndarray]] = []
    all_positions: list[dict[str, int]] = []
    n_total_rows = len(prompts)
    for batch_start in range(0, n_total_rows, max(1, batch_size)):
        batch_end = min(batch_start + max(1, batch_size), n_total_rows)
        if batch_start % (max(1, batch_size) * 5) == 0:
            logger.info(
                "[%s] capture batch rows %d:%d/%d", log_label, batch_start, batch_end, n_total_rows
            )
        batch_ids: list[list[int]] = []
        positions: list[dict[str, int]] = []
        for local_i in range(batch_start, batch_end):
            row_ids, pos = capture_row_ids_and_positions(
                tokenizer,
                prefix_texts[local_i],
                prompts[local_i],
                completions[local_i],
                boundary,
                row_label=str(local_i),
                max_model_len=max_model_len,
                max_formatted_tokens=max_formatted_tokens,
            )
            batch_ids.append(row_ids)
            positions.append(pos)

        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            # logits_to_keep=1: hidden-state-only forward — skip the full-vocab
            # logits materialization (#779 OOM class); introspection-guarded.
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
            }
            try:
                outputs = model(**kwargs, logits_to_keep=1)
            except TypeError:
                outputs = model(**kwargs)
        hidden_states = outputs.hidden_states[1:]  # post-block states (skip embeddings)
        if len(hidden_states) != n_layers:
            raise ValueError(f"model returned {len(hidden_states)} layers, expected {n_layers}")
        if hidden_states[0].shape[-1] != hidden_dim:
            raise ValueError(
                f"model hidden dim {hidden_states[0].shape[-1]} != expected {hidden_dim}"
            )

        for local_i, pos in enumerate(positions):

            def extract_pos(
                position: int, *, row_i: int = local_i, hs_layers=hidden_states
            ) -> np.ndarray:
                return np.stack(
                    [hs[row_i, position, :].to(torch.float16).cpu().numpy() for hs in hs_layers],
                    axis=0,
                )

            def extract_span(
                start: int,
                end: int,
                *,
                row_i: int = local_i,
                n_total: int = pos["n_total"],
                hs_layers=hidden_states,
            ) -> np.ndarray:
                start = min(max(0, start), n_total - 1)
                end = min(max(start + 1, end), n_total)
                return np.stack(
                    [
                        hs[row_i, start:end, :].mean(dim=0).to(torch.float16).cpu().numpy()
                        for hs in hs_layers
                    ],
                    axis=0,
                )

            row_summary = {
                "prefix_end": extract_pos(pos["prefix_end"]),
                "context_end": extract_pos(pos["context_end"]),
                "t1": extract_span(pos["answer_start"], pos["answer_end"]),
            }
            for kind, arr in row_summary.items():
                assert arr.shape == (n_layers, hidden_dim), (kind, arr.shape)
            summaries.append(row_summary)
            all_positions.append(pos)
    return summaries, all_positions


def write_store_shard(
    store_dir: Path | str,
    shard_idx: int,
    summaries: list[dict[str, np.ndarray]],
    meta_rows: list[dict],
    *,
    kinds: tuple[str, ...] = SUMMARY_KINDS,
) -> list[Path]:
    """Write one capture shard in the layout ``store_io.load_summaries`` reads.

    Emits ``{kind}_L{layer:02d}_shard{NN:02d}.npy`` per (kind, layer) plus the
    ``row_index_shard{NN:02d}.jsonl`` sidecar (atomic tmp+replace writes).
    Asserts summary/meta row-count parity.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    if len(summaries) != len(meta_rows):
        raise ValueError(f"summaries ({len(summaries)}) != meta_rows ({len(meta_rows)})")
    if not summaries:
        raise ValueError("refusing to write an empty shard")
    n_layers = summaries[0][kinds[0]].shape[0]
    written: list[Path] = []
    for kind in kinds:
        for layer in range(n_layers):
            arr = np.stack([s[kind][layer] for s in summaries], axis=0).astype(np.float16)
            path = store_dir / f"{kind}_L{layer:02d}_shard{shard_idx:02d}.npy"
            # Dot-prefixed tmp: must NOT match the loader's `{kind}_L*_shard*.npy`
            # glob (a crash-surviving tmp would otherwise enter the shard set);
            # ends in .npy so np.save does not append a suffix.
            tmp = path.with_name(".tmp_" + path.name)
            np.save(tmp, arr)
            os.replace(tmp, path)
            written.append(path)
    index_path = store_dir / f"row_index_shard{shard_idx:02d}.jsonl"
    tmp = index_path.with_name(index_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in meta_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, index_path)
    written.append(index_path)
    return written


def load_capture_model(device: str = "cuda", dtype: str = "bfloat16"):
    """Load the pinned instruct model for teacher-forced capture (lazy torch)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=INSTRUCT_REVISION,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def _shard_meta_path(store_dir: Path, shard_idx: int) -> Path:
    return Path(store_dir) / f"_capture_meta_shard{shard_idx:02d}.json"


def shard_done(store_dir: Path | str, shard_idx: int, fingerprint: str) -> bool:
    """Resume predicate: shard complete iff its meta sidecar matches the
    fingerprint AND the row_index sidecar exists (npy files written first)."""
    meta_path = _shard_meta_path(Path(store_dir), shard_idx)
    index_path = Path(store_dir) / f"row_index_shard{shard_idx:02d}.jsonl"
    if not meta_path.exists() or not index_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text()).get("fingerprint") == fingerprint
    except (json.JSONDecodeError, OSError):
        return False


def capture_rollout_files(
    rollout_paths: list[Path],
    *,
    store_dir: Path | str,
    model,
    tokenizer,
    n_layers: int = N_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
    device: str = "cuda",
    batch_size: int = DEFAULT_CAPTURE_BATCH_SIZE,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    fingerprint: str = "",
    max_model_len: int = MAX_MODEL_LEN,
    max_formatted_tokens: int = MAX_FORMATTED_TOKENS,
) -> dict:
    """Capture summaries for generation rollout JSONs into store shards.

    Each rollout JSON (``generation.generate_labeling`` output) contributes one
    row: (prefix_text, prompt_text, completion). Sharded per ``shard_rows``
    with per-shard resume (checkpoint-per-unit; the shard is the unit).
    Over-budget rows are DROPPED with a digest count (the capture itself fails
    loud, so the filter runs at load). Returns the capture manifest.
    """
    store_dir = Path(store_dir)
    rows: list[tuple[dict, dict]] = []  # (payload, meta_row)
    n_over_budget = 0
    for path in rollout_paths:
        payload = json.loads(Path(path).read_text())
        try:
            _ = capture_row_ids_and_positions(
                tokenizer,
                payload["prefix_text"],
                payload["prompt_text"],
                payload["completion"],
                max_model_len=max_model_len,
                max_formatted_tokens=max_formatted_tokens,
            )
        except ValueError:
            n_over_budget += 1
            continue
        meta_row = {
            "context_id": payload.get("context_id"),
            "behavior": payload.get("behavior"),
            "split": payload.get("split"),
            "rung": payload.get("rung"),
            "group_key": payload.get("group_key"),
            "rollout_k": payload.get("rollout_k"),
            "is_eval_only": False,
            "source_file": Path(path).name,
        }
        rows.append((payload, meta_row))
    if not rows:
        raise RuntimeError(
            f"capture has 0 in-budget rows from {len(rollout_paths)} rollout files "
            f"({n_over_budget} over budget)"
        )

    n_shards = (len(rows) + shard_rows - 1) // shard_rows
    t0 = time.time()
    n_captured = 0
    n_resumed = 0
    for shard_idx in range(n_shards):
        if shard_done(store_dir, shard_idx, fingerprint):
            n_resumed += 1
            continue
        shard = rows[shard_idx * shard_rows : (shard_idx + 1) * shard_rows]
        summaries, positions = capture_batch(
            [p["prefix_text"] for p, _ in shard],
            [p["prompt_text"] for p, _ in shard],
            [p["completion"] for p, _ in shard],
            model=model,
            tokenizer=tokenizer,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            device=device,
            batch_size=batch_size,
            log_label=f"capture-shard{shard_idx:02d}",
            max_model_len=max_model_len,
            max_formatted_tokens=max_formatted_tokens,
        )
        meta_rows = [dict(meta, **pos) for (_, meta), pos in zip(shard, positions, strict=True)]
        write_store_shard(store_dir, shard_idx, summaries, meta_rows)
        meta_path = _shard_meta_path(store_dir, shard_idx)
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "n_rows": len(shard)}))
        os.replace(tmp, meta_path)
        n_captured += len(shard)
        logger.info(
            "[capture] shard %d/%d rows=%d elapsed=%.0fs",
            shard_idx + 1,
            n_shards,
            len(shard),
            time.time() - t0,
        )
    manifest = {
        "n_rollout_files": len(rollout_paths),
        "n_rows": len(rows),
        "n_over_budget": n_over_budget,
        "n_shards": n_shards,
        "n_shards_resumed": n_resumed,
        "n_rows_captured": n_captured,
        "fingerprint": fingerprint,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = store_dir / "_capture_manifest.json"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    return manifest


def teacher_forced_ln_logp(
    pairs: list[tuple[str, str]],
    *,
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = DEFAULT_CAPTURE_BATCH_SIZE,
    max_model_len: int = MAX_MODEL_LEN,
) -> list[float]:
    """Length-normalized teacher-forced log P(completion | prompt) per pair.

    The TF fixed +/- pool margin scorer (llm-judging.md rule 19; dv_build
    consumes this). Per-segment token-id concatenation (same BPE-seam
    discipline as capture); right padding; the log-softmax gather runs
    GPU-resident and only per-pair scalars transfer to CPU.
    """
    import torch  # lazy: GPU dep

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("teacher_forced_ln_logp requires RIGHT padding")

    out: list[float] = []
    for start in range(0, len(pairs), max(1, batch_size)):
        batch = pairs[start : start + max(1, batch_size)]
        batch_ids: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for prompt, completion in batch:
            prompt_ids = _token_ids(tokenizer, prompt)
            completion_ids = _token_ids(tokenizer, completion)
            if not completion_ids:
                raise ValueError("empty completion in teacher_forced_ln_logp pair")
            row_ids = prompt_ids + completion_ids
            if len(row_ids) > max_model_len:
                raise ValueError(
                    f"TF pair has {len(row_ids)} tokens, exceeding max_model_len={max_model_len}"
                )
            batch_ids.append(row_ids)
            spans.append((len(prompt_ids), len(row_ids)))
        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for row_i, (comp_start, row_end) in enumerate(spans):
            # Token at position t is predicted by logits at t-1.
            tgt = input_ids[row_i, comp_start:row_end]
            lp = (
                logprobs[row_i, comp_start - 1 : row_end - 1, :]
                .gather(-1, tgt.unsqueeze(-1))
                .squeeze(-1)
            )
            out.append(float(lp.mean().item()))
    return out
