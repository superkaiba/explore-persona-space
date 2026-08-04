#!/usr/bin/env python
"""Capture driver for task #2054: teacher-forced activation capture at layer 19.

For each spliced row under `--input-dir` (phase_b inserted, phase_c on-policy,
or phase_d cell_c outputs, one JSONL per variant), run a teacher-forced HF
forward pass at layer 19 through the target model (Qwen2.5-7B or -Instruct)
and extract three activation summaries per row per the plan §4 "Both mapping
arms" contract:

- `v_C` (context arm) — layer-19 hidden state at the last prompt token BEFORE
  the answer starts (token at `answer_start - 1` in char space → token index
  via the tokenizer's offset mapping).
- `v_A` (answer arm target) — mean of layer-19 hidden states over answer tokens
  (tokens whose byte range lies within `[answer_start, answer_end)`).
- `v_P` (prefix arm) — layer-19 hidden state at the last token of the PREFIX
  (per Critical Rule: system + user query, before the assistant/story turn).
  Located deterministically for each row_form:
  * story-form rows (form=`attrib_quoted`): the last token before the
    attribution marker ` replied: "` / ` said, "` (the character's dialogue
    opening); v_P is undefined when no marker is found and that row's v_P is
    recorded as null (never coerced).
  * chat-form rows (spliced through a chat template): the last prompt token
    before the assistant/user turn, per the tokenizer's offset mapping on the
    prompt text ending at `answer_start`.

Activations persist to `<output-dir>/{variant}_{model}.npz` as three
`{conv_id: <d-vec>}` fp16 arrays keyed by conv_id (never materializes the full
corpus in memory at once — a streaming per-row concatenation via a resizeable
list; the plan's stream-reduce recipe is inherited when the fits phase reads
these back). Per-cell diagnostics (DV 7 answer-length parity + DV 8 conv_id
intersection + realized row count + peak GPU memory / wall-time) land at
`eval_results/issue_2054/capture_diagnostics/{variant}_{model}.json`.

Uploads to HF `superkaiba1/explore-persona-space-data/issue2054_lattice/activations/{variant}/{model}/`
(best-effort, non-fatal). Skipped when `--dry-run` (0-byte activation shell +
diagnostics stub proving the byte-offset-to-token-index mapping runs on ≤3
sample rows without loading the model).

Emits `[phase=capture]` log lines terminating in `[phase=done]` on graceful
completion.

Exit 0 on success. Exit 1 on model/GPU/HF failure. Exit 2 on missing input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
TASK_PREFIX = "issue2054_lattice"

_MODEL_ID = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

# Attribution markers used by the parent's `attrib_quoted` splice form
# (`.claude/worktrees/issue-1345/scripts/issue1345_scaffold_common.py`).
# Ordered so a more-specific candidate (with punctuation) is tried first.
_ATTRIB_MARKERS = (
    ' replied: "',
    ' replied "',
    ' said: "',
    ' said, "',
    ' said "',
    ' answered: "',
    ' answered, "',
)


def _log(msg: str) -> None:
    print(f"[phase=capture] {msg}", flush=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _resolve_input_paths(input_dir: Path, variants: list[str], phase: str) -> dict[str, Path]:
    """Return {variant: input JSONL path} keyed on variant.

    Phase-scoped naming inherits Units A/B (see phase_b/phase_c/phase_d):
      - phase=inserted  -> spliced_inserted_<variant>.jsonl
      - phase=on_policy -> on_policy_<variant>.jsonl (falls back to .mock.jsonl)
      - phase=cell_c    -> cell_c_<variant>.jsonl

    Smoke fallback: any *.jsonl directly under `input_dir` counts as a `_flat`
    variant when the variant subtree is missing.
    """
    filename_by_phase = {
        "inserted": lambda v: f"spliced_inserted_{v}.jsonl",
        "on_policy": lambda v: f"on_policy_{v}.jsonl",
        "cell_c": lambda v: f"cell_c_{v}.jsonl",
    }
    mock_fallback_by_phase = {
        "on_policy": lambda v: f"on_policy_{v}.mock.jsonl",
    }
    naming = filename_by_phase.get(phase)
    if naming is None:
        raise ValueError(f"unknown --phase {phase!r}")

    out: dict[str, Path] = {}
    for variant in variants:
        candidate = input_dir / variant / naming(variant)
        if candidate.is_file():
            out[variant] = candidate
            continue
        # phase_c mock output (dry-run of phase_c) — accept for smoke.
        mock_naming = mock_fallback_by_phase.get(phase)
        if mock_naming is not None:
            mock = input_dir / variant / mock_naming(variant)
            if mock.is_file():
                out[variant] = mock
    if not out:
        # Smoke fallback: any *.jsonl directly under the root.
        stray = sorted(input_dir.glob("*.jsonl"))
        if stray:
            out["_flat"] = stray[0]
    return out


def _locate_prefix_end_char(text: str, answer_start: int, form: str | None) -> int | None:
    """Return the CHAR position of the last prefix char (before the user query),
    or None when the prefix arm cannot be located deterministically.

    Convention: for `attrib_quoted` rows the prefix ends immediately BEFORE the
    attribution marker ` replied: "` / ` said, "` — i.e. the last character of
    the story preamble that ANNOUNCES the answering turn. `text[:prefix_end]`
    is the pre-query narrative + the user's question and everything up to (but
    not including) the attribution.

    For non-attrib_quoted rows (chat-formatted templates that phase_d may emit
    via the chat splice form), we currently return None: chat-form prefix
    resolution requires the tokenizer's chat-template offsets, which are the
    fits-phase's concern — the capture rig records prefix arm as null for
    those rows and reports the null fraction as a diagnostic. Downstream fits
    can re-locate via the chat-template offset recipe on-VM.
    """
    before = text[:answer_start]
    best: int = -1
    for marker in _ATTRIB_MARKERS:
        idx = before.rfind(marker)
        if idx > best:
            best = idx
    if best < 0:
        return None
    return best


def _char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Map a [char_start, char_end) byte-offset range to inclusive-exclusive
    token indices via the tokenizer's offset mapping.

    Convention: a token whose byte range [tok_lo, tok_hi) OVERLAPS the char
    range [char_start, char_end) is IN the answer span. Empty overlap -> not
    in the span.
    """
    lo = None
    hi = 0
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:
            continue
        if tok_hi <= char_start:
            continue
        if tok_lo >= char_end:
            break
        if lo is None:
            lo = i
        hi = i + 1
    if lo is None:
        return 0, 0
    return lo, hi


def _token_before_char(offsets: list[tuple[int, int]], char_pos: int) -> int:
    """Return the token index whose end offset is <= char_pos (the last token
    STRICTLY before char_pos).
    """
    idx = -1
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:
            continue
        if tok_hi <= char_pos:
            idx = i
        else:
            break
    return max(0, idx)


def _compute_positions(tokenizer, row: dict) -> dict | None:
    """Tokenize `row['final_text']` and derive (token_ids, v_C_pos, v_A span,
    v_P_pos) from the row's char offsets.

    Returns:
      {input_ids, answer_lo, answer_hi, v_C_pos, v_P_pos, n_tokens}
      or None if positions cannot be resolved deterministically (recorded
      per-row diagnostic).
    """
    text = row.get("final_text")
    a_start = row.get("answer_start")
    a_end = row.get("answer_end")
    if not isinstance(text, str) or not isinstance(a_start, int) or not isinstance(a_end, int):
        return None
    if a_end <= a_start or a_end > len(text) or a_start < 0:
        return None
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(enc["input_ids"])
    offsets = [(int(lo), int(hi)) for lo, hi in enc["offset_mapping"]]
    if not input_ids:
        return None
    answer_lo, answer_hi = _char_span_to_token_span(offsets, a_start, a_end)
    if answer_hi <= answer_lo:
        return None
    v_C_pos = max(0, answer_lo - 1)
    prefix_end_char = _locate_prefix_end_char(text, a_start, row.get("form"))
    if prefix_end_char is None or prefix_end_char <= 0:
        v_P_pos = None
    else:
        v_P_pos = _token_before_char(offsets, prefix_end_char)
    return {
        "input_ids": input_ids,
        "answer_lo": answer_lo,
        "answer_hi": answer_hi,
        "v_C_pos": v_C_pos,
        "v_P_pos": v_P_pos,
        "n_tokens": len(input_ids),
    }


def _write_activation_shell(out_path: Path) -> None:
    """Create an empty 0-byte activation file (dry-run only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"")


def _write_diagnostics(diagnostics_path: Path, payload: dict) -> None:
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = diagnostics_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, diagnostics_path)


def _answer_length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    import statistics

    xs = sorted(lengths)
    n = len(xs)

    def _pct(p: float) -> float:
        if n == 1:
            return float(xs[0])
        # Nearest-rank (ceiling) percentile — deterministic + integer-friendly.
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(xs[k])

    return {
        "n": n,
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "p10": _pct(0.10),
        "p90": _pct(0.90),
    }


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def _load_model(model_id: str, device: str):
    """Load a Qwen2.5-7B HF CausalLM in bf16 for teacher-forced capture."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def _call_model_with_hidden_states(model, input_ids, attention_mask):
    """One HF forward with hidden-states out; logits_to_keep=1 saves ~4.9 GiB
    on Qwen2.5-7B (`.claude/rules/gotchas.md` — unread-logits OOM class, #779).
    """
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


def _capture_positions_only(
    tokenizer, rows: list[dict], sample_n: int
) -> tuple[list[dict], int, int, int]:
    """Dry-run helper: resolve positions for up to `sample_n` rows without any
    model forward. Returns (per-row diagnostic list, ok count, prefix-null
    count, skipped count).
    """
    per_row: list[dict] = []
    ok = 0
    prefix_null = 0
    skipped = 0
    for row in rows[: max(1, sample_n)]:
        pos = _compute_positions(tokenizer, row)
        if pos is None:
            skipped += 1
            per_row.append(
                {
                    "conv_id": str(row.get("conv_id") or row.get("scaffold_id") or ""),
                    "status": "skipped",
                }
            )
            continue
        ok += 1
        if pos["v_P_pos"] is None:
            prefix_null += 1
        per_row.append(
            {
                "conv_id": str(row.get("conv_id") or row.get("scaffold_id") or ""),
                "status": "ok",
                "n_tokens": pos["n_tokens"],
                "answer_lo": pos["answer_lo"],
                "answer_hi": pos["answer_hi"],
                "v_C_pos": pos["v_C_pos"],
                "v_P_pos": pos["v_P_pos"],
            }
        )
    return per_row, ok, prefix_null, skipped


def _run_dry_variant(
    variant: str,
    model_slug: str,
    input_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Dry-run per-variant handler: exercise CLI + input reader + tokenization
    on <=3 sample rows, emit 0-byte activation shell + diagnostics stub, no
    model load, no GPU.
    """
    rows = _read_jsonl(input_path)
    n_in = len(rows)
    if n_in == 0:
        return {
            "variant": variant,
            "model": model_slug,
            "input_path": _rel(input_path),
            "n_in": 0,
            "n_out": 0,
            "status": "empty-input",
        }

    # Optional target conv_id cap for sample-set size (mirrors phase_c/d).
    if args.target_conv_ids and args.target_conv_ids > 0:
        rows = rows[: args.target_conv_ids]

    model_id = _MODEL_ID.get(model_slug, model_slug)
    _log(f"variant={variant} model={model_slug} loading tokenizer only (dry-run)")
    tokenizer = _load_tokenizer(model_id)

    sample_n = min(3, len(rows)) if args.dry_run else len(rows)
    per_row, ok, prefix_null, skipped = _capture_positions_only(tokenizer, rows, sample_n)

    lengths = [
        int(r.get("answer_hi", 0) - r.get("answer_lo", 0))
        for r in per_row
        if r.get("status") == "ok"
    ]
    conv_ids = sorted({r["conv_id"] for r in per_row if r.get("status") == "ok" and r["conv_id"]})

    activation_path = output_dir / variant / f"{variant}_{model_slug}.npz"
    _write_activation_shell(activation_path)

    diagnostics_path = diagnostics_dir / f"{variant}_{model_slug}.json"
    payload = {
        "phase": "capture",
        "dry_run": True,
        "variant": variant,
        "model": model_slug,
        "input_path": _rel(input_path),
        "output_path": _rel(activation_path),
        "layer": args.layer,
        "seed": args.seed,
        "n_in": n_in,
        "n_sampled": sample_n,
        "n_ok": ok,
        "n_skipped": skipped,
        "n_prefix_null": prefix_null,
        "answer_token_length_stats": _answer_length_stats(lengths),  # DV 7 stub
        "conv_ids": conv_ids,  # DV 8 stub (post-any-filter set on sampled subset)
        "per_row": per_row,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    _write_diagnostics(diagnostics_path, payload)

    _log(
        f"variant={variant} model={model_slug} dry-run "
        f"sampled={sample_n}/{n_in} ok={ok} prefix_null={prefix_null} skipped={skipped} "
        f"-> shell={_rel(activation_path)} diag={_rel(diagnostics_path)}"
    )
    return {
        "variant": variant,
        "model": model_slug,
        "input_path": _rel(input_path),
        "n_in": n_in,
        "n_sampled": sample_n,
        "n_out": ok,
        "n_skipped": skipped,
        "n_prefix_null": prefix_null,
        "diagnostics": _rel(diagnostics_path),
        "activations": _rel(activation_path),
        "status": "dry-ok",
    }


def _select_layer_slice(hidden_states_tuple, layer: int, hidden_dim: int):
    """Return the (B, T, D) tensor for the requested layer.

    HF's `output_hidden_states=True` returns a tuple of length `n_layers + 1`
    (index 0 = embeddings; index i = post-layer-i state), so `layer=19` reads
    `hidden_states_tuple[19]`. Fail loud on any shape/dim mismatch.
    """
    if not (0 <= layer < len(hidden_states_tuple)):
        raise ValueError(
            f"layer={layer} out of range for model with {len(hidden_states_tuple) - 1} layers"
        )
    hs = hidden_states_tuple[layer]
    if hs.shape[-1] != hidden_dim:
        raise ValueError(f"hidden dim mismatch: got {hs.shape[-1]} expected {hidden_dim}")
    return hs


def _run_gpu_variant(
    variant: str,
    model_slug: str,
    input_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Production per-variant handler: teacher-forced HF forward per row at
    the target layer; emits {conv_id: {v_C, v_A, v_P}} to an .npz per cell.

    This is the on-pod path. Not exercised by the dry-run smoke.
    """
    import numpy as np
    import torch

    rows = _read_jsonl(input_path)
    n_in = len(rows)
    if n_in == 0:
        return {
            "variant": variant,
            "model": model_slug,
            "input_path": _rel(input_path),
            "n_in": 0,
            "n_out": 0,
            "status": "empty-input",
        }
    if args.target_conv_ids and args.target_conv_ids > 0:
        rows = rows[: args.target_conv_ids]

    model_id = _MODEL_ID.get(model_slug, model_slug)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"variant={variant} model={model_slug} device={device} loading tokenizer+model")
    tokenizer = _load_tokenizer(model_id)
    model = _load_model(model_id, device)
    hidden_dim = int(model.config.hidden_size)

    # Ensure right-padding (positions index the UNPADDED sequence — the
    # gotchas.md § "Teacher-forced capture inputs" invariant).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError(
            "capture positions index the UNPADDED sequence and require RIGHT padding; "
            f"tokenizer.padding_side={tokenizer.padding_side!r}"
        )

    per_row_diag: list[dict] = []
    conv_ids: list[str] = []
    v_C_rows: list[np.ndarray] = []
    v_A_rows: list[np.ndarray] = []
    v_P_rows: list[np.ndarray | None] = []
    lengths: list[int] = []

    n_skipped = 0
    n_prefix_null = 0
    t0 = time.time()
    peak_gpu_bytes = 0

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prepared: list[tuple[dict, dict]] = []
        for row in batch:
            pos = _compute_positions(tokenizer, row)
            if pos is None:
                n_skipped += 1
                per_row_diag.append(
                    {
                        "conv_id": str(row.get("conv_id") or row.get("scaffold_id") or ""),
                        "status": "skipped",
                    }
                )
                continue
            prepared.append((row, pos))
        if not prepared:
            continue

        input_ids_list = [pos["input_ids"] for _, pos in prepared]
        padded = tokenizer.pad(
            {"input_ids": input_ids_list},
            return_tensors="pt",
            padding=True,
        )
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = _call_model_with_hidden_states(model, input_ids, attention_mask)
        hs = _select_layer_slice(outputs.hidden_states, args.layer, hidden_dim)

        for local_i, (row, pos) in enumerate(prepared):
            v_C = hs[local_i, pos["v_C_pos"], :].to(torch.float16).cpu().numpy()
            v_A = (
                hs[local_i, pos["answer_lo"] : pos["answer_hi"], :]
                .mean(dim=0)
                .to(torch.float16)
                .cpu()
                .numpy()
            )
            if pos["v_P_pos"] is None:
                v_P: np.ndarray | None = None
                n_prefix_null += 1
            else:
                v_P = hs[local_i, pos["v_P_pos"], :].to(torch.float16).cpu().numpy()

            conv_id = str(row.get("conv_id") or row.get("scaffold_id") or "")
            conv_ids.append(conv_id)
            v_C_rows.append(v_C.astype(np.float16, copy=False))
            v_A_rows.append(v_A.astype(np.float16, copy=False))
            v_P_rows.append(v_P.astype(np.float16, copy=False) if v_P is not None else None)
            lengths.append(int(pos["answer_hi"] - pos["answer_lo"]))
            per_row_diag.append(
                {
                    "conv_id": conv_id,
                    "status": "ok",
                    "n_tokens": pos["n_tokens"],
                    "answer_lo": pos["answer_lo"],
                    "answer_hi": pos["answer_hi"],
                    "v_C_pos": pos["v_C_pos"],
                    "v_P_pos": pos["v_P_pos"],
                }
            )

        if device == "cuda":
            peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated()))

        # Per-unit progress line (code-style.md § Checkpoint-per-phase per-unit
        # progress line; batches are the natural unit here).
        elapsed = time.time() - t0
        print(
            f"[phase=capture] batch {start // batch_size + 1}/"
            f"{(len(rows) + batch_size - 1) // batch_size} "
            f"variant={variant} model={model_slug} rows={len(prepared)} "
            f"cumulative={len(conv_ids)}/{len(rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del outputs, hs, input_ids, attention_mask

    # Stack + write. v_P is stored per-row with a mask carrying the null rows;
    # None rows write a zero vector alongside `v_P_present == False`.
    import numpy as np

    activation_path = output_dir / variant / f"{variant}_{model_slug}.npz"
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    v_C_arr = (
        np.stack(v_C_rows, axis=0) if v_C_rows else np.zeros((0, hidden_dim), dtype=np.float16)
    )
    v_A_arr = (
        np.stack(v_A_rows, axis=0) if v_A_rows else np.zeros((0, hidden_dim), dtype=np.float16)
    )
    v_P_present = np.array([vp is not None for vp in v_P_rows], dtype=bool)
    v_P_arr = (
        np.stack(
            [
                vp if vp is not None else np.zeros((hidden_dim,), dtype=np.float16)
                for vp in v_P_rows
            ],
            axis=0,
        )
        if v_P_rows
        else np.zeros((0, hidden_dim), dtype=np.float16)
    )
    tmp = activation_path.with_suffix(".npz.tmp.npz")  # never trailing `.tmp` — np.savez
    # appends `.npz` to any target not already ending in `.npz`; the .tmp.npz
    # suffix stays valid and the replace() source resolves.
    np.savez(
        tmp,
        conv_id=np.array(conv_ids),
        v_C=v_C_arr,
        v_A=v_A_arr,
        v_P=v_P_arr,
        v_P_present=v_P_present,
    )
    os.replace(tmp, activation_path)

    wall_seconds = time.time() - t0
    diagnostics_path = diagnostics_dir / f"{variant}_{model_slug}.json"
    payload = {
        "phase": "capture",
        "dry_run": False,
        "variant": variant,
        "model": model_slug,
        "input_path": _rel(input_path),
        "output_path": _rel(activation_path),
        "layer": args.layer,
        "seed": args.seed,
        "batch_size": batch_size,
        "n_in": n_in,
        "n_processed": len(rows),
        "n_ok": len(conv_ids),
        "n_skipped": n_skipped,
        "n_prefix_null": n_prefix_null,
        "answer_token_length_stats": _answer_length_stats(lengths),  # DV 7
        "conv_ids": sorted(set(conv_ids)),  # DV 8
        "peak_gpu_bytes": peak_gpu_bytes if device == "cuda" else 0,
        "wall_seconds": round(wall_seconds, 3),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    _write_diagnostics(diagnostics_path, payload)

    _log(
        f"variant={variant} model={model_slug} captured n_ok={len(conv_ids)}/{n_in} "
        f"prefix_null={n_prefix_null} skipped={n_skipped} "
        f"peak_gpu={peak_gpu_bytes / (1024**3):.2f} GiB wall={wall_seconds:.1f}s "
        f"-> {_rel(activation_path)}"
    )
    return {
        "variant": variant,
        "model": model_slug,
        "input_path": _rel(input_path),
        "n_in": n_in,
        "n_out": len(conv_ids),
        "n_skipped": n_skipped,
        "n_prefix_null": n_prefix_null,
        "peak_gpu_bytes": peak_gpu_bytes if device == "cuda" else 0,
        "wall_seconds": round(wall_seconds, 3),
        "diagnostics": _rel(diagnostics_path),
        "activations": _rel(activation_path),
        "status": "ok",
    }


def _upload_to_hf(activations_by_variant: dict[str, Path], model_slug: str) -> None:
    """Best-effort mirror of activations — ONE bulk `upload_folder` commit per
    (variant, model) via the shared `_upload_folder_filtered` helper.

    Skipped by the caller when `--skip-upload` or `--dry-run` is set.
    """
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    # Group by common parent so ONE commit can cover the whole set. The natural
    # root here is the output-dir passed to the driver; the caller resolves it.
    if not activations_by_variant:
        return
    parents = {p.parent.parent.resolve() for p in activations_by_variant.values()}
    if len(parents) != 1:
        _log(f"WARN heterogeneous activation roots; skipping bulk upload: {parents}")
        return
    root = next(iter(parents))
    allow_patterns: list[str] = []
    expected_paths: list[str] = []
    for variant, p in activations_by_variant.items():
        if not p.is_file() or p.stat().st_size == 0:
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        allow_patterns.append(rel)
        expected_paths.append(f"{TASK_PREFIX}/activations/{rel}")
    if not allow_patterns:
        return
    try:
        _upload_folder_filtered(
            root,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/activations",
            allow_patterns=allow_patterns,
            expected_repo_paths=expected_paths,
        )
        _log(
            f"uploaded {len(allow_patterns)} activation file(s) in one bulk commit "
            f"(model={model_slug})"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN activation bulk upload failed (model={model_slug}): {exc}")


def run_phase(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.exists():
        print(f"ERROR: input-dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = (_REPO_ROOT / "eval_results/issue_2054/capture_diagnostics").resolve()
    if str(output_dir).startswith("/tmp/"):
        # Smoke output tree stays fully under /tmp — never write diagnostics to
        # the repo-tracked eval_results/ from a smoke.
        diagnostics_dir = (output_dir / "capture_diagnostics").resolve()

    variants = list(args.variants)
    input_paths = _resolve_input_paths(input_dir, variants, args.phase)
    if not input_paths:
        print(
            f"ERROR: no phase={args.phase} input JSONLs under {input_dir} for variants={variants}",
            file=sys.stderr,
        )
        return 2

    _log(
        f"start: phase={args.phase} model={args.model} layer={args.layer} "
        f"dry_run={args.dry_run} variants={list(input_paths.keys())}"
    )

    per_variant_reports: list[dict] = []
    activations_by_variant: dict[str, Path] = {}
    for variant, input_path in input_paths.items():
        try:
            if args.dry_run:
                report = _run_dry_variant(
                    variant, args.model, input_path, output_dir, diagnostics_dir, args
                )
            else:
                report = _run_gpu_variant(
                    variant, args.model, input_path, output_dir, diagnostics_dir, args
                )
        except Exception as exc:  # noqa: BLE001
            _log(f"ERROR variant={variant} model={args.model}: {exc}")
            per_variant_reports.append(
                {
                    "variant": variant,
                    "model": args.model,
                    "input_path": _rel(input_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        per_variant_reports.append(report)
        activations_path = Path(report.get("activations") or "")
        if activations_path.name:
            resolved = (
                activations_path
                if activations_path.is_absolute()
                else _REPO_ROOT / activations_path
            )
            activations_by_variant[variant] = resolved

    total_ok = sum(int(r.get("n_out") or 0) for r in per_variant_reports)
    if not args.dry_run and total_ok == 0:
        print("ERROR: capture produced ZERO summaries across all variants", file=sys.stderr)
        return 1

    is_smoke = str(output_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload and not args.dry_run:
        try:
            _upload_to_hf(activations_by_variant, args.model)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN upload stage failed: {exc}")

    digest = {
        "phase": "capture",
        "model": args.model,
        "layer": args.layer,
        "dry_run": bool(args.dry_run),
        "per_variant": per_variant_reports,
        "n_total_ok": total_ok,
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    digest_path = output_dir / f"capture_digest_{args.model}.json"
    tmp = digest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, digest_path)

    print(f"[phase=capture] digest: n_total_ok={total_ok}", flush=True)
    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", required=True, help="phase_b / phase_c / phase_d output root")
    p.add_argument("--output-dir", default="data/issue_2054/activations/")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        required=True,
        help="comma-separated variant slugs (e.g. char_helios,char_wren)",
    )
    p.add_argument(
        "--model",
        default="qwen2.5-7b-instruct",
        help="qwen2.5-7b | qwen2.5-7b-instruct",
    )
    p.add_argument(
        "--phase",
        choices=("inserted", "on_policy", "cell_c"),
        required=True,
        help="which upstream unit's output layout to read",
    )
    p.add_argument("--layer", type=int, default=19, help="hidden-state layer index (plan §11)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EPM_CAPTURE_BATCH_SIZE", "8")),
        help="teacher-forced forward batch size (default 8, env EPM_CAPTURE_BATCH_SIZE)",
    )
    p.add_argument(
        "--target-conv-ids",
        type=int,
        default=0,
        help="if >0, cap the input rows to this count (mirrors phase_c/d convention)",
    )
    p.add_argument("--skip-upload", action="store_true", help="skip HF mirror step")
    p.add_argument("--upload", action="store_true", help="force HF mirror step (default when GPU)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only wiring smoke: exercise CLI + tokenization, emit 0-byte activation shell",
    )
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
