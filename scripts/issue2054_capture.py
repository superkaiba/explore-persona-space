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
- `v_P` (prefix arm) — layer-19 hidden state at the last token BEFORE the user
  query (plan §6 pooling row: last-token pooling at the PRE-QUERY position).
  Located per row:
  * rows carrying the renderer-recorded `prefix_end_char` field (every
    phase_b/c/d row as of the framing-axis build — `issue2054_forms` records
    the pre-query boundary BY CONSTRUCTION for all forms): use it directly.
  * rows lacking the field (parent-legacy / round-1 outputs): a form-aware
    fallback — chat / bare_text rows read the fixed turn-header length;
    attrib_quoted (or form-less) rows keep the legacy attribution-marker
    search ` replied: "` / ` said, "` (NOTE: pre-ATTRIBUTION, not pre-query —
    a plan-divergent legacy convention, surfaced per-row as
    `prefix_src="legacy_marker"` so a mixed-convention cell is visible);
    bare_label rows locate the `{character}: ` label; v_P is undefined when
    nothing resolves and that row's v_P is recorded as null (never coerced).

Activations persist to
`<output-dir>/{variant}/{variant}__{phase}__{form}__{model}.npz` as three
`{conv_id: <d-vec>}` fp16 arrays keyed by conv_id (never materializes the full
corpus in memory at once — a streaming per-row concatenation via a resizeable
list; the plan's stream-reduce recipe is inherited when the fits phase reads
these back). The filename is the canonical 4-axis cell key
(`issue2054_forms.cell_key` — identity x condition x framing x model), so two
runs differing only in `--phase` / `--form` land on DISTINCT files (C6: the
pre-fix `{variant}_{model}` naming let `--phase on_policy` overwrite cell (b)
with cell (d)). Per-cell diagnostics (DV 7 answer-length parity + DV 8 conv_id
intersection + realized row count + peak GPU memory / wall-time + the per-row
position block kill-gate 5 reads) land at
`eval_results/issue_2054/capture_diagnostics/{cell_key}.json`.

Uploads to HF `superkaiba1/explore-persona-space-data/issue2054_lattice/activations/{variant}/`
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

import issue2054_forms as forms  # noqa: E402
import issue2054_resume as resume  # noqa: E402

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
    """Tolerant JSONL reader; undecodable lines are COUNTED + warned (M3 —
    never silently skipped)."""
    rows: list[dict] = []
    n_bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        _log(f"WARN {n_bad} undecodable JSONL line(s) skipped in {path}")
    return rows


def _resolve_input_paths(
    input_dir: Path, variants: list[str], phase: str, form: str
) -> dict[str, Path]:
    """Return {variant: input JSONL path} keyed on variant.

    Condition+form-scoped naming shares ONE source with the producing units
    (`issue2054_forms.phase_output_name` — phase_b/phase_c/phase_d write the
    same names), e.g. phase=inserted form=chat ->
    `spliced_inserted_<variant>__chat.jsonl`; phase=on_policy falls back to
    the `.mock.jsonl` twin (phase_c dry-run output) for smoke.

    Smoke fallback: any *.jsonl directly under `input_dir` counts as a `_flat`
    variant when the variant subtree is missing.
    """
    out: dict[str, Path] = {}
    for variant in variants:
        candidate = input_dir / variant / forms.phase_output_name(phase, variant, form)
        if candidate.is_file():
            out[variant] = candidate
            continue
        # phase_c mock output (dry-run of phase_c) — accept for smoke.
        if phase == "on_policy":
            mock = input_dir / variant / forms.phase_output_name(phase, variant, form, mock=True)
            if mock.is_file():
                out[variant] = mock
    if not out:
        # Smoke fallback: any *.jsonl directly under the root.
        stray = sorted(input_dir.glob("*.jsonl"))
        if stray:
            out["_flat"] = stray[0]
    return out


def _locate_prefix_end_char(text: str, answer_start: int, form: str | None) -> int | None:
    """LEGACY attribution-marker search (rows lacking `prefix_end_char` only).

    Returns the CHAR position immediately BEFORE the attribution marker
    ` replied: "` / ` said, "` — the last character of the story preamble that
    ANNOUNCES the answering turn. NOTE this is the pre-ATTRIBUTION boundary
    (round-1 convention), NOT the plan §6 pre-query boundary the renderers now
    record; rows resolved through this path are tagged
    `prefix_src="legacy_marker"` in the diagnostics so a mixed-convention cell
    is visible. None when no marker is found.
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


def _prefix_end_char_for_row(row: dict, text: str, answer_start: int) -> tuple[int | None, str]:
    """(pre-query char boundary, source tag) for one row — every form covered.

    Preference order:
      1. `prefix_end_char` recorded by the renderer (`issue2054_forms` — the
         plan §6 pre-query convention, by construction) -> "recorded".
      2. Form-aware fallback for legacy rows:
         chat / bare_text -> the fixed turn-header length ("form_header");
         attrib_quoted (or form-less) -> legacy attribution-marker search
         ("legacy_marker" — pre-attribution, plan-divergent; see
         `_locate_prefix_end_char`);
         bare_label -> the `{character}: ` label before the answer
         ("legacy_label");
         bare_paragraph -> end of the pre-answer prose, trailing newlines
         stripped ("legacy_paragraph").
      3. (None, "none") — v_P recorded null, never coerced.
    """
    rec = row.get("prefix_end_char")
    if isinstance(rec, int) and not isinstance(rec, bool) and 0 <= rec <= answer_start:
        return rec, "recorded"
    form = row.get("form")
    if form == "chat":
        if text.startswith(forms.CHAT_USER_HEADER):
            return len(forms.CHAT_USER_HEADER), "form_header"
        return None, "none"
    if form == "bare_text":
        if text.startswith(forms.BARE_USER_PREFIX):
            return len(forms.BARE_USER_PREFIX), "form_header"
        return None, "none"
    if form in (None, "attrib_quoted"):
        idx = _locate_prefix_end_char(text, answer_start, form)
        if idx is not None:
            return idx, "legacy_marker"
        return None, "none"
    if form == "bare_label":
        character = str(row.get("character") or "").strip()
        if character:
            label = f"{character}: "
            if text[:answer_start].endswith(label):
                return answer_start - len(label), "legacy_label"
        return None, "none"
    if form == "bare_paragraph":
        stripped = len(text[:answer_start].rstrip("\n"))
        if 0 < stripped < answer_start:
            return stripped, "legacy_paragraph"
        return None, "none"
    return None, "none"


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


def _token_before_char(offsets: list[tuple[int, int]], char_pos: int) -> int | None:
    """Return the token index whose end offset is <= char_pos (the last token
    STRICTLY before char_pos), or None when NO token ends before char_pos —
    the caller records v_P as null (M3: the pre-fix `max(0, idx)` silently
    substituted token 0, coercing an undefined prefix boundary to a wrong
    activation instead of the null-recording contract).
    """
    idx = -1
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:
            continue
        if tok_hi <= char_pos:
            idx = i
        else:
            break
    return None if idx < 0 else idx


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
    prefix_end_char, prefix_src = _prefix_end_char_for_row(row, text, a_start)
    if prefix_end_char is None or prefix_end_char <= 0:
        v_P_pos = None
        prefix_src = "none"
    else:
        v_P_pos = _token_before_char(offsets, prefix_end_char)
        if v_P_pos is None:
            # No token ends before the recorded boundary: v_P is undefined for
            # this row — record null, never coerce to token 0 (M3).
            prefix_src = "none"
    return {
        "input_ids": input_ids,
        "answer_lo": answer_lo,
        "answer_hi": answer_hi,
        "v_C_pos": v_C_pos,
        "v_P_pos": v_P_pos,
        "prefix_src": prefix_src,
        "n_tokens": len(input_ids),
    }


def _write_activation_shell(out_path: Path) -> None:
    """Create an empty 0-byte activation file (dry-run only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"")


# Blocks every consumer gate reads from the capture diagnostics: `per_row` is
# kill-gate 5's answer-length source (`issue2054_fits._answer_length_ks_from_
# diagnostics` reads `per_row[*].answer_hi - answer_lo`), the stats block is
# DV 7, `conv_ids` is DV 8. A payload missing any of them is a defect, never a
# mode difference (C7: the pre-fix production payload omitted `per_row`, so
# gate 5 always read empty-length-arrays -> KS=NaN -> could never fire).
_GATE_SOURCE_KEYS = ("per_row", "answer_token_length_stats", "conv_ids")


def _write_diagnostics(diagnostics_path: Path, payload: dict) -> None:
    """Atomic diagnostics write; REFUSES a payload missing a gate source (C7)."""
    for key in _GATE_SOURCE_KEYS:
        if payload.get(key) is None:
            raise ValueError(
                f"capture diagnostics payload missing {key!r} — the downstream "
                f"kill-gate/DV sources {_GATE_SOURCE_KEYS} are non-optional (C7)"
            )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = diagnostics_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, diagnostics_path)


def _diagnostics_payload(
    *,
    dry_run: bool,
    variant: str,
    condition: str,
    form: str,
    model_slug: str,
    input_path: Path,
    activation_path: Path,
    layer: int,
    seed: int,
    n_in: int,
    n_ok: int,
    n_skipped: int,
    n_prefix_null: int,
    per_row: list[dict],
    lengths: list[int],
    conv_ids: list[str],
    extra: dict | None = None,
) -> dict:
    """Single serialization site for capture diagnostics.

    BOTH the dry-run and the production (GPU) handlers build their payload
    here, so the `per_row` block kill-gate 5 reads is carried UNCONDITIONALLY
    (C7: pre-fix, only the dry-run branch serialized `per_row`).
    """
    payload = {
        "phase": "capture",
        "dry_run": bool(dry_run),
        "variant": variant,
        "condition": condition,
        "form": form,
        "model": model_slug,
        "cell": forms.cell_key(variant, condition, form, model_slug),
        "input_path": _rel(input_path),
        "output_path": _rel(activation_path),
        "layer": layer,
        "seed": seed,
        "n_in": n_in,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_prefix_null": n_prefix_null,
        "prefix_src_counts": _prefix_src_counts(per_row),
        "answer_token_length_stats": _answer_length_stats(lengths),  # DV 7
        "conv_ids": conv_ids,  # DV 8
        "per_row": per_row,  # kill-gate 5 length source (C7)
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


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


def _prefix_src_counts(per_row: list[dict]) -> dict[str, int]:
    """Per-cell tally of prefix-boundary sources (mixed-convention visibility:
    `recorded` = plan §6 pre-query, `legacy_*` = fallback conventions)."""
    counts: dict[str, int] = {}
    for r in per_row:
        if r.get("status") != "ok":
            continue
        src = str(r.get("prefix_src") or "none")
        counts[src] = counts.get(src, 0) + 1
    return counts


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
                "prefix_src": pos["prefix_src"],
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

    # 4-axis cell naming (C6): condition + form are part of the output identity.
    cell = forms.cell_key(variant, args.phase, args.form, model_slug)
    activation_path = output_dir / variant / f"{cell}.npz"
    _write_activation_shell(activation_path)

    diagnostics_path = diagnostics_dir / f"{cell}.json"
    payload = _diagnostics_payload(
        dry_run=True,
        variant=variant,
        condition=args.phase,
        form=args.form,
        model_slug=model_slug,
        input_path=input_path,
        activation_path=activation_path,
        layer=args.layer,
        seed=args.seed,
        n_in=n_in,
        n_ok=ok,
        n_skipped=skipped,
        n_prefix_null=prefix_null,
        per_row=per_row,
        lengths=lengths,
        conv_ids=conv_ids,
        extra={"n_sampled": sample_n},
    )
    _write_diagnostics(diagnostics_path, payload)

    _log(
        f"variant={variant} model={model_slug} dry-run "
        f"sampled={sample_n}/{n_in} ok={ok} prefix_null={prefix_null} skipped={skipped} "
        f"-> shell={_rel(activation_path)} diag={_rel(diagnostics_path)}"
    )
    return {
        "variant": variant,
        "model": model_slug,
        "condition": args.phase,
        "form": args.form,
        "cell": cell,
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

    Resume (C9/M6): a cell whose .npz + regime-matching done sidecar already
    exist is SKIPPED — a crash on variant 4/5 of the ~100 GPU-h capture no
    longer re-forwards variants 1-3. Regime = the FULL 4-axis cell key
    (variant/condition/form/model — the Unit C constraint) + layer + seed +
    target_conv_ids; a CHANGED input JSONL recomputes with a log line; a
    DIFFERENT regime at the same path refuses (RegimeMismatch).
    """
    # 4-axis cell naming (C6): condition + form are part of the output identity.
    cell = forms.cell_key(variant, args.phase, args.form, model_slug)
    activation_path = output_dir / variant / f"{cell}.npz"
    regime = {
        "cell": cell,
        "layer": int(args.layer),
        "seed": int(args.seed),
        "target_conv_ids": int(args.target_conv_ids),
    }
    inputs = {"input_sha256": resume.file_sha256(input_path)}
    disposition, reason = resume.resume_disposition(
        activation_path, regime, inputs, overwrite=args.overwrite
    )
    if disposition == resume.SKIP:
        done = resume.read_done(activation_path) or {}
        n_ok = int((done.get("extra") or {}).get("n_ok") or 0)
        _log(f"variant={variant} model={model_slug} RESUME skip ({reason}) cell={cell}")
        return {
            "variant": variant,
            "model": model_slug,
            "condition": args.phase,
            "form": args.form,
            "cell": cell,
            "input_path": _rel(input_path),
            "n_in": n_ok,
            "n_out": n_ok,
            "diagnostics": _rel(diagnostics_dir / f"{cell}.json"),
            "activations": _rel(activation_path),
            "status": "resumed",
        }
    if disposition == resume.RECOMPUTE:
        _log(f"variant={variant} model={model_slug} recompute: {reason}")

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
                    "prefix_src": pos["prefix_src"],
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
    # (`cell` + `activation_path` were resolved at the resume check above.)
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
    diagnostics_path = diagnostics_dir / f"{cell}.json"
    payload = _diagnostics_payload(
        dry_run=False,
        variant=variant,
        condition=args.phase,
        form=args.form,
        model_slug=model_slug,
        input_path=input_path,
        activation_path=activation_path,
        layer=args.layer,
        seed=args.seed,
        n_in=n_in,
        n_ok=len(conv_ids),
        n_skipped=n_skipped,
        n_prefix_null=n_prefix_null,
        per_row=per_row_diag,  # kill-gate 5 length source — production too (C7)
        lengths=lengths,
        conv_ids=sorted(set(conv_ids)),
        extra={
            "batch_size": batch_size,
            "n_processed": len(rows),
            "peak_gpu_bytes": peak_gpu_bytes if device == "cuda" else 0,
            "wall_seconds": round(wall_seconds, 3),
        },
    )
    _write_diagnostics(diagnostics_path, payload)
    # Done sidecar LAST — written only after BOTH the .npz and the diagnostics
    # landed, so a crash between the two can never mint a false skip (C9/M6).
    resume.write_done(
        activation_path,
        regime,
        inputs,
        extra={"n_ok": len(conv_ids), "wall_seconds": round(wall_seconds, 3)},
    )

    _log(
        f"variant={variant} model={model_slug} captured n_ok={len(conv_ids)}/{n_in} "
        f"prefix_null={n_prefix_null} skipped={n_skipped} "
        f"peak_gpu={peak_gpu_bytes / (1024**3):.2f} GiB wall={wall_seconds:.1f}s "
        f"-> {_rel(activation_path)}"
    )
    return {
        "variant": variant,
        "model": model_slug,
        "condition": args.phase,
        "form": args.form,
        "cell": cell,
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
    """Mirror activations — ONE bulk `upload_folder` commit per (variant,
    model) via the shared `_upload_folder_filtered` helper. FATAL on failure
    (M2): the capture store is ~100 GPU-h of plan-declared downstream input —
    a swallowed upload failure + `[phase=done]` + Step-8 teardown is the #521
    loss class, a direct Upload Policy violation.

    Skipped by the caller when `--skip-upload` or `--dry-run` is set.
    """
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    # Group by common parent so ONE commit can cover the whole set. The natural
    # root here is the output-dir passed to the driver; the caller resolves it.
    if not activations_by_variant:
        return
    parents = {p.parent.parent.resolve() for p in activations_by_variant.values()}
    if len(parents) != 1:
        raise RuntimeError(
            f"heterogeneous activation roots — cannot compose one bulk upload: {parents}"
        )
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
        # Declared outputs but nothing upload-eligible: an empty-set verify is
        # vacuous (#1482) — fail loud, never pass silently.
        raise RuntimeError(
            f"upload set resolved EMPTY against declared activations: {activations_by_variant}"
        )
    _upload_folder_filtered(
        root,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{TASK_PREFIX}/activations",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    _log(
        f"uploaded {len(allow_patterns)} activation file(s) in one bulk commit (model={model_slug})"
    )


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
    input_paths = _resolve_input_paths(input_dir, variants, args.phase, args.form)
    if not input_paths:
        print(
            f"ERROR: no phase={args.phase} form={args.form} input JSONLs under {input_dir} "
            f"for variants={variants}",
            file=sys.stderr,
        )
        return 2

    # GPU sharding (Unit F): stride the SORTED resolved variant list. Per-cell
    # writes are disjoint BY CONSTRUCTION ({variant}/{cell_key}.npz +
    # capture_diagnostics/{cell_key}.json — C6), so concurrent shards of one
    # (phase, form, model) invocation never collide on cell outputs; the
    # per-(phase, form, model) DIGEST is the one shared write, so it gains a
    # shard suffix below when shard_count > 1.
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        print(
            f"ERROR: invalid shard spec --shard-index={args.shard_index} "
            f"--shard-count={args.shard_count} (need 0 <= index < count, count >= 1)",
            file=sys.stderr,
        )
        return 2
    if args.shard_count > 1:
        all_resolved = sorted(input_paths)
        shard_variants = all_resolved[args.shard_index :: args.shard_count]
        input_paths = {v: input_paths[v] for v in shard_variants}
        _log(
            f"shard {args.shard_index}/{args.shard_count}: variants={shard_variants} "
            f"(resolved pool={all_resolved})"
        )
        if not input_paths:
            # A composition bug (more shards than resolved variants) — fail
            # loud rather than exit 0 with a vacuous digest.
            print(
                f"ERROR: shard {args.shard_index}/{args.shard_count} resolved EMPTY "
                f"against variants={all_resolved} — size --shard-count <= variant count",
                file=sys.stderr,
            )
            return 2

    _log(
        f"start: phase={args.phase} form={args.form} model={args.model} layer={args.layer} "
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
        except resume.RegimeMismatch:
            # A regime refusal is a WHOLE-invocation defect (wrong flags /
            # colliding output dir), never a per-variant skip — propagate.
            raise
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
    errored = [r["variant"] for r in per_variant_reports if r.get("status") == "error"]
    if not args.dry_run and errored:
        # Never exit 0 with cells missing — the digest records the errors, and
        # the resume sidecars make the re-run skip the completed cells (C9/M6).
        print(
            f"ERROR: capture errored on {len(errored)} variant(s): {errored} "
            "(completed cells persisted; re-run resumes them)",
            file=sys.stderr,
        )
        return 1

    is_smoke = str(output_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload and not args.dry_run:
        # FATAL on failure (M2): `[phase=done]` must never report done with
        # the 100 GPU-h activation store un-persisted (#521 class).
        _upload_to_hf(activations_by_variant, args.model)

    digest = {
        "phase": "capture",
        "condition": args.phase,
        "form": args.form,
        "model": args.model,
        "layer": args.layer,
        "dry_run": bool(args.dry_run),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "per_variant": per_variant_reports,
        "n_total_ok": total_ok,
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    # Digest keyed on (condition, form, model) — C6: two runs differing only in
    # --phase/--form must not overwrite each other's digest either. Under
    # sharding the digest additionally carries a shard suffix (Unit F): two
    # concurrent shards of ONE (phase, form, model) invocation must not both
    # write capture_digest__{phase}__{form}__{model}.json — the composer
    # (scripts/issue2054_shard_launch.py) aggregates the shard digests into
    # the canonical un-suffixed name post-hoc.
    sep = forms.CELL_KEY_SEP
    shard_suffix = (
        f"{sep}shard{args.shard_index}of{args.shard_count}" if args.shard_count > 1 else ""
    )
    digest_path = (
        output_dir
        / f"capture_digest{sep}{args.phase}{sep}{args.form}{sep}{args.model}{shard_suffix}.json"
    )
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
        choices=forms.CONDITIONS,
        required=True,
        help="condition axis: which upstream unit's output layout to read",
    )
    p.add_argument(
        "--form",
        required=True,
        choices=forms.FORMS,
        help=(
            "framing axis (plan §4; REQUIRED, no default — C6): selects the "
            "form-keyed input JSONLs and keys the .npz / diagnostics cell names"
        ),
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
        "--shard-index",
        type=int,
        default=0,
        help=(
            "0-based shard id (Unit F GPU sharding): this invocation captures "
            "sorted(resolved variants)[index::count]; per-cell writes are "
            "disjoint by construction, the digest gains a shard suffix"
        ),
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "total concurrent shards over the resolved variant list (default 1 "
            "= unsharded, byte-identical legacy behavior). Launch one process "
            "per shard with CUDA_VISIBLE_DEVICES pinned per GPU — see "
            "scripts/issue2054_shard_launch.py"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only wiring smoke: exercise CLI + tokenization, emit 0-byte activation shell",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "re-capture cells even when a regime-matching done sidecar exists "
            "(default resumes completed cells — C9/M6)"
        ),
    )
    args = p.parse_args()
    try:
        return run_phase(args)
    except resume.RegimeMismatch as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
