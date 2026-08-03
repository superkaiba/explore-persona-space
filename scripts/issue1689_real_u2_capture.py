"""Issue #1689 follow-up round ``real-u2-capture`` — Phases A2+B (render + capture).

Self-contained render + teacher-forced L19 activation capture over the
real-u2 corpus (Phase A0) + Haiku companion (Phase A1). Emits per-cell
stores at ``data/issue_1689/real_u2_capture/store/<model_slug>/<cell>/L19.pt``
carrying the SUBSET of the parent user_slot_recapture schema this round's
fits (Phase C) reads: ``slots`` dict (X_prefix / X_context / Y at
parent_answer_end / turn_end / mean_over_turn), ``conv_ids``, ``n_tokens``,
``unit`` metadata.

Cells (12 total): user_realu2 + user_haikuu2 × chat/naturalistic/story × base/instruct.

Framings + slots (verbatim from the parent's user_slot_recapture):

  chat  (text = apply_chat_template([u1, a1, u2], add_generation_prompt=True))
    prev_turn_end       end of a1's turn block — X_prefix arm (u1+a1 context)
    u2_end              end of u2's content — X_context arm (u1+a1+u2 context)
    parent_answer_end   end of the text (assistant-start header) — Y target

  naturalistic (text = "User: {u1}\\n\\nAssistant: {a1}\\n\\nUser: {u2}")
    prev_turn_end       end of "{a1}\\n\\n" — X_prefix
    u2_end              end of u2 (== end of text) — X_context / Y

  story (text = STORY_USER_TEMPLATE with "Alex" as the character)
    prev_turn_end       end of a1's closing ." — X_prefix
    u2_end              end of u2's content — X_context
    parent_answer_end   end of text — Y

Discipline:
  * Layer 19 ONLY (project HEADLINE_LAYER).
  * RIGHT padding + attention_mask (no left-pad RoPE trap, #502).
  * ``logits_to_keep=1`` when the model exposes it (avoids full-vocab
    allocation, #779) — probed via inspect.signature at load.
  * Char-offset → token-index via ``return_offsets_mapping=True`` — the
    #1092/#1315 recipe — with straddler policy EXCLUDE for X-side, INCLUDE
    for end-of-content slots.
  * Token-id concatenation invariant NOT enforced (this rig teacher-forces
    the full render as one sequence; no per-segment id concat, no seam
    merge misalignment risk from that direction).

Content hygiene: never prints row text (LMSYS/WildChat are unscreened
real-user text; digest-only per CLAUDE.md § harmful-content data hygiene).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from scripts.issue1689_common import (  # noqa: E402
    HEADLINE_LAYER,
    MODEL_BASE,
    MODEL_INSTRUCT,
)

# --- Framings + slot names --------------------------------------------------

CAPTURE_LAYER = HEADLINE_LAYER  # 19

FRAMINGS: tuple[str, ...] = ("chat", "naturalistic", "story")
MODELS: tuple[str, ...] = (MODEL_BASE, MODEL_INSTRUCT)
PROVENANCES: tuple[str, ...] = ("realu2", "haikuu2")

SLOT_NAMES: tuple[str, ...] = ("prev_turn_end", "u2_end", "parent_answer_end")

# Straddler policy — X-side slots EXCLUDE (no leak), end-of-content INCLUDE.
SLOT_STRADDLER: dict[str, str] = {
    "prev_turn_end": "exclude",
    "u2_end": "include",
    "parent_answer_end": "include",
}

# Fit pairs the Phase-C fits will run per cell (name, x_slot, y_slot).
PRIMARY_FIT_BY_FRAMING: dict[str, tuple[str, str, str]] = {
    "chat": ("prev_to_answer", "prev_turn_end", "parent_answer_end"),
    "naturalistic": ("prev_to_u2end", "prev_turn_end", "u2_end"),
    "story": ("prev_to_answer_story", "prev_turn_end", "parent_answer_end"),
}

# Story template (verbatim from parent user_slot_recapture render, "Alex" variant).
STORY_USER_TEMPLATE = (
    "A user named Alex sat down at their computer to talk with an AI assistant. "
    'Alex typed: "{u1}". The assistant replied: "{a1}". Alex thought about the '
    'response, then typed a follow-up: "{u2}". The assistant wrote back: '
)

NATURALISTIC_USER_TAG = "User: "
NATURALISTIC_ASSISTANT_TAG = "Assistant: "
NATURALISTIC_SEP = "\n\n"

MAX_TOKENS = 7104  # matches PROMPT_TOKEN_BUDGET

# Round-2 Minor #1 halt floor for per-cell empty-u2 drops. A rate above this
# floor means the haiku-gen fail-fast floor was misconfigured (haikuu2 arm)
# or the corpus's u2_real column is broken (realu2 arm) — either way, a
# silent shrink is unsafe. Matches the DROP_RATE_HALT_FLOOR in haiku_gen.
EMPTY_U2_HALT_FLOOR = 0.05


# --- Rendering --------------------------------------------------------------


def render_chat(u1: str, a1: str, u2: str, tokenizer) -> tuple[str, dict[str, int]]:
    """Chat template render; slot char offsets derived from the content-independent
    template tail (never `text.find(content)` — the #1776 rule)."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Slot boundaries: anchor from the fixed template tail (assistant start
    # header) working backwards.
    # `<|im_start|>assistant\n` marks parent_answer_end == len(text).
    parent_answer_end = len(text)
    # Last `<|im_end|>\n` before assistant start marks u2 turn terminator; the
    # SECOND-to-last marks a1 terminator.
    im_end_marker = "<|im_end|>\n"
    tail_positions: list[int] = []
    start = 0
    while True:
        pos = text.find(im_end_marker, start)
        if pos == -1:
            break
        tail_positions.append(pos + len(im_end_marker))
        start = pos + len(im_end_marker)
    # Expected: at least 3 markers (system?, user, assistant) or (u1, a1, u2).
    if len(tail_positions) < 2:
        raise RuntimeError(f"chat template lacks >=2 <|im_end|> markers: n={len(tail_positions)}")
    # a1's terminator = second-to-last <|im_end|>
    prev_turn_end = tail_positions[-2]
    # u2's terminator = last <|im_end|>
    u2_end = tail_positions[-1]
    return text, {
        "prev_turn_end": prev_turn_end,
        "u2_end": u2_end,
        "parent_answer_end": parent_answer_end,
    }


def render_naturalistic(u1: str, a1: str, u2: str, tokenizer) -> tuple[str, dict[str, int]]:
    """Plain "User:/Assistant:" render — no chat template."""
    prefix = f"{NATURALISTIC_USER_TAG}{u1}{NATURALISTIC_SEP}{NATURALISTIC_ASSISTANT_TAG}{a1}{NATURALISTIC_SEP}"
    text = f"{prefix}{NATURALISTIC_USER_TAG}{u2}"
    return text, {
        "prev_turn_end": len(prefix),
        "u2_end": len(text),
        "parent_answer_end": len(text),  # naturalistic has no separate answer slot; alias
    }


def render_story(u1: str, a1: str, u2: str, tokenizer) -> tuple[str, dict[str, int]]:
    """Story template ("Alex" variant, verbatim)."""
    text = STORY_USER_TEMPLATE.format(u1=u1, a1=a1, u2=u2)
    # prev_turn_end = end of a1's closing `".` in template
    # Find the marker `assistant replied: "` and use its position + len(a1) + len('".')
    m = 'assistant replied: "'
    start = text.find(m)
    if start == -1:
        raise RuntimeError("story template render missing marker")
    a1_start = start + len(m)
    a1_end = a1_start + len(a1)
    # Closing quote+dot: `".`
    prev_turn_end = a1_end + len('".')

    # u2_end = end of u2 content (before its closing quote+dot).
    m2 = 'typed a follow-up: "'
    u2_start = text.find(m2)
    if u2_start == -1:
        raise RuntimeError("story template missing u2 marker")
    u2_content_start = u2_start + len(m2)
    u2_end = u2_content_start + len(u2)

    return text, {
        "prev_turn_end": prev_turn_end,
        "u2_end": u2_end,
        "parent_answer_end": len(text),
    }


RENDERERS = {
    "chat": render_chat,
    "naturalistic": render_naturalistic,
    "story": render_story,
}


# --- Token index resolution -------------------------------------------------


def char_to_token_index(offset_mapping: list[tuple[int, int]], char_pos: int, policy: str) -> int:
    """Resolve a char offset to a token index under a straddler policy.

    ``exclude`` = if a token STRADDLES char_pos, drop it (return the last token
      strictly BEFORE char_pos); guarantees no later content leaks into X.
    ``include`` = if a token straddles char_pos, keep it (return that token);
      guarantees the content up to char_pos is retained.
    """
    if char_pos == 0:
        return 0
    # Find the token whose (start, end) contains char_pos or ends at it.
    for i, (s, e) in enumerate(offset_mapping):
        if e == 0 and s == 0:
            # special token (BOS/pad); skip
            continue
        if e >= char_pos >= s:
            if e == char_pos:
                # Boundary lands exactly at token end — no straddle; return token i.
                return i
            # Straddle: token i covers char_pos strictly inside.
            if policy == "exclude":
                # Return the previous non-special token.
                for j in range(i - 1, -1, -1):
                    ss, ee = offset_mapping[j]
                    if not (ss == 0 and ee == 0):
                        return j
                return 0
            else:
                return i
        if s > char_pos:
            # No token straddles; return previous.
            for j in range(i - 1, -1, -1):
                ss, ee = offset_mapping[j]
                if not (ss == 0 and ee == 0):
                    return j
            return 0
    # Fell off end; return last non-special token.
    for j in range(len(offset_mapping) - 1, -1, -1):
        ss, ee = offset_mapping[j]
        if not (ss == 0 and ee == 0):
            return j
    return len(offset_mapping) - 1


# --- Capture ----------------------------------------------------------------


def _logits_to_keep_kwargs(model) -> dict:
    """Return {"logits_to_keep": 1} when the forward signature explicitly names
    it — else {}. Guards against unread-logits full-vocab allocation (#779).
    A wrapper's **kwargs does NOT count (silent swallow).
    """
    try:
        sig = inspect.signature(model.forward)
    except Exception:
        try:
            sig = inspect.signature(model.__call__)
        except Exception:
            return {}
    params = sig.parameters
    if "logits_to_keep" in params and params["logits_to_keep"].kind not in (
        inspect.Parameter.VAR_KEYWORD,
    ):
        return {"logits_to_keep": 1}
    return {}


def capture_cell(
    *,
    model,
    tokenizer,
    rows: list[dict],
    framing: str,
    provenance: str,
    device: str,
    batch_size: int,
    out_path: Path,
) -> dict:
    """Capture L19 activations at every slot for one cell.

    Returns a manifest dict; writes the store to ``out_path``.
    """
    import numpy as np
    import torch

    renderer = RENDERERS[framing]
    u2_key = "u2_real" if provenance == "realu2" else "u2_haiku"

    # 1. Render every row + resolve slot char offsets.
    #
    # Round-2 Minor #1: split empty-u2 drops from token-budget drops so a
    # spike in either is visible per cell (and halt loud when empty-u2 rows
    # exceed 5% of the corpus — the sibling of the haiku-gen fail-fast
    # floor). ``isinstance(u2, str)`` handles ``None`` (the haiku_gen failure
    # sentinel) — those rows are already accounted for in the haiku-gen
    # drop-report and should never dominate here.
    n_dropped_empty_u2 = 0
    n_dropped_over_budget = 0
    rendered_texts: list[str] = []
    slot_char_offsets: list[dict[str, int]] = []
    kept_rows: list[dict] = []
    for row in rows:
        u2 = row.get(u2_key, "")
        if not isinstance(u2, str) or not u2.strip():
            n_dropped_empty_u2 += 1
            continue
        text, offsets = renderer(row["u1"], row["a1"], u2, tokenizer)
        # Token-budget filter — a rendered row over MAX_TOKENS is dropped.
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tok > MAX_TOKENS:
            n_dropped_over_budget += 1
            continue
        rendered_texts.append(text)
        slot_char_offsets.append(offsets)
        kept_rows.append(row)

    n = len(rendered_texts)
    if n == 0:
        raise RuntimeError(f"cell {framing}/{provenance}: zero rows after rendering/filtering")

    # Fail-fast on a per-cell empty-u2 rate above the 5% floor — a spike here
    # means the haiku-gen fail-fast floor was misconfigured or the corpus's
    # u2_real column is broken (realu2 arm) and a silent shrink is unsafe.
    n_input = len(rows)
    empty_u2_rate = n_dropped_empty_u2 / max(1, n_input)
    if empty_u2_rate > EMPTY_U2_HALT_FLOOR:
        raise RuntimeError(
            f"cell {framing}/{provenance}: empty-u2 rate {empty_u2_rate:.4f} exceeds "
            f"floor {EMPTY_U2_HALT_FLOOR} "
            f"(n_dropped_empty_u2={n_dropped_empty_u2} of n={n_input}). "
            "A silent-shrink guardrail: check the haiku-gen drop-report for "
            "the corresponding failure_ids on the haikuu2 arm, or check "
            "u2_real coverage in the corpus manifest on the realu2 arm."
        )

    print(
        f"[capture] cell={framing}/{provenance} rows={n} "
        f"(from {n_input} corpus rows; "
        f"n_dropped_empty_u2={n_dropped_empty_u2} "
        f"n_dropped_over_budget={n_dropped_over_budget})",
        flush=True,
    )

    d_model = model.config.hidden_size
    slots_out = {s: np.zeros((n, d_model), dtype=np.float32) for s in SLOT_NAMES}
    slot_token_pos = {s: np.zeros(n, dtype=np.int32) for s in SLOT_NAMES}
    n_tokens = np.zeros(n, dtype=np.int32)

    keep_kwargs = _logits_to_keep_kwargs(model)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_texts = rendered_texts[start:end]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            return_offsets_mapping=True,
            add_special_tokens=False,  # apply_chat_template already includes template specials
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        offsets_batch = enc["offset_mapping"].tolist()

        # Resolve slot token index per row.
        batch_positions: list[dict[str, int]] = []
        for i, off in enumerate(offsets_batch):
            row_positions: dict[str, int] = {}
            row_char_offsets = slot_char_offsets[start + i]
            valid_len = int(attn[i].sum().item())
            trimmed_off = off[:valid_len]
            for s in SLOT_NAMES:
                token_idx = char_to_token_index(trimmed_off, row_char_offsets[s], SLOT_STRADDLER[s])
                row_positions[s] = int(token_idx)
            batch_positions.append(row_positions)
            n_tokens[start + i] = valid_len

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                **keep_kwargs,
            )
        hs = out.hidden_states[CAPTURE_LAYER]  # (B, T, D)
        for i in range(end - start):
            for s in SLOT_NAMES:
                idx = batch_positions[i][s]
                slots_out[s][start + i] = hs[i, idx].float().cpu().numpy()
                slot_token_pos[s][start + i] = idx

        # Free hidden states before next batch.
        del out, hs

    store = {
        "slots": slots_out,
        "slot_token_pos": slot_token_pos,
        "n_tokens": n_tokens,
        "conv_ids": np.array([r["conv_id"] for r in kept_rows], dtype=object),
        "corpus": np.array([r["corpus"] for r in kept_rows], dtype=object),
        "unit": {
            "framing": framing,
            "provenance": provenance,
            "model": model.config.name_or_path,
            "unit_id": f"user_{provenance}_{framing}",
        },
        "slot_names": list(SLOT_NAMES),
        "straddler_policy": dict(SLOT_STRADDLER),
        "primary_fit": PRIMARY_FIT_BY_FRAMING[framing],
        "layer": CAPTURE_LAYER,
        "d_model": int(d_model),
        "capture_layer": CAPTURE_LAYER,
        "n_rows": n,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")

    import torch as _torch

    _torch.save(store, tmp)
    os.replace(tmp, out_path)

    return {
        "framing": framing,
        "provenance": provenance,
        "n_rows": n,
        "n_dropped_at_render": len(rows) - n,
        "d_model": int(d_model),
        "out_path": str(out_path),
    }


# --- Top-level driver -------------------------------------------------------


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def capture_all_cells(
    *,
    corpus_path: Path,
    haiku_path: Path,
    out_root: Path,
    models: list[str],
    framings: list[str],
    provenances: list[str],
    batch_size: int,
    device: str | None,
    smoke: bool,
    smoke_rows: int,
) -> dict:
    """Load corpus + haiku sidecar; capture every (model, framing, provenance) cell."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load corpus + haiku fill; merge by conv_id.
    corpus_rows: dict[str, dict] = {}
    with corpus_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            corpus_rows[str(r["conv_id"])] = r

    haiku_rows: dict[str, dict] = {}
    with haiku_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            haiku_rows[str(r["conv_id"])] = r

    merged: list[dict] = []
    for cid, cr in corpus_rows.items():
        hr = haiku_rows.get(cid, {})
        row = dict(cr)
        row["u2_haiku"] = hr.get("u2_haiku", "")
        merged.append(row)

    if smoke:
        merged = merged[:smoke_rows]

    print(f"[capture] merged {len(merged)} rows (corpus x haiku)", flush=True)

    if device is None:
        device = _resolve_device()

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cells: list[dict] = []
    for model_name in models:
        model_slug = model_name.replace("/", "_")
        print(f"[capture] loading model {model_name} on {device} dtype={dtype}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            model_kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device == "cpu":
            model = model.to(device)
        model.eval()

        try:
            for framing in framings:
                for prov in provenances:
                    unit_id = f"user_{prov}_{framing}"
                    out_path = out_root / model_slug / unit_id / f"L{CAPTURE_LAYER}.pt"
                    if out_path.exists():
                        print(f"[capture] SKIP {model_slug}/{unit_id}: already exists", flush=True)
                        continue
                    manifest_entry = capture_cell(
                        model=model,
                        tokenizer=tok,
                        rows=merged,
                        framing=framing,
                        provenance=prov,
                        device=device,
                        batch_size=batch_size,
                        out_path=out_path,
                    )
                    manifest_entry["model"] = model_name
                    manifest_entry["model_slug"] = model_slug
                    cells.append(manifest_entry)
        finally:
            del model
            del tok
            import gc

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "cells": cells,
        "layer": CAPTURE_LAYER,
        "framings": framings,
        "provenances": provenances,
        "models": models,
        "smoke": smoke,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "issue_1689"
        / "real_u2_capture"
        / "corpus"
        / "real_multiturn_first_exchange.jsonl",
    )
    ap.add_argument(
        "--haiku",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "issue_1689"
        / "real_u2_capture"
        / "raw_completions"
        / "haiku_u2.jsonl",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / "real_u2_capture" / "store",
    )
    ap.add_argument(
        "--models",
        default=",".join(MODELS),
        help="comma-separated model ids",
    )
    ap.add_argument(
        "--framings",
        default=",".join(FRAMINGS),
        help="comma-separated framing names",
    )
    ap.add_argument(
        "--provenances",
        default=",".join(PROVENANCES),
        help="comma-separated provenance names",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-rows", type=int, default=20)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit",
    )
    args = ap.parse_args()

    if args.import_check:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        print("[capture] import-check OK", flush=True)
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    provenances = [p.strip() for p in args.provenances.split(",") if p.strip()]

    print(
        f"[phase=capture] models={models} framings={framings} provenances={provenances} "
        f"smoke={args.smoke}",
        flush=True,
    )

    manifest = capture_all_cells(
        corpus_path=args.corpus,
        haiku_path=args.haiku,
        out_root=args.out_root,
        models=models,
        framings=framings,
        provenances=provenances,
        batch_size=args.batch_size,
        device=args.device,
        smoke=args.smoke,
        smoke_rows=args.smoke_rows,
    )

    manifest_path = args.out_root / "capture_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[phase=capture] done: {len(manifest['cells'])} cells captured", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
