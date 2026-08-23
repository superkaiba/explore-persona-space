"""issue #2378 capture rig — teacher-forced activation capture + P1 layer sweep (plan §4.3).

One teacher-forced forward per row over the FINAL measured text (prime stripped;
opener + answer included; user cells: the rendered template through the user_2
span). Three slots per (row × layer), #2054 offset-mapping conventions:

- ``v_C``  last-token hidden state BEFORE the answer's first character
           (token index ``answer_lo - 1``; rows with ``answer_lo == 0`` are
           DROPPED with reason ``no_context_token`` — stricter than #2054's
           ``max(0, lo-1)`` clamp, documented deviation).
- ``v_A``  token-mean over the answer (resp. user-turn) span.
- ``v_P``  last-token hidden state before the mined utterance / question /
           assistant_1-reply begins (rows where no token ends before that char
           are DROPPED with reason ``prefix_pos_unresolved`` — #2054 records
           None; we drop, documented deviation).

Phases (``--phase`` over ``PHASES``):
- ``pilot``         P1: N pilot chat rows, ALL 65 hidden states captured to
                    ``--pilot-out-root`` (plan §10 DECLARED DISCARD; regen via
                    this phase / ``--all-layers``), then the batched layer
                    sweep -> ``layer_sweep.json`` (G1(c) gate input).
- ``sweep``         layer sweep only, from an existing pilot store.
- ``capture``       production capture at L* + flanks {L*-8,L*-4,L*+4,L*+8}
                    clamped to [1,63], over capture_ready kept ids per cell.
- ``capture_fresh`` fresh-draw captures (seeds 138-141): ``fresh_draws`` cells
                    + ``user_sim_fresh``.
- ``probe_span`` / ``probe_npz`` / ``probe_sweep``  CPU self-verification
                    probes (no model, no GPU, no network).

Storage: bf16-as-uint16 bit arrays (numpy has no bfloat16; fp16 would overflow
Qwen massive activations) in PLAIN ``np.savez`` npz per (cell x chunk x layer)
(#813: never savez_compressed), atomic via ``<stem>.tmp.npz`` + ``os.replace``
(#1092: np.savez appends ``.npz`` to non-.npz names). Decode via
``decode_bf16``. Resume: ``cm.StageLedger`` keyed on generating parameters
only. All drops are counted with named reasons and persisted per chunk —
never silent (zero-failure contract, plan §4.3).

Model: ``Qwen3_5ForConditionalGeneration`` (fallback ``AutoModelForImageTextToText``)
bf16, text-only, explicit ``.to(device)`` (never ``device_map='auto'`` — #825
silent CPU offload). The physical GPU is CVD-pinned by the LAUNCHER; ``--device``
stays ``cuda``. Asserts at load: 64 text layers, hidden 5120, >= --min-free-hbm-gb
free; at first forward: 65 hidden states (plan §12 assumption 4).

Smoke = the same entrypoint at small counts (``--rows 16 --skip-capture-ready``;
PASS_UNIFIED — no smoke-only branches).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Script-mode sys.path bootstrap (#823; r5 model-venv fix): under the dedicated
# model venv (/root/eps-model-venv — no editable install of this repo) neither
# `explore_persona_space` nor the scripts/ siblings are importable unless the
# repo's src/ + scripts/ dirs are on sys.path. Mirrors issue2378_dispatch.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Thread caps freeze at first BLAS import — load_dotenv() BEFORE numpy (#847;
# pinned by tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2378_common as cm  # noqa: E402
import issue2378_gen as gen  # noqa: E402

N_LAYERS = 64  # config.text_config.num_hidden_layers (plan §12 assumption 4)
HIDDEN_SIZE = 5120
FLANK_OFFSETS = (-8, -4, 4, 8)  # production flanks around L*, clamped to [1, 63]
REDUCED_BASIS_K = 1024  # mirrors scripts/issue2054_fits.py REDUCED_BASIS_K
LAMBDA_GRID_PARAMS = (-2.0, 4.0, 13)  # np.logspace generating params (machine-stable key)
DOF_CAP = 0.9  # #1887: GCV ridge dof cap
G1C_R2_FLOOR = 0.05  # plan §7 G1(c): max held-out reduced-basis R^2 >= 0.05
SLOTS = ("v_C", "v_A", "v_P")

_ANCHOR_CACHE: dict[str, int] = {}
_POOL_CACHE: dict[str, dict[str, dict]] = {}


# ---------------------------------------------------------------------------
# bf16 <-> uint16 storage codec
# ---------------------------------------------------------------------------


def _encode_bf16(torch, t) -> np.ndarray:
    """bf16 tensor -> uint16 bit array (bit-exact; fp16 would overflow Qwen
    massive activations, fp32 doubles the store)."""
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()


def decode_bf16(a: np.ndarray, torch_mod):
    """Decode a bf16-as-uint16 array back to a torch bf16 tensor (bit-exact)."""
    assert a.dtype == np.uint16, a.dtype
    arr = np.ascontiguousarray(a).view(np.int16)
    return torch_mod.from_numpy(arr.copy()).view(torch_mod.bfloat16)


def _atomic_savez(path: Path, **arrays) -> None:
    """Plain np.savez (#813) via open handle + os.replace; tmp named
    ``<stem>.tmp.npz`` so np.savez cannot append a second ``.npz`` (#1092)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[: -len(".npz")] + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Span/position helpers (ported from scripts/issue2054_capture.py, adapted)
# ---------------------------------------------------------------------------


def _char_span_to_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int]:
    """#2054 port: token i in span iff [tok_lo, tok_hi) overlaps
    [char_start, char_end); zero-width offset rows skipped; (0, 0) = no overlap.
    Callers must reject empty char spans BEFORE calling (an empty span between
    token boundaries is not an overlap question)."""
    lo: int | None = None
    hi = 0
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:  # zero-width rows (some specials)
            continue
        if tok_hi <= char_start:
            continue
        if tok_lo >= char_end:
            break
        if lo is None:
            lo = i
        hi = i + 1
    if lo is None:
        return (0, 0)
    return (lo, hi)


def _token_before_char(offsets, char_pos: int) -> int | None:
    """#2054 port: index of the LAST token ending at or before ``char_pos``;
    None when no token ends before it (M3: never coerce to 0)."""
    idx = -1
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:
            continue
        if tok_hi <= char_pos:
            idx = i
    return None if idx < 0 else idx


def _divergence_anchor(render_fn, text: str) -> int | None:
    """Content-independent content-start anchor (#1776: never ``text.find``).

    Render with ``text`` and with a first-char-flipped sentinel; the common
    prefix length of the two RENDERED strings is the char index where the
    content begins. Returns None for empty text / degenerate templates.
    """
    if not text:
        return None
    flipped = ("X" if text[0] != "X" else "Y") + text[1:]
    a = render_fn(text)
    b = render_fn(flipped)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i >= len(a):
        return None
    return i


def _chat_anchor(tok) -> int:
    """Constant question-content start offset of the chat render (template
    preamble is content-independent, so one probe render pins it)."""
    if "chat" not in _ANCHOR_CACHE:
        pos = _divergence_anchor(lambda t: gen._render_chat(tok, t), "probe question")
        if pos is None:
            raise RuntimeError("chat template divergence anchor failed (fail loud)")
        _ANCHOR_CACHE["chat"] = pos
    return _ANCHOR_CACHE["chat"]


# ---------------------------------------------------------------------------
# Per-cell final-text assembly (returns payload | (None, drop_reason))
# ---------------------------------------------------------------------------

PLAIN_PREFIX = "User: "


def _assemble_chat(tok, template_sha: str, question: str, answer: str | None, row_sha):
    if row_sha is not None and row_sha != template_sha:
        return None, "template_sha_mismatch"
    if not answer:
        return None, "empty_answer"
    prompt = gen._render_chat(tok, question)
    pos = _chat_anchor(tok)
    if prompt[pos : pos + len(question)] != question:
        return None, "anchor_slice_mismatch"
    final = prompt + answer  # stored answer is stripped; direct join (deviation, documented)
    return {
        "final_text": final,
        "answer_lo_char": len(prompt),
        "answer_hi_char": len(final),
        "prefix_char": pos,
    }, None


def _assemble_plain(question: str, answer: str | None):
    if not answer:
        return None, "empty_answer"
    prompt = f"{PLAIN_PREFIX}{question}\n\nAssistant:"
    final = prompt + " " + answer  # single-space join; stored answer is stripped (documented)
    pos = len(PLAIN_PREFIX)
    if final[pos : pos + len(question)] != question:
        return None, "anchor_slice_mismatch"
    return {
        "final_text": final,
        "answer_lo_char": len(prompt) + 1,
        "answer_hi_char": len(final),
        "prefix_char": pos,
    }, None


def _assemble_story(mined_row: dict, opener_text: str, answer: str | None):
    if answer is None:
        return None, "no_close_quote"
    if not answer:
        return None, "empty_answer"
    pre = mined_row["scene_pre_answer"]
    final = pre + "\n\n" + opener_text + answer  # opener ends with an opening quote
    utter_cs, utter_ce = mined_row["utter_span"]  # gen_text coords; pre = scene_seed + gen_text[:q]
    pos = len(mined_row["scene_seed"]) + utter_cs
    if final[pos : pos + (utter_ce - utter_cs)] != mined_row["utterance"]:
        return None, "anchor_slice_mismatch"
    return {
        "final_text": final,
        "answer_lo_char": len(pre) + 2 + len(opener_text),
        "answer_hi_char": len(final),
        "prefix_char": pos,
    }, None


def _a1_anchor(tok, u1: str, a1: str) -> int | None:
    """Char index where assistant_1's reply content begins in the rendered prefix."""
    return _divergence_anchor(lambda t: gen._render_user_prefix(tok, u1, t), a1)


def _assemble_user_real(tok, row: dict, pool_row: dict):
    prefix = gen._render_user_prefix(tok, pool_row["u1"], pool_row["a1"])
    if len(prefix) != row["header_end"] or not row["rendered_text"].startswith(prefix):
        return None, "prefix_render_mismatch"
    lo, hi = row["u2_span"]
    final = row["rendered_text"][:hi]  # truncate at u2 end: teacher-forced through user_2 only
    a1 = pool_row["a1"]
    pos = _a1_anchor(tok, pool_row["u1"], a1)
    if pos is None or final[pos : pos + len(a1)] != a1:
        return None, "anchor_slice_mismatch"
    return {
        "final_text": final,
        "answer_lo_char": lo,
        "answer_hi_char": hi,
        "prefix_char": pos,
    }, None


def _assemble_user_sim(tok, row: dict, pool_row: dict):
    """DEVIATION (declared, r1 review g2 concern 5): the producer-STRIPPED
    ``sim_turn`` is joined DIRECTLY onto the rendered prefix — same class as
    the chat/plain direct-join deviations in the module docstring."""
    prefix = gen._render_user_prefix(tok, pool_row["u1"], pool_row["a1"])
    if len(prefix) != row["prefix_chars"] or cm.text_digest(prefix) != row["prefix_digest"]:
        return None, "prefix_digest_mismatch"
    sim = row.get("sim_turn")
    if not sim:
        return None, "empty_answer"
    final = prefix + sim
    a1 = pool_row["a1"]
    pos = _a1_anchor(tok, pool_row["u1"], a1)
    if pos is None or final[pos : pos + len(a1)] != a1:
        return None, "anchor_slice_mismatch"
    return {
        "final_text": final,
        "answer_lo_char": len(prefix),
        "answer_hi_char": len(final),
        "prefix_char": pos,
    }, None


# ---------------------------------------------------------------------------
# Row-source resolution (unit 1 artifacts; local-first -> --stage-raw-from-hf)
# ---------------------------------------------------------------------------

STORY_PROV_KEYS = (
    "character",
    "family",
    "wave",
    "setting_id",
    "situation_id",
    "register_id",
    "final_seed_id",
    "char_intro_id",
    "opener_id",
    "prime_exemplar_ids",
)


def _user_pool(args) -> dict[str, dict]:
    if "user" not in _POOL_CACHE:
        rows = gen._load_pool(gen._resolve_pools_dir(args), "user_draw")
        _POOL_CACHE["user"] = {r["conv_id"]: r for r in rows}
    return _POOL_CACHE["user"]


def _capture_ready_ids(args, cell: str) -> set[str]:
    """Kept ids from the gen capture_ready gate; user cells use the pair
    intersection (plan §4.2b). Fail-loud when absent (--skip-capture-ready
    is the smoke escape)."""
    path = Path(args.ledger_root) / "capture_ready" / f"{cell}.json"
    if not path.exists():
        raise RuntimeError(
            f"missing capture_ready gate {path} — run issue2378_gen --phase capture_ready "
            "first (or pass --skip-capture-ready for the smoke)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if cell in cm.USER_CELLS:
        inter = payload.get("pair_intersection") or {}
        ids = inter.get("intersection_ids")
        if not ids:
            raise RuntimeError(f"empty/missing pair_intersection.intersection_ids in {path}")
        return set(ids)
    ids = payload.get("kept_ids")
    if not ids:
        raise RuntimeError(f"empty kept_ids in {path} (fail loud)")
    return set(ids)


def _fill_to_cap(ids, cap: int):
    """Deterministic sorted-id iteration; cap 0 = no cap."""
    for k, rid in enumerate(sorted(ids)):
        if cap and k >= cap:
            return
        yield rid


def _story_aux(args):
    mined = gen._load_mined_rows(gen._rows_dir(args, "sega_mined", args.mined_dir))
    return mined


def _collect_cell_rows(args, tok, template_sha: str, cell: str, rows_cap: int):
    """Production capture rows for one cell: kept rows (∩ capture_ready unless
    --skip-capture-ready), assembled to (final_text, spans, prefix_char)."""
    drops: Counter = Counter()
    out: list[dict] = []
    kept = None if args.skip_capture_ready else _capture_ready_ids(args, cell)

    def _admit(rid) -> bool:
        if kept is not None and rid not in kept:
            drops["not_capture_ready"] += 1
            return False
        return True

    if cell in ("chat", "plain_text"):
        stage = "chat" if cell == "chat" else "plain"
        base = gen._stage_kept_rows(gen._rows_dir(args, stage), cell)
        pool_ids = [r for r in base if _admit(r)]
        for rid in _fill_to_cap(pool_ids, rows_cap):
            r = base[rid]
            if cell == "chat":
                payload, reason = _assemble_chat(
                    tok, template_sha, r["question"], r["answer"], r.get("template_sha")
                )
            else:
                payload, reason = _assemble_plain(r["question"], r["answer"])
            if reason:
                drops[reason] += 1
                continue
            out.append({"row_id": rid, "prov": {"conv_id": rid, "seed": r.get("seed")}, **payload})
    elif cell in cm.STORY_CELLS:
        base = gen._stage_kept_rows(gen._rows_dir(args, "segb"), cell)
        mined = _story_aux(args)
        pool_ids = [r for r in base if _admit(r)]
        for rid in _fill_to_cap(pool_ids, rows_cap):
            r = base[rid]
            m = mined.get(rid)
            if m is None:
                drops["mined_row_missing"] += 1
                continue
            if m.get("opener_id") != r.get("opener_id"):
                drops["opener_id_mismatch"] += 1
                continue
            payload, reason = _assemble_story(m, r["opener_text"], r["answer"])
            if reason:
                drops[reason] += 1
                continue
            prov = {k: m.get(k) for k in STORY_PROV_KEYS}
            out.append({"row_id": rid, "prov": prov, **payload})
    elif cell == "chat_user_real":
        base = gen._stage_kept_rows(gen._rows_dir(args, "user_real_render"), cell)
        pool = _user_pool(args)
        pool_ids = [r for r in base if _admit(r)]
        for rid in _fill_to_cap(pool_ids, rows_cap):
            pr = pool.get(rid)
            if pr is None:
                drops["pool_row_missing"] += 1
                continue
            payload, reason = _assemble_user_real(tok, base[rid], pr)
            if reason:
                drops[reason] += 1
                continue
            out.append({"row_id": rid, "prov": {"conv_id": rid}, **payload})
    elif cell == "chat_user_sim":
        base = gen._stage_kept_rows(gen._rows_dir(args, "user_sim"), cell)
        pool = _user_pool(args)
        pool_ids = [r for r in base if _admit(r)]
        for rid in _fill_to_cap(pool_ids, rows_cap):
            pr = pool.get(rid)
            if pr is None:
                drops["pool_row_missing"] += 1
                continue
            payload, reason = _assemble_user_sim(tok, base[rid], pr)
            if reason:
                drops[reason] += 1
                continue
            out.append({"row_id": rid, "prov": {"conv_id": rid}, **payload})
    else:
        raise SystemExit(f"unknown cell {cell}")
    if not out:
        raise RuntimeError(
            f"empty capture row set for cell={cell} (fail loud); drops={dict(drops)}"
        )
    return out, drops


def _collect_fresh_rows(args, tok, template_sha: str, cell: str, draw_seed: int, rows_cap: int):
    """Fresh-draw capture rows (seeds 138-141). Fresh rows carry no ``keep``
    (answer cells) — keep is computed IN-capture; user_sim_fresh rows carry
    keep from gen and are honored."""
    drops: Counter = Counter()
    out: list[dict] = []
    kept = None if args.skip_capture_ready else _capture_ready_ids(args, cell)

    def _admit(rid) -> bool:
        if kept is not None and rid not in kept:
            drops["not_capture_ready"] += 1
            return False
        return True

    if cell == "chat_user_sim":
        rows_dir = gen._rows_dir(args, "user_sim_fresh")
        pool = _user_pool(args)
        cand: dict[str, dict] = {}
        for path in sorted(rows_dir.glob("*.jsonl")):
            for row in cm.iter_jsonl(path):
                if row.get("cell") != cell or row.get("draw_seed") != draw_seed:
                    continue
                if not row.get("keep"):
                    drops[row.get("drop_reason") or "gen_dropped"] += 1
                    continue
                cand[row["conv_id"]] = row
        if not cand:
            raise RuntimeError(
                f"no kept user_sim_fresh rows for d{draw_seed} under {rows_dir} (fail loud)"
            )
        for rid in _fill_to_cap([r for r in cand if _admit(r)], rows_cap):
            pr = pool.get(rid)
            if pr is None:
                drops["pool_row_missing"] += 1
                continue
            payload, reason = _assemble_user_sim(tok, cand[rid], pr)
            if reason:
                drops[reason] += 1
                continue
            out.append({"row_id": rid, "prov": {"conv_id": rid, "draw_seed": draw_seed}, **payload})
    else:
        rows_dir = gen._rows_dir(args, "fresh_draws")
        files = sorted(rows_dir.glob(f"{cell}_d{draw_seed}_s*.jsonl"))
        if not files:
            raise RuntimeError(f"no fresh_draws files for {cell} d{draw_seed} under {rows_dir}")
        fresh: dict[str, dict] = {}
        for path in files:
            for row in cm.iter_jsonl(path):
                if row.get("cell") == cell and row.get("draw_seed") == draw_seed:
                    fresh[row["row_id"]] = row
        if cell in ("chat", "plain_text"):
            stage = "chat" if cell == "chat" else "plain"
            base = gen._stage_kept_rows(gen._rows_dir(args, stage), cell)
            mined = None
        else:
            base = gen._stage_kept_rows(gen._rows_dir(args, "segb"), cell)
            mined = _story_aux(args)
        for rid in _fill_to_cap([r for r in fresh if _admit(r)], rows_cap):
            fr = fresh[rid]
            br = base.get(rid)
            if br is None:
                drops["base_row_missing"] += 1
                continue
            if cell == "chat":
                payload, reason = _assemble_chat(
                    tok, template_sha, br["question"], fr.get("answer"), fr.get("template_sha")
                )
            elif cell == "plain_text":
                payload, reason = _assemble_plain(br["question"], fr.get("answer"))
            else:
                m = mined.get(rid) if mined else None
                if m is None:
                    drops["mined_row_missing"] += 1
                    continue
                payload, reason = _assemble_story(m, br["opener_text"], fr.get("answer"))
            if reason:
                drops[reason] += 1
                continue
            prov = {"draw_seed": draw_seed}
            if cell in cm.STORY_CELLS and mined is not None:
                prov.update({k: mined[rid].get(k) for k in STORY_PROV_KEYS})
            else:
                prov["conv_id"] = rid
            out.append({"row_id": rid, "prov": prov, **payload})
    if not out:
        raise RuntimeError(
            f"empty fresh capture row set cell={cell} d{draw_seed} (fail loud); drops={dict(drops)}"
        )
    return out, drops


# ---------------------------------------------------------------------------
# Tokenization + positions
# ---------------------------------------------------------------------------


def _tokenize_and_positions(tok, chunk_rows: list[dict], max_tokens: int):
    """ONE tokenization of each final_text with offsets (#2054 convention: the
    forward consumes these exact ids, so positions are internally consistent).
    Rows whose positions do not resolve are dropped with named reasons."""
    texts = [r["final_text"] for r in chunk_rows]
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    kept: list[dict] = []
    drops: Counter = Counter()
    for r, ids, offsets in zip(chunk_rows, enc["input_ids"], enc["offset_mapping"]):
        if not ids:
            drops["empty_tokenization"] += 1
            continue
        if len(ids) > max_tokens:
            drops["over_length"] += 1
            continue
        if r["answer_hi_char"] <= r["answer_lo_char"]:
            drops["answer_span_empty"] += 1  # #825 zero-width span class
            continue
        lo, hi = _char_span_to_token_span(offsets, r["answer_lo_char"], r["answer_hi_char"])
        if hi <= lo:
            drops["answer_span_empty"] += 1
            continue
        if lo == 0:
            drops["no_context_token"] += 1  # v_C undefined (deviation vs #2054 clamp)
            continue
        v_p = _token_before_char(offsets, r["prefix_char"])
        if v_p is None:
            drops["prefix_pos_unresolved"] += 1
            continue
        kept.append(
            {
                **r,
                "input_ids": list(ids),
                "n_tokens": len(ids),
                "v_C_pos": lo - 1,
                "v_P_pos": v_p,
                "ans_lo": lo,
                "ans_hi": hi,
            }
        )
    return kept, drops


# ---------------------------------------------------------------------------
# Model load + batched forwards
# ---------------------------------------------------------------------------


def _load_model_ctx(args) -> dict:
    """bf16 model on an explicit device with fail-loud config + HBM asserts."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()  # thread caps + HF env BEFORE torch import (#847)
    import torch

    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable (fail loud)")
        free, total = torch.cuda.mem_get_info()
        need = int(args.min_free_hbm_gb * 2**30)
        if free < need:
            raise RuntimeError(
                f"HBM preflight failed: free={free / 2**30:.1f} GiB < "
                f"required {args.min_free_hbm_gb:.1f} GiB (total {total / 2**30:.1f} GiB) — "
                "27B bf16 weights are ~55.6 GB; free the device or lower --min-free-hbm-gb"
            )
    try:
        from transformers import Qwen3_5ForConditionalGeneration as _ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as _ModelCls
    model = _ModelCls.from_pretrained(cm.MODEL_ID, dtype=torch.bfloat16)
    model.to(device)  # explicit placement — never device_map="auto" (#825)
    model.eval()
    tcfg = getattr(model.config, "text_config", model.config)
    if tcfg.num_hidden_layers != N_LAYERS:
        raise RuntimeError(f"num_hidden_layers={tcfg.num_hidden_layers}, expected {N_LAYERS}")
    if tcfg.hidden_size != HIDDEN_SIZE:
        raise RuntimeError(f"hidden_size={tcfg.hidden_size}, expected {HIDDEN_SIZE}")
    import inspect

    params = inspect.signature(model.forward).parameters
    # logits are unread — skip full-vocab logits when the EXPLICIT param exists
    # (bare **kwargs does NOT count; gotchas.md logits_to_keep entry).
    lk = {"logits_to_keep": 1} if "logits_to_keep" in params else {}
    tok = gen._get_tokenizer()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
    return {
        "torch": torch,
        "model": model,
        "device": device,
        "logits_kwargs": lk,
        "pad_id": int(pad_id),
        "checked": False,
    }


def _ensure_model(args, holder: dict) -> dict:
    if "ctx" not in holder:
        holder["ctx"] = _load_model_ctx(args)
    return holder["ctx"]


def _pack_batches(recs: list[dict], batch_tokens: int, max_batch_rows: int) -> list[list[int]]:
    """Longest-first length-bucket packing under a token budget (padding waste
    bounded; the longest bucket runs FIRST so OOM fails fast)."""
    order = sorted(range(len(recs)), key=lambda i: (-recs[i]["n_tokens"], recs[i]["row_id"]))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        if not cur:
            cur = [i]
            cur_max = recs[i]["n_tokens"]
            continue
        if (len(cur) + 1) * cur_max > batch_tokens or len(cur) >= max_batch_rows:
            batches.append(cur)
            cur = [i]
            cur_max = recs[i]["n_tokens"]
        else:
            cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def _forward_chunk(args, mctx: dict, recs: list[dict], layers: list[int]):
    """Teacher-forced forwards for one chunk; returns {layer: {slot: (n,d) uint16}}
    aligned with ``recs`` order. Manual RIGHT padding (positions index the
    unpadded prefix; causal mask + attention_mask keep pads inert)."""
    torch = mctx["torch"]
    model = mctx["model"]
    dev = mctx["device"]
    n = len(recs)
    out = {
        layer: {s: np.empty((n, HIDDEN_SIZE), dtype=np.uint16) for s in SLOTS} for layer in layers
    }
    for batch in _pack_batches(recs, args.batch_tokens, args.max_batch_rows):
        bsz = len(batch)
        t_max = max(recs[i]["n_tokens"] for i in batch)
        ids = torch.full((bsz, t_max), mctx["pad_id"], dtype=torch.long)
        mask = torch.zeros((bsz, t_max), dtype=torch.long)
        for j, ri in enumerate(batch):
            ln = recs[ri]["n_tokens"]
            ids[j, :ln] = torch.tensor(recs[ri]["input_ids"], dtype=torch.long)
            mask[j, :ln] = 1
        with torch.no_grad():
            res = model(
                input_ids=ids.to(dev),
                attention_mask=mask.to(dev),
                output_hidden_states=True,
                **mctx["logits_kwargs"],
            )
        hs = res.hidden_states
        if not mctx["checked"]:
            if len(hs) != N_LAYERS + 1:
                raise RuntimeError(
                    f"expected {N_LAYERS + 1} hidden states (embeddings + {N_LAYERS} layers), "
                    f"got {len(hs)} (plan §12 assumption 4)"
                )
            assert hs[0].shape[-1] == HIDDEN_SIZE, tuple(hs[0].shape)
            mctx["checked"] = True
        hdev = hs[0].device
        rows = torch.arange(bsz, device=hdev)
        pos_c = torch.tensor([recs[ri]["v_C_pos"] for ri in batch], device=hdev)
        pos_p = torch.tensor([recs[ri]["v_P_pos"] for ri in batch], device=hdev)
        lo = torch.tensor([recs[ri]["ans_lo"] for ri in batch], device=hdev)
        hi = torch.tensor([recs[ri]["ans_hi"] for ri in batch], device=hdev)
        t_idx = torch.arange(t_max, device=hdev)
        span = (t_idx[None, :] >= lo[:, None]) & (t_idx[None, :] < hi[:, None])
        denom = span.sum(1).clamp_min(1).float()
        bidx = np.asarray(batch)
        for layer in layers:
            h = hs[layer]
            v_c = h[rows, pos_c]
            v_p = h[rows, pos_p]
            v_a = ((h.float() * span[..., None]).sum(1) / denom[:, None]).to(torch.bfloat16)
            for slot, tens in (("v_C", v_c), ("v_A", v_a), ("v_P", v_p)):
                out[layer][slot][bidx] = _encode_bf16(torch, tens)
        del res, hs
    return out


# ---------------------------------------------------------------------------
# Chunked capture with StageLedger resume
# ---------------------------------------------------------------------------


def _row_record(r: dict) -> dict:
    rec = {k: r[k] for k in ("row_id", "n_tokens", "v_C_pos", "v_P_pos", "ans_lo", "ans_hi")}
    rec.update(r.get("prov") or {})
    return rec


def _capture_cell(
    args,
    holder: dict,
    tok,
    cell: str,
    rows: list[dict],
    layers: list[int],
    out_root: Path,
    tag: str,
    phase_name: str,
    assembly_drops: Counter,
    draw_seed: int | None = None,
) -> dict:
    """Capture one cell (or one fresh draw) chunk-by-chunk; per-chunk npz per
    layer + rows.json, ledger-resumable; final per-tag meta aggregates counts."""
    out_root.mkdir(parents=True, exist_ok=True)
    regime = {
        "phase": phase_name,
        "cell": cell,
        "tag": tag,
        "layers": [int(x) for x in layers],
        "model": cm.MODEL_ID,
        "rows_cap": int(args.rows),
        "n_rows": len(rows),
        # Input CONTENT fingerprint (r1 review g2 concern 1): a post-regen
        # recapture into the same out_root must NOT resume onto the pre-regen
        # store — string digests, machine-stable (never float hashes).
        "rows_fingerprint": cm.text_digest(
            "\n".join(cm.text_digest(r["final_text"]) for r in rows)
        ),
        "max_capture_tokens": int(args.max_capture_tokens),
        "chunk_rows": int(args.chunk_rows),
        "draw_seed": draw_seed,
    }
    ledger_path = out_root / "ledgers" / f"{tag}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = cm.StageLedger(ledger_path, regime)
    n_chunks = math.ceil(len(rows) / args.chunk_rows)
    t0 = time.time()
    for ci in range(n_chunks):
        key = f"c{ci:04d}"
        if ledger.is_done(key):
            continue
        chunk = rows[ci * args.chunk_rows : (ci + 1) * args.chunk_rows]
        recs, pos_drops = _tokenize_and_positions(tok, chunk, args.max_capture_tokens)
        if recs:
            mctx = _ensure_model(args, holder)
            arrays = _forward_chunk(args, mctx, recs, layers)
        else:
            arrays = {
                layer: {s: np.empty((0, HIDDEN_SIZE), dtype=np.uint16) for s in SLOTS}
                for layer in layers
            }
        row_ids = np.array([r["row_id"] for r in recs])
        for layer in layers:
            meta = json.dumps(
                {
                    "encoding": "bf16_as_uint16",
                    "cell": cell,
                    "layer": layer,
                    "draw_seed": draw_seed,
                    "hidden_size": HIDDEN_SIZE,
                }
            )
            _atomic_savez(
                out_root / f"{tag}__part{ci:04d}__L{layer}.npz",
                v_C=arrays[layer]["v_C"],
                v_A=arrays[layer]["v_A"],
                v_P=arrays[layer]["v_P"],
                row_ids=row_ids,
                meta=np.array(meta),
            )
        cm.atomic_write_json(
            out_root / f"{tag}__part{ci:04d}__rows.json",
            {
                "cell": cell,
                "tag": tag,
                "part": ci,
                "draw_seed": draw_seed,
                "rows": [_row_record(r) for r in recs],
                "position_drops": dict(pos_drops),
                "metadata": cm.run_metadata(),
            },
        )
        ledger.mark_done(key)
        cm.progress(f"{phase_name}:{tag}", ci + 1, n_chunks, key, t0)
    # Aggregate from the durable chunk records (resume-correct: skipped chunks
    # contribute via their persisted rows.json, never in-memory state).
    pos_drops_total: Counter = Counter()
    n_captured = 0
    for ci in range(n_chunks):
        payload = json.loads(
            (out_root / f"{tag}__part{ci:04d}__rows.json").read_text(encoding="utf-8")
        )
        pos_drops_total.update(payload["position_drops"])
        n_captured += len(payload["rows"])
    meta = {
        "regime": regime,
        "n_rows_in": len(rows),
        "n_captured": n_captured,
        "assembly_drops": dict(assembly_drops),
        "position_drops": dict(pos_drops_total),
        "n_chunks": n_chunks,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(out_root / f"{tag}__meta.json", meta)
    print(
        f"[{phase_name}] {tag}: captured {n_captured}/{len(rows)} rows x {len(layers)} layers "
        f"(assembly_drops={dict(assembly_drops)}, position_drops={dict(pos_drops_total)})",
        flush=True,
    )
    return meta


def _headroom(args, out_root: Path, n_rows: int, n_layers: int, phase: str) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    need_gb = n_rows * n_layers * len(SLOTS) * HIDDEN_SIZE * 2 / 1e9 * 1.5 + 1.0
    assert_out_root_headroom(out_root, need_gb, phase=phase)


def _resolve_layers(args) -> list[int]:
    """Explicit --layers csv beats --all-layers beats L* + flanks ∩ [1,63]."""
    if args.layers:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        for layer in layers:
            if not 0 <= layer <= N_LAYERS:
                raise SystemExit(f"--layers value {layer} outside [0, {N_LAYERS}]")
        return layers
    if args.all_layers:
        return list(range(N_LAYERS + 1))
    lstar = args.layer_star
    if lstar is None and args.layer_star_from:
        payload = json.loads(Path(args.layer_star_from).read_text(encoding="utf-8"))
        lstar = int(payload["selected_layer"])
    if lstar is None:
        raise SystemExit("capture needs --layers, --all-layers, --layer-star, or --layer-star-from")
    if not 1 <= lstar <= 63:
        raise SystemExit(f"--layer-star {lstar} outside [1, 63]")
    layers = {lstar} | {lstar + o for o in FLANK_OFFSETS if 1 <= lstar + o <= 63}
    return sorted(layers)


# ---------------------------------------------------------------------------
# Batched layer sweep (P1) — reduced-basis GCV ridge, batched across layers
# ---------------------------------------------------------------------------


def _eigh_robust(torch, a):
    """torch.linalg.eigh with the cuSOLVER non-convergence CPU fallback
    (gotchas.md; never jitter the Gram)."""
    try:
        return torch.linalg.eigh(a)
    except torch.linalg.LinAlgError:
        if a.is_cuda:
            print(
                f"[eigh-robust] cuda eigh non-convergence -> CPU fallback {tuple(a.shape)}",
                flush=True,
            )
            w, v = torch.linalg.eigh(a.cpu())
            return w.to(a.device), v.to(a.device)
        raise


def _batched_reduced_ridge(torch, x_tr, y_tr, x_te, y_te, k: int, lambdas, dof_cap: float):
    """Batched-over-layers reduced-basis (top-k PCA) GCV ridge, mirroring the
    scripts/issue2054_fits.py conventions: center X by the train mean, top-k
    principal basis (Gram-eigh route), standardize the reduced features
    (population std + 1e-9), center Y, GCV over the λ grid with the #1887 dof
    cap (degenerate fallback = largest λ), uniform-SS multivariate held-out R².

    Shapes: x/y (L, n, d) float64. Returns (r2, lam_idx, dof_at_best,
    degenerate) each (L,)."""
    n_layers, n_tr, d = x_tr.shape
    k_use = min(k, n_tr, d)
    mu_x = x_tr.mean(1, keepdim=True)
    xc = x_tr - mu_x
    gram = xc @ xc.transpose(1, 2)  # (L, n_tr, n_tr)
    w_g, u_g = _eigh_robust(torch, gram)  # ascending eigenvalues
    w_k = w_g[:, -k_use:].clamp_min(1e-12)
    u_k = u_g[:, :, -k_use:]
    basis = xc.transpose(1, 2) @ (u_k / w_k.sqrt().unsqueeze(1))  # (L, d, k)
    z_tr = xc @ basis
    z_te = (x_te - mu_x) @ basis
    mu_z = z_tr.mean(1, keepdim=True)
    sd_z = z_tr.std(1, unbiased=False, keepdim=True) + 1e-9
    zs_tr = (z_tr - mu_z) / sd_z
    zs_te = (z_te - mu_z) / sd_z
    ybar = y_tr.mean(1, keepdim=True)
    yc = y_tr - ybar
    ata = zs_tr.transpose(1, 2) @ zs_tr  # (L, k, k)
    w, q = _eigh_robust(torch, ata)
    c = q.transpose(1, 2) @ (zs_tr.transpose(1, 2) @ yc)  # (L, k, d)
    row_energy = (c**2).sum(-1)  # (L, k)
    y2 = (yc**2).sum((1, 2))  # (L,)
    lam = torch.as_tensor(np.asarray(lambdas), dtype=x_tr.dtype, device=x_tr.device)
    inv = 1.0 / (w.unsqueeze(-1) + lam)  # (L, k, m)
    rss = y2[:, None] - ((2.0 * inv - w.unsqueeze(-1) * inv**2) * row_energy.unsqueeze(-1)).sum(1)
    dof = (w.unsqueeze(-1) * inv).sum(1)  # (L, m)
    gcv = rss / (n_tr * (1.0 - dof / n_tr).clamp_min(1e-12) ** 2)
    allowed = dof <= dof_cap * n_tr
    degenerate = ~allowed.any(1)
    gcv_masked = torch.where(allowed, gcv, torch.full_like(gcv, float("inf")))
    best = gcv_masked.argmin(1)
    best = torch.where(degenerate, torch.full_like(best, lam.shape[0] - 1), best)
    inv_best = 1.0 / (w + lam[best].unsqueeze(-1))  # (L, k)
    coef = q @ (c * inv_best.unsqueeze(-1))  # (L, k, d)
    y_hat = zs_te @ coef + ybar
    resid = ((y_te - y_hat) ** 2).sum((1, 2))
    ss_tot = ((y_te - y_te.mean(1, keepdim=True)) ** 2).sum((1, 2))
    r2 = 1.0 - resid / ss_tot
    dof_best = dof.gather(1, best[:, None]).squeeze(1)
    return r2, best, dof_best, degenerate


def _serial_reduced_ridge_oracle(x_tr, y_tr, x_te, y_te, k: int, lambdas, dof_cap: float):
    """Self-contained SERIAL numpy oracle for one layer, mirroring the same
    conventions via the SVD route (probe_sweep equivalence reference; kept
    separate from issue2054_fits: its module top is import-heavy)."""
    mu = x_tr.mean(0)
    xc = x_tr - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k_use = min(k, *xc.shape)
    basis = vt[:k_use].T
    z_tr = xc @ basis
    z_te = (x_te - mu) @ basis
    mu_z = z_tr.mean(0)
    sd = z_tr.std(0) + 1e-9
    zs_tr = (z_tr - mu_z) / sd
    zs_te = (z_te - mu_z) / sd
    ybar = y_tr.mean(0)
    yc = y_tr - ybar
    w, q = np.linalg.eigh(zs_tr.T @ zs_tr)
    c = q.T @ (zs_tr.T @ yc)
    row_energy = (c**2).sum(-1)
    y2 = (yc**2).sum()
    n_tr = x_tr.shape[0]
    best_idx, best_gcv = None, np.inf
    for i, lam in enumerate(lambdas):
        inv = 1.0 / (w + lam)
        dof = float((w * inv).sum())
        if dof > dof_cap * n_tr:
            continue
        rss = y2 - float(((2.0 * inv - w * inv**2) * row_energy).sum())
        gcv = rss / (n_tr * (1.0 - dof / n_tr) ** 2)
        if gcv < best_gcv:
            best_idx, best_gcv = i, gcv
    degenerate = best_idx is None
    if degenerate:
        best_idx = len(lambdas) - 1
    coef = q @ (c * (1.0 / (w + lambdas[best_idx]))[:, None])
    y_hat = zs_te @ coef + ybar
    r2 = 1.0 - ((y_te - y_hat) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum()
    return float(r2), int(best_idx), bool(degenerate)


def _pilot_layer_arrays(root: Path, tag: str, layer: int):
    files = sorted(root.glob(f"{tag}__part*__L{layer}.npz"))
    if not files:
        raise RuntimeError(f"no pilot npz for layer {layer} under {root} — run --phase pilot")
    v_c, v_a, ids = [], [], []
    for f in files:
        with np.load(f) as z:
            meta = json.loads(z["meta"].item())
            if meta["encoding"] != "bf16_as_uint16":
                raise RuntimeError(f"unexpected encoding in {f}: {meta}")
            v_c.append(z["v_C"])
            v_a.append(z["v_A"])
            ids.extend(z["row_ids"].tolist())
    return np.concatenate(v_c), np.concatenate(v_a), ids


def _pilot_layers_present(root: Path, tag: str) -> list[int]:
    layers = set()
    for f in root.glob(f"{tag}__part0000__L*.npz"):
        layers.add(int(f.name.rsplit("__L", 1)[1][: -len(".npz")]))
    if not layers:
        raise RuntimeError(f"no pilot store under {root} for tag={tag} — run --phase pilot")
    return sorted(layers)


def _u16_to_f64(torch, a: np.ndarray, device: str):
    return decode_bf16(a, torch).to(device=device, dtype=torch.float64)


def _run_sweep(args) -> dict:
    """Batched all-layer reduced-basis ridge sweep over the pilot store;
    selects L* = argmax held-out R² restricted to [1, 63]; persists a
    self-describing layer_sweep.json (G1(c) gate input)."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()  # thread caps BEFORE torch import (#847)
    import torch

    root = Path(args.pilot_out_root)
    tag = "chat"
    layers = _pilot_layers_present(root, tag)
    lambdas = np.logspace(*LAMBDA_GRID_PARAMS)
    dev = args.sweep_device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    ref_v_c, ref_v_a, ref_ids = _pilot_layer_arrays(root, tag, layers[0])
    n = len(ref_ids)
    if n < 64:
        raise RuntimeError(f"pilot store has only {n} rows — too few for the 80/20 sweep")
    rng = np.random.default_rng(cm.derived_seed(cm.SEED, "layer_sweep"))
    perm = rng.permutation(n)
    n_tr = int(round(0.8 * n))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    per_layer: list[dict] = []
    t0 = time.time()
    chunk = max(1, int(args.sweep_layer_chunk))
    for c0 in range(0, len(layers), chunk):
        group = layers[c0 : c0 + chunk]
        xs, ys = [], []
        for layer in group:
            if layer == layers[0]:
                v_c, v_a, ids = ref_v_c, ref_v_a, ref_ids
            else:
                v_c, v_a, ids = _pilot_layer_arrays(root, tag, layer)
            if ids != ref_ids:
                raise RuntimeError(f"row_ids mismatch between layer {layer} and {layers[0]}")
            xs.append(_u16_to_f64(torch, v_c, dev))
            ys.append(_u16_to_f64(torch, v_a, dev))
        x = torch.stack(xs)
        y = torch.stack(ys)
        del xs, ys
        r2, lam_idx, dof, degen = _batched_reduced_ridge(
            torch,
            x[:, tr_idx],
            y[:, tr_idx],
            x[:, te_idx],
            y[:, te_idx],
            int(args.reduced_k),
            lambdas,
            DOF_CAP,
        )
        for j, layer in enumerate(group):
            per_layer.append(
                {
                    "layer": int(layer),
                    "r2": float(r2[j]),
                    "lambda": float(lambdas[int(lam_idx[j])]),
                    "dof": float(dof[j]),
                    "degenerate": bool(degen[j]),
                }
            )
        del x, y
        cm.progress("sweep", min(c0 + chunk, len(layers)), len(layers), f"L{group[-1]}", t0)

    r2_by_layer = {row["layer"]: row["r2"] for row in per_layer}
    domain = [layer for layer in layers if 1 <= layer <= 63]
    if not domain:
        raise RuntimeError(f"no layers in the selection domain [1,63] among {layers}")
    lstar = max(domain, key=lambda layer: r2_by_layer[layer])
    max_r2 = r2_by_layer[lstar]
    payload = {
        "cell": tag,
        "layers": [int(x) for x in layers],
        "selected_layer": int(lstar),
        "selection_domain": [1, 63],
        "per_layer": per_layer,
        "n_rows": n,
        "n_train": int(n_tr),
        "n_test": int(n - n_tr),
        "k": int(min(args.reduced_k, n_tr, HIDDEN_SIZE)),
        "lambda_grid": {"kind": "logspace", "params": list(LAMBDA_GRID_PARAMS)},
        "dof_cap": DOF_CAP,
        "split_seed": cm.derived_seed(cm.SEED, "layer_sweep"),
        "gate_g1c": {
            "threshold": G1C_R2_FLOOR,
            "max_r2": float(max_r2),
            "passes": bool(max_r2 >= G1C_R2_FLOOR),
            # Stated narrowing vs plan §7 wording (r1 review g2 concern 4):
            # the max is over the SELECTION domain [1,63]; embedding-adjacent
            # layers 0/64 are excluded from both selection and the gate.
            "domain_note": "max over selection domain [1,63]; layers 0/64 excluded",
        },
        "metadata": cm.run_metadata(),
    }
    out = Path(args.layer_sweep_out)
    cm.atomic_write_json(out, payload)
    print(
        f"[sweep] L*={lstar} max_r2={max_r2:.4f} "
        f"(G1c floor {G1C_R2_FLOOR}: {'PASS' if max_r2 >= G1C_R2_FLOOR else 'FAIL'}) -> {out}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_pilot(args) -> None:
    """P1: capture --pilot-rows chat rows at ALL 65 hidden states into the
    pilot store (plan §10 declared discard; never uploaded), then sweep."""
    tok = gen._get_tokenizer()
    template_sha = gen._assert_chat_template(tok)
    base = gen._stage_kept_rows(gen._rows_dir(args, "chat"), "chat")
    drops: Counter = Counter()
    rows: list[dict] = []
    for rid in sorted(base):
        if len(rows) >= args.pilot_rows:
            break
        r = base[rid]
        payload, reason = _assemble_chat(
            tok, template_sha, r["question"], r["answer"], r.get("template_sha")
        )
        if reason:
            drops[reason] += 1
            continue
        rows.append({"row_id": rid, "prov": {"conv_id": rid, "seed": r.get("seed")}, **payload})
    if not rows:
        raise RuntimeError(f"pilot: empty chat row set (fail loud); drops={dict(drops)}")
    layers = list(range(N_LAYERS + 1))
    out_root = Path(args.pilot_out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _headroom(args, out_root, len(rows), len(layers), phase="pilot")
    holder: dict = {}
    _capture_cell(
        args,
        holder,
        tok,
        "chat",
        rows,
        layers,
        out_root,
        tag="chat",
        phase_name="pilot",
        assembly_drops=drops,
    )
    if not args.skip_sweep:
        _run_sweep(args)
    print("[phase=pilot] done", flush=True)


def phase_sweep(args) -> None:
    """Sweep-only re-run over an existing pilot store."""
    _run_sweep(args)
    print("[phase=sweep] done", flush=True)


def phase_capture(args) -> None:
    """Production capture: all 9 active cells (v7; or --cells) at L* + flanks."""
    layers = _resolve_layers(args)
    tok = gen._get_tokenizer()
    template_sha = gen._assert_chat_template(tok)
    cells = [c for c in args.cells.split(",") if c] or list(cm.ALL_CELLS)
    for cell in cells:
        if cell not in cm.ALL_CELLS:
            raise SystemExit(f"unknown cell {cell}")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    holder: dict = {}
    summary: dict[str, dict] = {}
    for cell in cells:
        rows, asm_drops = _collect_cell_rows(args, tok, template_sha, cell, args.rows)
        _headroom(args, out_root, len(rows), len(layers), phase=f"capture:{cell}")
        meta = _capture_cell(
            args,
            holder,
            tok,
            cell,
            rows,
            layers,
            out_root,
            tag=cell,
            phase_name="capture",
            assembly_drops=asm_drops,
        )
        summary[cell] = {
            k: meta[k] for k in ("n_rows_in", "n_captured", "assembly_drops", "position_drops")
        }
    # Shard-keyed filename (r1 review g2 concern 3): parallel --cells shards
    # share one out_root; a fixed name would be last-writer-wins. Per-tag
    # __meta.json stays authoritative.
    cm.atomic_write_json(
        out_root / f"capture_summary__{cells[0]}.json",
        {"cells": summary, "layers": layers, "metadata": cm.run_metadata()},
    )
    if not args.skip_upload:
        uploaded = cm.upload_stage_dir(out_root, f"{cm.HF_PREFIX}/analysis_tensors/activations")
        print(f"[capture] uploaded+verified {len(uploaded)} files", flush=True)
    print("[phase=capture] done", flush=True)


FRESH_DEFAULT_CELLS = ("chat", "plain_text", *cm.STORY_CELLS, "chat_user_sim")


def phase_capture_fresh(args) -> None:
    """Fresh-draw captures (seeds 138-141): fresh_draws cells + user_sim_fresh.
    chat_user_real has no fresh draws (deterministic render)."""
    layers = _resolve_layers(args)
    tok = gen._get_tokenizer()
    template_sha = gen._assert_chat_template(tok)
    cells = [c for c in args.cells.split(",") if c] or list(FRESH_DEFAULT_CELLS)
    for cell in cells:
        if cell not in FRESH_DEFAULT_CELLS:
            raise SystemExit(f"cell {cell} has no fresh draws (chat_user_real is render-only)")
    seeds = [int(s) for s in cm.FRESH_SEEDS[: args.fresh_draws]]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    holder: dict = {}
    summary: dict[str, dict] = {}
    for cell in cells:
        for seed in seeds:
            rows, asm_drops = _collect_fresh_rows(args, tok, template_sha, cell, seed, args.rows)
            _headroom(args, out_root, len(rows), len(layers), phase=f"capture_fresh:{cell}")
            tag = f"{cell}__fresh_d{seed}"
            meta = _capture_cell(
                args,
                holder,
                tok,
                cell,
                rows,
                layers,
                out_root,
                tag=tag,
                phase_name="capture_fresh",
                assembly_drops=asm_drops,
                draw_seed=seed,
            )
            summary[tag] = {
                k: meta[k] for k in ("n_rows_in", "n_captured", "assembly_drops", "position_drops")
            }
    # Shard-keyed filename (r1 review g2 concern 3; per-tag __meta.json is
    # the authoritative record).
    cm.atomic_write_json(
        out_root / f"capture_fresh_summary__{cells[0]}.json",
        {"tags": summary, "layers": layers, "metadata": cm.run_metadata()},
    )
    if not args.skip_upload:
        uploaded = cm.upload_stage_dir(out_root, f"{cm.HF_PREFIX}/analysis_tensors/activations")
        print(f"[capture_fresh] uploaded+verified {len(uploaded)} files", flush=True)
    print("[phase=capture_fresh] done", flush=True)


# ---------------------------------------------------------------------------
# CPU self-verification probes (no model, no GPU, no network)
# ---------------------------------------------------------------------------


def phase_probe_span(args) -> None:
    """Span-mapping + anchor probes on synthetic tokenizer output."""
    offsets = [(0, 4), (4, 5), (5, 11), (11, 11), (11, 15)]  # incl. one zero-width row
    assert _char_span_to_token_span(offsets, 5, 11) == (2, 3)
    assert _char_span_to_token_span(offsets, 11, 15) == (4, 5)
    assert _char_span_to_token_span(offsets, 3, 6) == (0, 3)  # straddling both boundaries
    assert _char_span_to_token_span(offsets, 20, 25) == (0, 0)  # beyond text -> no overlap
    assert _token_before_char(offsets, 5) == 1
    assert _token_before_char(offsets, 4) == 0
    assert _token_before_char(offsets, 0) is None  # M3: never coerce to 0
    assert _token_before_char(offsets, 15) == 4

    def render(t: str) -> str:
        return f"<sys>hello</sys><user>{t}</user><asst>"

    pos = _divergence_anchor(render, "question text")
    assert pos == len("<sys>hello</sys><user>"), pos
    assert render("question text")[pos : pos + len("question text")] == "question text"
    assert _divergence_anchor(render, "") is None
    # plain-cell arithmetic mirrors _assemble_plain
    payload, reason = _assemble_plain("q1?", "an answer")
    assert reason is None
    ft = payload["final_text"]
    assert ft[payload["answer_lo_char"] : payload["answer_hi_char"]] == "an answer"
    assert ft[payload["prefix_char"] :].startswith("q1?")
    payload, reason = _assemble_plain("q1?", "")
    assert payload is None and reason == "empty_answer"
    print("[probe_span] PASS", flush=True)


def phase_probe_npz(args) -> None:
    """bf16-as-uint16 npz round-trip: bit-exact, atomic tmp naming (#1092)."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    import torch

    with tempfile.TemporaryDirectory(prefix="i2378_probe_npz_") as td:
        t = (torch.randn(4, 16, dtype=torch.float32) * 1e4).to(torch.bfloat16)
        a = _encode_bf16(torch, t)
        assert a.dtype == np.uint16 and a.shape == (4, 16)
        path = Path(td) / "probe__part0000__L1.npz"
        _atomic_savez(
            path,
            v_C=a,
            row_ids=np.array(["r0", "r1", "r2", "r3"]),
            meta=np.array(json.dumps({"encoding": "bf16_as_uint16"})),
        )
        assert path.exists()
        stray = list(Path(td).glob("*.tmp.npz")) + list(Path(td).glob("*.npz.npz"))
        assert not stray, stray  # np.savez .npz-append trap avoided
        with np.load(path) as z:
            back = z["v_C"]
            assert json.loads(z["meta"].item())["encoding"] == "bf16_as_uint16"
        t2 = decode_bf16(back, torch)
        assert torch.equal(t2, t), "bf16 round-trip not bit-exact"
    print("[probe_npz] PASS", flush=True)


def phase_probe_sweep(args) -> None:
    """Batched-rewrite equivalence at fp64 tiny shapes: batched layer sweep vs
    the self-contained serial numpy oracle (tol 1e-7; λ index exact)."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    import torch

    rng = np.random.default_rng(0)
    n_layers, n_tr, n_te, d, k = 3, 48, 16, 16, 8
    x = rng.standard_normal((n_layers, n_tr + n_te, d))
    b_true = rng.standard_normal((n_layers, d, d))
    y = x @ b_true + 0.1 * rng.standard_normal((n_layers, n_tr + n_te, d))
    lambdas = np.logspace(*LAMBDA_GRID_PARAMS)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    r2_b, lam_b, _, degen_b = _batched_reduced_ridge(
        torch, xt[:, :n_tr], yt[:, :n_tr], xt[:, n_tr:], yt[:, n_tr:], k, lambdas, DOF_CAP
    )
    for li in range(n_layers):
        r2_s, lam_s, degen_s = _serial_reduced_ridge_oracle(
            x[li, :n_tr], y[li, :n_tr], x[li, n_tr:], y[li, n_tr:], k, lambdas, DOF_CAP
        )
        diff = abs(float(r2_b[li]) - r2_s)
        assert diff < 1e-7, f"layer {li}: batched {float(r2_b[li])} vs serial {r2_s}"
        assert int(lam_b[li]) == lam_s, f"layer {li}: λ index {int(lam_b[li])} vs {lam_s}"
        assert bool(degen_b[li]) == degen_s
        print(f"[probe_sweep] layer {li}: r2={r2_s:.6f} |Δ|={diff:.2e} λ_idx={lam_s}", flush=True)
    print("[probe_sweep] PASS", flush=True)


def phase_probe_gating(args) -> None:
    """Capture-ready admission-gate probe (r1 codex blocker smoke-capture-
    ready-bypass): synthetic plain-text fixtures drive BOTH collectors through
    the REAL ``_collect_cell_rows``/``_collect_fresh_rows`` bodies, once WITH
    the capture_ready gate (exclusions counted ``not_capture_ready``) and once
    under ``--skip-capture-ready`` (the pre-admission P1 pilot escape). No
    model, no network; plain_text needs no tokenizer."""
    _ = args
    import argparse as _ap
    import tempfile

    with tempfile.TemporaryDirectory(prefix="i2378_gating_") as td:
        root = Path(td)
        raw_root = root / "raw"
        ledger_root = root / "ledger"
        rows = [
            {
                "cell": "plain_text",
                "conv_id": f"mt_{i:03d}",
                "question": f"What is {i} plus {i}?",
                "answer": f"The answer is {2 * i}.",
                "finish_reason": "stop",
                "seed": i,
                "regen": False,
                "keep": True,
                "drop_reason": None,
                "template_sha": None,
            }
            for i in range(3)
        ]
        gen._write_chunk_jsonl(raw_root / "plain" / "plain_text_w1_s0_c0000.jsonl", rows)
        fresh = [
            {
                "cell": "plain_text",
                "row_id": f"mt_{i:03d}",
                "draw_seed": 138,
                "gen_text": f"Still {2 * i}.",
                "finish_reason": "stop",
                "seed": i,
                "regen": False,
                "answer": f"Still {2 * i}.",
                "template_sha": None,
            }
            for i in range(3)
        ]
        gen._write_chunk_jsonl(raw_root / "fresh_draws" / "plain_text_d138_s0.jsonl", fresh)
        cm.atomic_write_json(
            ledger_root / "capture_ready" / "plain_text.json",
            {"cell": "plain_text", "kept_ids": ["mt_000", "mt_001"]},
        )

        def _ns(skip: bool) -> _ap.Namespace:
            return _ap.Namespace(
                raw_root=str(raw_root),
                ledger_root=str(ledger_root),
                skip_capture_ready=skip,
                stage_raw_from_hf=False,
                mined_dir=str(raw_root / "sega_mined"),
            )

        out, drops = _collect_cell_rows(_ns(False), None, "", "plain_text", 0)
        assert {r["row_id"] for r in out} == {"mt_000", "mt_001"}, sorted(r["row_id"] for r in out)
        assert drops["not_capture_ready"] == 1, dict(drops)
        out, drops = _collect_cell_rows(_ns(True), None, "", "plain_text", 0)
        assert len(out) == 3 and drops["not_capture_ready"] == 0, (len(out), dict(drops))
        out, drops = _collect_fresh_rows(_ns(False), None, "", "plain_text", 138, 0)
        assert {r["row_id"] for r in out} == {"mt_000", "mt_001"}, sorted(r["row_id"] for r in out)
        assert drops["not_capture_ready"] == 1, dict(drops)
        out, drops = _collect_fresh_rows(_ns(True), None, "", "plain_text", 138, 0)
        assert len(out) == 3 and drops["not_capture_ready"] == 0, (len(out), dict(drops))
    print("[probe_gating] PASS (gated + skip-capture-ready, both collectors)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = {
    "pilot": phase_pilot,
    "sweep": phase_sweep,
    "capture": phase_capture,
    "capture_fresh": phase_capture_fresh,
    "probe_span": phase_probe_span,
    "probe_npz": phase_probe_npz,
    "probe_sweep": phase_probe_sweep,
    "probe_gating": phase_probe_gating,
}


def _import_check() -> None:
    """Argparse-attribute completeness over THIS module (#2163; module-level
    function, never inline in main — a bare in-function import would shadow
    module symbols function-wide, #1739). The gen helpers we call read
    args.{raw_root,stage_raw_from_hf,pools_dir,stage_pools_from_hf} inside
    gen's module; our parser defines all four."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue #2378 capture rig (plan §4.3)")
    ap.add_argument("--phase", choices=sorted(PHASES), default=None)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="verify deferred imports + argparse-attribute completeness, then exit 0",
    )
    # row sources (unit 1 artifacts; local-first -> HF staging fallback)
    ap.add_argument("--raw-root", default=str(cm.RAW_ROOT_DEFAULT))
    ap.add_argument("--stage-raw-from-hf", action="store_true")
    ap.add_argument("--pools-dir", default=str(cm.POOLS_DIR))
    ap.add_argument("--stage-pools-from-hf", action="store_true")
    ap.add_argument(
        "--mined-dir", default=None, help="sega_mined rows dir (default: <raw-root>/sega_mined)"
    )
    ap.add_argument(
        "--ledger-root",
        default=str(cm.LEDGER_ROOT),
        help="capture_ready gate root (gen --phase capture_ready output)",
    )
    # outputs
    ap.add_argument("--out-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations"))
    ap.add_argument(
        "--pilot-out-root",
        default=str(cm.PILOT_STORE_DEFAULT),
        help="pilot all-layer store (plan §10 declared discard; never uploaded). "
        "Dispatch threads a ROUND-SCOPED root here (cm.pilot_capture_out_root) so "
        "a round>=2 pilot never resumes into round 1's StageLedger (r12 fix).",
    )
    ap.add_argument("--layer-sweep-out", default=str(cm.LEDGER_ROOT / "pilot" / "layer_sweep.json"))
    # sizing
    ap.add_argument("--rows", type=int, default=0, help="per-cell row cap (0 = all; smoke: 16)")
    ap.add_argument("--pilot-rows", type=int, default=2500)
    ap.add_argument("--cells", default="", help="csv of cells (default: phase-specific full set)")
    ap.add_argument("--fresh-draws", type=int, default=len(cm.FRESH_SEEDS))
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--max-batch-rows", type=int, default=64)
    ap.add_argument("--max-capture-tokens", type=int, default=8192)
    # layers
    ap.add_argument("--layers", default="", help="explicit csv of layer indices (0..64)")
    ap.add_argument("--all-layers", action="store_true", help="capture all 65 hidden states")
    ap.add_argument("--layer-star", type=int, default=None)
    ap.add_argument(
        "--layer-star-from", default="", help="layer_sweep.json path to read selected_layer from"
    )
    # devices / sweep
    ap.add_argument(
        "--device", default="cuda", help="model device (physical GPU is CVD-pinned by the launcher)"
    )
    ap.add_argument("--min-free-hbm-gb", type=float, default=60.0)
    ap.add_argument("--sweep-device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--sweep-layer-chunk", type=int, default=16)
    ap.add_argument("--reduced-k", type=int, default=REDUCED_BASIS_K)
    # escapes
    ap.add_argument(
        "--skip-capture-ready",
        action="store_true",
        help="smoke escape: skip the capture_ready kept-id gate",
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--skip-sweep",
        action="store_true",
        help="pilot: capture only (sweep separately via --phase sweep)",
    )
    return ap


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    PHASES[args.phase](args)
    # Explicit exit after flush: heavy C-extension atexit teardown can rewrite
    # the rc of a COMPLETED phase under `set -euo pipefail` (#1689 gotcha).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
