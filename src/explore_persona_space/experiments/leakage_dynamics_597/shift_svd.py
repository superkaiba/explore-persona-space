# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, − and ※ legitimately
"""Phases A/B (#597 follow-up `svd-per-checkpoint-titration-read`) — activation shifts.

Per-checkpoint per-context activation-shift extraction over the parent's 25
contexts × 50 questions probe rows (plan §4):

  Phase A (``--mode base``, once): teacher-forced base-model forwards with
  residuals captured via FORWARD HOOKS on the decoder block modules (the
  #493 round-6 mechanism — see ``_LayerHookCapture``) over every (context,
  question) row; store per row the slot + mean-over-response residuals at
  layers {7, 14, 21, 27} (fp16), the slot logits' four floats (storage
  contract: logp, z_marker, z_eos, logZ — incident #530), and derive the
  base context bank (per-context means) with the #536 ``global_mean``
  centered cosine matrix + provenance. Includes the base-vs-base zero-shift
  sanity and the one-row ``lm_head(final_norm(hook(last_block)))``
  logits-reproduction check (plan §12.4 — verifies the captured residual is
  genuinely PRE-final-norm; NOTE ``out.hidden_states[-1]`` is POST-final-norm
  in transformers ≥4.5x, so the tuple path double-norms at the last layer:
  round-1 of this rig did exactly that and failed on the real bf16 model at
  cos 0.812, epm:failure v3).

  Phase B (``--mode unit``, per arm × source): enumerate the downloaded
  checkpoint ladder (the parent's ``enumerate_ladder``), hot-swap each LoRA
  checkpoint (``PeftModel.from_pretrained`` → read → ``unload()``, threading
  the returned handle — the parent's pattern), run trained-side forwards on
  the SAME rows, subtract the cached Phase-A base reads, and persist per
  checkpoint THE MOMENT it completes (checkpoint-per-phase): per-context mean
  Δv at 4 layers × 2 poolings, the split-half mean pair at the layer-14 slot,
  per-question Δv norms, and the trained-side four floats. After the last
  checkpoint the FIRST-read checkpoint is re-read and its four floats must
  reproduce (end-of-ladder hot-swap invariant, atol 1e-3 — parent's check).

Four-float reproduction gate: for any checkpoint step with a downloaded
parent reference file (``panel_trajectories_raw/<arm>/<source>/step_*.json``)
in ``--refs-dir``, this module's own forward must reproduce the stored
``log P(※)`` to ≤ 0.1 nat per row on the trained side (base side gated in
``--mode base`` via ``--base-ref``). FAIL is a hard error — slot/tokenization
drift, wrong adapter, or indexing bugs are caught before any grid spend.

Batching note: rows are LEFT-padded per sub-batch with no explicit
``position_ids`` — the EXACT convention of the parent's
``compute_marker_slot_stats`` (whose stored records are the reproduction
reference). Qwen-2.5 uses RoPE (relative), so a uniform left-pad position
offset is a no-op for masked attention; equivalence vs the serial unpadded
read is pinned by ``tests/test_issue597_leakage_dynamics.py`` (cosine ≥ 0.999,
measured ~4e-7 max abs diff on a tiny Qwen2).

Residual mechanism note (round-2 fix, schema v2): residual reads come from
forward hooks on ``decoder.layers[L]`` — NOT ``output_hidden_states=True`` —
because the hidden-states tuple's TAIL entry is post-final-norm, so the
``hidden_states[L+1]`` read silently changes space at the LAST layer (27 of
28 on Qwen-2.5-7B). Hook output ≡ ``hidden_states[L+1]`` for L ≤ 26 (#493
GPU-verified), so the parent's {7, 14, 21} reads are reproduced exactly;
layer 27 is now genuinely pre-final-norm, matching the persona-distance
sweep family's canonical mechanism (#404/#406/#493).

Run as a SUBPROCESS from the dispatcher (``scripts/issue_597/
titration_svd_597.py``): ``uv run python -m
explore_persona_space.experiments.leakage_dynamics_597.shift_svd``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.shift_svd")

# v2 = layer-27 residual mechanism change (forward hooks, pre-final-norm —
# round-2 fix): v1 artifacts carry a POST-final-norm layer-27 read and must
# never mix with v2 via resume-skip (_stored_ckpt_is_current) or bank load.
SCHEMA_CKPT = "i597_svd_ckpt_v2"
SCHEMA_UNIT = "i597_svd_unit_v2"
SCHEMA_BANK = "i597_svd_base_bank_v2"

# Read layers (plan §11: {7,14,21} from #521/#551 + 27 from the persona-distance
# sweep set; primary 14 = the producing pipeline's DEFAULT_LAYER).
LAYERS: tuple[int, ...] = (7, 14, 21, 27)
PRIMARY_LAYER: int = 14
POOLINGS: tuple[str, ...] = ("slot", "mean_resp")

# Plan §7 gate tolerances.
FOURFLOAT_TOL_NATS = 0.1  # vs the parent's stored records (its own gate hit 0.067)
ZERO_SHIFT_TOL = 1e-3  # max ||Δv|| on the base-vs-base pass
INVARIANT_ATOL = 1e-3  # end-of-ladder four-float re-read (parent's tolerance)

# Both arms' ladders are downloaded from the Hub (immutable published
# artifacts), so the provenance run-id is a stable literal — the analogue of
# the parent's ARM_A_IMMUTABLE_RUN_ID for its re-downloadable arm.
HF_IMMUTABLE_RUN_ID = "hf-immutable-titration"

FOURFLOAT_KEYS = ("logp", "z_marker", "z_eos", "logZ")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _metadata(extra: dict | None = None) -> dict:
    meta = {
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "ts": datetime.now(UTC).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


# ── residual capture (forward hooks; #493 round-6 mechanism) ─────────────────


def _resolve_decoder(model):
    """The inner decoder module carrying ``.layers`` + ``.norm``.

    ``Qwen2ForCausalLM.get_decoder()`` → ``Qwen2Model``; ``PeftModel``
    forwards the attribute, so the SAME resolution works on hot-swapped
    LoRA checkpoints (Phase B).
    """
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model


class _LayerHookCapture:
    """Forward hooks on decoder block modules: PRE-final-norm residuals.

    The #493 round-6 mechanism (GPU-verified 2026-06-05): the
    ``output_hidden_states=True`` tuple's entry ``[L+1]`` equals the block-L
    output only for L ≤ n_blocks−2; the TAIL entry is the post-final-norm
    tensor (it is exactly ``lm_head``'s input). Hooking the block modules
    captures the raw residual stream uniformly at EVERY layer, eliminating
    the last-layer post-norm quirk (round-1 failure class, epm:failure v3).

    ``captured[layer]`` holds the (B, T, H) output of the MOST RECENT
    forward; callers clear between forwards via ``captured.clear()``.
    """

    def __init__(self, model, layers: tuple[int, ...]):
        self._decoder = _resolve_decoder(model)
        self._layers = tuple(dict.fromkeys(layers))
        n_blocks = len(self._decoder.layers)
        for layer in self._layers:
            if not 0 <= layer < n_blocks:
                raise IndexError(f"layer={layer} out of range (model has {n_blocks} blocks)")
        self.captured: dict[int, object] = {}
        self._handles: list = []

    def _make_hook(self, layer: int):
        def _hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.captured[layer] = hs.detach()

        return _hook

    def __enter__(self):
        for layer in self._layers:
            self._handles.append(
                self._decoder.layers[layer].register_forward_hook(self._make_hook(layer))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ── serial reference read ────────────────────────────────────────────────────


def _read_residuals_serial(model, full_ids, layers, response_start):
    """ONE teacher-forced forward; per-layer slot + mean-over-response reads.

    Ported from ``issue-602@d2e0bdf21:src/explore_persona_space/analysis/
    activation_shift.py::_read_residuals`` with ONE mechanism change (round-2
    fix): residuals come from :class:`_LayerHookCapture` forward hooks instead
    of ``output_hidden_states=True``. Identical at every layer the parent
    actually read ({7, 14, 21} of 28 — hook ≡ ``hidden_states[L+1]`` for
    L ≤ 26, #493 GPU-verified); pre-final-norm at the LAST block where the
    tuple's tail is post-norm. Kept as the SERIAL reference the batched
    ``compute_panel_reads`` is equivalence-pinned against (cosine ≥ 0.999 per
    layer × pooling).

    Returns ``{layer: {"slot": (H,), "mean_resp": (H,)}}`` — fp32 CPU.
    ``slot`` is the residual at the LAST token of the sequence
    (the parent's ``slot=-1`` read); ``mean_resp`` is the mean over
    ``[response_start:]``.
    """
    import torch

    with torch.no_grad(), _LayerHookCapture(model, tuple(layers)) as cap:
        ids = full_ids.unsqueeze(0).to(model.device)
        model(ids)
        n_t = ids.shape[1]
        assert 0 < response_start < n_t, (
            f"empty response segment: response_start={response_start}, T={n_t}"
        )
        reads: dict[int, dict[str, object]] = {}
        for layer in layers:
            h = cap.captured[layer]
            assert h.dim() == 3, f"expected (B, T, H), got {h.shape}"
            reads[layer] = {
                "slot": h[0, -1].detach().float().cpu(),
                "mean_resp": h[0, response_start:].mean(dim=0).detach().float().cpu(),
            }
    return reads


# ── row preparation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeRow:
    """One (context, question) teacher-forced row, pre-tokenized."""

    context: str
    q_idx: int
    full_ids: tuple[int, ...]
    prompt_len: int  # response tokens start here (== len(prompt_ids))


def limit_probe_contexts(
    probe_contexts: dict[str, dict], limit_contexts: int | None
) -> dict[str, dict]:
    """Deterministic context subset: the FIRST N names in stored insertion order."""
    if limit_contexts is None:
        return probe_contexts
    names = list(probe_contexts)[:limit_contexts]
    return {n: probe_contexts[n] for n in names}


def _build_prompt_text(tokenizer, panel_system_prompt: str, q: str) -> str:
    """The chat-template prompt render — the prompt part of ``build_slot_context``.

    Byte-identity with the parent: ``build_slot_context(tok, sp, q, r) ==
    _build_prompt_text(tok, sp, q) + r`` is ASSERTED per row in
    :func:`prepare_rows` (so the slot convention can never silently drift).
    """
    msgs_prompt: list[dict[str, str]] = []
    if panel_system_prompt and panel_system_prompt != "":
        msgs_prompt.append({"role": "system", "content": panel_system_prompt})
    msgs_prompt.append({"role": "user", "content": q})
    return tokenizer.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)


def prepare_rows(tokenizer, probe_contexts: dict[str, dict]) -> list[ProbeRow]:
    """Tokenize every (context, question) row with the parent's slot convention.

    The full string comes from the parent's byte-identity-pinned
    ``build_slot_context`` (``T_panel(q) + r_base``, encoded verbatim with
    ``add_special_tokens=False`` — the ``compute_marker_slot_stats``
    convention). The prefix-decomposition assert (plan §12.3) guarantees
    ``full_ids[:prompt_len] == prompt_ids`` so the mean-over-response pooling
    has a recoverable ``response_start``.
    """
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        build_slot_context,
    )

    rows: list[ProbeRow] = []
    for name, info in probe_contexts.items():
        sp = info["system_prompt"] or ""
        for qi, row in enumerate(info["rows"]):
            prompt_text = _build_prompt_text(tokenizer, sp, row["q"])
            full_text = build_slot_context(tokenizer, sp, row["q"], row["r_base"])
            assert full_text == prompt_text + row["r_base"], (
                f"({name}, q{qi}): build_slot_context output is not prompt + r_base — "
                "the parent's slot convention drifted"
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            # Prefix-decomposition assert (plan §12.3): response_start must be
            # recoverable, i.e. no BPE merge across the prompt/response seam.
            assert full_ids[: len(prompt_ids)] == prompt_ids, (
                f"({name}, q{qi}): tokenization does not decompose as prefix + response "
                f"(prompt_len={len(prompt_ids)}); mean-resp pooling would mis-index"
            )
            assert len(full_ids) > len(prompt_ids), f"({name}, q{qi}): empty response segment"
            rows.append(
                ProbeRow(
                    context=name,
                    q_idx=qi,
                    full_ids=tuple(full_ids),
                    prompt_len=len(prompt_ids),
                )
            )
    assert rows, "no probe rows prepared"
    return rows


# ── batched panel read ───────────────────────────────────────────────────────


def compute_panel_reads(
    model,
    rows: list[ProbeRow],
    *,
    layers: tuple[int, ...],
    marker_id: int,
    eos_token_id: int,
    batch_size: int,
    device: str,
    pad_token_id: int,
    check_lm_head: bool = False,
):
    """Batched teacher-forced reads: residuals + slot four floats, ONE forward each.

    Left-pad per sub-batch (the ``compute_marker_slot_stats`` convention; no
    explicit ``position_ids`` — RoPE relative invariance, equivalence-pinned
    vs :func:`_read_residuals_serial`). Residuals via :class:`_LayerHookCapture`
    (pre-final-norm at every layer; round-2 fix).

    Returns a dict:
      ``slot[L]``: (R, H) fp32 torch CPU tensor — residual at the last token;
      ``mean_resp[L]``: (R, H) fp32 — mean over the response segment;
      ``fourfloat``: (R, 4) float64 numpy — (logp, z_marker, z_eos, logZ);
      ``argmax_id``: (R,) int64 numpy — slot argmax (emission read).
    """
    import numpy as np
    import torch

    n_rows = len(rows)
    slot: dict[int, list] = {layer: [] for layer in layers}
    mean_resp: dict[int, list] = {layer: [] for layer in layers}
    fourfloat = np.empty((n_rows, 4), dtype=np.float64)
    argmax_id = np.empty((n_rows,), dtype=np.int64)

    # The lm_head check needs the LAST block's hook regardless of the read
    # layers (production layers include it: 27 == n_blocks-1 on Qwen-2.5-7B).
    last_block = len(_resolve_decoder(model).layers) - 1
    hook_layers = tuple(dict.fromkeys((*layers, last_block))) if check_lm_head else tuple(layers)

    lm_head_checked = not check_lm_head
    with torch.no_grad(), _LayerHookCapture(model, hook_layers) as cap:
        for start in range(0, n_rows, batch_size):
            chunk = rows[start : start + batch_size]
            max_len = max(len(r.full_ids) for r in chunk)
            padded, attn = [], []
            for r in chunk:
                pad_len = max_len - len(r.full_ids)
                padded.append([pad_token_id] * pad_len + list(r.full_ids))
                attn.append([0] * pad_len + [1] * len(r.full_ids))
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
            cap.captured.clear()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            assert logits.ndim == 3, logits.shape
            missing = [layer for layer in hook_layers if layer not in cap.captured]
            assert not missing, f"layer hooks did not fire for {missing}"
            for i, r in enumerate(chunk):
                pad_len = max_len - len(r.full_ids)
                resp_start_abs = pad_len + r.prompt_len
                assert resp_start_abs < max_len, (r.context, r.q_idx, resp_start_abs, max_len)
                for layer in layers:
                    h = cap.captured[layer]
                    assert h.dim() == 3, h.shape
                    slot[layer].append(h[i, -1].detach().float().cpu())
                    mean_resp[layer].append(
                        h[i, resp_start_abs:].mean(dim=0).detach().float().cpu()
                    )
                raw = logits[i, -1, :].float()
                log_z = float(torch.logsumexp(raw, dim=-1).item())
                z_marker = float(raw[marker_id].item())
                z_eos = float(raw[eos_token_id].item())
                fourfloat[start + i] = (z_marker - log_z, z_marker, z_eos, log_z)
                argmax_id[start + i] = int(torch.argmax(raw).item())
            if not lm_head_checked:
                _assert_lm_head_reproduces(model, logits, cap.captured[last_block], row_index=0)
                lm_head_checked = True
            del out, logits

    return {
        "slot": {layer: torch.stack(slot[layer]) for layer in layers},
        "mean_resp": {layer: torch.stack(mean_resp[layer]) for layer in layers},
        "fourfloat": fourfloat,
        "argmax_id": argmax_id,
    }


def _assert_lm_head_reproduces(model, logits, h_last_block, row_index: int) -> None:
    """One-row check: ``lm_head(final_norm(hook(last_block)))`` == model logits.

    Plan §12.4 verify — pins that the HOOK-captured residual is genuinely
    PRE-final-norm and row/slot-aligned with the forward's own logits:
    applying the model's final norm ONCE to the captured last-block slot
    output must land on the model's own slot logits. Run on the BASE model
    only (Phase A); bf16 reduction-order noise is absorbed by the cosine +
    argmax form of the assert.

    Round-1 regression note (epm:failure v3): the tuple path fed
    ``out.hidden_states[-1]`` — which is POST-final-norm (it is exactly
    ``lm_head``'s input) — through ``final_norm`` again, double-norming.
    Cos read 0.812 on the real bf16 Qwen-2.5-7B; the CPU smoke passed only
    because a random-init tiny model's RMSNorm weights are ones (uniform →
    direction-preserving). Pinned by
    ``tests/test_issue597_leakage_dynamics.py::
    test_panel_reads_pre_final_norm_and_double_norm_regression``.

    ``h_last_block`` is the hook capture of ``decoder.layers[-1]`` (B, T, H).
    """
    import torch

    inner = _resolve_decoder(model)
    recomputed = model.get_output_embeddings()(inner.norm(h_last_block[row_index, -1]))
    own = logits[row_index, -1]
    cos = torch.nn.functional.cosine_similarity(recomputed.float(), own.float(), dim=0).item()
    same_argmax = int(recomputed.argmax()) == int(own.argmax())
    if cos < 0.9999 or not same_argmax:
        raise RuntimeError(
            f"lm_head reproduction check FAILED: cos={cos:.6f}, same_argmax={same_argmax} — "
            "the hook-captured last-block residual is not the pre-final-norm input of the "
            "model's own slot logits (indexing or norm-space drift)"
        )
    log.info("[phase=lm_head_check] cos=%.6f argmax_match=%s", cos, same_argmax)


# ── four-float reproduction gate ─────────────────────────────────────────────


def compare_fourfloat_to_reference(
    rows: list[ProbeRow],
    fourfloat,
    ref_payload: dict,
    *,
    side: str,
    tol_nats: float = FOURFLOAT_TOL_NATS,
) -> dict:
    """Gate: this pipeline's ``log P(※)`` vs the parent's stored records (≤ tol).

    ``ref_payload`` is one parent ``step_*.json`` (schema ``i597_panel_ckpt_v1``,
    per-row four floats both sides). Matches on (context, q_idx); every one of
    OUR rows must resolve in the reference. ``side`` ∈ {"trained", "base"}
    selects which stored side to compare. Gates on logp (the plan's ≤0.1-nat
    criterion); z/logZ max diffs are reported for the smoke report.
    """
    ref_rows = {(r["context"], r["q_idx"]): r for r in ref_payload["rows"]}
    logp_key = f"logp_{side}"
    diffs = {"logp": 0.0, "z_marker": 0.0, "z_eos": 0.0, "logZ": 0.0}
    worst_row = None
    for i, row in enumerate(rows):
        key = (row.context, row.q_idx)
        if key not in ref_rows:
            raise RuntimeError(
                f"four-float gate: reference has no row for {key} "
                f"(arm/source/step mismatch or wrong reference file)"
            )
        ref = ref_rows[key]
        # Plain-float casts: this report lands in JSON meta payloads (numpy
        # scalars would crash json.dump) and the pass flag must be a real bool.
        d_logp = float(abs(fourfloat[i, 0] - float(ref[logp_key])))
        if d_logp > diffs["logp"]:
            diffs["logp"] = d_logp
            worst_row = key
        diffs["z_marker"] = float(
            max(diffs["z_marker"], abs(fourfloat[i, 1] - ref[f"z_marker_{side}"]))
        )
        diffs["z_eos"] = float(max(diffs["z_eos"], abs(fourfloat[i, 2] - ref[f"z_eos_{side}"])))
        diffs["logZ"] = float(max(diffs["logZ"], abs(fourfloat[i, 3] - ref[f"logZ_{side}"])))
    result = {
        "side": side,
        "n_rows_compared": len(rows),
        "max_abs_diff": diffs,
        "worst_row": list(worst_row) if worst_row else None,
        "tol_nats": tol_nats,
        "pass": bool(diffs["logp"] <= tol_nats),
    }
    if not result["pass"]:
        raise RuntimeError(
            f"FOUR-FLOAT REPRODUCTION GATE FAILED ({side}): max |Δ logp| = "
            f"{diffs['logp']:.4f} nat > {tol_nats} at row {worst_row} — slot indexing, "
            "tokenization, or adapter-loading drift vs the parent's stored records."
        )
    log.info(
        "[phase=fourfloat_gate] %s side PASSED: max |dlogp|=%.4f nat over %d rows",
        side,
        diffs["logp"],
        len(rows),
    )
    return result


# ── base bank (Phase A) ──────────────────────────────────────────────────────


def context_means(reads_tensor, rows: list[ProbeRow], context_names: list[str]):
    """Per-context mean over questions: (R, H) → (C, H) in ``context_names`` order."""
    import torch

    by_ctx: dict[str, list[int]] = {c: [] for c in context_names}
    for i, r in enumerate(rows):
        by_ctx[r.context].append(i)
    means = []
    for c in context_names:
        idx = by_ctx[c]
        assert idx, f"context {c} has no rows"
        means.append(reads_tensor[idx].mean(dim=0))
    return torch.stack(means)


def centered_cosine_matrix(bank):
    """#536 canonical bank cosine: global-mean-center → L2-normalize → cosine.

    Delegates to ``representation_shift.compute_cosine_matrix(C,
    centering="global_mean")`` — the rule's named implementation.
    """
    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    return compute_cosine_matrix(bank, centering="global_mean")


def run_base_mode(args) -> int:
    """Phase A: base side once — residuals, four floats, bank, sanity gates."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        load_probe_rows,
    )

    t0 = time.time()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model_path = args.base_model or BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> "
            f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}, expected [{MARKER_ID}]"
        )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    probe_contexts = load_probe_rows(args.probe_rows, limit_questions=args.limit_questions)
    probe_contexts = limit_probe_contexts(probe_contexts, args.limit_contexts)
    context_names = list(probe_contexts)
    rows = prepare_rows(tokenizer, probe_contexts)
    log.info(
        "[phase=base_setup] %d contexts x %d questions = %d rows",
        len(context_names),
        len(next(iter(probe_contexts.values()))["rows"]),
        len(rows),
    )

    layers = tuple(int(x) for x in args.layers.split(","))
    log.info("[phase=base_load] loading %s on %s (%s)", base_model_path, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    reads = compute_panel_reads(
        model,
        rows,
        layers=layers,
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=args.batch_size,
        device=device,
        pad_token_id=pad_id,
        check_lm_head=True,
    )
    log.info("[phase=base_reads] %d rows read in %.1fs", len(rows), time.time() - t0)

    # ── base-side four-float reproduction gate (plan §7a, base leg) ──
    gate_base = None
    if args.base_ref is not None:
        ref_payload = json.loads(Path(args.base_ref).read_text())
        gate_base = compare_fourfloat_to_reference(
            rows, reads["fourfloat"], ref_payload, side="base"
        )

    # ── base-vs-base zero-shift sanity (plan §7b) ──
    n_zero = min(args.zero_shift_rows, len(rows))
    zero_rows = rows[:n_zero]
    reads2 = compute_panel_reads(
        model,
        zero_rows,
        layers=layers,
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=args.batch_size,
        device=device,
        pad_token_id=pad_id,
    )
    max_norm = 0.0
    for layer in layers:
        for pooling in POOLINGS:
            d = reads2[pooling][layer] - reads[pooling][layer][:n_zero]
            max_norm = max(max_norm, float(d.norm(dim=1).max().item()))
    zero_report = {"n_rows": n_zero, "max_delta_norm": max_norm, "tol": ZERO_SHIFT_TOL}
    if max_norm > ZERO_SHIFT_TOL:
        raise RuntimeError(
            f"ZERO-SHIFT SANITY FAILED: base-vs-base max ||Δv|| = {max_norm:.2e} > "
            f"{ZERO_SHIFT_TOL} — extraction/indexing artifacts masquerade as shifts."
        )
    log.info("[phase=base_zero_shift] PASSED max ||dv|| = %.2e <= %s", max_norm, ZERO_SHIFT_TOL)

    # ── bank + centered cosines (#536) ──
    arrays: dict[str, np.ndarray] = {}
    for pooling in POOLINGS:
        for layer in layers:
            arrays[f"resid_{pooling}_l{layer}"] = reads[pooling][layer].to(torch.float16).numpy()
            bank = context_means(reads[pooling][layer], rows, context_names)
            arrays[f"bank_{pooling}_l{layer}"] = bank.to(torch.float16).numpy()
            arrays[f"bank_cos_centered_{pooling}_l{layer}"] = (
                centered_cosine_matrix(bank).numpy().astype(np.float32)
            )
    arrays["fourfloat_base"] = reads["fourfloat"]
    arrays["argmax_id_base"] = reads["argmax_id"]
    arrays["row_context_idx"] = np.array(
        [context_names.index(r.context) for r in rows], dtype=np.int32
    )
    arrays["row_q_idx"] = np.array([r.q_idx for r in rows], dtype=np.int32)

    meta = {
        "schema": SCHEMA_BANK,
        "base_model": base_model_path,
        "layers": list(layers),
        "poolings": list(POOLINGS),
        "context_names": context_names,
        "n_contexts": len(context_names),
        "n_questions": len(next(iter(probe_contexts.values()))["rows"]),
        "n_rows": len(rows),
        "centering": "global_mean",  # #536 provenance — bank_cos_centered_* only
        "bank_persona_names": context_names,
        "marker_id": MARKER_ID,
        "eos_token_id": IM_END_ID,
        "zero_shift": zero_report,
        "fourfloat_gate_base": gate_base,
        "device": device,
        "metadata": _metadata({"wall_seconds": round(time.time() - t0, 1)}),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, out)
    log.info("[phase=base_persist] base bank -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return 0


# ── ladder reads (Phase B) ───────────────────────────────────────────────────


def _load_bank(bank_path: Path):
    """Load the Phase-A bank npz; return (meta, arrays-lazy npz handle)."""
    import numpy as np

    npz = np.load(bank_path, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    assert meta["schema"] == SCHEMA_BANK, meta["schema"]
    return meta, npz


def _ckpt_arrays(
    reads,
    bank_npz,
    rows: list[ProbeRow],
    context_names: list[str],
    layers: tuple[int, ...],
):
    """Per-checkpoint deltas: context means, split-half pair, per-q norms.

    Split halves = even / odd question indices (deterministic; plan §4
    split-half pair at the layer-14 slot).
    """
    import numpy as np
    import torch

    n_ctx = len(context_names)
    n_layers = len(layers)
    out: dict[str, np.ndarray] = {}

    delta_mean = np.empty((n_layers, len(POOLINGS), n_ctx, reads["slot"][layers[0]].shape[1]))
    for li, layer in enumerate(layers):
        for pi, pooling in enumerate(POOLINGS):
            base = torch.from_numpy(bank_npz[f"resid_{pooling}_l{layer}"]).float()
            delta = reads[pooling][layer] - base  # (R, H) fp32
            delta_mean[li, pi] = context_means(delta, rows, context_names).numpy()
            if layer == PRIMARY_LAYER and pooling == "slot":
                # Split-half pair + per-question norms at the primary read.
                by_ctx: dict[str, list[int]] = {c: [] for c in context_names}
                for i, r in enumerate(rows):
                    by_ctx[r.context].append(i)
                n_q = max(len(v) for v in by_ctx.values())
                halves = np.empty((2, n_ctx, delta.shape[1]))
                per_q_norm = np.full((n_ctx, n_q), np.nan)
                for ci, c in enumerate(context_names):
                    idx = by_ctx[c]
                    even = [i for i in idx if rows[i].q_idx % 2 == 0]
                    odd = [i for i in idx if rows[i].q_idx % 2 == 1]
                    assert even and odd, (
                        f"context {c}: need >=1 question in each split half "
                        f"(got {len(even)}/{len(odd)}) — raise --limit-questions to >=2"
                    )
                    halves[0, ci] = delta[even].mean(dim=0).numpy()
                    halves[1, ci] = delta[odd].mean(dim=0).numpy()
                    for j, i in enumerate(idx):
                        per_q_norm[ci, j] = float(delta[i].norm().item())
                out["split_half_l14_slot"] = halves.astype(np.float16)
                out["per_q_norm_l14_slot"] = per_q_norm.astype(np.float32)
    out["delta_mean"] = delta_mean.astype(np.float16)
    out["fourfloat_trained"] = reads["fourfloat"]
    out["argmax_id_trained"] = reads["argmax_id"]
    return out


def _stored_ckpt_is_current(path: Path, run_config_id: str) -> bool:
    """Resume-skip: a stored per-checkpoint npz must match this run's config id."""
    import numpy as np

    try:
        npz = np.load(path, allow_pickle=False)
        meta = json.loads(str(npz["meta"]))
    except Exception:
        return False
    return meta.get("schema") == SCHEMA_CKPT and meta.get("run_config_id") == run_config_id


def read_one_checkpoint(
    base_model,
    tokenizer,
    ckpt_dir: Path,
    rows: list[ProbeRow],
    *,
    layers: tuple[int, ...],
    marker_id: int,
    eos_token_id: int,
    batch_size: int,
    device: str,
    pad_token_id: int,
):
    """Gauge-assert + hot-swap + read ONE checkpoint; return (reads, base_model).

    The returned ``base_model`` is the post-``unload()`` reference (PEFT
    mutates the wrapped model in place; callers must thread the returned
    handle forward — the parent's ``probe_one_checkpoint`` contract).
    """
    from peft import PeftModel

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    cfg_path = ckpt_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json missing at {cfg_path}")
    assert_gauge_free_adapter_config(json.loads(cfg_path.read_text()), context=str(cfg_path))

    peft_model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=False)
    peft_model.eval()
    try:
        reads = compute_panel_reads(
            peft_model,
            rows,
            layers=layers,
            marker_id=marker_id,
            eos_token_id=eos_token_id,
            batch_size=batch_size,
            device=device,
            pad_token_id=pad_token_id,
        )
    finally:
        base_model = peft_model.unload()
        del peft_model
    return reads, base_model


def run_unit_mode(args) -> int:
    """Phase B: one arm × source ladder — per-checkpoint deltas + consolidation."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        enumerate_ladder,
        load_probe_rows,
    )

    t0 = time.time()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model_path = args.base_model or BASE_MODEL

    bank_meta, bank_npz = _load_bank(Path(args.base_bank))
    layers = tuple(bank_meta["layers"])
    context_names: list[str] = bank_meta["context_names"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError("marker token id drifted vs MARKER_ID")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    probe_contexts = load_probe_rows(args.probe_rows, limit_questions=args.limit_questions)
    probe_contexts = limit_probe_contexts(probe_contexts, args.limit_contexts)
    rows = prepare_rows(tokenizer, probe_contexts)
    # The unit's rows MUST be the bank's rows (same subtraction index space).
    assert list(probe_contexts) == context_names, (
        f"unit context set {list(probe_contexts)} != bank context set {context_names}"
    )
    assert len(rows) == bank_meta["n_rows"], (len(rows), bank_meta["n_rows"])

    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    ladder = enumerate_ladder(Path(args.ckpt_root), steps)
    unit = f"{args.arm}_{args.source}"
    log.info(
        "[phase=unit_setup_%s] %d checkpoints %s; %d rows",
        unit,
        len(ladder),
        [s for s, _ in ladder],
        len(rows),
    )

    run_config_id = json.dumps(
        {
            "unit": unit,
            "n_rows": len(rows),
            "contexts": context_names,
            "layers": list(layers),
            "ladder_run_id": HF_IMMUTABLE_RUN_ID,
            "schema": SCHEMA_CKPT,
        },
        sort_keys=True,
    )

    refs_dir = Path(args.refs_dir) if args.refs_dir else None

    out_dir = Path(args.out_dir)
    ckpt_dir_out = out_dir / f"per_checkpoint_{unit}"
    ckpt_dir_out.mkdir(parents=True, exist_ok=True)

    log.info("[phase=unit_load_base_%s] loading %s on %s", unit, base_model_path, device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_model.eval()

    first_step: int | None = None
    first_ckpt_dir: Path | None = None
    first_fourfloat = None
    gate_reports: list[dict] = []
    step_files: dict[int, Path] = {}
    for step, ckpt_dir in ladder:
        out_path = ckpt_dir_out / f"step_{step:05d}.npz"
        step_files[step] = out_path
        if out_path.exists() and _stored_ckpt_is_current(out_path, run_config_id):
            log.info("[phase=unit_ckpt_%s] step %d already read; skipping", unit, step)
            if first_step is None:
                first_step, first_ckpt_dir = step, ckpt_dir
                first_fourfloat = np.load(out_path, allow_pickle=False)["fourfloat_trained"]
            continue
        t_ck = time.time()
        reads, base_model = read_one_checkpoint(
            base_model,
            tokenizer,
            ckpt_dir,
            rows,
            layers=layers,
            marker_id=MARKER_ID,
            eos_token_id=IM_END_ID,
            batch_size=args.batch_size,
            device=device,
            pad_token_id=pad_id,
        )
        # Four-float reproduction gate where a parent reference exists.
        ref_path = refs_dir / f"step_{step:05d}.json" if refs_dir else None
        if ref_path is not None and ref_path.exists():
            ref_payload = json.loads(ref_path.read_text())
            gate = compare_fourfloat_to_reference(
                rows, reads["fourfloat"], ref_payload, side="trained"
            )
            gate["step"] = step
            gate_reports.append(gate)
        arrays = _ckpt_arrays(reads, bank_npz, rows, context_names, layers)
        meta = {
            "schema": SCHEMA_CKPT,
            "run_config_id": run_config_id,
            "arm": args.arm,
            "source": args.source,
            "step": step,
            "unit": unit,
            "layers": list(layers),
            "poolings": list(POOLINGS),
            "context_names": context_names,
            "n_rows": len(rows),
            "ladder_run_id": HF_IMMUTABLE_RUN_ID,
            "fourfloat_keys": list(FOURFLOAT_KEYS),
            "metadata": _metadata({"wall_seconds": round(time.time() - t_ck, 1)}),
        }
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, out_path)
        if first_step is None:
            first_step, first_ckpt_dir = step, ckpt_dir
            first_fourfloat = reads["fourfloat"].copy()
        log.info(
            "[phase=unit_ckpt_%s] step %d done in %.1fs -> %s",
            unit,
            step,
            time.time() - t_ck,
            out_path,
        )

    # End-of-ladder hot-swap invariant (parent's check): re-read the FIRST
    # checkpoint; its four floats must reproduce within INVARIANT_ATOL.
    assert first_step is not None and first_ckpt_dir is not None and first_fourfloat is not None
    log.info("[phase=unit_invariant_%s] re-reading first checkpoint step %d", unit, first_step)
    reads_recheck, base_model = read_one_checkpoint(
        base_model,
        tokenizer,
        first_ckpt_dir,
        rows,
        layers=layers,
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=args.batch_size,
        device=device,
        pad_token_id=pad_id,
    )
    worst = float(np.max(np.abs(reads_recheck["fourfloat"] - first_fourfloat)))
    if worst > INVARIANT_ATOL:
        raise RuntimeError(
            f"END-OF-LADDER HOT-SWAP INVARIANT FAILED ({unit}): step {first_step} re-read "
            f"drifted by {worst:.6f} (> {INVARIANT_ATOL}) — cumulative adapter unload-state "
            "corruption; the ladder's reads are not trustworthy."
        )
    log.info("[phase=unit_invariant_%s] PASSED (max |diff| = %.2e)", unit, worst)

    # Consolidate per-ckpt files → one unit npz (the §6.5 deliverable shape).
    steps_sorted = sorted(step_files)
    stacked: dict[str, list] = {}
    for step in steps_sorted:
        npz = np.load(step_files[step], allow_pickle=False)
        for key in (
            "delta_mean",
            "split_half_l14_slot",
            "per_q_norm_l14_slot",
            "fourfloat_trained",
            "argmax_id_trained",
        ):
            stacked.setdefault(key, []).append(npz[key])
    unit_arrays = {k: np.stack(v) for k, v in stacked.items()}
    unit_meta = {
        "schema": SCHEMA_UNIT,
        "run_config_id": run_config_id,
        "arm": args.arm,
        "source": args.source,
        "unit": unit,
        "steps": steps_sorted,
        "layers": list(layers),
        "poolings": list(POOLINGS),
        "context_names": context_names,
        "n_rows": len(rows),
        "ladder_run_id": HF_IMMUTABLE_RUN_ID,
        "invariant_max_abs_diff": worst,
        "fourfloat_gates": gate_reports,
        "fourfloat_keys": list(FOURFLOAT_KEYS),
        "metadata": _metadata({"wall_seconds": round(time.time() - t0, 1), "device": device}),
    }
    unit_path = out_dir / f"{unit}.npz"
    tmp = unit_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(unit_meta)), **unit_arrays)
    os.replace(tmp, unit_path)
    # Verify the consolidated file loads before deleting per-ckpt shards.
    verify = np.load(unit_path, allow_pickle=False)
    assert verify["delta_mean"].shape[0] == len(steps_sorted), verify["delta_mean"].shape
    for step in steps_sorted:
        step_files[step].unlink()
    log.info(
        "[phase=unit_persist_%s] unit npz -> %s (%.1f MB, %d steps); per-ckpt shards removed",
        unit,
        unit_path,
        unit_path.stat().st_size / 1e6,
        len(steps_sorted),
    )

    # Per-unit scalar summary JSON (slab; synced to git by the orchestrator).
    if args.summary_out:
        summary = {
            "schema": "i597_svd_unit_summary_v1",
            "unit": unit,
            "arm": args.arm,
            "source": args.source,
            "steps": steps_sorted,
            "n_rows": len(rows),
            "invariant_max_abs_diff": worst,
            "fourfloat_gates": gate_reports,
            "unit_npz": str(unit_path),
            "metadata": _metadata({"wall_seconds": round(time.time() - t0, 1)}),
        }
        sp = Path(args.summary_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp_s = sp.with_suffix(".tmp")
        tmp_s.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        os.replace(tmp_s, sp)
        log.info("[phase=unit_summary_%s] -> %s", unit, sp)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Phase A/B subprocess CLI (called by titration_svd_597.py)."""
    parser = argparse.ArgumentParser(
        description="#597 SVD-titration shift extraction (Phases A/B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("base", "unit"), required=True)
    parser.add_argument("--probe-rows", type=Path, required=True)
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model (CPU smoke: tiny random-weight model + real tokenizer "
        "— the parent panel_probe's smoke pattern). Production always uses the default.",
    )
    parser.add_argument("--layers", type=str, default=",".join(str(x) for x in LAYERS))
    # base mode
    parser.add_argument("--out", type=Path, default=None, help="base mode: base_bank.npz path")
    parser.add_argument("--zero-shift-rows", type=int, default=50)
    parser.add_argument(
        "--base-ref",
        type=Path,
        default=None,
        help="base mode: one parent step_*.json for the base-side four-float gate.",
    )
    # unit mode
    parser.add_argument("--arm", choices=("a", "b"), default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--ckpt-root", type=Path, default=None)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--base-bank", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=None,
        help="unit mode: dir of parent step_*.json files; every step with a ref "
        "present is four-float gated (>=the unit's first step in production).",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    if args.mode == "base":
        if args.out is None:
            raise SystemExit("--mode base requires --out")
        return run_base_mode(args)
    for req in ("arm", "source", "ckpt_root", "base_bank", "out_dir"):
        if getattr(args, req) is None:
            raise SystemExit(f"--mode unit requires --{req.replace('_', '-')}")
    return run_unit_mode(args)


if __name__ == "__main__":
    sys.exit(main())
