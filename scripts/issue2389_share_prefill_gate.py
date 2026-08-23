#!/usr/bin/env python3
"""Issue #2389 — gate-4b item-5 pod-side ``share_prefill`` equivalence battery (M-N2).

Produces ``gates/share_prefill_equivalence.json`` — the ONLY artifact that can
arm ``share_prefill=True`` in production generation (plan §4.7 item 5 pin 2:
``issue2389_run.py --share-prefill auto`` reads this verdict at phase entry;
no artifact / FAIL / ``off`` ⇒ serial, FAIL-OPEN). Runs on ONE GPU inside the
gate-4b window (workers 0–2, concurrent with gate-3-slice generation).

Battery legs (pre-registered, plan §4.7 item 5; per batch × {unhooked, hooked}):

- **leg 0 — identical-run nondeterminism, measured FIRST:** two IDENTICAL
  unperturbed shared-prefill greedy runs, per-step logits compared bitwise.
  This is the plan's mandated precondition for leg (e2)'s exactness read and
  the calibration basis for leg (b)'s bf16 tolerance.
- **leg (b) — per-step logit equivalence (K_eq = 8):** the shared-prefill
  path vs the REAL per-draw-prefill ``model.generate()`` reference, greedy,
  hooked AND unhooked, unequal-length LEFT-padded batches. Exactness regime:
  fp32 rigs (``--offline-tiny`` / ``--tiny``) require BITWISE equality; the
  production bf16 rig uses the calibrated two-tier convention (gotchas.md
  #779/#1005 family) — binding tier: per-step argmax agreement == 100% AND
  max |Δ log-softmax| <= max(0.05, 10 × the leg-0 measured identical-run
  log-softmax drift); reported tier: raw fp32 max-abs logit diff.
- **leg (e2) — BRANCH INDEPENDENCE (the direct aliasing probe):** perturb
  draw 0's first decode token (``_force_first_token``); sibling draws'
  per-step logits must be BITWISE unchanged through decode steps 2..K_eq —
  EXACT on every rig, tolerance NEVER permitted here. When leg 0 measures
  the rig NOT bitwise-deterministic, (e2) falls back to the direct
  cache-byte / storage-isolation assert (deepcopied per-draw cache storage
  pointers disjoint from the base cache's) — never a widened threshold.
- **storage isolation (always recorded):** the per-draw ``copy.deepcopy``
  cache's reachable tensor storages are disjoint from the base prefill
  cache's.
- **leg (f) — measured wall (reported, non-binding):** public
  ``generate_batch`` serial vs ``share_prefill=True`` at temp 1.0, one
  LONG-context and one SHORT-context cell batch (``--skip-wall`` for smoke).

Verdict: PASS ⇔ every (batch × variant) leg (b) AND leg (e2) passed. The
script always exits 0 on a completed battery (FAIL-OPEN — the verdict rides
the artifact; the run proceeds serial on FAIL). Legs (a)/(c)/(d) are covered
by the CPU acceptance battery (``tests/test_issue2389_steering_share_prefill``)
and steering's own asserts; this script is their production-device sibling.

Modes:
  --offline-tiny   no-network CPU rig (offline WordLevel tokenizer + tiny
                   random-config qwen3_next hybrid) — the tiny-real CPU e2e
                   smoke; writes the same artifact shape to --out-root.
  --tiny           run.py's tiny from-config model over the REAL bank
                   contexts (fetches config/tokenizer at the pinned revision).
  (default)        the production bf16 27B at the pinned revision (CUDA).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402
import torch.nn.functional as tF  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2389_run as R  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    _encode_left_padded,
    _generate_batch_shared_prefill,
    _shared_prefill_forward,
    generate_batch,
)

logger = logging.getLogger("issue2389.share_prefill_gate")

GATE_NAME = "share_prefill_equivalence.json"
# Round-5 H (concern share-prefill-battery-domain-blind): bump on ANY change
# to the battery's legs / tolerance convention / verdict criterion — adoption
# refuses a report from a different protocol (re-measure, never adopt).
GATE_PROTOCOL_VERSION = 1
K_EQ_DEFAULT = 8
# Binding leg-(b) bf16 tier: max |Δ log-softmax| floor (probability-space,
# scale-free) — the realized tolerance is max(this, 10 × leg-0 drift).
LOGSOFTMAX_TOL_FLOOR = 0.05


# ── offline tiny rig (no network; mirrors the CPU acceptance battery) ──

_WORDS = [
    "hello", "world", "tell", "me", "about", "cats", "dogs", "the", "a",
    "story", "you", "are", "helpful", "pirate", "short", "long", "answer",
    "question", "sky", "blue", "green", "red", "one", "two", "three", "four",
    "user", "system", "assistant",
]  # fmt: skip

OFFLINE_BATCHES = {
    "short": [
        {"system": None, "user": "tell me about cats"},
        {"system": None, "user": "one two three four question"},
    ],
    "long": [
        {"system": "you are a helpful pirate", "user": "tell me a long story about the blue sky"},
        {"system": None, "user": "tell me about dogs and cats and the green sky"},
        {"system": None, "user": "hello world"},
    ],
}


def _offline_tiny_model_and_tok():
    """Offline WordLevel chat tokenizer + tiny random qwen3_next hybrid
    (3 linear + 1 full-attention layers) — no HF fetch, fp32 CPU."""
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers import models as tmodels
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

    vocab = {"<|pad|>": 0, "<|im_end|>": 1, "<|im_start|>": 2, "<unk>": 3}
    for w in _WORDS:
        vocab[w] = len(vocab)
    tok_obj = Tokenizer(tmodels.WordLevel(vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="<|pad|>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
        "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant{% endif %}"
    )
    mcfg = Qwen3NextConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        full_attention_interval=4,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    torch.manual_seed(20260819)
    model = Qwen3NextForCausalLM(mcfg).eval()
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


# ── battery legs ───────────────────────────────────────────────────────


@torch.no_grad()
def _serial_step_logits(model, tok, ctxs, hook, k, render_fn, ids_fn) -> list[torch.Tensor]:
    """Per-draw-prefill reference: REAL ``model.generate()`` raw per-step logits."""
    input_ids, attention_mask, _ = _encode_left_padded(model, tok, ctxs, render_fn, ids_fn)
    if hook is not None:
        hook.arm(expected_prompt_len=input_ids.shape[1])
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=k,
        pad_token_id=tok.pad_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    return [step.float() for step in out.logits]


def _shared_step_logits(
    model, tok, ctxs, hook, k, draws, render_fn, ids_fn, force_first: dict | None = None
) -> list[list[torch.Tensor]]:
    """Shared-prefill greedy per-step logits via the battery seams."""
    _, step_logits = _generate_batch_shared_prefill(
        model,
        tok,
        ctxs,
        n=draws,
        hook=hook,
        max_new_tokens=k,
        temperature=0.0,
        render_fn=render_fn,
        ids_fn=ids_fn,
        _collect_step_logits=k,
        _force_first_token=force_first,
    )
    return step_logits


def _pairwise_drift(a: list[list[torch.Tensor]], b: list[list[torch.Tensor]]) -> dict:
    """Max abs logit + log-softmax drift and bitwise flag between two shared runs."""
    max_abs = 0.0
    max_dls = 0.0
    bitwise = True
    for da, db in zip(a, b, strict=True):
        for sa, sb in zip(da, db, strict=True):
            if not torch.equal(sa, sb):
                bitwise = False
            max_abs = max(max_abs, float((sa - sb).abs().max()))
            max_dls = max(
                max_dls,
                float((tF.log_softmax(sa, dim=-1) - tF.log_softmax(sb, dim=-1)).abs().max()),
            )
    return {"bitwise": bitwise, "max_abs_logit": max_abs, "max_abs_log_softmax": max_dls}


def _cache_storage_ptrs(obj, seen: set[int] | None = None, depth: int = 0) -> set[int]:
    """Reachable tensor storage pointers (cache-byte / storage-isolation probe)."""
    if seen is None:
        seen = set()
    ptrs: set[int] = set()
    if depth > 6 or id(obj) in seen:
        return ptrs
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        ptrs.add(obj.untyped_storage().data_ptr())
        return ptrs
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            ptrs |= _cache_storage_ptrs(v, seen, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            ptrs |= _cache_storage_ptrs(v, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            ptrs |= _cache_storage_ptrs(v, seen, depth + 1)
    return ptrs


@torch.no_grad()
def _storage_isolation(model, tok, ctxs, hook, render_fn, ids_fn) -> dict:
    """Deepcopied per-draw cache storages disjoint from the base prefill cache."""
    input_ids, attention_mask, _ = _encode_left_padded(model, tok, ctxs, render_fn, ids_fn)
    _, base_past = _shared_prefill_forward(model, input_ids, attention_mask, hook)
    base = _cache_storage_ptrs(base_past)
    dup = _cache_storage_ptrs(copy.deepcopy(base_past))
    shared = base & dup
    return {
        "n_base_storages": len(base),
        "n_copy_storages": len(dup),
        "n_shared_storages": len(shared),
        "disjoint": not shared,
    }


def _run_variant(
    model, tok, ctxs, *, hook_factory, k: int, draws: int, exact: bool, render_fn, ids_fn
) -> dict:
    """One (batch × {unhooked|hooked}) battery pass; hooks freshly installed
    per run (a prefill consumes the armed edit)."""
    assert draws >= 2, "leg (e2) needs >= 2 draws (a sibling must exist)"

    def _with_hook(fn, *fargs, **fkw):
        hook = hook_factory(model) if hook_factory else None
        if hook is not None:
            hook.install()
        try:
            return fn(*fargs, hook=hook, **fkw)
        finally:
            if hook is not None:
                hook.remove()

    # leg 0 — identical-run nondeterminism, measured FIRST.
    run1 = _with_hook(
        _shared_step_logits, model, tok, ctxs, k=k, draws=2, render_fn=render_fn, ids_fn=ids_fn
    )
    run2 = _with_hook(
        _shared_step_logits, model, tok, ctxs, k=k, draws=2, render_fn=render_fn, ids_fn=ids_fn
    )
    leg0 = _pairwise_drift(run1, run2)

    # leg (b) — shared vs the REAL generate() per-draw-prefill reference.
    serial = _with_hook(
        _serial_step_logits, model, tok, ctxs, k=k, render_fn=render_fn, ids_fn=ids_fn
    )
    shared = _with_hook(
        _shared_step_logits, model, tok, ctxs, k=k, draws=draws, render_fn=render_fn, ids_fn=ids_fn
    )
    n_common = min(len(serial), min(len(s) for s in shared))
    assert n_common >= 2, (len(serial), [len(s) for s in shared])
    b_bitwise = True
    b_max_abs = 0.0
    b_max_dls = 0.0
    b_argmax_agree = 0
    b_argmax_total = 0
    for i in range(draws):
        for t in range(n_common):
            sa, sb = shared[i][t], serial[t]
            if not torch.equal(sa, sb):
                b_bitwise = False
            b_max_abs = max(b_max_abs, float((sa - sb).abs().max()))
            b_max_dls = max(
                b_max_dls,
                float((tF.log_softmax(sa, dim=-1) - tF.log_softmax(sb, dim=-1)).abs().max()),
            )
            agree = (sa.argmax(dim=-1) == sb.argmax(dim=-1)).all()
            b_argmax_agree += int(bool(agree))
            b_argmax_total += 1
    tol = max(LOGSOFTMAX_TOL_FLOOR, 10.0 * leg0["max_abs_log_softmax"])
    if exact:
        b_pass = b_bitwise
    else:
        b_pass = (b_argmax_agree == b_argmax_total) and (b_max_dls <= tol)
    leg_b = {
        "n_steps_compared": n_common,
        "draws": draws,
        "bitwise": b_bitwise,
        "max_abs_logit": b_max_abs,
        "max_abs_log_softmax": b_max_dls,
        "argmax_agreement": f"{b_argmax_agree}/{b_argmax_total}",
        "tolerance_regime": "exact-bitwise" if exact else "bf16-two-tier",
        "log_softmax_tol": None if exact else tol,
        "passed": bool(b_pass),
    }

    # leg (e2) — branch independence, EXACT (fallback: storage isolation).
    isolation = _with_hook(_storage_isolation, model, tok, ctxs, render_fn=render_fn, ids_fn=ids_fn)
    if leg0["bitwise"]:
        base = shared  # the unperturbed run above
        vocab = base[0][0].shape[-1]
        forced = (base[0][0].argmax(dim=-1) + 1) % vocab
        perturbed = _with_hook(
            _shared_step_logits,
            model,
            tok,
            ctxs,
            k=k,
            draws=draws,
            render_fn=render_fn,
            ids_fn=ids_fn,
            force_first={0: forced},
        )
        # Draw 0 must actually have been perturbed (steps > 0 diverge or texts
        # differ); siblings must be BITWISE unchanged through steps 2..k.
        perturb_took = any(not torch.equal(base[0][t], perturbed[0][t]) for t in range(1, n_common))
        sib_exact = all(
            torch.equal(base[i][t], perturbed[i][t])
            for i in range(1, draws)
            for t in range(n_common)
        )
        leg_e2 = {
            "mode": "bitwise-sibling",
            "perturbation_effective": bool(perturb_took),
            "siblings_bitwise_through_steps": n_common,
            "passed": bool(sib_exact and perturb_took),
        }
    else:
        # Plan-sanctioned fallback (never a widened threshold): the rig is not
        # bitwise-deterministic across identical runs, so the sibling read is
        # replaced by the direct cache-byte / storage-isolation assert.
        leg_e2 = {
            "mode": "cache-isolation-fallback",
            "identical_run_drift": leg0,
            "passed": bool(isolation["disjoint"]),
        }

    return {
        "leg0_identical_run": leg0,
        "leg_b": leg_b,
        "leg_e2": leg_e2,
        "storage_isolation": isolation,
        "passed": bool(leg_b["passed"] and leg_e2["passed"]),
    }


def _leg_f_wall(model, tok, ctxs, *, draws: int, max_new_tokens: int, render_fn, ids_fn) -> dict:
    """Reported-only wall comparison through the PUBLIC generate_batch surface."""
    t0 = time.monotonic()
    generate_batch(
        model, tok, ctxs, n=draws, max_new_tokens=max_new_tokens, temperature=1.0,
        seed_base=2389, render_fn=render_fn, ids_fn=ids_fn, share_prefill=False,
    )  # fmt: skip
    serial_s = time.monotonic() - t0
    t0 = time.monotonic()
    generate_batch(
        model, tok, ctxs, n=draws, max_new_tokens=max_new_tokens, temperature=1.0,
        seed_base=2389, render_fn=render_fn, ids_fn=ids_fn, share_prefill=True,
    )  # fmt: skip
    shared_s = time.monotonic() - t0
    return {
        "draws": draws,
        "max_new_tokens": max_new_tokens,
        "serial_s": serial_s,
        "shared_s": shared_s,
        "speedup": (serial_s / shared_s) if shared_s > 0 else None,
    }


def run_battery(
    model,
    tok,
    batches: dict[str, list[dict]],
    *,
    k_eq: int,
    draws: int,
    exact: bool,
    render_fn=None,
    ids_fn=None,
    hook_layer: int | None = None,
    wall_draws: int = 0,
    wall_max_new_tokens: int = 256,
) -> dict:
    """The full battery over every (batch × {unhooked, hooked}) variant."""
    n_layers = model.config.num_hidden_layers

    def _hook_factory(m) -> DeltaHook:
        torch.manual_seed(7)
        delta = torch.randn(m.config.hidden_size, dtype=torch.float32)
        layer = hook_layer if hook_layer is not None else max(0, n_layers // 2 - 1)
        return DeltaHook(m, layer=layer, delta=delta, alpha=1.0)

    variants: dict[str, dict] = {}
    walls: dict[str, dict] = {}
    for name, ctxs in batches.items():
        for variant, factory in (("unhooked", None), ("hooked", _hook_factory)):
            logger.info("[battery] batch=%s variant=%s n_ctx=%d", name, variant, len(ctxs))
            variants[f"{name}.{variant}"] = _run_variant(
                model,
                tok,
                ctxs,
                hook_factory=factory,
                k=k_eq,
                draws=draws,
                exact=exact,
                render_fn=render_fn,
                ids_fn=ids_fn,
            )
        if wall_draws > 0:
            walls[name] = _leg_f_wall(
                model,
                tok,
                ctxs,
                draws=wall_draws,
                max_new_tokens=wall_max_new_tokens,
                render_fn=render_fn,
                ids_fn=ids_fn,
            )
    verdict = "PASS" if all(v["passed"] for v in variants.values()) else "FAIL"
    return {
        "verdict": verdict,
        "criterion": (
            "PASS <=> leg (b) AND leg (e2) pass for every (batch x {unhooked, hooked}) "
            "variant; leg (b) exact-bitwise on fp32 rigs, bf16 two-tier (argmax agreement "
            "== 100% AND max |dlog-softmax| <= max(0.05, 10x leg-0 drift)) on the "
            "production rig; leg (e2) EXACT always (cache-isolation fallback only when "
            "leg 0 measures the rig non-bitwise); FAIL-OPEN — run proceeds serial"
        ),
        "k_eq": k_eq,
        "draws": draws,
        "exact_regime": exact,
        "variants": variants,
        "wall_leg_f": walls or None,
    }


# ── batch selection over the real bank ────────────────────────────────


def _select_cell_batches(tok, contexts: dict, batch_size: int) -> dict[str, list[dict]]:
    """One LONG-context and one SHORT-context cell batch (leg f, plan item 5),
    picked mechanically by rendered token length of each cell's first context;
    batches keep unequal lengths so LEFT padding is engaged."""
    by_cell: dict[str, list[str]] = {}
    for cid in sorted(contexts):
        by_cell.setdefault(contexts[cid]["cell"], []).append(cid)
    probe_len = {
        cell: len(R.BANK29.context_token_ids_2389(tok, contexts[cids[0]]))
        for cell, cids in by_cell.items()
    }
    long_cell = max(probe_len, key=lambda c: probe_len[c])
    short_cell = min(probe_len, key=lambda c: probe_len[c])
    out: dict[str, list[dict]] = {}
    for name, cell in (("long", long_cell), ("short", short_cell)):
        ctxs = [contexts[cid] for cid in by_cell[cell][: max(batch_size, 2)]]
        lens = {len(R.BANK29.context_token_ids_2389(tok, c)) for c in ctxs}
        for cid in by_cell[cell][max(batch_size, 2) :]:
            if len(lens) >= 2:
                break
            c = contexts[cid]
            ctxs.append(c)
            lens.add(len(R.BANK29.context_token_ids_2389(tok, c)))
        assert len(lens) >= 2, f"{cell}: could not build an unequal-length batch"
        out[name] = ctxs
    logger.info(
        "[battery] cells: long=%s (%d tok) short=%s (%d tok)",
        long_cell,
        probe_len[long_cell],
        short_cell,
        probe_len[short_cell],
    )
    return out


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2389 gate-4b share_prefill battery (M-N2).")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--offline-tiny", action="store_true", help="no-network CPU tiny rig")
    ap.add_argument("--tiny", action="store_true", help="run.py tiny model over real contexts")
    ap.add_argument("--k-eq", type=int, default=K_EQ_DEFAULT)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--f-draws", type=int, default=10, help="leg (f) draw count (production K)")
    # 2048 = the production default cap (leg (f) is a PRODUCTION-shape wall
    # comparison; 256 inflated the prefill fraction ~8-16x and overstated
    # share_prefill's benefit — r1 efficiency concern).
    ap.add_argument("--f-max-new-tokens", type=int, default=2048)
    ap.add_argument("--skip-wall", action="store_true", help="skip leg (f) (smoke)")
    # B2 (r1 review): the SAME enum as the callee `issue2389_run.parse_args` —
    # the prior `full` default could not bind through `R.parse_args` and every
    # real (non --offline-tiny) invocation died in argparse.
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    # Round-5 A (r4 review): the battery is IDEMPOTENT — a matching completed
    # report is adopted (skip re-measure + rewrite) so the plan §9
    # same-command resume never churns the artifact; --force re-measures.
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-measure even when a matching completed report exists",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve deferred imports + args-attribute completeness, then exit 0",
    )
    return ap.parse_args(argv)


def _compose_run_cfg(args: argparse.Namespace) -> "R.RunConfig":
    """Build the shared RunConfig through run.py's OWN parser (the reused-
    module contract) — extracted so the default-argument composition is
    testable end-to-end (B2: a `--upload` default that cannot bind through
    `R.parse_args` must fail a unit test, not the real gate invocation)."""
    rargv = ["--phase", "anchors", "--out-root", str(args.out_root), "--upload", args.upload]
    if args.tiny:
        rargv.append("--tiny")
    return R.build_config(R.parse_args(rargv))


def _runtime_identity() -> dict[str, str]:
    """torch/transformers versions — the kernel-level runtime identity the
    battery's bf16 numerics + tolerance calibration are functions of
    (round-5 H)."""
    import transformers

    return {"torch": str(torch.__version__), "transformers": str(transformers.__version__)}


def _impl_digest() -> str:
    """sha256 of the CERTIFIED shared-prefill implementation's source
    (round-6 hardening of round-5 H, concern
    ``share-prefill-battery-domain-blind``): the git identity stays
    WARN-only — crash-fix commits ELSEWHERE must not force re-measures (the
    16<->32 fingerprint-flip quarantine risk) — but an edit to the function
    the PASS actually certifies (``_generate_batch_shared_prefill``)
    invalidates the equivalence evidence itself, so it binds HARD:
    re-measure + overwrite (a same-decision overwrite keeps the
    verdict+mode digest, so live freezes are unaffected — round-5 A)."""
    import hashlib
    import inspect

    return hashlib.sha256(inspect.getsource(_generate_batch_shared_prefill).encode()).hexdigest()


def _adoptable_gate_report(
    out_path: Path, mode: str, cfg: "R.RunConfig | None", args: argparse.Namespace
) -> dict | None:
    """An existing gate report THIS invocation may ADOPT instead of re-measuring.

    Round-5 A (r4 review) — the ``_reusable_pilot_report`` idiom applied to
    the battery driver: the dispatcher re-runs the battery unconditionally on
    the plan §9 same-command resume, and every re-run used to rewrite the
    artifact with a fresh ``ts`` (different bytes at an identical verdict),
    which the freeze's old raw-byte digest read as vanished evidence
    (spurious family disarm -> ``regime_fingerprint`` flip -> banked-shard
    quarantine). Adoption requires the SAME mode (offline-tiny / tiny /
    production — a mode mismatch is the designed B6 smoke->production
    upgrade path and re-measures + overwrites, never raises) and, for
    non-offline modes, the same regime (model id@revision + tiny/smoke
    bits); an unrecognized verdict re-measures.

    Round-5 H (concern ``share-prefill-battery-domain-blind``): adoption is
    additionally EVIDENCE-STRENGTH-aware — a report whose recorded battery
    inputs are WEAKER than this invocation's (fewer equivalence steps /
    draws, a different batch shape, no wall leg when this invocation would
    measure one — the canonical ``--skip-wall``-produced PASS), a different
    battery protocol version, a different torch/transformers runtime
    (the bf16 tolerance calibration is a kernel-version property), or a
    different certified-implementation source digest (``impl_sha256``,
    round-6) is NOT
    called "matching": the battery re-measures + overwrites (a same-decision
    overwrite keeps the verdict+mode digest, so live freezes are unaffected
    — round-5 A). A recorded git commit differing from this checkout's is
    WARN-only: crash-fix commits legitimately land between a battery run
    and the plan §9 same-command resume, and a hard git pin would force a
    re-measure on every resume — the runtime the numerics depend on is
    pinned by the torch/transformers legs above. Returns the adoptable
    record or None (= run the battery)."""
    if not out_path.exists():
        return None
    rec = json.loads(out_path.read_text())
    if rec.get("mode") != mode:
        return None
    if rec.get("verdict") not in ("PASS", "FAIL"):
        return None
    if cfg is not None:
        repro = rec.get("repro") or {}
        if (
            repro.get("model_id") != cfg.model_id
            or repro.get("model_revision") != cfg.model_revision
        ):
            return None
        if bool(repro.get("tiny")) != bool(cfg.tiny) or bool(repro.get("smoke")) != bool(cfg.smoke):
            return None
    weaker: list[str] = []
    if rec.get("protocol_version") != GATE_PROTOCOL_VERSION:
        weaker.append(
            f"protocol_version={rec.get('protocol_version')!r} != {GATE_PROTOCOL_VERSION}"
        )
    if int(rec.get("k_eq") or -1) < int(args.k_eq):
        weaker.append(f"k_eq={rec.get('k_eq')} < this invocation's {args.k_eq}")
    if int(rec.get("draws") or -1) < int(args.draws):
        weaker.append(f"draws={rec.get('draws')} < this invocation's {args.draws}")
    want_batch = None if args.offline_tiny else int(args.batch_size)
    if rec.get("batch_size", "<unrecorded>") != want_batch:
        weaker.append(
            f"batch_size={rec.get('batch_size', '<unrecorded>')!r} != this invocation's "
            f"{want_batch!r} (padding shape is part of what the legs certify)"
        )
    if not args.skip_wall:
        if int(rec.get("wall_draws") or -1) < int(args.f_draws) or not rec.get("wall_leg_f"):
            weaker.append(
                f"wall evidence weaker/absent (wall_draws={rec.get('wall_draws')}, "
                f"wall_leg_f={'present' if rec.get('wall_leg_f') else 'absent'}) — a "
                f"--skip-wall PASS must not stand in for a wall-measuring invocation"
            )
        elif int(rec.get("wall_max_new_tokens") or -1) != int(args.f_max_new_tokens):
            weaker.append(
                f"wall_max_new_tokens={rec.get('wall_max_new_tokens')} != this invocation's "
                f"{args.f_max_new_tokens} (leg f is a production-shape wall comparison)"
            )
    repro = rec.get("repro") or {}
    cur_rt = _runtime_identity()
    if repro.get("torch") != cur_rt["torch"] or repro.get("transformers") != cur_rt["transformers"]:
        weaker.append(
            f"runtime torch={repro.get('torch')}/transformers={repro.get('transformers')} != "
            f"this env's {cur_rt['torch']}/{cur_rt['transformers']} (bf16 tolerance "
            "calibration is kernel-version-dependent)"
        )
    impl = _impl_digest()
    if rec.get("impl_sha256", "<unrecorded>") != impl:
        weaker.append(
            f"impl_sha256={str(rec.get('impl_sha256', '<unrecorded>'))[:12]!r} != this "
            f"checkout's {impl[:12]!r} (round-6, concern "
            "share-prefill-battery-domain-blind: the certified shared-prefill "
            "implementation changed — its equivalence evidence is implementation-scoped)"
        )
    if weaker:
        logger.info(
            "[battery] existing %s-mode report NOT adoptable (%s) — re-measuring",
            mode,
            "; ".join(weaker),
        )
        return None
    if cfg is not None and repro.get("git_commit") != R._git_sha():
        logger.warning(
            "[battery] adopting a report recorded at git %s != this checkout %s "
            "(WARN-only: runtime identity is pinned by torch/transformers + the "
            "certified implementation by impl_sha256; --force re-measures)",
            repro.get("git_commit"),
            R._git_sha(),
        )
    return rec


def _import_check() -> int:
    from tokenizers import Tokenizer  # noqa: F401
    from transformers import PreTrainedTokenizerFast  # noqa: F401
    from transformers.models.qwen3_next import Qwen3NextConfig  # noqa: F401

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.out_root is None:
        raise SystemExit("--out-root is required (unless --import-check)")

    mode = "offline-tiny" if args.offline_tiny else ("tiny" if args.tiny else "production")
    cfg = None if args.offline_tiny else _compose_run_cfg(args)
    gates_dir = Path(args.out_root) / "gates"
    out_path = gates_dir / GATE_NAME
    if not args.force:
        existing = _adoptable_gate_report(out_path, mode, cfg, args)
        if existing is not None:
            # Round-5 A: idempotent same-command resume — NO model load, NO
            # re-measure, and crucially NO artifact rewrite (a fresh-``ts``
            # rewrite would churn the bytes under any live family freeze).
            logger.info(
                "[battery] matching %s-mode report already at %s (verdict=%s) — skipping "
                "re-measure (--force re-runs)",
                mode,
                out_path,
                existing["verdict"],
            )
            if cfg is not None and args.upload != "none":
                # Idempotent durability: re-verify the unchanged artifact is
                # on the Hub (covers a prior crash between write and upload).
                R._upload_dir(
                    cfg, cfg.gates_dir, f"{R.HF_PREFIX}/analysis_tensors/gates", [GATE_NAME]
                )
            print(f"[phase=share_prefill_gate_done] verdict={existing['verdict']} (adopted)")
            return 0

    if args.offline_tiny:
        model, tok = _offline_tiny_model_and_tok()
        batches = OFFLINE_BATCHES
        render_fn = ids_fn = None
        exact = True
    else:
        model, tok = R.load_model_and_tokenizer(cfg)
        batches = _select_cell_batches(tok, R.BANK.build_contexts(), args.batch_size)
        render_fn, ids_fn = R.BANK29.render_context_2389, R.BANK29.context_token_ids_2389
        exact = bool(args.tiny)  # tiny = fp32 CPU; production = bf16 CUDA

    report = run_battery(
        model,
        tok,
        batches,
        k_eq=args.k_eq,
        draws=args.draws,
        exact=exact,
        render_fn=render_fn,
        ids_fn=ids_fn,
        wall_draws=0 if args.skip_wall else args.f_draws,
        wall_max_new_tokens=args.f_max_new_tokens,
    )
    report["mode"] = mode
    report["ts"] = datetime.now(UTC).isoformat()
    # Round-5 H: the evidence-strength inputs _adoptable_gate_report compares
    # (protocol version, batch shape, wall-leg strength; k_eq/draws are
    # already inside run_battery's report). repro carries torch/transformers
    # in EVERY mode so runtime identity is always checkable.
    report["protocol_version"] = GATE_PROTOCOL_VERSION
    report["batch_size"] = None if args.offline_tiny else int(args.batch_size)
    report["wall_draws"] = 0 if args.skip_wall else int(args.f_draws)
    report["wall_max_new_tokens"] = int(args.f_max_new_tokens)
    # Round-6 (concern share-prefill-battery-domain-blind): the certified
    # implementation's source digest — binds HARD at adoption while the git
    # identity stays WARN-only (see _impl_digest).
    report["impl_sha256"] = _impl_digest()
    report["repro"] = R._repro(cfg) if cfg is not None else {"mode": mode, **_runtime_identity()}

    gates_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(out_path, report)
    logger.info("[battery] verdict=%s -> %s", report["verdict"], out_path)
    if cfg is not None and args.upload != "none":
        R._upload_dir(cfg, cfg.gates_dir, f"{R.HF_PREFIX}/analysis_tensors/gates", [GATE_NAME])
    # FAIL-OPEN: the verdict rides the artifact; a completed battery exits 0.
    print(f"[phase=share_prefill_gate_done] verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
