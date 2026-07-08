#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 3 (GPU): matched-probe teacher-forced v0(C,B) capture.

Teacher-forces the base model over the on-policy completions
``issue763_generate_completions.py`` produced, captures the per-layer answer-span
MEAN over the model's OWN answer tokens, and probe-means them to ``v0(C,B)`` per
(context, behavior) at all 28 layers. VECTORIZED via
``BatchedAnswerSpanCapture`` (left-padded batch, the ONE vectorization piece,
plan §4.5 + ``.claude/rules/vectorize-many-cell-fits.md``).

The MATCHED-PROBE INVARIANT (the whole point): ``v0(C,B)`` is averaged over
EXACTLY the probes whose completions exist for (C, B) — the join is exact by
construction (#763 generates then captures the SAME completions; no cross-source
join). Per-row answer-span assert catches teacher-forcing truncation.

Adaptive T-cap + B-reduction (plan §4.5, methodology-critic concern): deception
generates to ``max_new_tokens=1024`` (> the §4.5 T≈512 capture-buffer
assumption), so long answers are TRUNCATED to ``--max-answer-tokens`` (default
1024, honoring the full deception cap) and the per-forward batch is reduced when
the padded length × batch would exceed the capture-buffer budget; the per-row
assert verifies the captured span length equals the (possibly-truncated) answer
token count.

Writes ``eval_results/issue_763/v0_matched_by_behavior.json`` (per-(C,B)
metadata + matched_n) and per-behavior ``.pt`` shards (the (n_ctx, 28, H)
tensors) under ``eval_results/issue_763/v0_shards/`` for the analysis +
HF analysis-tensor upload.

``--smoke`` runs the IDENTICAL path on the smoke gen slice with a tiny CPU model
(``--device cpu``), and runs an extra batched-vs-serial cosine equivalence check
(the batched-rewrite-equivalence requirement) when ``--check-equivalence`` is set.

Usage::

    uv run python scripts/issue763_capture_v0_matched.py --behaviors deception ...
    uv run python scripts/issue763_capture_v0_matched.py --smoke --behaviors deception \
        --device cpu --model-name Qwen/Qwen2.5-0.5B-Instruct --check-equivalence
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import torch  # noqa: E402
from issue594_common import messages_for_instance  # noqa: E402
from issue658_common import EXPECTED_HIDDEN, EXPECTED_LAYERS  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    GEN_DIR,
    BatchedAnswerSpanCapture,
    dump_json,
    ensure_smoke_scope,
    load_json,
    reproducibility_metadata,
    smoke_scoped,
)

logger = logging.getLogger("issue763_capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the batched answer-span offsets are pad_left + prompt_len ..
    tokenizer.padding_side = "left"
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _encode_pair(tokenizer, instance: dict, probe: str, answer: str, max_answer_tokens: int):
    """Return (prompt_ids (Lp,), ans_ids (La,)) with the answer T-capped."""
    messages = messages_for_instance(instance, probe)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]
    ans_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ans_ids.shape[0] > max_answer_tokens:
        ans_ids = ans_ids[:max_answer_tokens]  # adaptive T-cap (plan §4.5)
    return prompt_ids, ans_ids


def _batched_capture_mean(
    model,
    tokenizer,
    capture: BatchedAnswerSpanCapture,
    instance: dict,
    probes: list[str],
    answers: list[str],
    n_layers: int,
    *,
    batch_size: int,
    max_answer_tokens: int,
):
    """Probe-mean of the answer-span mean per layer -> (n_layers, H), matched_n.

    Skips probes with an empty answer span (logged). Per-row answer-span assert
    is implicit in the left-pad offset math (start/end derived from token lengths).
    """
    pad_id = tokenizer.pad_token_id
    device = model.device
    accum = torch.zeros(n_layers, model.config.hidden_size, dtype=torch.float32)
    n_used = 0

    # Build per-probe (prompt_ids, ans_ids); drop empties.
    pairs = []
    for q, a in zip(probes, answers, strict=True):
        prompt_ids, ans_ids = _encode_pair(tokenizer, instance, q, a, max_answer_tokens)
        if ans_ids.shape[0] == 0:
            logger.warning("[capture] empty answer span ctx=%s probe=%r", instance["id"], q[:40])
            continue
        pairs.append((prompt_ids, ans_ids))

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        full_seqs = [torch.cat([p, a]) for p, a in batch]
        max_len = max(s.shape[0] for s in full_seqs)
        b = len(batch)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((b, max_len), dtype=torch.long)
        spans: list[tuple[int, int]] = []
        for r, (full, (p, a)) in enumerate(zip(full_seqs, batch, strict=True)):
            pad_left = max_len - full.shape[0]
            input_ids[r, pad_left:] = full
            attn[r, pad_left:] = 1
            ans_start = pad_left + p.shape[0]
            ans_end = pad_left + p.shape[0] + a.shape[0]
            spans.append((ans_start, ans_end))
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)
        per_row = capture.mean_answer_spans(spans)  # list of (L, H) fp32 CPU
        for vec in per_row:
            accum += vec
            n_used += 1

    if n_used == 0:
        raise RuntimeError(f"ctx {instance['id']}: every probe produced an empty answer span")
    return (accum / n_used), n_used


def _serial_capture_reference(
    model, tokenizer, instance, probe, answer, n_layers, max_answer_tokens
):
    """Batch-1 reference for the equivalence check (the #658 capture math)."""
    from issue763_common import EXPECTED_LAYERS as _L  # noqa: F401

    prompt_ids, ans_ids = _encode_pair(tokenizer, instance, probe, answer, max_answer_tokens)
    full = torch.cat([prompt_ids, ans_ids]).unsqueeze(0).to(model.device)
    cap = BatchedAnswerSpanCapture(model, n_layers)
    with torch.no_grad():
        model(input_ids=full)
    s = prompt_ids.shape[0]
    e = prompt_ids.shape[0] + ans_ids.shape[0]
    vec = cap.mean_answer_spans([(s, e)])[0]
    cap.remove()
    return vec  # (L, H)


def check_batched_serial_equivalence(
    model, tokenizer, instance, probes, answers, n_layers
) -> float:
    """Batched-rewrite-equivalence: cosine(batched_row, serial_row) per (probe, layer).

    Returns the MIN cosine across all (probe × layer). The smoke asserts >= 0.999
    (the batched-rewrite-equivalence requirement: left-pad must not perturb the
    captured residuals vs the batch-1 path).
    """
    import torch.nn.functional as fnl

    cap = BatchedAnswerSpanCapture(model, n_layers)
    # Batched: capture all probes in ONE left-padded forward (the per-(C,B) path).
    pad_id = tokenizer.pad_token_id
    pairs = [
        _encode_pair(tokenizer, instance, q, a, 1024) for q, a in zip(probes, answers, strict=True)
    ]
    pairs = [(p, a) for p, a in pairs if a.shape[0] > 0]
    full_seqs = [torch.cat([p, a]) for p, a in pairs]
    max_len = max(s.shape[0] for s in full_seqs)
    b = len(pairs)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    spans = []
    for r, (full, (p, a)) in enumerate(zip(full_seqs, pairs, strict=True)):
        pad_left = max_len - full.shape[0]
        input_ids[r, pad_left:] = full
        attn[r, pad_left:] = 1
        spans.append((pad_left + p.shape[0], pad_left + p.shape[0] + a.shape[0]))
    with torch.no_grad():
        model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device))
    batched_rows = cap.mean_answer_spans(spans)
    cap.remove()

    min_cos = 1.0
    for (q, a), brow in zip(zip(probes, answers, strict=False), batched_rows, strict=False):
        srow = _serial_capture_reference(model, tokenizer, instance, q, a, n_layers, 1024)
        cos = fnl.cosine_similarity(brow, srow, dim=-1)  # (L,)
        min_cos = min(min_cos, float(cos.min()))
    return min_cos


# ── `neutral-contrast-and-cofit` round: prompt-side c(C,B) capture (plan §4.1.4) ──


def _batched_prompt_mean(
    model, tokenizer, capture, prompt_texts: list[str], *, batch_size: int
) -> tuple[torch.Tensor, int]:
    """Probe-mean of the PROMPT-token mean per layer -> (n_layers, H), n_used.

    Prompt-ONLY forwards (no answer): each row's span is its full non-pad
    region (the chat-templated system prompt + probe + generation header), so
    ``c(C,B)[l]`` mirrors ``v0``'s probe-averaged construction on the prompt
    side. Left-padded batches through the same ``BatchedAnswerSpanCapture``.
    """
    pad_id = tokenizer.pad_token_id
    device = model.device
    n_layers = capture.n_layers
    accum = torch.zeros(n_layers, model.config.hidden_size, dtype=torch.float32)
    n_used = 0
    encoded = [
        tokenizer(t, return_tensors="pt", padding=False)["input_ids"][0] for t in prompt_texts
    ]
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        max_len = max(s.shape[0] for s in batch)
        b = len(batch)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((b, max_len), dtype=torch.long)
        spans: list[tuple[int, int]] = []
        for r, seq in enumerate(batch):
            pad_left = max_len - seq.shape[0]
            input_ids[r, pad_left:] = seq
            attn[r, pad_left:] = 1
            spans.append((pad_left, max_len))
        with torch.no_grad():
            model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        for vec in capture.mean_answer_spans(spans):
            accum += vec
            n_used += 1
    if n_used == 0:
        raise RuntimeError("prompt-span capture: no prompts produced tokens")
    return (accum / n_used), n_used


def _run_prompt_span(args, model, tokenizer, capture, n_layers: int, hidden: int) -> int:
    """--span prompt: c(C,B) prompt-side context capture -> c0 shards (plan §4.1.4).

    ``c(C,B)[l]`` = mean residual over the PROMPT tokens (chat-templated system
    prompt + probe), averaged over the behavior's frozen-pool probes — probe-
    averaged (NOT system-prompt-only) so the construction mirrors ``v0``'s
    probe-averaged answer-side mean. 50 x (60+60+60+20+20) = 11,000 prompt-only
    forwards, batched. Writes ``<COFIT_DIR>/c0_shards/c0_<b>.pt`` [n_ctx, L, H].
    """
    from issue594_common import load_battery
    from issue763_common import COFIT_DIR, load_frozen_pool_staged

    _, instances = load_battery()
    if args.smoke:
        instances = instances[:3]
    shard_dir = COFIT_DIR / "c0_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    c0_meta: dict[str, dict] = {}
    for behavior in args.behaviors:
        probes = load_frozen_pool_staged(behavior)["probes"]
        if args.smoke:
            probes = probes[:5]
        rows, ctx_ids = [], []
        for inst in instances:
            prompt_texts = [
                tokenizer.apply_chat_template(
                    messages_for_instance(inst, probe), tokenize=False, add_generation_prompt=True
                )
                for probe in probes
            ]
            c0, n_used = _batched_prompt_mean(
                model, tokenizer, capture, prompt_texts, batch_size=args.batch_size
            )
            assert c0.shape == (n_layers, hidden), c0.shape
            rows.append(c0)
            ctx_ids.append(inst["id"])
            c0_meta.setdefault(behavior, {})[inst["id"]] = {"n_probes": n_used}
        tensor = torch.stack(rows)  # (n_ctx, n_layers, H)
        torch.save(
            {"tensor": tensor, "context_ids": ctx_ids, "behavior": behavior, "span": "prompt"},
            shard_dir / f"c0_{behavior}.pt",
        )
        logger.info("[capture:prompt] %s -> c0 shard %s", behavior, tuple(tensor.shape))
    out = {
        "n_layers": n_layers,
        "hidden": hidden,
        "span": "prompt",
        "by_behavior": c0_meta,
        "metadata": reproducibility_metadata({"phase": "capture_context_side"}),
    }
    dump_json(out, COFIT_DIR / "c0_matched_by_behavior.json")
    capture.remove()
    print(f"[issue763.capture] wrote c0 for {len(c0_meta)} behaviors ({len(instances)} contexts)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: matched-probe v0 capture.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-answer-tokens", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-equivalence", action="store_true")
    ap.add_argument(
        "--span",
        choices=["answer", "prompt"],
        default="answer",
        help="answer = the v0 answer-span capture (default); prompt = the round's "
        "prompt-side c(C,B) capture (plan §4.1.4)",
    )
    args = ap.parse_args()
    # FIRST: --smoke re-execs with EPM_ISSUE763_SMOKE_SCOPE=1 (write paths
    # rebind under smoke_scope/); the env WITHOUT --smoke fails loud.
    ensure_smoke_scope(args.smoke)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    model, tokenizer = _load_model(args.model_name, device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    if not args.smoke:
        assert n_layers == EXPECTED_LAYERS, f"expected {EXPECTED_LAYERS} layers, got {n_layers}"
        assert hidden == EXPECTED_HIDDEN, f"expected hidden {EXPECTED_HIDDEN}, got {hidden}"

    capture = BatchedAnswerSpanCapture(model, n_layers)

    if args.span == "prompt":
        return _run_prompt_span(args, model, tokenizer, capture, n_layers, hidden)

    # smoke_scoped: a --span answer --smoke run must never clobber the REAL v0
    # shards at the canonical path (same class as review r1 C1(iii)).
    shard_dir = smoke_scoped(EVAL_RESULTS_DIR / "v0_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    v0_meta: dict[str, dict] = {}
    equivalence_min_cos: float | None = None

    for behavior in args.behaviors:
        gen_dir = GEN_DIR / behavior
        if not gen_dir.is_dir():
            raise FileNotFoundError(f"no generated completions for {behavior}: {gen_dir}")
        ctx_files = sorted(gen_dir.glob("*.json"))
        rows = []
        ctx_ids = []
        for cf in ctx_files:
            gen = load_json(cf)
            ctx_id = gen["context_id"]
            # the battery instance for messages_for_instance — reconstruct the
            # minimal instance shape (id + system_prompt + prefix_messages) from
            # the battery; we re-load it lazily from the battery below.
            probes = [c["probe"] for c in gen["cells"]]
            # one completion per probe (n_samples=1 for all 5 behaviors)
            answers = [c["completions"][0]["text"] for c in gen["cells"]]
            instance = _instance_for_ctx(ctx_id)
            if args.check_equivalence and equivalence_min_cos is None:
                equivalence_min_cos = check_batched_serial_equivalence(
                    model, tokenizer, instance, probes[:3], answers[:3], n_layers
                )
                logger.info("[equivalence] min cosine(batched, serial) = %.6f", equivalence_min_cos)
            v0, matched_n = _batched_capture_mean(
                model,
                tokenizer,
                capture,
                instance,
                probes,
                answers,
                n_layers,
                batch_size=args.batch_size,
                max_answer_tokens=args.max_answer_tokens,
            )
            assert v0.shape == (n_layers, hidden), v0.shape
            # matched-probe invariant: matched_n must equal the number of
            # non-empty-answer probes (the join is exact by construction).
            assert matched_n <= len(probes), (matched_n, len(probes))
            rows.append(v0)
            ctx_ids.append(ctx_id)
            v0_meta.setdefault(behavior, {})[ctx_id] = {
                "matched_n": matched_n,
                "n_probes": len(probes),
            }
        tensor = torch.stack(rows)  # (n_ctx, n_layers, H)
        torch.save(
            {"tensor": tensor, "context_ids": ctx_ids, "behavior": behavior},
            shard_dir / f"v0_{behavior}.pt",
        )
        logger.info("[capture] %s -> v0 shard %s", behavior, tuple(tensor.shape))

    out = {
        "n_layers": n_layers,
        "hidden": hidden,
        "by_behavior": v0_meta,
        "equivalence_min_cosine": equivalence_min_cos,
        "metadata": reproducibility_metadata({"phase": "capture"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "v0_matched_by_behavior.json")
    capture.remove()

    if args.check_equivalence and equivalence_min_cos is not None:
        assert equivalence_min_cos >= 0.999, (
            f"batched-vs-serial equivalence FAILED: min cosine {equivalence_min_cos:.6f} < 0.999 "
            "(left-pad perturbed the captured residuals)"
        )
    print(
        f"[issue763.capture] wrote v0 for {len(v0_meta)} behaviors; min_cos={equivalence_min_cos}"
    )
    return 0


_BATTERY_CACHE: dict[str, dict] = {}


def _instance_for_ctx(ctx_id: str) -> dict:
    """Resolve a context id to its battery instance (id/system_prompt/prefix)."""
    if not _BATTERY_CACHE:
        from issue594_common import load_battery

        _, instances = load_battery()
        for inst in instances:
            _BATTERY_CACHE[inst["id"]] = inst
    if ctx_id not in _BATTERY_CACHE:
        raise KeyError(f"context {ctx_id!r} not in the 50-context battery")
    return _BATTERY_CACHE[ctx_id]


if __name__ == "__main__":
    raise SystemExit(main())
